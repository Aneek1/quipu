"""Run a StepModel over the benchmark apps and score it.

    python -m stepbuild.bench.run --model reference [--apps todo notes]
        [--out results/stepbuild] [--library data/stepbuild/train-000.jsonl ...]

For each app: a fresh sandbox (create_sandbox with the shared npm cache) ->
run_app -> if every step passed, the hidden acceptance tests (run_acceptance) ->
a trace written atomically to out/<model>/<app>.json (the runner's full trace plus
the acceptance result) -> the sandbox removed with remove_sandbox, always, in a
finally, so a crash never leaves a sandbox (or a link into the npm cache) behind.
After each app out/<model>/summary.md is rewritten, so an interrupted run still
leaves a summary of the apps it finished.

Per-app models. A model that needs to know which app it is serving (the
ReferenceModel, which replays that app's reference replies) exposes
`for_app(app) -> StepModel`; run_bench calls it once per app when present and
uses the returned model for that app, and uses the model itself otherwise. The
hook is optional so an ordinary backend (a Qwen, Quipu) is just a StepModel and
knows nothing about the benchmark; the per-app model's calls stay separate, so
one app's replies can never bleed into another's.

Leakage. The example library shown to the model must never contain a benchmark
app's own reference solution, or the benchmark would measure copying. The
library is meant to be the dataset's train split only (--library loads split
"train" from the given JSONL shards, and defaults to empty); on top of that,
run_bench refuses (LeakageError) any library example whose reply equals any
app's reference reply for any step, whichever apps are being run. Replies are
compared with line endings normalised and outer whitespace stripped; this
catches a verbatim copy, not a near-copy.

Backends and the reply budget. The runner sizes prompts assuming a reply never
exceeds its reply_reserve (runner.REPLY_RESERVE), so a real backend must cap its
max new tokens at that value. The ReferenceModel has no context window.

Scores (summarise), computed from the per-app records:
- apps passing acceptance: apps whose every step passed AND whose acceptance
  tests passed, over all apps run;
- first-try step pass rate: model steps (not the checks-only run step) that
  passed on their first attempt, over the model steps that ran; steps never
  reached after an earlier failure are not counted;
- mean retries per model step: attempts beyond the first, over the model steps
  that ran;
- failing steps: for each app that did not fully pass, the key of the step it
  failed at, or "acceptance" when every step passed but acceptance did not;
- wall time: total and mean seconds per app (the runner's time plus acceptance).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from quipu.fsio import write_text_atomic
from stepbuild.bench.acceptance import list_apps, load_reference, load_spec, run_acceptance
from stepbuild.harness.model import ScriptedModel, StepModel
from stepbuild.harness.plan import make_plan
from stepbuild.harness.retrieve import ExampleLibrary
from stepbuild.harness.runner import result_to_dict, run_app
from stepbuild.harness.sandbox import create_sandbox, default_cache_dir, remove_sandbox

DEFAULT_OUT = Path("results") / "stepbuild"
SUMMARY_NAME = "summary.md"


class LeakageError(ValueError):
    """The example library contains a benchmark app's reference reply."""


class ReferenceModel:
    """Replays an app's reference replies (steps 1..5) in order: a perfect model,
    used to prove the benchmark end to end (it must score 100%).

    ReferenceModel() is unbound; run_bench calls for_app(app) to get a model bound
    to one app. An unbound model refuses to answer, and a bound one raises
    RuntimeError if asked for more replies than the reference has (a retry would
    mean the reference failed a step, and replaying a later step's reply in its
    place would only hide that).
    """

    name = "reference"
    context_tokens: int | None = None

    def __init__(self, app: str | None = None) -> None:
        self.app = app
        self._script = None if app is None else ScriptedModel(
            load_reference(app), name=self.name
        )

    def for_app(self, app: str) -> "ReferenceModel":
        return ReferenceModel(app)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self._script is None:
            raise RuntimeError(
                "ReferenceModel is not bound to an app; use ReferenceModel().for_app(app)"
            )
        return self._script.complete(messages)


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def check_no_leakage(library: ExampleLibrary) -> None:
    """Raise LeakageError if any library reply equals any app's reference reply."""
    refs: dict[str, str] = {}
    for app in list_apps():
        for n, reply in enumerate(load_reference(app), start=1):
            refs.setdefault(_norm(reply), f"{app} step {n}")
    for i, example in enumerate(library.examples):
        where = refs.get(_norm(example.reply))
        if where is not None:
            raise LeakageError(
                f"library example {i} ({example.step[:60]!r}) is the reference reply of "
                f"{where}; the example library must be the dataset's train split only"
            )


def _step_summary(result, plan) -> list[dict[str, Any]]:
    model_steps = {s.number: s.model_step for s in plan.steps}
    return [
        {
            "number": s.number,
            "key": s.key,
            "model_step": model_steps[s.number],
            "passed": s.passed,
            "attempts": len(s.attempts),
        }
        for s in result.steps
    ]


def _select_apps(apps: Sequence[str] | None) -> list[str]:
    known = list_apps()
    if apps is None:
        return known
    if isinstance(apps, str):
        raise TypeError("apps must be a sequence of app names, not a single str")
    unknown = [a for a in apps if a not in known]
    if unknown:
        raise ValueError(f"unknown app(s) {unknown}; known apps: {', '.join(known)}")
    return list(apps)


def run_bench(
    model: StepModel,
    apps: Sequence[str] | None,
    out_dir: Path,
    library: ExampleLibrary,
    *,
    max_retries: int = 3,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `model` over `apps` (None = all) and return one record per app; see the
    module docstring. Writes out_dir/<model.name>/<app>.json and summary.md."""
    selected = _select_apps(apps)
    check_no_leakage(library)  # before any sandbox work
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    model_dir = Path(out_dir) / model.name
    records: list[dict[str, Any]] = []
    work = Path(tempfile.mkdtemp(prefix="stepbuild-bench-"))
    try:
        for app in selected:
            app_model = model.for_app(app) if hasattr(model, "for_app") else model
            plan = make_plan(app, load_spec(app))
            start = time.monotonic()
            sandbox = create_sandbox(work, cache_dir)
            try:
                result = run_app(app_model, plan, sandbox, library, max_retries=max_retries)
                acceptance = (
                    run_acceptance(app, sandbox.root)
                    if result.status == "passed_steps"
                    else None
                )
            finally:
                remove_sandbox(sandbox)
            seconds = time.monotonic() - start
            trace_path = model_dir / f"{app}.json"
            trace = result_to_dict(result)
            trace["acceptance"] = None if acceptance is None else dataclasses.asdict(acceptance)
            trace["total_seconds"] = seconds
            write_text_atomic(
                trace_path, json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
            )
            records.append({
                "app": app,
                "model": model.name,
                "status": result.status,
                "acceptance_passed": None if acceptance is None else acceptance.passed,
                "steps": _step_summary(result, plan),
                "seconds": seconds,
                "trace": str(trace_path),
            })
            write_text_atomic(model_dir / SUMMARY_NAME, summarise(records))
    finally:
        try:
            os.rmdir(work)  # empty unless a sandbox removal failed; then leave it
        except OSError:
            pass
    return records


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / den:.1f}%)" if den else f"{num}/{den} (n/a)"


def _failing_step(record: dict[str, Any]) -> str | None:
    if record["status"] == "passed_steps":
        return None if record["acceptance_passed"] else "acceptance"
    failed = [s for s in record["steps"] if not s["passed"]]
    return failed[0]["key"] if failed else record["status"]


def summarise(records: Sequence[dict[str, Any]]) -> str:
    """Markdown: one row per app, then the totals (see the module docstring)."""
    models = sorted({r["model"] for r in records})
    lines = [f"# Stepbuild benchmark: {', '.join(models) or '-'}", ""]
    if not records:
        return "\n".join(lines + ["no apps were run", ""])
    lines += [
        "| app | status | acceptance | attempts per step | retries | seconds |",
        "|---|---|---|---|---|---|",
    ]
    model_steps = first_try = retries = full_pass = 0
    failing: Counter[str] = Counter()
    total_seconds = 0.0
    for r in records:
        steps = r["steps"]
        ms = [s for s in steps if s["model_step"]]
        app_retries = sum(s["attempts"] - 1 for s in ms)
        model_steps += len(ms)
        first_try += sum(1 for s in ms if s["passed"] and s["attempts"] == 1)
        retries += app_retries
        acc = r["acceptance_passed"]
        if r["status"] == "passed_steps" and acc:
            full_pass += 1
        key = _failing_step(r)
        if key is not None:
            failing[key] += 1
        total_seconds += r["seconds"]
        acc_text = "-" if acc is None else ("pass" if acc else "fail")
        attempts = " ".join(str(s["attempts"]) for s in steps)
        lines.append(
            f"| {r['app']} | {r['status']} | {acc_text} | {attempts} | {app_retries} "
            f"| {r['seconds']:.1f} |"
        )
    n = len(records)
    mean_retries = retries / model_steps if model_steps else 0.0
    hist = ", ".join(f"{k} {v}" for k, v in sorted(failing.items())) or "none"
    lines += [
        "",
        "## Totals",
        "",
        f"- Apps passing acceptance: {_pct(full_pass, n)}",
        f"- First-try step pass rate (model steps): {_pct(first_try, model_steps)}",
        f"- Mean retries per model step: {mean_retries:.2f} "
        f"({retries} retries over {model_steps} model steps)",
        f"- Failing steps: {hist}",
        f"- Wall time: {total_seconds:.1f} s total, {total_seconds / n:.1f} s mean per app",
        "",
    ]
    return "\n".join(lines)


_MODELS = {"reference": ReferenceModel}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stepbuild.bench.run",
        description="Run a model over the stepbuild benchmark apps and score it.",
    )
    parser.add_argument("--model", required=True, choices=sorted(_MODELS))
    parser.add_argument("--apps", nargs="+", default=None, help="default: all apps")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--library", type=Path, nargs="+", default=[],
        help="dataset JSONL shards; only the train split is loaded (default: no examples)",
    )
    args = parser.parse_args(argv)
    library = ExampleLibrary.from_jsonl(args.library, split="train")
    records = run_bench(_MODELS[args.model](), args.apps, args.out, library)
    sys.stdout.write(summarise(records))
    sys.stdout.flush()
    ok = bool(records) and all(
        r["status"] == "passed_steps" and r["acceptance_passed"] for r in records
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
