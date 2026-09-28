"""Run a StepModel over the benchmark apps and score it.

    python -m stepbuild.bench.run --model reference [--apps todo notes]
        [--out results/stepbuild] [--library data/stepbuild/train-000.jsonl ...]
        [--max-retries 3] [--require-pass]

For each app: a fresh sandbox (create_sandbox with the shared npm cache) ->
run_app -> if every step passed, the hidden acceptance tests (run_acceptance) ->
a trace written atomically to out/<model slug>/<app>.json -> the sandbox removed
with remove_sandbox in a finally. After each app out/<model slug>/summary.md is
rewritten, so an interrupted run still leaves a summary of the apps it finished.

Harness errors. Anything that raises while an app is being built (the sandbox,
the runner, the acceptance run) is recorded for that app as status
"harness_error" (the exception and traceback go in its JSON) and the bench moves
on to the next app, as spec §4 asks: one broken app must not hide the scores of
the others. A harness_error app counts as not passing. KeyboardInterrupt is not
caught. If remove_sandbox itself fails, the failure is logged and recorded
(`cleanup_error`) and the run continues; it never replaces the app's own error.

Per-app models. A model that needs to know which app it is serving (the
ReferenceModel, which replays that app's reference replies) exposes
`for_app(app) -> StepModel`; run_bench calls it once per app when present and
uses the returned model for that app, and uses the model itself otherwise. The
hook is optional so an ordinary backend (a Qwen, Quipu) is just a StepModel and
knows nothing about the benchmark.

Leakage. The example library shown to the model must never contain a benchmark
app's own reference solution, or the benchmark would measure copying. The
library is meant to be the dataset's train split only (--library loads split
"train" from the given JSONL shards, and defaults to empty). On top of that,
run_bench refuses (LeakageError) any library example that, against ANY app's
reference replies (not just the apps being run):

- equals a reference reply once all whitespace runs are collapsed (so
  re-indented or re-wrapped copies are caught), or
- has a FILE block whose 5-token-shingle Jaccard similarity with a reference
  block for the same path is >= LEAK_JACCARD (0.9). Tokens are identifiers,
  numbers and single punctuation characters, so layout does not matter.
  Calibration on the reference solutions themselves: the most similar blocks of
  two different apps score 0.815 (notes vs todo models.py), and a re-indented
  copy scores 1.0. (todo and todo_auth share byte-identical List/Form components;
  they are one solution, not two apps that happen to agree.)

What is NOT caught: a copy with its identifiers renamed (a models.py with every
`note` changed to `memo`) changes most shingles and passes. The train-split rule
is the real protection; this guard catches accidents.

Backends and the reply budget. The runner sizes prompts assuming a reply never
exceeds its reply_reserve (runner.REPLY_RESERVE), so a real backend must cap its
max new tokens at that value; the provenance header records it as "max new
tokens". The ReferenceModel has no context window.

Provenance. Every summary.md starts with, and every per-app JSON carries under
"run", what produced the numbers: the model (raw name, backend class, context
window, max new tokens), which apps ran (with a SUBSET warning when not all),
the example library (split, shards with sha256, example count, leakage guard
result), the harness (git commit, "-dirty" if tracked files are modified,
retry and token budgets, examples per prompt), the machine (start/end times with
timezone, hostname, GPU, Python and Node versions) and the command line. A
summary reflects only the records of the run that wrote it.

Scores (summarise), for N apps with M = 5 model steps each (the checks-only run
step is not a model step):
- Apps passing acceptance (headline): status passed_steps AND acceptance passed,
  over N (a harness_error app is not passing);
- Apps completing all steps: status passed_steps, over N;
- Steps passed: model steps passed on any attempt, over N*M (a step never reached
  counts as not passed);
- Steps passed first try: model steps passed on attempt 1, over N*M;
- Steps reached: model steps attempted at least once, over N*M;
- First-try rate among reached steps: conditional on the step being reached;
- Mean retries per passed step: attempts beyond the first, summed over passed
  model steps, over the passed model steps;
- Steps that exhausted the retry budget: failed model steps with 1 + max_retries
  attempts;
- Attempts lost to infrastructure: model-step attempts that failed with
  model_error or prompt_too_long, over all model-step attempts (flagged when not
  zero, since they measure the backend, not the model);
- Where non-passing apps stopped: "step N key", "acceptance (all steps
  passed)" or "harness_error", with counts;
- Wall time (steps + acceptance): from sandbox setup to the end of acceptance,
  total and mean per app.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from quipu.fsio import write_text_atomic
from stepbuild.bench.acceptance import list_apps, load_reference, load_spec, run_acceptance
from stepbuild.harness.blocks import BlockError, parse_blocks
from stepbuild.harness.model import ScriptedModel, StepModel
from stepbuild.harness.plan import make_plan
from stepbuild.harness.prompt import FEEDBACK_MAX_TOKENS
from stepbuild.harness.retrieve import ExampleLibrary
from stepbuild.harness.runner import (
    N_EXAMPLES,
    REPLY_RESERVE,
    RETRY_RESERVE,
    result_to_dict,
    run_app,
)
from stepbuild.harness.sandbox import create_sandbox, default_cache_dir, remove_sandbox

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("results") / "stepbuild"
SUMMARY_NAME = "summary.md"
HARNESS_ERROR = "harness_error"
INFRA_CHECKS = ("model_error", "prompt_too_long")
N_MODEL_STEPS = sum(1 for s in make_plan("app", "spec").steps if s.model_step)
LEAK_JACCARD = 0.9
SHINGLE = 5
REPO_ROOT = Path(__file__).resolve().parents[2]

_TOKEN = re.compile(r"\w+|[^\w\s]")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


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


# ---------------------------------------------------------------- leakage guard

def _collapse(text: str) -> str:
    return " ".join(text.split())


def _shingles(text: str) -> frozenset[tuple[str, ...]]:
    tokens = _TOKEN.findall(text)
    if len(tokens) < SHINGLE:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i:i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _blocks(reply: str):
    try:
        return parse_blocks(reply)
    except BlockError:
        return []  # not FILE blocks: only the whole-reply comparison applies


class LeakageGuard:
    """The reference index of the leakage guard (see the module docstring), built
    once from every app's reference replies, so one guard can screen many replies:
    check_no_leakage screens a library with it, and the dataset build
    (stepbuild.dataset.build) skips mined examples it flags, so a leaked reference
    never reaches the shards in the first place."""

    def __init__(self) -> None:
        self._whole: dict[str, str] = {}
        self._by_path: dict[str, list[tuple[str, frozenset]]] = {}
        for app in list_apps():
            for n, reply in enumerate(load_reference(app), start=1):
                where = f"{app} step {n}"
                self._whole.setdefault(_collapse(reply), where)
                for block in _blocks(reply):
                    self._by_path.setdefault(block.path, []).append(
                        (where, _shingles(block.content))
                    )

    def find(self, reply: str) -> str | None:
        """None if `reply` is clean, else what it copies, worded to follow a label
        (" is the reference reply of ..." or ": its <path> is a near-copy ...")."""
        where = self._whole.get(_collapse(reply))
        if where is not None:
            return f" is the reference reply of {where} (whitespace ignored)"
        for block in _blocks(reply):
            refs = self._by_path.get(block.path)
            if not refs:
                continue
            mine = _shingles(block.content)
            for where, theirs in refs:
                score = jaccard(mine, theirs)
                if score >= LEAK_JACCARD:
                    return (
                        f": its {block.path} is a near-copy of the reference for "
                        f"{where} ({SHINGLE}-shingle Jaccard {score:.3f} >= {LEAK_JACCARD})"
                    )
        return None


def check_no_leakage(library: ExampleLibrary) -> str:
    """Raise LeakageError if any library example copies a reference reply (see the
    module docstring); otherwise return a one-line description of what was checked."""
    guard = LeakageGuard()
    for i, example in enumerate(library.examples):
        found = guard.find(example.reply)
        if found is not None:
            label = f"library example {i} ({example.step[:60]!r})"
            raise LeakageError(
                f"{label}{found}; the example library must be the dataset's train split only"
            )
    return (
        f"passed ({len(library.examples)} examples vs every app's reference: whitespace-"
        f"collapsed exact match, per-file {SHINGLE}-shingle Jaccard >= {LEAK_JACCARD})"
    )


# ---------------------------------------------------------------- provenance

def model_slug(name: str) -> str:
    """The model name made safe as one directory name ("Qwen/Qwen2.5" -> "Qwen_Qwen2.5")."""
    slug = _UNSAFE.sub("_", name or "")
    if slug in ("", ".", ".."):
        raise ValueError(f"model name {name!r} cannot be used as a results directory")
    return slug


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _run_text(cmd: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            list(cmd), cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown (git not found)"
    head = _run_text([git, "rev-parse", "--short=12", "HEAD"], REPO_ROOT)
    if not head:
        return "unknown (not a git checkout)"
    dirty = _run_text([git, "status", "--porcelain", "--untracked-files=no"], REPO_ROOT)
    return head + ("-dirty" if dirty else "")


def gpu_name() -> str:
    """The GPU the model could use. torch is asked only if something already
    imported it (a CPU-only reference run should not pay for importing it);
    otherwise nvidia-smi; otherwise "cpu". A hidden GPU (CUDA_VISIBLE_DEVICES
    empty) is reported as cpu, since nothing in the run can use it."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu (CUDA_VISIBLE_DEVICES is empty)"
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return torch.cuda.get_device_name(0)
            return "cpu"
        except Exception:
            pass
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run_text([smi, "--query-gpu=name", "--format=csv,noheader"])
        if out:
            return out.splitlines()[0].strip()
    return "cpu"


def node_version() -> str:
    node = shutil.which("node")
    return (_run_text([node, "--version"]) if node else None) or "not found"


def shard_info(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Path and short sha256 of each library shard, so a summary names its data."""
    info = []
    for p in paths:
        digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
        info.append({"path": str(p), "sha256": digest})
    return info


def _backend(model: StepModel) -> str:
    """The model's class as module.qualname. Under `python -m stepbuild.bench.run`
    the built-in models live in __main__, so name the module it was run as."""
    cls = type(model)
    module = cls.__module__
    if module == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        module = getattr(spec, "name", None) or module
    return f"{module}.{cls.__qualname__}"


# ---------------------------------------------------------------- running

def _step_summary(result, plan) -> list[dict[str, Any]]:
    model_steps = {s.number: s.model_step for s in plan.steps}
    out = []
    for s in result.steps:
        kinds = Counter(c.name for a in s.attempts for c in a.checks if not c.passed)
        out.append({
            "number": s.number,
            "key": s.key,
            "model_step": model_steps[s.number],
            "passed": s.passed,
            "attempts": len(s.attempts),
            "failure_kinds": dict(sorted(kinds.items())),
        })
    return out


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


def _error_info(e: BaseException) -> dict[str, str]:
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": "".join(traceback.format_exception(e)),
    }


def run_bench(
    model: StepModel,
    apps: Sequence[str] | None,
    out_dir: Path,
    library: ExampleLibrary,
    *,
    max_retries: int = 3,
    cache_dir: Path | None = None,
    library_info: dict[str, Any] | None = None,
    command: str | None = None,
) -> list[dict[str, Any]]:
    """Run `model` over `apps` (None = all) and return one record per app; see the
    module docstring. Writes out_dir/<model slug>/<app>.json and summary.md.

    `library_info` ({"split", "shards": shard_info(...)}) and `command` only feed
    the provenance header; main() passes them."""
    selected = _select_apps(apps)
    model_dir = Path(out_dir) / model_slug(model.name)
    leakage = check_no_leakage(library)  # before any sandbox work
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    lib = dict(library_info or {"split": None, "shards": []})
    lib.update(examples=len(library.examples), leakage_guard=leakage)
    run: dict[str, Any] = {
        "model": {
            "name": model.name,
            "backend": _backend(model),
            "context_tokens": getattr(model, "context_tokens", None),
            "max_new_tokens": REPLY_RESERVE,
        },
        "apps": {"run": selected, "available": len(list_apps())},
        "library": lib,
        "harness": {
            "git_commit": git_commit(),
            "max_retries": max_retries,
            "reply_reserve": REPLY_RESERVE,
            "retry_reserve": RETRY_RESERVE,
            "feedback_max_tokens": FEEDBACK_MAX_TOKENS,
            "examples_per_prompt": N_EXAMPLES,
        },
        "environment": {
            "started": _now(),
            "ended": None,
            "hostname": socket.gethostname(),
            "gpu": gpu_name(),
            "python": platform.python_version(),
            "node": node_version(),
        },
        "command": command or "stepbuild.bench.run.run_bench() (library call)",
    }
    records: list[dict[str, Any]] = []
    work = Path(tempfile.mkdtemp(prefix="stepbuild-bench-"))
    try:
        for i, app in enumerate(selected):
            records.append(_run_one(model, app, library, max_retries, cache_dir, work,
                                    model_dir, run))
            if i == len(selected) - 1:
                run["environment"]["ended"] = _now()
            write_text_atomic(model_dir / SUMMARY_NAME, summarise(records, run))
    finally:
        try:
            os.rmdir(work)  # empty unless a sandbox removal failed; then leave it
        except OSError:
            pass
    return records


def _run_one(model, app, library, max_retries, cache_dir, work, model_dir, run):
    app_model = model.for_app(app) if hasattr(model, "for_app") else model
    start = time.monotonic()
    sandbox = result = acceptance = error = cleanup_error = None
    plan = None
    try:
        plan = make_plan(app, load_spec(app))
        sandbox = create_sandbox(work, cache_dir)
        result = run_app(app_model, plan, sandbox, library, max_retries=max_retries)
        if result.status == "passed_steps":
            acceptance = run_acceptance(app, sandbox.root)
    except Exception as e:  # KeyboardInterrupt still stops the whole bench
        error = _error_info(e)
        log.error("harness error in app %s: %s: %s", app, error["type"], error["message"])
    finally:
        if sandbox is not None:
            try:
                remove_sandbox(sandbox)
            except Exception as e:  # never mask the app's own outcome or error
                cleanup_error = f"{type(e).__name__}: {e}"
                log.warning("could not remove sandbox %s: %s", sandbox.root, cleanup_error)
    total = time.monotonic() - start
    status = HARNESS_ERROR if error else result.status
    trace_path = model_dir / f"{app}.json"
    trace: dict[str, Any] = {
        "run": run,
        "app": app,
        "model": model.name,
        "status": status,
        "error": error,
        "cleanup_error": cleanup_error,
        "run_seconds": None if error else result.seconds,
        "total_seconds": total,
        "steps": [] if error else result_to_dict(result)["steps"],
        "acceptance": (
            None if error or acceptance is None else dataclasses.asdict(acceptance)
        ),
    }
    write_text_atomic(trace_path, json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    return {
        "app": app,
        "model": model.name,
        "status": status,
        "acceptance_passed": None if error or acceptance is None else acceptance.passed,
        "steps": [] if error else _step_summary(result, plan),
        "max_retries": max_retries,
        "error": None if error is None else f"{error['type']}: {error['message']}",
        "cleanup_error": cleanup_error,
        "run_seconds": None if error else result.seconds,
        "total_seconds": total,
        "trace": str(trace_path),
    }


# ---------------------------------------------------------------- scores

def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0 * num / den:.1f}%)" if den else f"{num}/{den} (n/a)"


def passed(record: dict[str, Any]) -> bool:
    return record["status"] == "passed_steps" and record["acceptance_passed"] is True


def stopped_at(record: dict[str, Any]) -> str | None:
    """Where a non-passing app stopped, or None if it passed."""
    if passed(record):
        return None
    if record["status"] == HARNESS_ERROR:
        return HARNESS_ERROR
    if record["status"] == "passed_steps":
        return "acceptance (all steps passed)"
    failed = [s for s in record["steps"] if not s["passed"]]
    return f"step {failed[0]['number']} {failed[0]['key']}" if failed else record["status"]


def _stop_order(label: str) -> tuple:
    m = re.match(r"step (\d+) ", label)
    if m:
        return (0, int(m.group(1)), label)
    return (1 if label.startswith("acceptance") else 2, 0, label)


def _header(run: dict[str, Any], n_records: int) -> list[str]:
    m, apps, lib = run["model"], run["apps"], run["library"]
    h, env = run["harness"], run["environment"]
    selected = apps["run"]
    subset = (
        f" -- SUBSET: only {len(selected)} of {apps['available']} apps; "
        "not comparable with a full run"
        if len(selected) < apps["available"] else ""
    )
    shards = ", ".join(f"{s['path']} (sha256 {s['sha256']})" for s in lib["shards"]) or "none"
    ended = env["ended"] or f"in progress ({n_records} of {len(selected)} apps done)"
    return [
        "## Run",
        "",
        f"- Model: {m['name']} (backend {m['backend']}; context tokens "
        f"{m['context_tokens']}; max new tokens {m['max_new_tokens']})",
        f"- Apps run: {len(selected)} of {apps['available']} ({', '.join(selected)}){subset}",
        f"- Example library: split {lib['split'] or 'n/a'}; {lib['examples']} examples; "
        f"shards: {shards}; leakage guard: {lib['leakage_guard']}",
        f"- Harness: commit {h['git_commit']}; max_retries {h['max_retries']}; "
        f"reply_reserve {h['reply_reserve']}; retry_reserve {h['retry_reserve']}; "
        f"feedback_max_tokens {h['feedback_max_tokens']}; examples per prompt "
        f"{h['examples_per_prompt']}",
        f"- Machine: {env['hostname']}; GPU {env['gpu']}; Python {env['python']}; "
        f"Node {env['node']}",
        f"- Started {env['started']}; ended {ended}",
        f"- Command: `{run['command']}`",
        "",
    ]


def summarise(records: Sequence[dict[str, Any]], run: dict[str, Any] | None = None) -> str:
    """Markdown: provenance (when `run` is given), one row per app, then the totals
    (see the module docstring). Only `records` are counted."""
    models = sorted({r["model"] for r in records})
    if run is not None:
        models = [run["model"]["name"]]
    lines = [f"# Stepbuild benchmark: {', '.join(models) or '-'}", ""]
    lines.append(
        f"Scope: only the {len(records)} app record(s) of this run; any other files "
        "in this directory are from earlier runs and are not counted."
    )
    lines.append("")
    if run is not None:
        lines += _header(run, len(records))
    if not records:
        return "\n".join(lines + ["no apps were run", ""])
    lines += [
        "## Apps",
        "",
        "| app | status | acceptance | steps passed (of 5) | stopped at "
        "| attempts per step | retries | seconds (steps + acceptance) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n = len(records)
    total_steps = n * N_MODEL_STEPS
    s_passed = s_first = s_reached = passed_retries = exhausted = 0
    attempts_all = attempts_infra = 0
    full_pass = completed = 0
    stops: Counter[str] = Counter()
    errors: list[str] = []
    total_seconds = 0.0
    for r in records:
        ms = [s for s in r["steps"] if s["model_step"]]
        app_passed = sum(1 for s in ms if s["passed"])
        s_passed += app_passed
        s_first += sum(1 for s in ms if s["passed"] and s["attempts"] == 1)
        s_reached += sum(1 for s in ms if s["attempts"] >= 1)
        passed_retries += sum(s["attempts"] - 1 for s in ms if s["passed"])
        exhausted += sum(
            1 for s in ms if not s["passed"] and s["attempts"] >= 1 + r["max_retries"]
        )
        attempts_all += sum(s["attempts"] for s in ms)
        attempts_infra += sum(
            s.get("failure_kinds", {}).get(k, 0) for s in ms for k in INFRA_CHECKS
        )
        if passed(r):
            full_pass += 1
        if r["status"] == "passed_steps":
            completed += 1
        stop = stopped_at(r)
        if stop is not None:
            stops[stop] += 1
        if r["status"] == HARNESS_ERROR:
            errors.append(f"{r['app']} ({r.get('error') or 'unknown error'})")
        total_seconds += r["total_seconds"]
        acc = r["acceptance_passed"]
        acc_text = "-" if acc is None else ("pass" if acc else "fail")
        attempts = " ".join(str(s["attempts"]) for s in r["steps"]) or "-"
        app_retries = sum(s["attempts"] - 1 for s in ms)
        lines.append(
            f"| {r['app']} | {r['status']} | {acc_text} | {app_passed} | {stop or '-'} "
            f"| {attempts} | {app_retries} | {r['total_seconds']:.1f} |"
        )
    mean_retries = f"{passed_retries / s_passed:.2f}" if s_passed else "n/a"
    stop_text = ", ".join(
        f"{k}: {stops[k]}" for k in sorted(stops, key=_stop_order)
    ) or "none"
    infra = (
        f"Attempts lost to infrastructure (model_error, prompt_too_long): "
        f"{_pct(attempts_infra, attempts_all)}"
    )
    if attempts_infra:
        infra = "WARNING: " + infra + " -- these measure the backend, not the model"
    lines += [
        "",
        "## Totals",
        "",
        f"- **Apps passing acceptance: {_pct(full_pass, n)}**",
        f"- Apps completing all steps: {_pct(completed, n)}",
        f"- Steps passed (of all {total_steps}): {_pct(s_passed, total_steps)}",
        f"- Steps passed first try (of all {total_steps}): {_pct(s_first, total_steps)}",
        f"- Steps reached (of all {total_steps}): {_pct(s_reached, total_steps)}",
        f"- First-try rate among reached steps (conditional): {_pct(s_first, s_reached)}",
        f"- Mean retries per passed step: {mean_retries} "
        f"({passed_retries} retries over {s_passed} passed steps)",
        f"- Steps that exhausted the retry budget: {exhausted}",
        f"- {infra}",
        f"- Where non-passing apps stopped: {stop_text}",
        f"- Harness errors: {'; '.join(errors) if errors else 'none'}",
        f"- Wall time (steps + acceptance, incl. sandbox setup): {total_seconds:.1f} s "
        f"total, {total_seconds / n:.1f} s mean per app",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

_MODELS = {"reference": ReferenceModel}


def main(argv: Sequence[str] | None = None) -> int:
    """Exit codes: 0 the run completed cleanly; 1 with --require-pass when any app
    did not pass acceptance; 2 for a leakage refusal, an unreadable library, or any
    harness_error (reported as one line on stderr, no traceback)."""
    argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--require-pass", action="store_true",
        help="exit 1 unless every app passes acceptance",
    )
    args = parser.parse_args(argv)
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.apps is not None:
        unknown = [a for a in args.apps if a not in list_apps()]
        if unknown:
            parser.error(f"unknown app(s) {unknown}; known: {', '.join(list_apps())}")
    try:
        library = ExampleLibrary.from_jsonl(args.library, split="train")
        info = {"split": "train", "shards": shard_info(args.library)}
        records = run_bench(
            _MODELS[args.model](), args.apps, args.out, library,
            max_retries=args.max_retries, library_info=info,
            command="python -m stepbuild.bench.run " + shlex.join(argv),
        )
    except (LeakageError, ValueError, OSError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    sys.stdout.write((Path(args.out) / model_slug(_MODELS[args.model].name) / SUMMARY_NAME)
                     .read_text(encoding="utf-8") if records else summarise(records))
    sys.stdout.flush()
    broken = [r["app"] for r in records if r["status"] == HARNESS_ERROR]
    if broken:
        print(f"error: harness_error in {', '.join(broken)} (see their JSON traces)",
              file=sys.stderr)
        return 2
    if args.require_pass and not all(passed(r) for r in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
