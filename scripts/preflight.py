"""Prove the full training path before leaving the real run unattended for a day.

Runs the REAL model on the REAL shards on CUDA, with the shipped config shortened
to five steps: train, eval, checkpoint, prune, finish, milestones; then a resume
from the latest checkpoint and one more step; then scripts/milestone_eval.py
against the preflight's own temp run; then scripts/weekend.py --dry-run against the
real config (reported, not gated). Everything is written to a temporary directory.
Needs data/shards and a GPU, so it is a script, not part of pytest.

Prints PASS/FAIL per check (SKIPPED for checks that depend on a phase that
raised) and exits non-zero on any failure.

Run: uv run python scripts/preflight.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from quipu.config import Config, TrainConfig, load_config
from quipu.model import Quipu
from quipu.train import LATEST, MILESTONE_DIR, Trainer

RUN_ID = "preflight"
_CKPT = re.compile(r"^step_\d+\.pt$")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_milestone_eval_module():
    """Import scripts/milestone_eval.py by path (it's a script, not a package
    member), the same way tests/test_milestone_eval.py does, so the fixed prompt
    list (ALL_PROMPTS) is read from one source of truth instead of being
    duplicated here."""
    spec = importlib.util.spec_from_file_location(
        "milestone_eval", REPO_ROOT / "scripts" / "milestone_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("milestone_eval", module)
    spec.loader.exec_module(module)
    return module


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.skipped = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        self.failed += not ok
        return ok

    def crashed(self, phase: str, exc: BaseException, dependent: list[str]) -> None:
        """A phase raised: report it as one FAIL and its dependent checks as SKIPPED."""
        self(f"{phase} raised", False, f"{type(exc).__name__}: {exc}")
        for name in dependent:
            print(f"  SKIPPED  {name}", flush=True)
            self.skipped += 1

    def skip_all(self, names: list[str], reason: str) -> None:
        for name in names:
            print(f"  SKIPPED  {name}  ({reason})", flush=True)
            self.skipped += 1


RUN_CHECKS = [
    "status completed", "reached the final step", "train losses finite",
    "at least one eval logged", "checkpoints on disk and <= ckpt_keep",
    "oldest checkpoint pruned", "latest.pt points to an existing file",
    "milestone files exist for configured steps + final",
    "milestones are bf16",
    "milestones load into Quipu(cfg.model) strict=True with tie intact",
    "milestones survive checkpoint pruning",
]
RESUME_CHECKS = [
    "resumed at the checkpointed step", "step continues from the checkpoint",
    "resumed loss finite", "log has a resumes entry", "no duplicate step numbers in the log",
]
MILESTONE_EVAL_CHECKS = [
    "milestone_eval.py exits 0",
    "metrics.json has one entry per checkpoint",
    "text_val_loss finite for every checkpoint",
    "code_val_loss finite for every checkpoint",
    "samples.md contains every fixed prompt",
]


# ---- milestone checks (phase 1) -------------------------------------------------


def expected_milestone_steps(tc: TrainConfig) -> list[int]:
    """Configured milestone steps plus the final step, which the trainer always
    saves too (see quipu/train.py's _run)."""
    return sorted(set(tc.milestones) | {tc.steps})


def check_milestones(check: Checks, cfg: Config, ckpt_dir: Path) -> None:
    """Pure CPU + filesystem: no CUDA and no real shards needed, so this is exactly
    what the CPU-only test below exercises directly with a fake ckpt_dir."""
    tc = cfg.train
    milestone_dir = ckpt_dir / MILESTONE_DIR
    expected = expected_milestone_steps(tc)

    found: dict[int, Path] = {}
    missing: list[int] = []
    for step in expected:
        p = milestone_dir / f"step_{step:06d}.pt"
        if p.is_file():
            found[step] = p
        else:
            missing.append(step)
    check(
        "milestone files exist for configured steps + final",
        not missing,
        f"expected {expected}" + (f", missing {missing}" if missing else ", all present"),
    )

    # Zero milestones found (e.g. the "milestone files exist" check above already
    # failed) must not read as "verified bf16/tie for all zero of them" -- an
    # empty loop below would otherwise leave both flags at their initial True and
    # pass vacuously, so seed them as already-failed in that case.
    bf16_ok, bf16_detail = (True, []) if found else (False, ["no milestones found"])
    load_ok, load_detail = (True, []) if found else (False, ["no milestones found"])
    for step, path in found.items():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            bf16_ok = load_ok = False
            bf16_detail.append(f"step {step}: {type(exc).__name__}: {exc}")
            load_detail.append(f"step {step}: could not read the file")
            continue

        floating = [v for v in state.values() if torch.is_tensor(v) and v.is_floating_point()]
        bad_dtypes = sorted({str(v.dtype) for v in floating if v.dtype != torch.bfloat16})
        if not floating or bad_dtypes:
            bf16_ok = False
            bf16_detail.append(f"step {step}: {bad_dtypes or 'no floating tensors found'}")

        try:
            model = Quipu(cfg.model)
            model.load_state_dict(state, strict=True)
            if model.lm_head.weight is not model.embed.weight:
                raise AssertionError("lm_head/embed tie was broken by load_state_dict")
        except Exception as exc:
            load_ok = False
            load_detail.append(f"step {step}: {type(exc).__name__}: {exc}")

    check("milestones are bf16", bf16_ok, "; ".join(bf16_detail) or "all bf16")
    check(
        "milestones load into Quipu(cfg.model) strict=True with tie intact",
        load_ok, "; ".join(load_detail) or "loaded and tied for every milestone",
    )

    # Demonstrate independence from ckpt_dir pruning: at least one milestone step
    # must have had its ckpt_dir/step_NNNNNN.pt (the resumable checkpoint) pruned
    # away by ckpt_keep, while the milestones/ copy survives. If the short run's
    # ckpt_keep never prunes a milestone step, the check can't prove anything, so
    # it fails loudly rather than passing on a config that doesn't exercise it.
    pruned_from_ckpt_dir = [step for step in found if not (ckpt_dir / f"step_{step:06d}.pt").is_file()]
    survived = [step for step in pruned_from_ckpt_dir if (milestone_dir / f"step_{step:06d}.pt").is_file()]
    check(
        "milestones survive checkpoint pruning",
        bool(pruned_from_ckpt_dir) and survived == pruned_from_ckpt_dir,
        f"pruned from ckpt_dir but present in milestones/: {survived}"
        if pruned_from_ckpt_dir else
        "no milestone step was pruned from ckpt_dir by this config; lower ckpt_keep "
        "or add an earlier milestone so this actually proves survival",
    )


# ---- milestone_eval.py subprocess (phase 3) -------------------------------------


def write_preflight_config_toml(cfg: Config, path: Path) -> None:
    """Serialise the already-validated (overridden) Config back to TOML, so
    milestone_eval.py -- which only takes --config, not programmatic overrides --
    can be pointed at the preflight run's temp ckpt_dir/shard_dir. Round-trips
    through load_config in the caller to prove it parses back to the same config."""

    def _lit(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_lit(v) for v in value) + "]"
        if isinstance(value, dict):
            # Inline table with quoted keys ("C++" is not a bare TOML key).
            return "{" + ", ".join(f"{json.dumps(k)} = {_lit(v)}" for k, v in value.items()) + "}"
        raise TypeError(f"cannot serialise {value!r} to TOML")

    lines = [f"name = {_lit(cfg.name)}", ""]
    for section_name, dc in (("model", cfg.model), ("data", cfg.data), ("train", cfg.train)):
        lines.append(f"[{section_name}]")
        for field in dc.__dataclass_fields__:
            # TrainConfig.context is populated FROM [model].context by load_config;
            # writing it back into [train] would trip load_config's own guard
            # ("context belongs in [model]; [train] inherits it").
            if section_name == "train" and field == "context":
                continue
            lines.append(f"{field} = {_lit(getattr(dc, field))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def check_milestone_eval_output(
    check: Checks,
    out_dir: Path,
    expected_labels: list[str],
    prompts: list[str],
) -> None:
    """Pure I/O over metrics.json / samples.md; no CUDA, no subprocess, so this is
    what the CPU-only test drives directly with hand-written fixture files."""
    metrics_path, samples_path = out_dir / "metrics.json", out_dir / "samples.md"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["checkpoints"]
    except (OSError, ValueError, KeyError) as exc:
        check("metrics.json has one entry per checkpoint", False,
              f"could not read {metrics_path}: {type(exc).__name__}: {exc}")
        check("text_val_loss finite for every checkpoint", False, "metrics.json unreadable")
        check("code_val_loss finite for every checkpoint", False, "metrics.json unreadable")
        metrics = None
    else:
        got_labels = [m.get("label") for m in metrics]
        check(
            "metrics.json has one entry per checkpoint",
            got_labels == expected_labels,
            f"expected {expected_labels}, got {got_labels}",
        )

        def _finite_check(name: str, key: str) -> None:
            successful = [m for m in metrics if "error" not in m]
            if not successful:
                # Zero successful checkpoints (an empty checkpoints list, or every
                # entry failed) must not read as "verified finite for all zero of
                # them" -- filtering an empty/all-error list leaves `bad` empty too,
                # which would otherwise pass vacuously.
                check(name, False, "no successful checkpoints to check")
                return
            bad = [
                f"{m.get('label')}: {m.get(key)!r}"
                for m in successful
                if not (isinstance(m.get(key), (int, float)) and math.isfinite(m[key]))
            ]
            check(name, not bad, "; ".join(bad) or "all finite")

        _finite_check("text_val_loss finite for every checkpoint", "text_val_loss")
        _finite_check("code_val_loss finite for every checkpoint", "code_val_loss")

    try:
        samples_text = samples_path.read_text(encoding="utf-8")
    except OSError as exc:
        check("samples.md contains every fixed prompt", False,
              f"could not read {samples_path}: {exc}")
        return
    missing_prompts = [p for p in prompts if f"`{p}`" not in samples_text]
    check(
        "samples.md contains every fixed prompt",
        not missing_prompts,
        f"missing: {missing_prompts}" if missing_prompts else f"all {len(prompts)} prompts present",
    )


def run_milestone_eval_check(check: Checks, cfg: Config, tmp_path: Path) -> None:
    me = _load_milestone_eval_module()
    config_path = tmp_path / "milestone_eval_config.toml"
    out_dir = tmp_path / "milestone_eval_out"
    write_preflight_config_toml(cfg, config_path)
    # Round-trip proof: if the hand-rolled TOML writer above produced something
    # load_config rejects or mis-parses, fail loudly here rather than inside the
    # subprocess with a confusing traceback.
    reloaded = load_config(config_path)
    if reloaded != cfg:
        check("milestone_eval.py exits 0", False,
              f"temp config at {config_path} does not round-trip to the same Config")
        check.skip_all(MILESTONE_EVAL_CHECKS[1:], "temp config did not round-trip")
        return

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "milestone_eval.py"),
        "--config", str(config_path), "--eval-batches", "2", "--out-dir", str(out_dir),
    ]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        check("milestone_eval.py exits 0", False, f"{type(exc).__name__}: {exc}")
        check.skip_all(MILESTONE_EVAL_CHECKS[1:], "milestone_eval.py did not run")
        return
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if not check("milestone_eval.py exits 0", result.returncode == 0, f"exit {result.returncode}"):
        check.skip_all(MILESTONE_EVAL_CHECKS[1:], "milestone_eval.py did not exit 0")
        return

    expected_steps = expected_milestone_steps(cfg.train)
    # No separate "final" label: the trainer always writes a milestone at the final
    # step, and milestone_eval drops the latest.pt entry when its weights equal that
    # milestone's, so the finished model is evaluated once, as step_<final>.
    expected_labels = [f"step_{s:06d}" for s in expected_steps]
    check_milestone_eval_output(check, out_dir, expected_labels, me.ALL_PROMPTS)


# ---- weekend.py --dry-run (phase 4, informational only) -------------------------


def print_launcher_guard_report(config_path: str) -> None:
    """Run the launcher's own --dry-run against the REAL config and print its guard
    lines. This is a report, not a gate: the owner may still have GPU-heavy apps
    open, be on battery, or be mid-way through the background data rebuild when
    preflight runs, and none of that should fail preflight."""
    print("\nlauncher guards (informational):")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "weekend.py"), "--dry-run",
           "--config", config_path]
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  could not run weekend.py --dry-run: {type(exc).__name__}: {exc}")
        return
    _print_guard_section(result.stdout, result.returncode)
    if result.stderr.strip():
        print("  stderr:")
        for line in result.stderr.splitlines():
            print(f"    {line}")


def _print_guard_section(stdout: str, returncode: int) -> None:
    """Pure formatting, split out so the CPU test can feed it a canned stdout
    string instead of running a real subprocess."""
    lines = stdout.splitlines()
    try:
        start = lines.index("start guards:")
    except ValueError:
        print("  (weekend.py --dry-run produced no 'start guards:' section; raw output below)")
        for line in lines:
            print(f"  {line}")
    else:
        for line in lines[start:]:
            print(f"  {line}")
    print(f"  (weekend.py --dry-run exit {returncode})")


def main() -> None:
    parser = argparse.ArgumentParser()
    # 5 steps saves at 2, 4 and 5, so pruning to ckpt_keep=2 actually runs, and
    # milestones [2, 4] land inside the run with the final step (5) as a third.
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    args = parser.parse_args()

    check = Checks()
    if not check("CUDA available", torch.cuda.is_available() and torch.cuda.device_count() > 0):
        sys.exit(1)

    started = time.perf_counter()
    # Windows can briefly lock a just-written checkpoint; a failed cleanup of the
    # temp dir must not turn a passing pre-flight into a traceback.
    with tempfile.TemporaryDirectory(prefix="quipu-preflight-", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        base = load_config(args.config)
        cfg = load_config(args.config, overrides={"train": {
            "total_tokens": args.steps * base.train.batch_tokens,
            "warmup_steps": 1,   # must be < steps; the schedule shape is not under test
            "eval_every": 2, "ckpt_every": 2, "ckpt_keep": 2,
            "ckpt_dir": str(tmp_path / "checkpoints"),
            "milestones": [2, 4],   # < steps (5), strictly increasing, at least one inside the run
        }})
        tc = cfg.train
        assert tc.steps == args.steps, (tc.steps, args.steps)
        shard_root = Path(cfg.data.shard_dir)
        run_dir = tmp_path / "runs"
        ckpt_dir = Path(tc.ckpt_dir)
        print(f"real config, micro_batch {tc.micro_batch}, grad_accum {tc.grad_accum}, "
              f"{tc.steps} steps, eval/ckpt every 2, keep 2, milestones {list(tc.milestones)}; "
              f"temp dir {tmp_path}")

        def trainer(resume: bool) -> Trainer:
            return Trainer(
                model_cfg=cfg.model, train_cfg=tc,
                shard_dir=shard_root / "train", val_dir=shard_root / "val",
                device="cuda", run_dir=run_dir, run_id=RUN_ID, resume=resume,
            )

        def read_log() -> dict:
            return json.loads((run_dir / f"{RUN_ID}.json").read_text(encoding="utf-8"))

        # ---- phase 1: a complete (short) run ---------------------------------
        print("\nrun():")
        run_ok = False
        t = None
        try:
            t = trainer(resume=False)
            t.run()
            run_ok = True
        except Exception as exc:
            check.crashed("run()", exc, RUN_CHECKS)
        if run_ok:
            try:
                record = read_log()
                losses = [s["train_loss"] for s in record["steps"]]
                check("status completed", record["status"] == "completed", record["status"])
                check("reached the final step", t.step == tc.steps,
                      f"step {t.step} of {tc.steps}")
                check("train losses finite",
                      bool(losses) and all(math.isfinite(x) for x in losses),
                      ", ".join(f"{x:.4f}" for x in losses))
                evals = record["evals"]
                check("at least one eval logged",
                      len(evals) >= 1 and all(math.isfinite(e["val_loss"]) for e in evals),
                      ", ".join(f"step {e['step']} val {e['val_loss']:.4f}" for e in evals))
                ckpts = sorted(p.name for p in ckpt_dir.iterdir() if _CKPT.match(p.name))
                check("checkpoints on disk and <= ckpt_keep",
                      1 <= len(ckpts) <= tc.ckpt_keep, f"keep {tc.ckpt_keep}: {', '.join(ckpts)}")
                check("oldest checkpoint pruned", "step_000002.pt" not in ckpts)
                pointer = torch.load(ckpt_dir / LATEST, weights_only=False)
                run_ok = check("latest.pt points to an existing file",
                               (ckpt_dir / pointer["file"]).is_file(), pointer["file"])
                check_milestones(check, cfg, ckpt_dir)
            except Exception as exc:
                # A missing log or checkpoint dir is a failed check, not a traceback.
                check("inspecting the run's log and checkpoints", False,
                      f"{type(exc).__name__}: {exc}")
                run_ok = False
        saved_step = t.step if t is not None else None
        del t
        torch.cuda.empty_cache()

        # ---- phase 2: resume and take one more step --------------------------
        print("\nresume:")
        if not run_ok:
            check.skip_all(RESUME_CHECKS, "run() did not leave a usable checkpoint")
        else:
            r = None
            try:
                r = trainer(resume=True)
                r.resume_from_latest()
            except Exception as exc:
                check.crashed("Trainer(resume=True) + resume_from_latest()", exc, RESUME_CHECKS)
            else:
                check("resumed at the checkpointed step", r.step == saved_step, f"step {r.step}")
                try:
                    loss = r.train_step()
                except Exception as exc:
                    check.crashed("train_step() after resume", exc, RESUME_CHECKS[1:])
                else:
                    check("step continues from the checkpoint", r.step == saved_step + 1,
                          f"step {r.step}")
                    check("resumed loss finite", math.isfinite(loss), f"{loss:.4f}")
                    record = read_log()
                    check("log has a resumes entry", len(record.get("resumes", [])) >= 1,
                          json.dumps(record.get("resumes")))
                    step_nums = [s["step"] for s in record["steps"]]
                    check("no duplicate step numbers in the log",
                          len(step_nums) == len(set(step_nums)), str(step_nums))
            del r
            torch.cuda.empty_cache()

        # ---- phase 3: milestone_eval.py against this preflight's temp run ----
        print("\nmilestone_eval.py:")
        if not run_ok:
            check.skip_all(MILESTONE_EVAL_CHECKS, "run() did not leave usable milestones/checkpoints")
        else:
            try:
                run_milestone_eval_check(check, cfg, tmp_path)
            except Exception as exc:
                check.crashed("milestone_eval.py check", exc, MILESTONE_EVAL_CHECKS)

        # ---- phase 4: weekend.py --dry-run against the real config (report) --
        print_launcher_guard_report(args.config)

    failed = check.failed or check.skipped
    print(f"\n{'FAIL' if failed else 'PASS'}: {check.failed} check(s) failed, "
          f"{check.skipped} skipped, {time.perf_counter() - started:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
