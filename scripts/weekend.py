"""The weekend launcher: start it once, come back Monday.

Runs `python -m quipu.train` as a child process, auto-resuming on failures that
are worth another try. The trainer's contract: 0 success; 2 a usage/config
error (reused run id without --resume, or --resume with a missing/corrupt run
log); 3 the non-finite stop; 130 a user interrupt. 2, 3 and interrupt codes are
never retried -- retrying a usage error or the non-finite stop would just fail
identically, and an interrupt means the owner asked to stop. Everything else
(1, or an unrelated crash such as a CUDA fault) is retried, up to a cap.

Ctrl+C is handled specially, because the launcher and the child share a
console: on Windows a Ctrl+C'd child can exit via STATUS_CONTROL_C_EXIT
(0xC000013A), which subprocess reports as -1073741510 (or its unsigned
equivalent, 3221225786, if something upstream treats it as unsigned) rather
than the POSIX-style 130. All of these are treated as "interrupted". If the
launcher's own process receives the Ctrl+C first (same console, most likely),
it is caught around the child call: the child is given a few seconds to exit
on its own, then terminated if it hasn't, and the run stops without retrying.

After a completed run it runs the evaluation scripts, each as its own process
so one failing step doesn't stop the rest. A one-page summary is (re)written
atomically after every single attempt, so a hard power-off mid-run still
leaves the latest state on disk, not just whatever was true at the last
successful step.

Run for real:      python -m uv run python scripts/weekend.py
Safe during a meeting (guards only, nothing started): add --dry-run
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from quipu.config import load_config
from quipu.fsio import replace_with_retry

REPO_ROOT = Path(__file__).resolve().parent.parent

# The trainer's own contract (spec section 4): 0 success, 2 usage/config
# error, 3 the non-finite stop, 130 a user interrupt (POSIX-style; see the
# module docstring for the Windows Ctrl+C exit codes the child can actually
# report). None of these are retried.
EXIT_USAGE_ERROR = 2
EXIT_NONFINITE = 3
EXIT_INTERRUPT = 130
# Raw Windows exit codes for a Ctrl+C'd child (STATUS_CONTROL_C_EXIT,
# 0xC000013A), signed and unsigned, on top of the POSIX-style 130.
WIN_CTRL_C_EXIT_CODES = frozenset({-1073741510, 3221225786})
INTERRUPT_CODES = frozenset({EXIT_INTERRUPT}) | WIN_CTRL_C_EXIT_CODES
NO_RETRY_CODES = frozenset({EXIT_USAGE_ERROR, EXIT_NONFINITE}) | INTERRUPT_CODES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass
class GuardResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


@dataclass
class Attempt:
    kind: str                      # "train" or an eval step's name
    args: list[str]
    returncode: int | None         # None means "not run" (e.g. script missing)
    started_at: str
    finished_at: str
    log_path: str | None = None
    retried: bool = False
    note: str = ""                 # e.g. "not present (skipped)", "failed"


@dataclass
class WeekendState:
    forced: bool = False
    guard_results: list[GuardResult] = field(default_factory=list)
    train_attempts: list[Attempt] = field(default_factory=list)
    eval_attempts: list[Attempt] = field(default_factory=list)
    final_status: str = "not started"


# --------------------------------------------------------------------------
# start guards
# --------------------------------------------------------------------------


def probe_other_gpu_vram_gb() -> float | None:
    """GB of GPU memory in use right now, summed across GPUs. We are not running
    yet, so every byte reported belongs to some other process. None means
    nvidia-smi isn't available (no NVIDIA GPU, or not on PATH)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    total_mib = 0.0
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            total_mib += float(line)
    return total_mib / 1024.0


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    # The real Win32 struct's fields are BYTE (unsigned); c_byte here would
    # read ACLineStatus 255 (unknown) back as -1 and break the 255 check below.
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("Reserved1", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def probe_on_battery() -> bool | None | str:
    """True on battery, False on AC, "unknown" if Windows itself reports
    ACLineStatus 255 (genuinely ambiguous -- must not be silently treated as
    AC), None on non-Windows or if Windows can't answer at all (in which case
    the guard treats it as "can't tell, don't block")."""
    if platform.system() != "Windows":
        return None
    status = _SYSTEM_POWER_STATUS()
    try:
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    if not ok:
        return None
    if status.ACLineStatus == 255:
        return "unknown"
    return status.ACLineStatus == 0


def probe_free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def probe_shards_present(shard_dir: Path) -> tuple[bool, bool]:
    """(train present and non-empty, val present and non-empty)."""

    def _has_files(d: Path) -> bool:
        return d.is_dir() and any(d.iterdir())

    return _has_files(shard_dir / "train"), _has_files(shard_dir / "val")


def run_guards(
    *,
    max_other_vram_gb: float,
    min_free_gb: float,
    shard_dir: Path,
    force: bool,
    probe_vram: Callable[[], float | None] = probe_other_gpu_vram_gb,
    probe_battery: Callable[[], bool | None | str] = probe_on_battery,
    probe_disk: Callable[[Path], float] = probe_free_disk_gb,
    probe_shards: Callable[[Path], tuple[bool, bool]] = probe_shards_present,
) -> list[GuardResult]:
    """Every probe still runs even under --force, so the summary shows what was
    bypassed rather than just skipping the check silently."""
    results: list[GuardResult] = []

    vram = probe_vram()
    if vram is None:
        results.append(GuardResult(
            "gpu-vram", True, "nvidia-smi not available; skipping this guard",
        ))
    else:
        ok = force or vram <= max_other_vram_gb
        results.append(GuardResult(
            "gpu-vram", ok, f"{vram:.2f} GB held by other processes (limit {max_other_vram_gb} GB)",
            fix="" if ok else (
                f"close GPU-heavy apps (Teams/Edge/WhatsApp/etc.) until other processes hold "
                f"<= {max_other_vram_gb} GB of GPU memory, or pass --force"
            ),
        ))

    battery = probe_battery()
    if battery is None:
        results.append(GuardResult(
            "power", True, "not on Windows, or power status unavailable; skipping this guard",
        ))
    elif battery == "unknown":
        ok = force
        results.append(GuardResult(
            "power", ok, "power state unknown (Windows reported ACLineStatus 255)",
            fix="" if ok else "power state unknown -- plug in and re-run, or use --force",
        ))
    else:
        ok = force or not battery
        results.append(GuardResult(
            "power", ok, "on battery" if battery else "on AC power",
            fix="" if ok else "plug the laptop into mains power, or pass --force",
        ))

    free_gb = probe_disk(REPO_ROOT)
    ok = force or free_gb >= min_free_gb
    results.append(GuardResult(
        "disk", ok, f"{free_gb:.1f} GB free on {REPO_ROOT.drive or REPO_ROOT.anchor} "
                    f"(need >= {min_free_gb} GB)",
        fix="" if ok else (
            f"free at least {min_free_gb} GB on the repo's drive (delete old checkpoints/"
            "results, empty Recycle Bin), or pass --force"
        ),
    ))

    train_present, val_present = probe_shards(shard_dir)
    ok = force or (train_present and val_present)
    missing = [d for d, present in (("train", train_present), ("val", val_present)) if not present]
    results.append(GuardResult(
        "shards", ok,
        "present" if not missing else
        f"missing or empty: {', '.join(f'data/shards/{d}' for d in missing)}",
        fix="" if ok else "run scripts/build_shards.py to build the shards first, or pass --force",
    ))

    return results


# --------------------------------------------------------------------------
# child processes
# --------------------------------------------------------------------------


def _stop_child(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Give a Ctrl+C'd child a moment to exit on its own -- it received the same
    console signal -- then escalate to terminate/kill if it's still alive."""
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def default_run_child(cmd: list[str], log_path: Path) -> int:
    """Run `cmd`, inheriting stdout/stderr to the console (so progress lines
    still show up live) while also teeing every line to `log_path`. Unbuffered
    so those lines arrive promptly instead of sitting in the child's stdio
    buffer for a 50-hour run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8") as logf:
        try:
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            msg = f"weekend.py: failed to launch {cmd!r}: {exc}\n"
            sys.stdout.write(msg)
            logf.write(msg)
            return 1
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
            proc.wait()
            return proc.returncode
        except KeyboardInterrupt:
            # The launcher's own process got the Ctrl+C too (same console);
            # make sure the child is actually gone before this propagates.
            _stop_child(proc)
            raise


ChildRunner = Callable[[list[str], Path], int]


def build_train_cmd(config: str, run_id: str, resume: bool) -> list[str]:
    cmd = [sys.executable, "-m", "quipu.train", "--config", config, "--run-id", run_id]
    if resume:
        cmd.append("--resume")
    return cmd


# --------------------------------------------------------------------------
# training with auto-resume
# --------------------------------------------------------------------------


def run_training(
    *,
    config: str,
    run_id: str,
    max_retries: int,
    retry_wait_s: float,
    log_dir: Path,
    run_log_exists: Callable[[str], bool],
    on_attempt: Callable[[Attempt], None],
    run_child: ChildRunner = default_run_child,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Returns the exit code of the final attempt (0 iff training completed).

    --resume is decided fresh before every attempt by asking `run_log_exists`,
    never assumed: a failed first attempt might have crashed before RunLog's
    __init__ ever flushed (e.g. a usage error, or a crash during model
    construction), in which case there is still no run log and forcing
    --resume would hand the trainer a --resume flag with nothing to resume."""
    retries_used = 0
    attempt_n = 0
    code = 1
    while True:
        attempt_n += 1
        resume = run_log_exists(run_id)
        cmd = build_train_cmd(config, run_id, resume)
        log_path = log_dir / f"train_attempt_{attempt_n}.log"
        started = _now()
        try:
            code = run_child(cmd, log_path)
        except KeyboardInterrupt:
            finished = _now()
            on_attempt(Attempt(
                kind="train", args=cmd, returncode=EXIT_INTERRUPT, started_at=started,
                finished_at=finished, log_path=str(log_path), retried=False,
                note="stopped by owner (Ctrl+C)",
            ))
            return EXIT_INTERRUPT
        finished = _now()
        will_retry = (
            code != 0
            and code not in NO_RETRY_CODES
            and retries_used < max_retries
        )
        note = "stopped by owner (Ctrl+C)" if code in INTERRUPT_CODES else (
            "usage/config error" if code == EXIT_USAGE_ERROR else (
                "non-finite stop" if code == EXIT_NONFINITE else ""
            )
        )
        on_attempt(Attempt(
            kind="train", args=cmd, returncode=code, started_at=started,
            finished_at=finished, log_path=str(log_path), retried=will_retry, note=note,
        ))
        if code == 0 or not will_retry:
            return code
        retries_used += 1
        sleep(retry_wait_s)


# --------------------------------------------------------------------------
# post-training evaluation
# --------------------------------------------------------------------------

_OPTIONAL_EVAL_SCRIPTS = ("milestone_eval", "needle_eval")


def run_evals(
    *,
    log_dir: Path,
    on_attempt: Callable[[Attempt], None],
    run_child: ChildRunner = default_run_child,
    script_exists: Callable[[Path], bool] = lambda p: p.is_file(),
) -> None:
    for name in _OPTIONAL_EVAL_SCRIPTS:
        script_path = REPO_ROOT / "scripts" / f"{name}.py"
        if not script_exists(script_path):
            on_attempt(Attempt(
                kind=name, args=[str(script_path)], returncode=None,
                started_at=_now(), finished_at=_now(), log_path=None,
                note="not present (skipped)",
            ))
            continue
        cmd = [sys.executable, str(script_path)]
        log_path = log_dir / f"{name}.log"
        started = _now()
        code = run_child(cmd, log_path)
        finished = _now()
        on_attempt(Attempt(
            kind=name, args=cmd, returncode=code, started_at=started,
            finished_at=finished, log_path=str(log_path),
            note="" if code == 0 else "failed",
        ))

    # Not an optional script: it's part of the package, always run.
    cmd = [sys.executable, "-m", "quipu.results_table", "results/runs", "--out", "RESULTS.md"]
    log_path = log_dir / "results_table.log"
    started = _now()
    code = run_child(cmd, log_path)
    finished = _now()
    on_attempt(Attempt(
        kind="results_table", args=cmd, returncode=code, started_at=started,
        finished_at=finished, log_path=str(log_path), note="" if code == 0 else "failed",
    ))


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def render_summary(state: WeekendState) -> str:
    lines = ["# Quipu weekend run summary", "", f"Generated: {_now()}", ""]

    lines.append("## Start guards")
    lines.append("")
    for g in state.guard_results:
        tag = "OK" if g.ok else ("BYPASSED (--force)" if state.forced else "FAILED")
        lines.append(f"- **{g.name}**: {tag} -- {g.detail}")
        if not g.ok and g.fix:
            lines.append(f"  - fix: {g.fix}")
    if not state.guard_results:
        lines.append("(not yet checked)")
    lines.append("")

    lines.append("## Training attempts")
    lines.append("")
    if not state.train_attempts:
        lines.append("(none yet)")
    for i, a in enumerate(state.train_attempts, 1):
        retry_note = "  -> retrying" if a.retried else ""
        lines.append(
            f"{i}. exit {a.returncode}  ({a.started_at} to {a.finished_at})  "
            f"log: {a.log_path}{retry_note}"
        )
    lines.append("")

    lines.append("## Evaluation steps")
    lines.append("")
    if not state.eval_attempts:
        lines.append("(none yet -- run only after training completes)")
    for a in state.eval_attempts:
        result = a.note if a.note else f"exit {a.returncode}"
        log_note = f"  log: {a.log_path}" if a.log_path else ""
        lines.append(f"- **{a.kind}**: {result}{log_note}")
    lines.append("")

    lines.append("## Final status")
    lines.append("")
    lines.append(state.final_status)
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append("- run log: `results/runs/<run-id>.json`")
    lines.append("- checkpoints: `checkpoints/` (resumable) and `checkpoints/milestones/`")
    lines.append("- results table: `RESULTS.md`")
    lines.append("- attempt logs: `results/weekend/*.log`")
    lines.append("")

    return "\n".join(lines)


def write_summary(state: WeekendState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(render_summary(state), encoding="utf-8")
        replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--run-id", default="quipu-114m-weekend")
    parser.add_argument("--force", action="store_true",
                         help="skip every start guard (owner only); bypasses are printed loudly")
    parser.add_argument("--retry-wait-s", type=float, default=120.0,
                         help="pause before relaunching after a retryable failure (tests use 0)")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-other-vram-gb", type=float, default=1.5)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--dry-run", action="store_true",
                         help="run the start guards, print the plan, exit -- safe to run in a meeting")
    return parser.parse_args(argv)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    summary_path = REPO_ROOT / "results" / "weekend_summary.md"
    log_dir = REPO_ROOT / "results" / "weekend"
    run_log_dir = REPO_ROOT / "results" / "runs"
    # The shard layout is whatever the run's own --config says, not a
    # hard-coded guess -- a differently-configured run must be guarded
    # against its own shards, not the default ones.
    cfg = load_config(args.config)
    shard_dir = _resolve(cfg.data.shard_dir)

    state = WeekendState(forced=args.force)

    state.guard_results = run_guards(
        max_other_vram_gb=args.max_other_vram_gb,
        min_free_gb=args.min_free_gb,
        shard_dir=shard_dir,
        force=args.force,
    )
    write_summary(state, summary_path)

    print("start guards:")
    for g in state.guard_results:
        tag = "ok" if g.ok else "FAILED"
        print(f"  [{tag}] {g.name}: {g.detail}")
        if not g.ok and g.fix:
            print(f"         fix: {g.fix}")

    failing = [g for g in state.guard_results if not g.ok]
    if failing and args.force:
        print()
        print("*** --force: the guard failure(s) above were BYPASSED. ***")
        print()
    elif failing:
        state.final_status = "refused to start: one or more start guards failed (see above)"
        write_summary(state, summary_path)
        return 1

    resume_now = (run_log_dir / f"{args.run_id}.json").exists()
    planned_train_cmd = build_train_cmd(args.config, args.run_id, resume_now)

    if args.dry_run:
        print()
        print("--dry-run: nothing started. Plan:")
        print(f"  1. train: {' '.join(planned_train_cmd)}")
        print(
            "  2. on exit 0: scripts/milestone_eval.py (if present), "
            "scripts/needle_eval.py (if present), "
            "python -m quipu.results_table results/runs --out RESULTS.md"
        )
        print(f"  summary will be kept at {summary_path}")
        state.final_status = "dry run only; no training was started"
        write_summary(state, summary_path)
        return 0

    def _on_train_attempt(a: Attempt) -> None:
        state.train_attempts.append(a)
        write_summary(state, summary_path)

    last_code = run_training(
        config=args.config, run_id=args.run_id,
        max_retries=args.max_retries, retry_wait_s=args.retry_wait_s,
        log_dir=log_dir,
        run_log_exists=lambda rid: (run_log_dir / f"{rid}.json").exists(),
        on_attempt=_on_train_attempt,
    )

    if last_code != 0:
        if last_code in INTERRUPT_CODES:
            state.final_status = "stopped by owner (Ctrl+C)"
            exit_code = EXIT_INTERRUPT
        elif last_code == EXIT_USAGE_ERROR:
            state.final_status = (
                "training did not complete: usage/config error (exit 2) -- "
                "not retried, a retry would fail identically"
            )
            exit_code = last_code
        else:
            state.final_status = (
                f"training did not complete; last exit code {last_code} "
                f"({len(state.train_attempts)} attempt(s))"
            )
            exit_code = last_code
        write_summary(state, summary_path)
        return exit_code

    def _on_eval_attempt(a: Attempt) -> None:
        state.eval_attempts.append(a)
        write_summary(state, summary_path)

    try:
        run_evals(log_dir=log_dir, on_attempt=_on_eval_attempt)
    except KeyboardInterrupt:
        # run_training already absorbs a Ctrl+C during training; this is the
        # backstop for one landing during the (much shorter) evaluation phase.
        state.final_status = "stopped by owner (Ctrl+C) during evaluation"
        write_summary(state, summary_path)
        return EXIT_INTERRUPT

    failed_evals = [a for a in state.eval_attempts if a.returncode not in (0, None)]
    if failed_evals:
        state.final_status = (
            "training completed; evaluation had failure(s): "
            + ", ".join(a.kind for a in failed_evals)
        )
    else:
        state.final_status = "training completed; evaluation done"
    write_summary(state, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
