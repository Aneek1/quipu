"""The rented-box launcher for the quipu-moe full run (spec sections 6.2-6.4 and 8,
plan Task M9).

    python -m quipu.spend start --usd-per-hour R          # once, when the box starts
    python scripts/remote/run_moe.py --config configs/quipu-moe.toml \\
        --winners results/ab/winners.toml --budget-usd 20 --usd-per-hour R

What it does, in order (everything lands in --out, default results/moe):

1. Resolve the run config: the --config file + the A/B winners + --override flags,
   written out whole as run_config.toml (the trainer, milestone_eval and later tools
   all read that one file). The launcher owns train.total_tokens, milestones,
   ckpt_dir, budget_usd and usd_per_hour; winners or overrides that set them are
   refused.
2. Throughput gate: the full-size run starts for real (run id "quipu-moe",
   checkpoints in --ckpt-dir) with milestones off, and is interrupted after
   --gate-minutes (SIGINT; Windows CTRL_BREAK): the trainer writes its interrupt
   checkpoint. Tokens/s is measured from the step lines' arrival times, leaving out
   everything before --warmup-skip-steps (startup, compile, the first steps) but
   keeping the evals and checkpoints in between, so it is the rate the long run will
   really get.
3. The plan (plan.md for the owner, plan.json for the launcher): hours and cost of
   the configured tokens at the measured rate (less --headroom), plus the box's
   spend so far (the shared ledger, quipu/spend.py) plus the long run's startup plus
   --reserve-usd (the chat SFT and the final eval/export, spec 13). If that is over
   --budget-usd, total_tokens is cut to what fits, rounded down to whole batches, and
   the milestones are rescaled by their fraction of the run (deduplicated, none past
   the end). Then it waits for the GO file (--go-file, default <out>/GO; the
   controller creates it once the owner approves), printing what each minute of
   waiting costs; the ledger keeps ticking. At GO the plan is checked again against
   the spend then: if the wait made it unaffordable it is trimmed again (plan.md
   says so), and if nothing fits it stops with exit 4.
4. The long run resumes from the gate's checkpoint, so the gate's steps are kept,
   not thrown away. The LR schedule depends on the run's length only after warm-up,
   so when the gate ended inside warm-up (the normal case: warmup_steps 500 is ~40
   min on a 5090, the gate 15) the reused steps are exactly the ones the trimmed run
   would have trained; plan.md reports it either way. Milestones that the rescale
   puts inside the gate move to the gate's checkpoint step (the resumed trainer
   writes that milestone from the weights it just loaded).
   Crashes are retried from the last checkpoint (like weekend.py): exit 1 or
   anything unexpected, up to --max-retries; 2 (usage), 3 (non-finite stop) and
   interrupts are not retried.
5. Spend guard: while a child trains (and before each launch) the launcher polls
   spend + cost of the next eval interval (min(eval_every, steps left) at the slower
   of the planned and the attempt's measured rate) + --reserve-usd; once that is
   above --budget-usd the child is interrupted (SIGINT -> interrupt checkpoint ->
   up to STOP_GRACE_S, then terminate, then kill) and the launcher exits 4. The
   budget is the box ledger's: what remains is --budget-usd minus everything the box
   has cost so far (setup, shards, the A/B runs, earlier invocations), so a restart
   with a larger --budget-usd resumes where it stopped.
6. At the end: scripts/milestone_eval.py on the run config, and summary.md. summary.md
   is written only when the run is over (completed, stopped by the guard, or failed
   for good); sync.sh stops syncing when it appears.

Restarts: plan.json is reused (no second gate) as long as the resolved config is the
same (anything else is exit 2: move <out> and the checkpoints aside to start over);
a GO from before the plan existed is removed.

Exit codes: 0 done, 1 training failed (after retries) or the gate could not measure,
2 usage error, 3 the trainer's non-finite stop, 4 stopped by the budget, 130
interrupted.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quipu.config import _merge, load_config, parse_overrides
from quipu.fsio import write_text_atomic
from quipu.spend import TICK_S, Ledger, LedgerError, Ticker

ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NONFINITE = 3
EXIT_BUDGET = 4
EXIT_INTERRUPTED = 130
# The trainer's exit codes (quipu.train): 0 ok, 1 crash, 2 usage, 3 non-finite stop,
# 130 interrupt; Windows reports a Ctrl+C'd child as STATUS_CONTROL_C_EXIT.
TRAIN_USAGE, TRAIN_NONFINITE = 2, 3
TRAIN_INTERRUPT_CODES = frozenset({130, -1073741510, 3221225786})

RUN_ID = "quipu-moe"
DEFAULT_OUT = "results/moe"
DEFAULT_CKPT_DIR = "checkpoints/quipu-moe"
GATE_MINUTES = 15.0
RESERVE_USD = 0.50          # chat SFT (~$0.25, spec 13) + final eval/export
HEADROOM = 0.03             # plan at 97% of the measured tokens/s
WARMUP_SKIP_STEPS = 20      # steps left out of the throughput measurement
STOP_GRACE_S = 300.0        # an interrupted child's time to write its checkpoint
KILL_WAIT_S = 30.0          # after terminate() before kill()
POLL_S = 5.0                # spend-guard and GO poll
WAIT_PRINT_S = 300.0        # how often the GO wait prints what it costs
MAX_RETRIES = 3
RETRY_WAIT_S = 120.0

# The launcher sets these per phase; winners / --override must not.
LAUNCHER_KEYS = {
    "total_tokens": "the throughput gate sets the token target",
    "milestones": "rescaled with the token target",
    "ckpt_dir": "use --ckpt-dir",
    "budget_usd": "use --budget-usd",
    "usd_per_hour": "use --usd-per-hour",
}

STEP_LINE = re.compile(r"^step (\d+)/(\d+)\s+loss\s+(\S+)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- measurement, projection, trimming ---------------------------------------------------

def effective_tokens_per_s(events: list[tuple[float, int]], batch_tokens: int,
                           skip_steps: int) -> float | None:
    """Tokens/s between the first step line at or after `skip_steps` and the last one
    (arrival times, so evals and checkpoints in between count). None if fewer than
    two step lines are past the warm-up."""
    kept = [(t, s) for t, s in events if s >= skip_steps]
    if len(kept) < 2:
        return None
    (t0, s0), (t1, s1) = kept[0], kept[-1]
    if t1 <= t0 or s1 <= s0:
        return None
    return (s1 - s0) * batch_tokens / (t1 - t0)


@dataclasses.dataclass(frozen=True)
class Projection:
    tokens_total: int
    tokens_done: int
    tokens_per_s: float
    usd_per_hour: float
    spent_usd: float
    reserve_usd: float
    startup_s: float
    budget_usd: float

    @property
    def tokens_left(self) -> int:
        return max(0, self.tokens_total - self.tokens_done)

    @property
    def hours(self) -> float:
        return (self.tokens_left / self.tokens_per_s + self.startup_s) / 3600

    @property
    def train_usd(self) -> float:
        return self.hours * self.usd_per_hour

    @property
    def total_usd(self) -> float:
        return self.spent_usd + self.train_usd + self.reserve_usd

    @property
    def remaining_after(self) -> float:
        """Budget left once the run is done (the reserve is still in it)."""
        return self.budget_usd - self.spent_usd - self.train_usd

    @property
    def fits(self) -> bool:
        return self.total_usd <= self.budget_usd + 1e-9


def project(*, tokens_total: int, tokens_done: int, tokens_per_s: float,
            usd_per_hour: float, spent_usd: float, reserve_usd: float, startup_s: float,
            budget_usd: float) -> Projection:
    return Projection(tokens_total, tokens_done, tokens_per_s, usd_per_hour, spent_usd,
                      reserve_usd, startup_s, budget_usd)


def fit_total_tokens(*, budget_usd: float, spent_usd: float, reserve_usd: float,
                     usd_per_hour: float, tokens_per_s: float, startup_s: float,
                     tokens_done: int, batch_tokens: int) -> int:
    """The largest total_tokens (whole batches) whose remaining training, plus the
    startup and the reserve, fits the budget. May be <= tokens_done: nothing fits."""
    usd = budget_usd - spent_usd - reserve_usd - startup_s / 3600 * usd_per_hour
    if usd <= 0 or usd_per_hour <= 0:
        return (tokens_done // batch_tokens) * batch_tokens
    more = math.floor(usd / usd_per_hour * 3600 * tokens_per_s * (1 - 1e-12))
    return ((tokens_done + more) // batch_tokens) * batch_tokens


def rescale_milestones(milestones, old_steps: int, new_steps: int,
                       gate_step: int = 0) -> tuple[int, ...]:
    """Each milestone keeps its fraction of the run (rounded), deduplicated, none at
    or past the end (the final step is always a milestone anyway). One that lands at
    or before `gate_step` (inside the gate, whose checkpoint the run resumes from)
    moves to gate_step itself."""
    out: set[int] = set()
    for m in milestones:
        v = max(1, round(m * new_steps / old_steps))
        if gate_step and v <= gate_step:
            v = gate_step
        if v < new_steps:
            out.add(v)
    return tuple(sorted(out))


def guard_margin_usd(steps_left: int, eval_every: int, batch_tokens: int,
                     tokens_per_s: float, usd_per_hour: float) -> float:
    """Cost of the next eval interval: min(eval_every, steps left) steps."""
    steps = max(0, min(eval_every, steps_left))
    return steps * batch_tokens / tokens_per_s / 3600 * usd_per_hour


def over_budget(spent_usd: float, margin_usd: float, reserve_usd: float,
                budget_usd: float) -> bool:
    return spent_usd + margin_usd + reserve_usd > budget_usd + 1e-9


def lr_at(step: int, lr: float, lr_min: float, warmup: int, steps: int) -> float:
    """quipu.train.Trainer.lr_at, for the plan's gate-reuse note."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= steps:
        return lr_min
    progress = (step - warmup) / max(1, steps - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


def gate_reuse_note(train: dict[str, Any], gate_step: int, old_steps: int,
                    new_steps: int) -> str:
    warmup = train["warmup_steps"]
    if gate_step <= warmup or old_steps == new_steps:
        return ("exact: the gate's steps all used the same learning rates the planned "
                "run uses" + (" (they were inside warm-up, which does not depend on "
                              "the run's length)" if old_steps != new_steps else ""))
    worst = max(abs(lr_at(s, train["lr"], train["lr_min"], warmup, old_steps)
                    / lr_at(s, train["lr"], train["lr_min"], warmup, new_steps) - 1)
                for s in range(warmup, gate_step))
    return (f"approximate: steps {warmup}-{gate_step} ran on the untrimmed cosine "
            f"(learning rate at most {worst:.2%} off the planned run's)")


# ---- the run config --------------------------------------------------------------------

def resolve_raw(config: str | Path, winners: str | Path | None,
                overrides: list[str]) -> dict[str, Any]:
    """The config file with the winners and --override flags merged in (as
    load_config merges them). Launcher-owned train keys are refused."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    layers: list[tuple[str, dict[str, Any]]] = []
    if winners:
        layers.append((f"winners {winners}", tomllib.loads(
            Path(winners).read_text(encoding="utf-8"))))
    if overrides:
        layers.append(("--override", parse_overrides(overrides)))
    for where, layer in layers:
        for key, why in LAUNCHER_KEYS.items():
            if key in layer.get("train", {}):
                raise ValueError(f"{where} sets train.{key}: the launcher owns it ({why})")
        raw = _merge(raw, layer)
    return raw


def with_train(raw: dict[str, Any], **train: Any) -> dict[str, Any]:
    return _merge(raw, {"train": train})


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"cannot write a non-finite number {v!r} to TOML")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise ValueError(f"cannot write {type(v).__name__} {v!r} to TOML")


def dump_toml(raw: dict[str, Any]) -> str:
    """A config dict (tables of scalars, lists and nested tables) as TOML that
    tomllib reads back equal."""
    lines: list[str] = []

    def table(d: dict[str, Any], prefix: list[str]) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix:
            lines.extend(["", "[" + ".".join(_key(p) for p in prefix) + "]"])
        for k, v in scalars.items():
            lines.append(f"{_key(k)} = {_value(v)}")
        for k, v in tables.items():
            table(v, prefix + [k])

    table(raw, [])
    return "\n".join(lines).lstrip("\n") + "\n"


def base_hash(raw: dict[str, Any]) -> str:
    """The resolved config without the launcher's keys: what a reused plan must match."""
    stripped = json.loads(json.dumps(raw))
    for key in LAUNCHER_KEYS:
        stripped.get("train", {}).pop(key, None)
    return hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()


# ---- the child process -----------------------------------------------------------------

@dataclasses.dataclass
class ChildOutcome:
    code: int
    stop_reason: str | None                 # set when the launcher interrupted it
    events: list[tuple[float, int]]         # (arrival time, step) of each step line
    started_at: float


# should_stop gets the attempt's step events so far and returns a reason to stop, or None.
StopCheck = Callable[[list[tuple[float, int]]], "str | None"]
ChildRunner = Callable[[list[str], Path, StopCheck], ChildOutcome]


class SubprocessChild:
    """Runs one trainer process, echoing and logging its output and recording when
    each step line arrived. A watcher thread polls should_stop every poll_s; when it
    returns a reason the child is interrupted (SIGINT; on Windows CTRL_BREAK to its
    own process group, which the trainer handles as SIGBREAK) and given grace_s to
    write its interrupt checkpoint, then terminated, then killed (the same ladder as
    ab_runs.SubprocessRunner). The children share a persistent inductor cache, so the
    long run reuses the gate's compile."""

    def __init__(self, cwd: str | Path = ROOT, echo: Callable[[str], None] | None = None,
                 grace_s: float = STOP_GRACE_S, kill_wait_s: float = KILL_WAIT_S,
                 poll_s: float = POLL_S, inductor_cache: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.cwd = Path(cwd)
        self.echo = echo or (lambda s: print(s, flush=True))
        self.grace_s = grace_s
        self.kill_wait_s = kill_wait_s
        self.poll_s = poll_s
        self.inductor_cache = inductor_cache
        self.clock = clock

    @staticmethod
    def _interrupt(proc: subprocess.Popen) -> bool:
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _wait(proc: Any, timeout: float | None) -> bool:
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
        except KeyboardInterrupt:
            return False

    def _stop_child(self, proc: Any, child_got_it: bool) -> None:
        if child_got_it:
            if self._wait(proc, self.grace_s):
                return
            if self._interrupt(proc) and self._wait(proc, self.kill_wait_s):
                return
        elif self._interrupt(proc) and self._wait(proc, self.grace_s):
            return
        proc.terminate()
        if self._wait(proc, self.kill_wait_s):
            return
        proc.kill()
        proc.wait()

    def __call__(self, cmd: list[str], log_path: Path, should_stop: StopCheck) -> ChildOutcome:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TORCHINDUCTOR_FX_GRAPH_CACHE="1",
                   TORCHINDUCTOR_AUTOGRAD_CACHE="1")
        if self.inductor_cache is not None:
            self.inductor_cache.mkdir(parents=True, exist_ok=True)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(self.inductor_cache)
        events: list[tuple[float, int]] = []
        stop: list[str] = []
        done = threading.Event()
        started = self.clock()
        proc = subprocess.Popen(
            cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)

        def watch() -> None:
            while not done.wait(self.poll_s):
                if proc.poll() is not None:
                    return
                try:
                    reason = should_stop(list(events))
                except Exception as exc:        # noqa: BLE001 - a failed check is not a stop
                    self.echo(f"[moe] warning: stop check failed ({exc})")
                    continue
                if reason:
                    stop.append(reason)
                    self.echo(f"[moe] {reason}: interrupting the trainer (checkpoint, up "
                              f"to {self.grace_s:.0f} s)")
                    self._stop_child(proc, child_got_it=False)
                    return

        watcher = threading.Thread(target=watch, name="moe-watch", daemon=True)
        watcher.start()
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                for line in proc.stdout:
                    log.write(line)
                    self.echo(line.rstrip("\n"))
                    m = STEP_LINE.match(line.strip())
                    if m:
                        events.append((self.clock(), int(m.group(1))))
            code = proc.wait()
        except KeyboardInterrupt:
            done.set()
            self.echo("[moe] interrupted: letting the trainer write its checkpoint")
            self._stop_child(proc, child_got_it=os.name != "nt")
            raise
        except BaseException:
            done.set()
            self._stop_child(proc, child_got_it=False)
            raise
        finally:
            done.set()
            if watcher.is_alive():
                watcher.join()
        return ChildOutcome(code=code, stop_reason=stop[0] if stop else None,
                            events=events, started_at=started)


def run_command(cmd: list[str], log_path: Path) -> int:
    """An evaluation step: output to the console and the log; its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in proc.stdout:
            log.write(line)
            print(line, end="", flush=True)
        return proc.wait()


# ---- the launcher ----------------------------------------------------------------------

class Stop(Exception):
    """End the launch with this exit code (summary written by run())."""

    def __init__(self, code: int, status: str) -> None:
        super().__init__(status)
        self.code = code
        self.status = status


class Launcher:
    def __init__(self, *, config: str | Path, winners: str | Path | None, budget_usd: float,
                 usd_per_hour: float, ledger: Ledger, run_child: ChildRunner,
                 reserve_usd: float = RESERVE_USD, gate_minutes: float = GATE_MINUTES,
                 out: str | Path = DEFAULT_OUT, ckpt_dir: str | Path = DEFAULT_CKPT_DIR,
                 run_id: str = RUN_ID, device: str = "cuda", overrides: list[str] | None = None,
                 go_file: str | Path | None = None,
                 run_cmd: Callable[[list[str], Path], int] = run_command,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 echo: Callable[[str], None] | None = None, poll_s: float = POLL_S,
                 wait_print_s: float = WAIT_PRINT_S, max_retries: int = MAX_RETRIES,
                 retry_wait_s: float = RETRY_WAIT_S, headroom: float = HEADROOM,
                 warmup_skip_steps: int = WARMUP_SKIP_STEPS) -> None:
        self.config = Path(config)
        self.winners = Path(winners) if winners else None
        self.budget = float(budget_usd)
        self.rate_arg = float(usd_per_hour)
        self.ledger = ledger
        self.run_child = run_child
        self.reserve = float(reserve_usd)
        self.gate_s = gate_minutes * 60
        self.out = Path(out)
        self.ckpt_dir = Path(ckpt_dir)
        self.run_id = run_id
        self.device = device
        self.overrides = list(overrides or [])
        self.go = Path(go_file) if go_file else self.out / "GO"
        self.run_cmd = run_cmd
        self.clock = clock
        self.sleep = sleep
        self.echo = echo or (lambda s: print(s, flush=True))
        self.poll_s = poll_s
        self.wait_print_s = wait_print_s
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self.headroom = headroom
        self.skip = warmup_skip_steps

        self.run_dir = self.out / "runs"
        self.run_log = self.run_dir / f"{run_id}.json"
        self.run_config = self.out / "run_config.toml"
        self.plan_json = self.out / "plan.json"
        self.plan_md = self.out / "plan.md"
        self.summary = self.out / "summary.md"
        self.attempts: list[dict[str, Any]] = []
        self.evals: list[dict[str, Any]] = []
        self.plan: dict[str, Any] | None = None
        self.spent_at_start: float | None = None
        self.raw: dict[str, Any] = {}

    # -- money --

    @property
    def rate(self) -> float:
        rate = self.ledger.usd_per_hour
        return self.rate_arg if rate is None else rate

    def _tick(self) -> None:
        try:
            self.ledger.tick_if_due(TICK_S)
        except (OSError, LedgerError) as exc:
            self.echo(f"[moe] warning: spend ledger tick failed ({exc})")

    def spent(self) -> float:
        return self.ledger.spent_usd()

    # -- helpers --

    def _train_cmd(self, resume: bool) -> list[str]:
        cmd = [sys.executable, "-m", "quipu.train", "--config", str(self.run_config),
               "--run-id", self.run_id, "--run-dir", str(self.run_dir), "--device", self.device]
        return cmd + (["--resume"] if resume else [])

    def _write_run_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.run_config, dump_toml(raw))
        try:
            load_config(self.run_config)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise Stop(EXIT_USAGE, f"the resolved run config is invalid: {exc}") from exc
        return raw

    def _last_logged_step(self) -> int:
        try:
            record = json.loads(self.run_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        return max((s.get("step", 0) for s in record.get("steps", [])), default=0)

    def _record_attempt(self, phase: str, outcome: ChildOutcome, note: str = "") -> None:
        self.attempts.append({
            "phase": phase, "code": outcome.code, "stop": outcome.stop_reason,
            "last_step": self._last_logged_step(), "note": note, "at": _now_iso()})

    @property
    def batch_tokens(self) -> int:
        return int(self.raw["train"]["batch_tokens"])

    # -- the whole launch --

    def run(self) -> int:
        try:
            code, status = self._run()
        except Stop as stop:
            code, status = stop.code, stop.status
        except KeyboardInterrupt:
            self.echo("[moe] interrupted (the trainer's checkpoint is kept; run again to resume)")
            return EXIT_INTERRUPTED
        if code == EXIT_USAGE:
            # Not the end of the run (a mistyped restart): no summary, so sync.sh goes on.
            self.echo(f"error: {status}")
        else:
            self._write_summary(status, code)
        return code

    def _run(self) -> tuple[int, str]:
        try:
            self.raw = resolve_raw(self.config, self.winners, self.overrides)
            load_config(self.config, None)          # the file itself must load
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise Stop(EXIT_USAGE, f"usage: {exc}") from exc
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            if self.ledger.ensure_session(self.rate_arg):
                self.echo(f"[moe] started a box session in {self.ledger.path} at "
                          f"${self.rate_arg:.2f}/h (run `python -m quipu.spend start` when "
                          "the box starts, so the setup counts too)")
            self.ledger.tick()
        except (OSError, ValueError, LedgerError) as exc:
            raise Stop(EXIT_USAGE, f"spend ledger: {exc}") from exc
        self.spent_at_start = self.spent()
        self.echo(f"[moe] box spend so far ${self.spent_at_start:.2f} of ${self.budget:.2f} "
                  f"(ledger {self.ledger.path}; ${self.rate:.2f}/h)")

        self.plan = self._load_plan()
        if self.summary.exists():
            self.summary.unlink()                   # a new launch; sync.sh waits for the new one
        if self.plan is None:
            if self.go.exists():
                self.go.unlink()
                self.echo(f"[moe] removed a stale {self.go} (it predates this plan)")
            self._gate()
        if not self.plan.get("completed"):
            self._wait_for_go()
            self._refit_at_go()
            self._long_run()
        self._run_evals()
        failed = [e["name"] for e in self.evals if e["code"] != 0]
        status = "completed" + (f"; evaluation failed: {', '.join(failed)}" if failed else "")
        return EXIT_OK, status

    # -- the plan file --

    def _load_plan(self) -> dict[str, Any] | None:
        try:
            plan = json.loads(self.plan_json.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise Stop(EXIT_USAGE, f"{self.plan_json} is unreadable ({exc})") from exc
        if plan.get("base_hash") != base_hash(self.raw):
            raise Stop(EXIT_USAGE, (
                f"{self.plan_json} was made for another config/winners/--override; the "
                f"checkpoints in {self.ckpt_dir} belong to that run. Move {self.out} and "
                f"{self.ckpt_dir} aside to start over, or run with the same settings"))
        self.echo(f"[moe] reusing the plan in {self.plan_json} (gate at step "
                  f"{plan['gate_step']}, {plan['tokens_per_s']:,.0f} tok/s)")
        return plan

    def _save_plan(self) -> None:
        write_text_atomic(self.plan_json, json.dumps(self.plan, indent=2))
        write_text_atomic(self.plan_md, self._plan_md())

    # -- 1. the throughput gate --

    def _gate(self) -> None:
        train = self.raw["train"]
        cfg_total = int(train["total_tokens"])
        gate_usd = self.gate_s / 3600 * self.rate
        if self.spent() + gate_usd + self.reserve > self.budget + 1e-9:
            raise Stop(EXIT_BUDGET, (
                f"stopped by the budget before the gate: spent ${self.spent():.2f} + gate "
                f"${gate_usd:.2f} + reserve ${self.reserve:.2f} > ${self.budget:.2f}"))
        self._write_run_config(with_train(
            self.raw, total_tokens=cfg_total, milestones=[],
            ckpt_dir=self.ckpt_dir.as_posix(), budget_usd=0.0, usd_per_hour=0.0))
        resume = self.run_log.exists()
        start_step = self._last_logged_step() if resume else 0
        deadline = self.clock() + self.gate_s
        self.echo(f"[moe] throughput gate: {self.gate_s / 60:.0f} min of the full-size run"
                  + (" (resuming)" if resume else ""))

        def should_stop(events: list[tuple[float, int]]) -> str | None:
            self._tick()
            if self.clock() >= deadline:
                return "gate time is up"
            if self.spent() + self.reserve > self.budget + 1e-9:
                return "budget reached during the gate"
            return None

        outcome = self.run_child(self._train_cmd(resume), self.out / "logs" / "gate.log",
                                 should_stop)
        self._record_attempt("gate", outcome)
        self._tick()
        if outcome.stop_reason and outcome.stop_reason.startswith("budget"):
            raise Stop(EXIT_BUDGET, f"stopped by the spend guard during the gate at step "
                                    f"{self._last_logged_step()}")
        if outcome.code != 0 and outcome.stop_reason is None:
            raise Stop(*self._failure(outcome.code, "the gate"))

        bt = self.batch_tokens
        tps = effective_tokens_per_s(outcome.events, bt, start_step + self.skip)
        if outcome.code == 0:
            # A run small enough to finish inside the gate (the laptop smoke): done.
            self.plan = {
                "base_hash": base_hash(self.raw), "made_at": _now_iso(),
                "tokens_per_s": tps or 0.0, "startup_s": 0.0,
                "gate_step": self._last_logged_step(), "gate_minutes": self.gate_s / 60,
                "config_total_tokens": cfg_total,
                "config_milestones": list(train.get("milestones", [])),
                "total_tokens": cfg_total, "milestones": [], "trimmed": False,
                "completed": True, "notes": ["the whole run finished inside the gate"]}
            self._save_plan()
            return
        if tps is None:
            raise Stop(EXIT_FAILED, (
                "the gate did not run long enough to measure tokens/s (fewer than two step "
                f"lines after the first {self.skip} steps); raise --gate-minutes"))
        # Startup (process, model, compile trial) = time to the first step line less
        # the steps before it at the measured rate.
        first_t, first_step = outcome.events[0]
        startup = max(0.0, first_t - outcome.started_at - (first_step - start_step) * bt / tps)
        gate_step = self._last_logged_step()
        if not (self.ckpt_dir / "latest.pt").exists():
            gate_step = 0          # no checkpoint to resume from: the run starts over
        self.plan = {
            "base_hash": base_hash(self.raw), "made_at": _now_iso(),
            "tokens_per_s": tps, "startup_s": startup, "gate_step": gate_step,
            "gate_minutes": self.gate_s / 60,
            "config_total_tokens": cfg_total, "config_milestones": list(train.get("milestones", [])),
            "completed": False, "notes": [],
        }
        self.echo(f"[moe] gate: {tps:,.0f} tok/s after the first {self.skip} steps; startup "
                  f"{startup:.0f} s; checkpoint at step {gate_step}")
        self._fit(first=True)

    def _fit(self, first: bool) -> None:
        """Set plan total_tokens / milestones to what fits now; write plan.md. Raises
        Stop(4) if not even one more step fits."""
        plan, train = self.plan, self.raw["train"]
        bt = self.batch_tokens
        tps = plan["tokens_per_s"] * (1 - self.headroom)
        gate_step = plan["gate_step"]
        want = plan["config_total_tokens"] if first else plan["total_tokens"]
        spent = self.spent()
        proj = project(tokens_total=want, tokens_done=gate_step * bt, tokens_per_s=tps,
                       usd_per_hour=self.rate, spent_usd=spent, reserve_usd=self.reserve,
                       startup_s=plan["startup_s"], budget_usd=self.budget)
        total = want
        if not proj.fits:
            total = fit_total_tokens(budget_usd=self.budget, spent_usd=spent,
                                     reserve_usd=self.reserve, usd_per_hour=self.rate,
                                     tokens_per_s=tps, startup_s=plan["startup_s"],
                                     tokens_done=gate_step * bt, batch_tokens=bt)
        steps = total // bt
        cfg_steps = plan["config_total_tokens"] // bt
        if steps <= max(gate_step, int(train["warmup_steps"])):
            plan["notes"].append(
                f"{_now_iso()}: nothing more fits: spent ${spent:.2f}, budget "
                f"${self.budget:.2f}, reserve ${self.reserve:.2f}")
            plan.update(total_tokens=want, trimmed=True, fits=False,
                        milestones=plan.get("milestones", []))
            self._save_plan()
            raise Stop(EXIT_BUDGET, "stopped by the budget: the run does not fit (plan.md)")
        if not first and total != want:
            plan["notes"].append(
                f"{_now_iso()}: re-trimmed at GO (the wait cost money): total_tokens "
                f"{want:,} -> {total:,}")
            self.echo(f"[moe] the approved plan no longer fits: re-trimmed to {total:,} tokens")
        plan.update(
            total_tokens=total, trimmed=total < plan["config_total_tokens"], fits=True,
            milestones=list(rescale_milestones(plan["config_milestones"], cfg_steps, steps,
                                               gate_step)),
            spent_at_plan=spent, lr_note=gate_reuse_note(train, gate_step, cfg_steps, steps))
        self._save_plan()
        final = self._projection(total)
        self.echo(f"[moe] plan: {total:,} tokens ({steps:,} steps), {final.hours:.1f} h, "
                  f"${final.train_usd:.2f} more; total ${final.total_usd:.2f} of "
                  f"${self.budget:.2f} with the ${self.reserve:.2f} reserve"
                  + (" (TRIMMED)" if plan["trimmed"] else "") + f" -> {self.plan_md}")

    def _projection(self, total: int) -> Projection:
        plan = self.plan
        return project(tokens_total=total, tokens_done=plan["gate_step"] * self.batch_tokens,
                       tokens_per_s=plan["tokens_per_s"] * (1 - self.headroom),
                       usd_per_hour=self.rate, spent_usd=self.spent(),
                       reserve_usd=self.reserve, startup_s=plan["startup_s"],
                       budget_usd=self.budget)

    def _plan_md(self) -> str:
        p, bt = self.plan, self.batch_tokens
        cfg_total = p["config_total_tokens"]
        total = p.get("total_tokens", cfg_total)
        per_min = self.rate / 60
        lines = [f"# quipu-moe full run: plan", "",
                 f"Written {_now_iso()} by scripts/remote/run_moe.py.", ""]
        if not p.get("fits", True):
            lines += ["**The run does not fit the budget.** " + p["notes"][-1], "",
                      "Raise --budget-usd (if the owner agrees) and run the launcher again.", ""]
        cfg = self._projection(cfg_total)
        plan = self._projection(total)
        lines += [
            f"- Measured throughput: **{p['tokens_per_s']:,.0f} tokens/s** (effective: step "
            f"lines' arrival times after the first {self.skip} steps, evals and checkpoints "
            f"included; {p['gate_minutes']:.0f}-minute gate); planned at "
            f"{(1 - self.headroom):.0%} of it.",
            f"- Startup of the long run (resume + compile), from the gate: {p['startup_s']:.0f} s.",
            f"- Box rate ${self.rate:.2f}/h = **${per_min:.4f}/min**. Box spend so far "
            f"${self.spent():.2f} (ledger {self.ledger.path}).",
            f"- Budget ${self.budget:.2f}; reserve ${self.reserve:.2f} kept for the chat SFT "
            "and the final eval/export.", "",
            "| | configured | planned |", "|---|---|---|",
            f"| tokens | {cfg_total:,} | {total:,} |",
            f"| steps | {cfg_total // bt:,} | {total // bt:,} |",
            f"| milestones | {p['config_milestones']} | {p.get('milestones')} |",
            f"| hours left | {cfg.hours:.2f} | {plan.hours:.2f} |",
            f"| cost of the rest of the run | ${cfg.train_usd:.2f} | ${plan.train_usd:.2f} |",
            f"| projected total spend (incl. reserve) | ${cfg.total_usd:.2f} | ${plan.total_usd:.2f} |",
            f"| budget left after the run (reserve included) | ${cfg.remaining_after:.2f} | "
            f"${plan.remaining_after:.2f} |", "",
        ]
        if p.get("trimmed"):
            lines += [f"Trimmed: yes. total_tokens {cfg_total:,} -> {total:,} "
                      f"({total / cfg_total:.1%} of the configured run, whole batches of "
                      f"{bt:,}); milestones rescaled by the same fraction "
                      f"{p['config_milestones']} -> {p.get('milestones')}.", ""]
        else:
            lines += ["Trimmed: no. The configured run fits.", ""]
        gate_step = p["gate_step"]
        if gate_step:
            lines += [f"Gate reuse: the long run resumes from the gate's checkpoint at step "
                      f"{gate_step} ({gate_step * bt:,} tokens already trained). "
                      f"{p.get('lr_note', '').capitalize()}. The run log's config block is the gate's "
                      f"(milestones off, untrimmed total); {self.run_config} is the run's.", ""]
        else:
            lines += ["Gate reuse: none (the gate left no checkpoint); the long run starts "
                      "from step 0.", ""]
        lines += [f"Waiting costs money: every minute before GO is ${per_min:.4f}. To start: "
                  f"`touch {self.go.as_posix()}` on the box. If the wait makes this plan "
                  "unaffordable, it is trimmed again at GO and this file says so.", ""]
        if p.get("notes"):
            lines += ["## Notes", ""] + [f"- {n}" for n in p["notes"]] + [""]
        return "\n".join(lines)

    # -- 2. GO --

    def _wait_for_go(self) -> None:
        if self.go.exists():
            return
        per_min = self.rate / 60
        t0 = self.clock()
        next_print = t0
        self.echo(f"[moe] plan written to {self.plan_md}; waiting for {self.go} "
                  f"(${per_min:.4f}/min while waiting)")
        while not self.go.exists():
            self._tick()
            now = self.clock()
            spent = self.spent()
            if now >= next_print:
                self.echo(f"[moe] waiting for {self.go}: {(now - t0) / 60:.0f} min so far, "
                          f"${per_min:.4f}/min (${self.rate:.2f}/h); spent ${spent:.2f} of "
                          f"${self.budget:.2f}")
                next_print = now + self.wait_print_s
            step_usd = self.batch_tokens / self.plan["tokens_per_s"] / 3600 * self.rate
            startup_usd = self.plan["startup_s"] / 3600 * self.rate
            if spent + self.reserve + startup_usd + step_usd > self.budget + 1e-9:
                self.plan["notes"].append(f"{_now_iso()}: the wait for GO used up the budget")
                self.plan["fits"] = False
                self._save_plan()
                raise Stop(EXIT_BUDGET, "stopped by the budget while waiting for GO")
            self.sleep(self.poll_s)
        self._tick()
        self.echo(f"[moe] GO after {(self.clock() - t0) / 60:.1f} min")

    def _refit_at_go(self) -> None:
        """Only before the long run has moved past the gate: trimming a run already
        under way would change its schedule mid-run (the guard covers that case)."""
        if self._last_logged_step() > self.plan["gate_step"]:
            return
        if not self._projection(self.plan["total_tokens"]).fits:
            self._fit(first=False)

    # -- 3. the long run --

    def _long_run(self) -> None:
        plan, train = self.plan, self.raw["train"]
        bt = self.batch_tokens
        steps = plan["total_tokens"] // bt
        planned_tps = plan["tokens_per_s"]
        remaining = max(0.0, self.budget - self.spent())
        self._write_run_config(with_train(
            self.raw, total_tokens=plan["total_tokens"], milestones=plan["milestones"],
            ckpt_dir=self.ckpt_dir.as_posix(), budget_usd=round(remaining, 4),
            usd_per_hour=self.rate))
        retries = 0
        while True:
            start = self._last_logged_step() if self.run_log.exists() else 0
            if over_budget(self.spent(), self._margin(steps, start, planned_tps), self.reserve,
                           self.budget):
                raise Stop(EXIT_BUDGET, (
                    f"stopped by the spend guard before launching (step {start}/{steps}): "
                    f"spent ${self.spent():.2f} + next interval "
                    f"${self._margin(steps, start, planned_tps):.2f} + reserve "
                    f"${self.reserve:.2f} > ${self.budget:.2f}"))

            def should_stop(events: list[tuple[float, int]], start: int = start) -> str | None:
                self._tick()
                live = effective_tokens_per_s(events, bt, start + self.skip)
                tps = planned_tps if live is None else min(planned_tps, live)
                # Step lines come every 10 steps: extrapolate from the last one (at the
                # conservative rate, so this never overestimates progress by much).
                step = start
                if events:
                    t_last, s_last = events[-1]
                    step = min(steps, s_last + int(max(0.0, self.clock() - t_last) * tps / bt))
                spent, margin = self.spent(), self._margin(steps, step, tps)
                if over_budget(spent, margin, self.reserve, self.budget):
                    return (f"spend guard: spent ${spent:.2f} + next interval ${margin:.2f} + "
                            f"reserve ${self.reserve:.2f} > budget ${self.budget:.2f} "
                            f"(step {step})")
                return None

            resume = self.run_log.exists()
            self.echo(f"[moe] long run: {'resuming at step ' + str(start) if resume else 'starting'}"
                      f" ({steps:,} steps; budget left ${self.budget - self.spent():.2f})")
            outcome = self.run_child(self._train_cmd(resume),
                                     self.out / "logs" / f"train_{len(self.attempts)}.log",
                                     should_stop)
            self._record_attempt("train", outcome)
            self._tick()
            if outcome.stop_reason:
                raise Stop(EXIT_BUDGET, (
                    f"stopped by the spend guard at step {self._last_logged_step()}/{steps} "
                    "(interrupt checkpoint written; run again with a larger --budget-usd to "
                    "continue)"))
            if outcome.code == 0:
                return
            if (outcome.code in (TRAIN_USAGE, TRAIN_NONFINITE)
                    or outcome.code in TRAIN_INTERRUPT_CODES or retries >= self.max_retries):
                raise Stop(*self._failure(outcome.code, "the long run"))
            retries += 1
            self.echo(f"[moe] trainer exited {outcome.code}; retry {retries}/"
                      f"{self.max_retries} from the last checkpoint in {self.retry_wait_s:.0f} s")
            self.sleep(self.retry_wait_s)
            self._tick()

    def _margin(self, steps: int, at_step: int, tokens_per_s: float) -> float:
        return guard_margin_usd(steps - at_step, int(self.raw["train"]["eval_every"]),
                                self.batch_tokens, tokens_per_s, self.rate)

    @staticmethod
    def _failure(code: int, what: str) -> tuple[int, str]:
        if code == TRAIN_USAGE:
            return EXIT_USAGE, f"{what} failed: the trainer's usage/config error (exit 2)"
        if code == TRAIN_NONFINITE:
            return EXIT_NONFINITE, f"{what} stopped: the trainer's non-finite stop (exit 3)"
        if code in TRAIN_INTERRUPT_CODES:
            return EXIT_INTERRUPTED, f"{what} was interrupted (exit {code})"
        return EXIT_FAILED, f"{what} failed (exit {code})"

    # -- 4. evaluation and summary --

    def _run_evals(self) -> None:
        cmd = [sys.executable, str(ROOT / "scripts" / "milestone_eval.py"), "--config",
               str(self.run_config), "--out-dir", str(self.out / "milestones"),
               "--device", self.device]
        self.echo("[moe] milestone evaluation")
        code = self.run_cmd(cmd, self.out / "logs" / "milestone_eval.log")
        self.evals.append({"name": "milestone_eval", "code": code, "cmd": cmd})
        self._tick()

    def _write_summary(self, status: str, code: int) -> None:
        try:
            self.ledger.tick()
        except (OSError, LedgerError) as exc:
            self.echo(f"[moe] warning: spend ledger tick failed ({exc})")
        try:
            spent = self.spent()
        except LedgerError:
            spent = float("nan")
        record: dict[str, Any] = {}
        try:
            record = json.loads(self.run_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        steps = record.get("steps") or []
        evals = record.get("evals") or []
        last = steps[-1] if steps else {}
        p = self.plan or {}
        bt = self.batch_tokens if self.raw else 0
        lines = ["# quipu-moe full run: summary", "", f"Written {_now_iso()}.", "",
                 f"**Status: {status}** (launcher exit {code})", "",
                 f"- Config {self.config}" + (f" + winners {self.winners}" if self.winners else "")
                 + (f" + overrides {self.overrides}" if self.overrides else "")
                 + f"; resolved in {self.run_config}.",
                 f"- Box spend ${spent:.2f} of the ${self.budget:.2f} budget "
                 f"(${self.spent_at_start or 0:.2f} when this launch started; reserve "
                 f"${self.reserve:.2f} for the chat SFT and eval/export; ledger {self.ledger.path}).",
                 f"- Budget left: ${self.budget - spent:.2f}."]
        if p:
            lines += [f"- Plan: {p.get('total_tokens', 0):,} tokens "
                      f"({p.get('total_tokens', 0) // max(bt, 1):,} steps)"
                      + (", trimmed from " + f"{p['config_total_tokens']:,}" if p.get("trimmed") else "")
                      + f"; gate {p.get('tokens_per_s', 0):,.0f} tok/s; milestones "
                      f"{p.get('milestones')}; see {self.plan_md}.",
                      f"- Gate reuse: resumed from the gate's checkpoint at step {p.get('gate_step')}"
                      f" ({p.get('lr_note', '-')})."]
        if steps:
            lines.append(f"- Trained to step {last.get('step')} ({last.get('tokens', 0):,} tokens), "
                         f"last train loss {last.get('train_loss', float('nan')):.4f}"
                         + (f", last val loss {evals[-1]['val_loss']:.4f} at step "
                            f"{evals[-1]['step']}" if evals else "") + ".")
        lines += ["", "## Attempts", ""]
        lines += [f"- {a['phase']}: exit {a['code']}" + (f", stopped ({a['stop']})" if a["stop"] else "")
                  + f", run log at step {a['last_step']} ({a['at']})" for a in self.attempts] or ["(none)"]
        lines += ["", "## Evaluation", ""]
        lines += [f"- {e['name']}: exit {e['code']}" for e in self.evals] or [
            "(not run: evaluation follows a completed run)"]
        lines += ["", "## Next", ""]
        if code == EXIT_OK:
            lines.append(f"Chat SFT (M12) in this session from the final checkpoint in "
                         f"{self.ckpt_dir}, inside the reserve; then eval, copy back, verify, "
                         "and destroy the box after the owner's go-ahead.")
        elif code == EXIT_BUDGET:
            lines.append("The checkpoint is kept. To continue, run the launcher again with a "
                         "larger --budget-usd (the owner decides); otherwise evaluate what is there.")
        else:
            lines.append("See the attempt logs in " + str(self.out / "logs") + ".")
        try:
            write_text_atomic(self.summary, "\n".join(lines) + "\n")
        except OSError as exc:
            self.echo(f"[moe] warning: could not write {self.summary} ({exc})")
        self.echo(f"[moe] {status}; summary in {self.summary}")


# ---- CLI --------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/quipu-moe.toml")
    p.add_argument("--winners", help="results/ab/winners.toml (the A/B overrides)")
    p.add_argument("--budget-usd", type=float, default=20.0,
                   help="cap on the box ledger's total spend this session (default 20)")
    p.add_argument("--usd-per-hour", type=float,
                   help="the box's rate (required; starts the box session if `python -m "
                        "quipu.spend start` was not run)")
    p.add_argument("--reserve-usd", type=float, default=RESERVE_USD,
                   help="kept back for the chat SFT and the final eval/export")
    p.add_argument("--gate-minutes", type=float, default=GATE_MINUTES)
    p.add_argument("--headroom", type=float, default=HEADROOM,
                   help="plan at (1 - headroom) x the measured tokens/s")
    p.add_argument("--warmup-skip-steps", type=int, default=WARMUP_SKIP_STEPS)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--go-file", help="default <out>/GO")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="cuda, never auto: a driver fault must fail loudly, not train on the CPU")
    p.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    p.add_argument("--retry-wait-s", type=float, default=RETRY_WAIT_S)
    p.add_argument("--ledger", help="spend ledger (default $QUIPU_SPEND_LEDGER or results/spend.json)")
    args = p.parse_args(argv)

    rate = args.usd_per_hour
    if rate is None or not math.isfinite(rate) or rate <= 0:
        print("error: pass --usd-per-hour > 0 (the box's rate)", file=sys.stderr)
        return EXIT_USAGE
    for name, allow_zero in (("budget_usd", False), ("reserve_usd", True), ("gate_minutes", False)):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
            print(f"error: --{name.replace('_', '-')} must be a number > 0"
                  + (" (or 0)" if allow_zero else ""), file=sys.stderr)
            return EXIT_USAGE
    if not 0 <= args.headroom < 1:
        print("error: --headroom must be in [0, 1)", file=sys.stderr)
        return EXIT_USAGE
    try:
        ledger = Ledger.load(args.ledger)
    except LedgerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    # Absolute paths: the trainer runs with the repo root as its working directory.
    out = Path(args.out).resolve()
    launcher = Launcher(
        config=Path(args.config).resolve(),
        winners=Path(args.winners).resolve() if args.winners else None,
        budget_usd=args.budget_usd, usd_per_hour=rate, ledger=ledger,
        reserve_usd=args.reserve_usd, gate_minutes=args.gate_minutes, out=out,
        ckpt_dir=Path(args.ckpt_dir).resolve(), run_id=args.run_id, device=args.device,
        overrides=args.override, go_file=Path(args.go_file).resolve() if args.go_file else None,
        run_child=SubprocessChild(inductor_cache=out / "inductor-cache"),
        max_retries=args.max_retries, retry_wait_s=args.retry_wait_s,
        headroom=args.headroom, warmup_skip_steps=args.warmup_skip_steps)
    with Ticker(ledger):
        return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
