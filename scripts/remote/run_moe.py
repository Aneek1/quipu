"""The rented-box launcher for the quipu-moe full run (spec sections 6.2-6.4 and 8,
plan Task M9).

    python -m quipu.spend start --usd-per-hour R          # once, when the box starts
    python scripts/remote/run_moe.py --config configs/quipu-moe.toml \\
        --winners results/ab/winners.toml --budget-usd 20 --reserve-usd 1.00 --usd-per-hour R

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
   the configured tokens at the planning rate, plus the box's spend so far (the
   shared ledger, quipu/spend.py) plus the long run's startup plus --reserve-usd
   (default $1.00: the chat SFT, evals and the copy-back, spec 13). The planning
   rate is the measured rate less --headroom (5%), with each eval / checkpoint
   interval's overhead added back: EVAL_OVERHEAD_S per eval_every steps and the
   longest checkpoint save the gate reported (else SAVE_OVERHEAD_S) per ckpt_every
   steps (a 15-minute gate usually sees neither). If that is over --budget-usd,
   total_tokens is cut to what fits, rounded down to whole batches, and the
   milestones are rescaled by their fraction of the run (deduplicated, none past
   the end). Then it waits for the GO file (--go-file, default <out>/GO; the
   controller creates it once the owner approves), printing what each minute of
   waiting costs; the ledger keeps ticking. The wait is bounded: after
   --go-max-wait-min (90), or as soon as what still fits falls below RETRIM_FLOOR
   (80%) of the approved tokens (the plan is then re-fitted for a fresh approval),
   it writes plan.md + summary.md (stop the instance from the Vast console; resume
   later with the same command) and exits 5. At GO the plan is checked again
   against the spend then: if the wait made it unaffordable it is trimmed again
   (plan.md says so), and if nothing fits it stops with exit 4.
4. The long run resumes from the gate's checkpoint, so the gate's steps are kept,
   not thrown away. The LR schedule depends on the run's length only after warm-up,
   so when the gate ended inside warm-up (the normal case: warmup_steps 500 is ~40
   min on a 5090, the gate 15) the reused steps are exactly the ones the trimmed run
   would have trained; plan.md reports it either way. Milestones that the rescale
   puts inside the gate move to the gate's checkpoint step (the resumed trainer
   writes that milestone from the weights it just loaded).
   Crashes are retried from the last checkpoint (like weekend.py): exit 1 or
   anything unexpected, up to --max-retries; 2 (usage), 3 (non-finite stop), 4 (the
   trainer's budget backstop) and interrupts are not retried. The gate is not
   retried: a crash there ends the launch (run it again). Once, after the long
   run's first full eval/checkpoint interval, the measured rate is compared with the
   planning rate: if it is more than REFIT_TOLERANCE (2%) lower, total_tokens is
   re-fitted at the measured rate (only ever trimmed, never extended), the trainer is
   interrupted (checkpoint) and resumed with the new length and rescaled milestones,
   and plan.md records it.
5. Spend guard: while a child trains (and before each launch) the launcher polls
   spend + cost of the next eval interval (min(eval_every, steps left) at the slower
   of the planned and the attempt's measured rate) + --reserve-usd; once that is
   above --budget-usd the child is interrupted (SIGINT -> interrupt checkpoint ->
   up to STOP_GRACE_S, or 3 x the longest checkpoint save the trainer has reported
   if that is longer, then terminate, then kill) and the launcher exits 4. The
   budget is the box ledger's: what remains is --budget-usd minus everything the box
   has cost so far (setup, shards, the A/B runs, earlier invocations), so a restart
   with a larger --budget-usd resumes where it stopped. A box session ended under the
   launcher (`spend stop --force`) is a budget stop too.
   If the launcher dies (SIGKILL, OOM): the trainer's output is a pipe to it, so the
   trainer's next output line (step lines every 10 steps) fails, which quipu.train
   takes as an orphan stop: it checkpoints and exits 130, appending its remaining
   output to the attempt's log (logs/gate.log, logs/train_N.log: the launcher tees
   every line there, flushed, and names the file in $QUIPU_TRAIN_LOG). Backstop,
   should it not notice: every trainer it launches gets train.budget_usd = the
   budget left at launch less the reserve, plus half the stop grace as slack past
   the guard's threshold (the gate: at most GATE_BACKSTOP_FACTOR x the gate's cost)
   and train.usd_per_hour = the box rate; it checkpoints and exits 4 on that. The
   slack means a live launcher's guard always stops the trainer first, and a signal
   that reaches a trainer already writing its backstop checkpoint is ignored. The
   trainer runs in a session of its own (POSIX; a process group on Windows), so a
   terminal hangup never reaches it directly; SIGTERM / SIGHUP to the launcher take
   Ctrl+C's path: the trainer is sent the interrupt and given its grace to write its
   checkpoint, then the launcher exits 130. However the launcher exits, it releases
   its tag in the ledger, so `spend stop` right after it needs no --force.
6. At the end: scripts/milestone_eval.py on the run config, and summary.md. summary.md
   is written only when the run is over (completed, stopped by the guard, or failed
   for good); sync.sh stops syncing when it appears. plan.json then records
   "completed" (and "evaluated"), so running the same command again does nothing.

Restarts: plan.json is reused (no second gate) as long as the resolved config is the
same (anything else is exit 2: move <out> and the checkpoints aside to start over);
a GO from before the plan existed is removed.

Exit codes: 0 done, 1 training failed (after retries) or the gate could not measure,
2 usage error, 3 the trainer's non-finite stop, 4 stopped by the budget, 5 no GO in
time (or the wait cost too much of the plan), 130 interrupted.
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

from quipu import childproc
from quipu.config import INHERIT_KEYS, _merge, load_config, parse_overrides
from quipu.fsio import write_text_atomic
from quipu.spend import TICK_S, Ledger, LedgerError, SessionEnded, Ticker

ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NONFINITE = 3
EXIT_BUDGET = 4
EXIT_NO_GO = 5
EXIT_INTERRUPTED = 130
# The trainer's exit codes (quipu.train): 0 ok, 1 crash, 2 usage, 3 non-finite stop,
# 4 its budget backstop, 130 interrupt; Windows reports a Ctrl+C'd child as
# STATUS_CONTROL_C_EXIT.
TRAIN_USAGE, TRAIN_NONFINITE, TRAIN_BUDGET = 2, 3, 4
TRAIN_INTERRUPT_CODES = frozenset({130, -1073741510, 3221225786})

RUN_ID = "quipu-moe"
DEFAULT_OUT = "results/moe"
DEFAULT_CKPT_DIR = "checkpoints/quipu-moe"
GATE_MINUTES = 15.0
RESERVE_USD = 1.00          # chat SFT (~$0.25, spec 13) + evals + copy-back
HEADROOM = 0.05             # plan at 95% of the measured tokens/s
EVAL_OVERHEAD_S = 30.0      # one eval (eval_batches forwards), per eval_every steps
SAVE_OVERHEAD_S = 60.0      # one checkpoint save when the gate saw none, per ckpt_every
REFIT_TOLERANCE = 0.02      # the long run's first interval this much slower: re-fit
GO_MAX_WAIT_MIN = 90.0      # the GO wait gives up after this long (exit 5)
RETRIM_FLOOR = 0.80         # ... or once less than this share of the approved run fits
GATE_BACKSTOP_FACTOR = 2.0  # an orphaned gate trainer stops after 2 x the gate's cost
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
# quipu.train's Trainer.save_checkpoint prints this after every checkpoint.
SAVE_LINE = re.compile(r"^checkpoint step (\d+) saved in ([\d.]+) s")
SAVE_GRACE_FACTOR = 3.0     # a stop waits at least this many x the longest save seen


def parse_save_s(line: str) -> float | None:
    """The save time from a trainer `checkpoint step N saved in X.X s` line, else None."""
    m = SAVE_LINE.match(line.strip())
    return float(m.group(2)) if m else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _echo(line: str) -> None:
    """print, but a closed terminal (a hangup: EIO, a broken pipe) does not stop the
    launcher mid-run: the log files still get everything."""
    try:
        print(line, flush=True)
    except (OSError, ValueError):
        pass


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


def planning_tokens_per_s(tokens_per_s: float, *, headroom: float, batch_tokens: int,
                          eval_every: int, ckpt_every: int, eval_s: float,
                          save_s: float) -> float:
    """The rate the plan is costed at: the measured tokens/s less `headroom`, with
    each interval's overhead spread over its steps (eval_s every eval_every steps,
    save_s every ckpt_every steps). A short gate sees few or no evals and saves, so
    its measured rate alone overstates what the long run gets."""
    if tokens_per_s <= 0:
        return 0.0
    step_s = (batch_tokens / (tokens_per_s * (1 - headroom))
              + eval_s / max(1, eval_every) + save_s / max(1, ckpt_every))
    return batch_tokens / step_s


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
        if not self.tokens_left:
            train_s = 0.0
        elif self.tokens_per_s <= 0:
            train_s = math.inf
        else:
            train_s = self.tokens_left / self.tokens_per_s
        return (train_s + self.startup_s) / 3600

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


def refit_milestones(config_milestones, cfg_steps: int, old_steps: int, new_steps: int,
                     gate_step: int, at_step: int) -> tuple[int, ...]:
    """Milestones after a mid-run trim at `at_step`, one per configured milestone: one
    the run already reached (its current position, as rescale_milestones placed it for
    old_steps, is at or before at_step) stays where it is; the others are rescaled to
    new_steps, and one that now falls before at_step moves to at_step (the resumed
    trainer writes it from the weights it loads). Deduplicated, none at or past the end."""
    out: set[int] = set()
    for m in config_milestones:
        old = max(1, round(m * old_steps / cfg_steps))
        if gate_step and old <= gate_step:
            old = gate_step
        if old <= at_step:
            if old < old_steps:
                out.add(old)
            continue
        new = max(at_step, round(m * new_steps / cfg_steps))
        if new < new_steps:
            out.add(new)
    return tuple(sorted(out))


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

def _refuse_inherit(where: str, table: dict[str, Any]) -> None:
    used = [k for k in INHERIT_KEYS if k in table]
    if used:
        raise ValueError(
            f"{where} uses {', '.join(used)}: the launcher takes a self-contained config "
            "(write the merged values into it; `inherit` is for quipu.train configs such "
            "as quipu-moe-sft.toml)")


def resolve_raw(config: str | Path, winners: str | Path | None,
                overrides: list[str]) -> dict[str, Any]:
    """The config file with the winners and --override flags merged in (as
    load_config merges them). Launcher-owned train keys are refused, and so is a
    config or winners file that uses `inherit` / `inherit_if_present`: this merge
    reads the TOML itself and writes the result elsewhere, where an inherit path
    would resolve against the wrong directory (or its base be dropped unseen)."""
    raw = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    _refuse_inherit(f"config {config}", raw)
    layers: list[tuple[str, dict[str, Any]]] = []
    if winners:
        layers.append((f"winners {winners}", tomllib.loads(
            Path(winners).read_text(encoding="utf-8"))))
        _refuse_inherit(*layers[-1])
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
    save_s: list[float] = dataclasses.field(default_factory=list)   # checkpoint save times


# should_stop gets the attempt's step events so far and returns a reason to stop, or None.
StopCheck = Callable[[list[tuple[float, int]]], "str | None"]
ChildRunner = Callable[[list[str], Path, StopCheck], ChildOutcome]


class SubprocessChild:
    """Runs one trainer process, echoing and logging its output and recording when
    each step line arrived. A watcher thread polls should_stop every poll_s; when it
    returns a reason the child is interrupted (SIGINT; on Windows CTRL_BREAK to its
    own process group, which the trainer handles as SIGBREAK) and given grace_s to
    write its interrupt checkpoint, then terminated, then killed (the same ladder as
    ab_runs.SubprocessRunner). The child runs in a session of its own on POSIX (a
    process group on Windows; quipu.childproc.popen_kwargs), so a terminal hangup or
    Ctrl+C never reaches it directly: a Ctrl+C here, or SIGTERM / SIGHUP (main
    installs handlers that raise KeyboardInterrupt), is forwarded to it explicitly,
    with the same grace. The children share a persistent inductor cache, so the
    long run reuses the gate's compile."""

    def __init__(self, cwd: str | Path = ROOT, echo: Callable[[str], None] | None = None,
                 grace_s: float = STOP_GRACE_S, kill_wait_s: float = KILL_WAIT_S,
                 poll_s: float = POLL_S, inductor_cache: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.cwd = Path(cwd)
        self.echo = echo or _echo
        self.grace_s = grace_s
        self.kill_wait_s = kill_wait_s
        self.poll_s = poll_s
        self.inductor_cache = inductor_cache
        self.clock = clock
        self.max_save_s = 0.0          # longest checkpoint save seen, over every attempt

    @property
    def effective_grace_s(self) -> float:
        """grace_s, raised to SAVE_GRACE_FACTOR x the longest checkpoint save seen."""
        return max(self.grace_s, SAVE_GRACE_FACTOR * self.max_save_s)

    def _saw_save(self, save_s: float) -> None:
        before = self.effective_grace_s
        self.max_save_s = max(self.max_save_s, save_s)
        if self.effective_grace_s > before:
            self.echo(f"[moe] checkpoint save took {save_s:.1f} s: stop grace now "
                      f"{self.effective_grace_s:.0f} s")

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
        grace = self.effective_grace_s
        if child_got_it:
            if self._wait(proc, grace):
                return
            if self._interrupt(proc) and self._wait(proc, self.kill_wait_s):
                return
        elif self._interrupt(proc) and self._wait(proc, grace):
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
        saves: list[float] = []
        stop: list[str] = []
        done = threading.Event()
        # An orphaned trainer (this launcher died) goes on writing to the same log.
        env[childproc.TRAIN_LOG_ENV] = os.path.abspath(log_path)
        started = self.clock()
        proc = subprocess.Popen(
            cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env, **childproc.popen_kwargs())

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
                              f"to {self.effective_grace_s:.0f} s)")
                    self._stop_child(proc, child_got_it=False)
                    return

        watcher = threading.Thread(target=watch, name="moe-watch", daemon=True)
        watcher.start()
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()             # on disk now: a killed launcher loses none
                    self.echo(line.rstrip("\n"))
                    m = STEP_LINE.match(line.strip())
                    if m:
                        events.append((self.clock(), int(m.group(1))))
                    save_s = parse_save_s(line)
                    if save_s is not None:
                        saves.append(save_s)
                        self._saw_save(save_s)
            code = proc.wait()
        except KeyboardInterrupt:
            done.set()
            self.echo("[moe] interrupted: letting the trainer write its checkpoint")
            # Its own session / process group: the child did not get the signal.
            self._stop_child(proc, child_got_it=False)
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
                            events=events, started_at=started, save_s=saves)


def run_command(cmd: list[str], log_path: Path) -> int:
    """An evaluation step: output to the console and the log; its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=dict(os.environ, PYTHONUNBUFFERED="1"))
        for line in proc.stdout:
            log.write(line)
            _echo(line.rstrip("\n"))
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
                 warmup_skip_steps: int = WARMUP_SKIP_STEPS,
                 go_max_wait_min: float = GO_MAX_WAIT_MIN, refit: bool = True,
                 eval_overhead_s: float = EVAL_OVERHEAD_S,
                 save_overhead_s: float = SAVE_OVERHEAD_S) -> None:
        self.config = Path(config)
        self.winners = Path(winners) if winners else None
        self.budget = float(budget_usd)
        self.rate_arg = float(usd_per_hour)
        self.ledger = ledger
        if ledger.tool is None:
            ledger.tool = "run_moe"          # `spend stop` refuses while this ticks
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
        self.echo = echo or _echo
        self.poll_s = poll_s
        self.wait_print_s = wait_print_s
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self.headroom = headroom
        self.skip = warmup_skip_steps
        self.go_max_wait_s = go_max_wait_min * 60
        self.refit = refit
        self.eval_overhead_s = eval_overhead_s
        self.save_overhead_s = save_overhead_s

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

    def _tick(self, stop_on_end: bool = True) -> None:
        """Tick the ledger (at most once per TICK_S). A box session ended under the
        launcher (`spend stop --force`: the box is going away) is a budget stop:
        Stop(EXIT_BUDGET), unless stop_on_end is False (then only reported)."""
        try:
            self.ledger.tick_if_due(TICK_S)
        except SessionEnded as exc:
            if stop_on_end:
                raise Stop(EXIT_BUDGET, f"stopped by the budget: the box session was ended "
                                        f"under the launcher ({exc})") from exc
            self.echo(f"[moe] warning: {exc}")
        except (OSError, LedgerError) as exc:
            self.echo(f"[moe] warning: spend ledger tick failed ({exc})")

    def _tick_in_check(self) -> str | None:
        """_tick for a should_stop check: a stop reason when the session was ended."""
        try:
            self._tick()
        except Stop as stop:
            return f"budget: {stop.status}"
        return None

    def spent(self) -> float:
        return self.ledger.spent_usd()

    def _backstop_slack_usd(self) -> float:
        """Half the child's stop grace in box time: the trainer's backstop is set this
        far beyond the guard's threshold, so a live launcher always stops the trainer
        first (its SIGINT is then the only stop) and an orphan still keeps half the
        grace for its checkpoint."""
        grace = getattr(self.run_child, "effective_grace_s", STOP_GRACE_S)
        return 0.5 * grace / 3600 * self.rate

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
            "last_step": self._last_logged_step(), "note": note, "at": _now_iso(),
            "max_save_s": max(outcome.save_s) if outcome.save_s else None})

    @property
    def batch_tokens(self) -> int:
        return int(self.raw["train"]["batch_tokens"])

    @property
    def interval_steps(self) -> int:
        """One full eval + checkpoint interval."""
        train = self.raw["train"]
        return max(int(train["eval_every"]), int(train["ckpt_every"]))

    def _tps_plan(self) -> float:
        """The planning rate: the gate's measured tokens/s less the headroom, with
        each eval and checkpoint interval's overhead added (plan.json keeps both)."""
        p, train = self.plan, self.raw["train"]
        return planning_tokens_per_s(
            p["tokens_per_s"], headroom=self.headroom, batch_tokens=self.batch_tokens,
            eval_every=int(train["eval_every"]), ckpt_every=int(train["ckpt_every"]),
            eval_s=p.get("eval_s", self.eval_overhead_s),
            save_s=p.get("save_s", self.save_overhead_s))

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
        elif status is None:
            pass                        # nothing to do: the summary on disk stands
        else:
            self._write_summary(status, code)
        return code

    def _run(self) -> tuple[int, str | None]:
        try:
            self.raw = resolve_raw(self.config, self.winners, self.overrides)
            load_config(self.config, None)          # the file itself must load
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise Stop(EXIT_USAGE, f"usage: {exc}") from exc
        if self.plan_json.exists():
            self.plan = done = self._load_plan()
            if done.get("completed") and done.get("evaluated"):
                self.echo(f"[moe] the run in {self.out} is complete and evaluated "
                          f"({self.summary}); nothing to do")
                return EXIT_OK, None
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

        if self.plan is None:
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
        if not failed:
            self.plan["evaluated"] = True
            self._save_plan()
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
        # The trainer's own backstop, should the launcher die during the gate: at most
        # GATE_BACKSTOP_FACTOR x the gate's cost, never past the budget less the reserve
        # (plus the slack that lets a live launcher stop it first).
        backstop = min(self.budget - self.spent() - self.reserve + self._backstop_slack_usd(),
                       GATE_BACKSTOP_FACTOR * gate_usd)
        self._write_run_config(with_train(
            self.raw, total_tokens=cfg_total, milestones=[],
            ckpt_dir=self.ckpt_dir.as_posix(), budget_usd=round(max(backstop, 0.01), 4),
            usd_per_hour=self.rate))
        resume = self.run_log.exists()
        start_step = self._last_logged_step() if resume else 0
        deadline = self.clock() + self.gate_s
        self.echo(f"[moe] throughput gate: {self.gate_s / 60:.0f} min of the full-size run"
                  + (" (resuming)" if resume else ""))

        def should_stop(events: list[tuple[float, int]]) -> str | None:
            ended = self._tick_in_check()
            if ended:
                return ended
            if self.clock() >= deadline:
                return "gate time is up"
            if self.spent() + self.reserve > self.budget + 1e-9:
                return "budget reached during the gate"
            return None

        outcome = self.run_child(self._train_cmd(resume), self.out / "logs" / "gate.log",
                                 should_stop)
        self._record_attempt("gate", outcome)
        if outcome.stop_reason and outcome.stop_reason.startswith("budget"):
            raise Stop(EXIT_BUDGET, f"stopped by the spend guard during the gate at step "
                                    f"{self._last_logged_step()} ({outcome.stop_reason})")
        if outcome.code == TRAIN_BUDGET and outcome.stop_reason is None:
            raise Stop(EXIT_BUDGET, f"the gate's trainer stopped on its budget backstop at "
                                    f"step {self._last_logged_step()} (exit 4)")
        if outcome.code != 0 and outcome.stop_reason is None:
            raise Stop(*self._failure(outcome.code, "the gate"))
        self._tick()

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
        seen_save = max(outcome.save_s) if outcome.save_s else None
        self.plan = {
            "base_hash": base_hash(self.raw), "made_at": _now_iso(),
            "tokens_per_s": tps, "startup_s": startup, "gate_step": gate_step,
            "gate_minutes": self.gate_s / 60,
            "config_total_tokens": cfg_total, "config_milestones": list(train.get("milestones", [])),
            # Per-interval overhead the gate's rate misses (see planning_tokens_per_s).
            "eval_s": self.eval_overhead_s,
            "save_s": seen_save if seen_save is not None else self.save_overhead_s,
            "save_observed": seen_save is not None,
            "completed": False, "notes": [],
        }
        self.echo(f"[moe] gate: {tps:,.0f} tok/s after the first {self.skip} steps; startup "
                  f"{startup:.0f} s; checkpoint at step {gate_step}; planning at "
                  f"{self._tps_plan():,.0f} tok/s (headroom {self.headroom:.0%}, eval "
                  f"{self.plan['eval_s']:.0f} s + save {self.plan['save_s']:.0f} s per interval)")
        self._fit(first=True)

    def _fit(self, first: bool, why: str = "re-trimmed at GO (the wait cost money)") -> None:
        """Set plan total_tokens / milestones to what fits now; write plan.md. Raises
        Stop(4) if not even one more step fits."""
        plan, train = self.plan, self.raw["train"]
        bt = self.batch_tokens
        gate_step = plan["gate_step"]
        want = plan["config_total_tokens"] if first else plan["total_tokens"]
        spent = self.spent()
        total = self._affordable_total(want, spent)
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
            plan["notes"].append(f"{_now_iso()}: {why}: total_tokens {want:,} -> {total:,}")
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

    def _affordable_total(self, want: int, spent: float) -> int:
        """`want` if the rest of it (from the gate's checkpoint) fits the budget at the
        planning rate with the startup and the reserve, else the largest whole-batch
        total that does (may be <= the gate's tokens: nothing fits)."""
        plan, bt = self.plan, self.batch_tokens
        tps = self._tps_plan()
        done = plan.get("done_step", plan["gate_step"]) * bt
        proj = project(tokens_total=want, tokens_done=done, tokens_per_s=tps,
                       usd_per_hour=self.rate, spent_usd=spent, reserve_usd=self.reserve,
                       startup_s=plan["startup_s"], budget_usd=self.budget)
        if proj.fits:
            return want
        return min(want, fit_total_tokens(
            budget_usd=self.budget, spent_usd=spent, reserve_usd=self.reserve,
            usd_per_hour=self.rate, tokens_per_s=tps, startup_s=plan["startup_s"],
            tokens_done=done, batch_tokens=bt))

    def _projection(self, total: int) -> Projection:
        plan = self.plan
        # From the gate's checkpoint; after a mid-run re-fit, from the step it was made at.
        done = plan.get("done_step", plan["gate_step"])
        return project(tokens_total=total, tokens_done=done * self.batch_tokens,
                       tokens_per_s=self._tps_plan(),
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
        train = self.raw["train"]
        save_src = ("the longest save the gate reported" if p.get("save_observed")
                    else "a default: the gate saved no checkpoint")
        lines += [
            f"- Measured throughput: **{p['tokens_per_s']:,.0f} tokens/s** (effective: step "
            f"lines' arrival times after the first {self.skip} steps, evals and checkpoints "
            f"in that window included; {p['gate_minutes']:.0f}-minute gate).",
            f"- Planned at **{self._tps_plan():,.0f} tokens/s**: {(1 - self.headroom):.0%} of "
            f"it, plus each interval's overhead: an eval ({p.get('eval_s', self.eval_overhead_s):.0f} s) "
            f"every {train['eval_every']} steps and a checkpoint save "
            f"({p.get('save_s', self.save_overhead_s):.0f} s, {save_src}) every "
            f"{train['ckpt_every']} steps. After the long run's first full interval "
            f"({self.interval_steps} steps) the measured rate is checked once: more than "
            f"{REFIT_TOLERANCE:.0%} below this, the run is re-fitted (trimmed, never "
            "extended) and this file says so.",
            f"- Startup of the long run (resume + compile), from the gate: {p['startup_s']:.0f} s.",
            f"- Box rate ${self.rate:.2f}/h = **${per_min:.4f}/min**. Box spend so far "
            f"${self.spent():.2f} (ledger {self.ledger.path}).",
            f"- Budget ${self.budget:.2f}; reserve ${self.reserve:.2f} kept for the chat SFT, "
            "the evals and the copy-back.", "",
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
                  "unaffordable, it is trimmed again at GO and this file says so. The launcher "
                  f"waits at most {self.go_max_wait_s / 60:.0f} min, and stops sooner if less "
                  f"than {RETRIM_FLOOR:.0%} of the planned tokens would still fit (exit 5): "
                  "then stop the instance from the Vast console and run the same command "
                  "later.", ""]
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
        approved = int(self.plan["total_tokens"])
        idle = (f"the box idles at ~${self.rate:.2f}/h: stop the instance from the Vast "
                "console (its disk is kept) and resume later with the same command")
        while not self.go.exists():
            self._tick()
            now = self.clock()
            spent = self.spent()
            if now >= next_print:
                self.echo(f"[moe] waiting for {self.go}: {(now - t0) / 60:.0f} min so far, "
                          f"${per_min:.4f}/min (${self.rate:.2f}/h); spent ${spent:.2f} of "
                          f"${self.budget:.2f}")
                next_print = now + self.wait_print_s
            step_usd = self.batch_tokens / self._tps_plan() / 3600 * self.rate
            startup_usd = self.plan["startup_s"] / 3600 * self.rate
            if spent + self.reserve + startup_usd + step_usd > self.budget + 1e-9:
                self.plan["notes"].append(f"{_now_iso()}: the wait for GO used up the budget")
                self.plan["fits"] = False
                self._save_plan()
                raise Stop(EXIT_BUDGET, "stopped by the budget while waiting for GO")
            if now - t0 >= self.go_max_wait_s:
                status = (f"GO not given within {self.go_max_wait_s / 60:.0f} min; {idle}")
                self.plan["notes"].append(f"{_now_iso()}: {status}")
                self._save_plan()
                raise Stop(EXIT_NO_GO, status)
            fits = self._affordable_total(approved, spent)
            if fits < RETRIM_FLOOR * approved:
                why = (f"while waiting for GO the plan shrank below {RETRIM_FLOOR:.0%} of the "
                       f"approved {approved:,} tokens (only {fits:,} fit now); re-fitted for a "
                       "fresh approval")
                self._fit(first=False, why=why)       # Stop(4) if nothing fits at all
                status = (f"GO not given before the wait cost more than {1 - RETRIM_FLOOR:.0%} "
                          f"of the approved run (plan.md is re-fitted to "
                          f"{self.plan['total_tokens']:,} tokens and needs a fresh approval); "
                          f"{idle}")
                self.plan["notes"].append(f"{_now_iso()}: {status}")
                self._save_plan()
                raise Stop(EXIT_NO_GO, status)
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
        plan = self.plan
        bt = self.batch_tokens
        retries = 0
        while True:
            self._tick()
            steps = plan["total_tokens"] // bt
            planned_tps = self._tps_plan()
            start = self._last_logged_step() if self.run_log.exists() else 0
            if over_budget(self.spent(), self._margin(steps, start, planned_tps), self.reserve,
                           self.budget):
                raise Stop(EXIT_BUDGET, (
                    f"stopped by the spend guard before launching (step {start}/{steps}): "
                    f"spent ${self.spent():.2f} + next interval "
                    f"${self._margin(steps, start, planned_tps):.2f} + reserve "
                    f"${self.reserve:.2f} > ${self.budget:.2f}"))
            # Written for every attempt: the trainer's backstop is the budget left NOW
            # (less the reserve, plus half the stop grace so the guard, above, always
            # stops a trainer first while the launcher lives), timed from its own
            # start, so no attempt, orphaned or not, can spend past it.
            backstop = self.budget - self.spent() - self.reserve + self._backstop_slack_usd()
            self._write_run_config(with_train(
                self.raw, total_tokens=plan["total_tokens"], milestones=plan["milestones"],
                ckpt_dir=self.ckpt_dir.as_posix(), budget_usd=round(max(backstop, 0.01), 4),
                usd_per_hour=self.rate))
            refit: dict[str, Any] = {}

            def should_stop(events: list[tuple[float, int]], start: int = start) -> str | None:
                ended = self._tick_in_check()
                if ended:
                    return ended
                live = effective_tokens_per_s(events, bt, start + self.skip)
                if live is not None and self.refit and not plan.get("refit_checked"):
                    kept = [s for _, s in events if s >= start + self.skip]
                    if kept[-1] - kept[0] >= self.interval_steps:
                        reason = self._check_refit(live, kept[-1], refit)
                        if reason:
                            return reason
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
            self._record_attempt("train", outcome, note=refit.get("note", ""))
            if refit and outcome.stop_reason == refit.get("reason"):
                self._apply_refit(refit)
                continue                         # resume with the re-fitted length
            if outcome.stop_reason:
                raise Stop(EXIT_BUDGET, (
                    f"stopped by the spend guard at step {self._last_logged_step()}/{steps} "
                    f"({outcome.stop_reason}; interrupt checkpoint written; run again with a "
                    "larger --budget-usd to continue)"))
            if outcome.code == 0:
                plan["completed"] = True         # a rerun of the same command is a no-op
                self._save_plan()
                self._tick(stop_on_end=False)
                return
            if outcome.code == TRAIN_BUDGET:
                raise Stop(EXIT_BUDGET, (
                    f"the trainer's budget backstop stopped it at step "
                    f"{self._last_logged_step()}/{steps} (exit 4, checkpoint written)"))
            if (outcome.code in (TRAIN_USAGE, TRAIN_NONFINITE)
                    or outcome.code in TRAIN_INTERRUPT_CODES or retries >= self.max_retries):
                raise Stop(*self._failure(outcome.code, "the long run"))
            retries += 1
            self.echo(f"[moe] trainer exited {outcome.code}; retry {retries}/"
                      f"{self.max_retries} from the last checkpoint in {self.retry_wait_s:.0f} s")
            self.sleep(self.retry_wait_s)

    def _check_refit(self, live: float, at_step: int, refit: dict[str, Any]) -> str | None:
        """Once, after the long run's first full interval: the measured rate against the
        planning rate. More than REFIT_TOLERANCE slower: what fits at the measured rate
        (less the headroom, one more startup) is computed, and if that is less than the
        plan, `refit` gets it and a stop reason is returned (the trainer checkpoints and
        is resumed with the new length). Never extends the run."""
        plan, bt = self.plan, self.batch_tokens
        planned = self._tps_plan()
        plan["refit_checked"] = True
        short = 1 - live / planned if planned > 0 else 0.0
        head = (f"first full interval of the long run (to step {at_step}): {live:,.0f} "
                f"tok/s measured vs {planned:,.0f} planned")
        if short <= REFIT_TOLERANCE:
            plan["notes"].append(f"{_now_iso()}: {head}: within {REFIT_TOLERANCE:.0%}, no re-fit")
            self._save_plan()
            return None
        total = plan["total_tokens"]
        new_total = min(total, fit_total_tokens(
            budget_usd=self.budget, spent_usd=self.spent(), reserve_usd=self.reserve,
            usd_per_hour=self.rate, tokens_per_s=live * (1 - self.headroom),
            startup_s=plan["startup_s"], tokens_done=at_step * bt, batch_tokens=bt))
        if new_total >= total:
            plan["notes"].append(f"{_now_iso()}: {head} ({short:.1%} slower); the plan "
                                 "still fits at the measured rate, no re-fit")
            self._save_plan()
            return None
        if new_total // bt <= at_step + int(self.raw["train"]["eval_every"]):
            plan["notes"].append(f"{_now_iso()}: {head} ({short:.1%} slower); too little "
                                 "would be left to re-fit, the spend guard handles it")
            self._save_plan()
            return None
        refit.update(total=new_total, live=live, planned=planned, at_step=at_step,
                     reason=(f"re-fit: {live:,.0f} tok/s measured, {short:.1%} below the "
                             f"planned {planned:,.0f}; total_tokens {total:,} -> {new_total:,}"),
                     note=f"interrupted for the re-fit to {new_total:,} tokens")
        self._save_plan()
        return refit["reason"]

    def _apply_refit(self, refit: dict[str, Any]) -> None:
        """The trainer has checkpointed: trim total_tokens, rescale the milestones still
        ahead, record it in plan.md. The next attempt resumes with them."""
        plan, bt = self.plan, self.batch_tokens
        at = self._last_logged_step()
        old_total = plan["total_tokens"]
        new_total = refit["total"]
        new_steps = new_total // bt
        if new_steps <= at:
            plan["notes"].append(f"{_now_iso()}: re-fit to {new_total:,} tokens dropped: the "
                                 f"run is already at step {at}")
            self._save_plan()
            return
        cfg_steps = plan["config_total_tokens"] // bt
        milestones = list(refit_milestones(plan["config_milestones"], cfg_steps,
                                           old_total // bt, new_steps, plan["gate_step"], at))
        plan["notes"].append(
            f"{_now_iso()}: re-fitted after the long run's first full interval: "
            f"{refit['live']:,.0f} tok/s measured vs {refit['planned']:,.0f} planned; "
            f"total_tokens {old_total:,} -> {new_total:,} ({new_steps:,} steps), milestones "
            f"{plan.get('milestones')} -> {milestones}; resumed from step {at} (the cosine "
            "schedule shortens from here)")
        plan.update(total_tokens=new_total, milestones=milestones, trimmed=True,
                    done_step=at, refit={"at_step": at, "from_total": old_total,
                                         "to_total": new_total, "measured_tps": refit["live"],
                                         "planned_tps": refit["planned"]})
        self._save_plan()
        self.echo(f"[moe] re-fitted: total_tokens {old_total:,} -> {new_total:,}; resuming "
                  f"from step {at} ({self.plan_md})")

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
        self._tick(stop_on_end=False)

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
            if p.get("refit"):
                r = p["refit"]
                lines.append(f"- Re-fitted at step {r['at_step']} after the first full interval: "
                             f"{r['measured_tps']:,.0f} tok/s measured vs {r['planned_tps']:,.0f} "
                             f"planned; total_tokens {r['from_total']:,} -> {r['to_total']:,}.")
        if steps:
            lines.append(f"- Trained to step {last.get('step')} ({last.get('tokens', 0):,} tokens), "
                         f"last train loss {last.get('train_loss', float('nan')):.4f}"
                         + (f", last val loss {evals[-1]['val_loss']:.4f} at step "
                            f"{evals[-1]['step']}" if evals else "") + ".")
        lines += ["", "## Attempts", ""]
        lines += [f"- {a['phase']}: exit {a['code']}" + (f", stopped ({a['stop']})" if a["stop"] else "")
                  + f", run log at step {a['last_step']}"
                  + (f", longest checkpoint save {a['max_save_s']:.1f} s"
                     if a.get("max_save_s") is not None else "")
                  + f" ({a['at']})" for a in self.attempts] or ["(none)"]
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
        elif code == EXIT_NO_GO:
            lines += [f"GO not given; box idle ~${self.rate:.2f}/h. Stop the instance from the "
                      "Vast console; its disk is kept. First end the box session so the stopped "
                      "time is not counted: `python -m quipu.spend stop`. To resume later: "
                      "start the instance, `python -m quipu.spend start --usd-per-hour R`, "
                      f"then the same run_moe.py command (it reuses {self.plan_json}; "
                      f"read {self.plan_md} again first, it may have been re-fitted)."]
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
                   help=f"kept back for the chat SFT, the evals and the copy-back "
                        f"(default {RESERVE_USD:.2f})")
    p.add_argument("--gate-minutes", type=float, default=GATE_MINUTES)
    p.add_argument("--headroom", type=float, default=HEADROOM,
                   help="plan at (1 - headroom) x the measured tokens/s, before the "
                        "per-interval eval/checkpoint overhead")
    p.add_argument("--go-max-wait-min", type=float, default=GO_MAX_WAIT_MIN,
                   help="give up waiting for GO after this many minutes (exit 5: stop the "
                        "instance, run the same command later)")
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
    for name, allow_zero in (("budget_usd", False), ("reserve_usd", True), ("gate_minutes", False),
                             ("go_max_wait_min", False)):
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
        headroom=args.headroom, warmup_skip_steps=args.warmup_skip_steps,
        go_max_wait_min=args.go_max_wait_min)
    return run_until_stopped(launcher, ledger)


def run_until_stopped(launcher: Launcher, ledger: Ledger) -> int:
    """launcher.run() with the ledger ticker and the stop signals: SIGTERM / SIGHUP
    (POSIX) take Ctrl+C's path, so the trainer is interrupted and checkpoints before
    the launcher exits 130. However it ends, the launcher's tag is released from the
    ledger, so `spend stop` right after it needs no --force."""
    previous = childproc.install_stop_signals(_echo, "[moe]")
    try:
        with Ticker(ledger):
            return launcher.run()
    finally:
        childproc.restore_signals(previous)
        release_tool(ledger, _echo, "[moe]")


def release_tool(ledger: Ledger, echo: Callable[[str], None], tag: str) -> None:
    """Ledger.release_tool, never raising (the tool is exiting anyway)."""
    try:
        ledger.release_tool()
    except (OSError, LedgerError) as exc:
        echo(f"{tag} warning: could not release the spend ledger ({exc}); `spend stop` "
             "refuses for 2 minutes (or pass --force)")


if __name__ == "__main__":
    sys.exit(main())
