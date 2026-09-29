"""The A/B runs for quipu-moe (spec sections 6.1 and 12, plan Task M8).

    python -m quipu.spend start --usd-per-hour 0.55      # once, when the box starts
    python scripts/ab_runs.py --config configs/quipu-moe-ab.toml --out results/ab \\
        --budget-usd 4 --usd-per-hour 0.55 [--with-fp8] [--dry-run]

What runs, in order (every run is `python -m quipu.train` with --override flags):
1. LR sweeps at SWEEP_TOKENS (100M): AdamW lr in ADAMW_LRS, then Muon muon_lr in
   MUON_LRS with the AdamW part at the best AdamW lr. Best = lowest final val loss.
   The config's own value runs first in each sweep: it is the divergence baseline
   for the other two.
2. Pairs at ARM_TOKENS (200M), one setting changed per pair:
     pair 1 optimizer   AdamW (best lr)  vs Muon (best muon_lr)
     pair 2 attnres     attnres_blocks 0 vs --attnres-blocks (4)
     pair 3 activation  swiglu           vs situ_glu
     pair 4 precision   bf16             vs fp8        (only with --with-fp8)
   The baseline of pairs 2-4 is the winner of pair 1 (its run is reused, not re-run).
   Pair 4 is off by default: results/fp8/laptop.md measured FP8 at 0.79-0.89x eager
   on the laptop and expects <= 1.1x on the 5090, below the 1.2x keep rule.
3. A seed re-run (seed + 1) of each pair's preliminary winner; noise = |val loss
   difference| between the two seeds. Identical re-runs (e.g. two pairs whose winner
   is the pair-1 winner) are the same cached run, so they cost once.
4. Decision rules (decide_pair). The simpler option is AdamW, no AttnRes, SwiGLU,
   bf16. The other option is kept only if its final val loss is lower by MORE than
   the noise, and: AttnRes costs <= 10% tokens/s; SiTU-GLU has no more loss spikes
   (a spike: step loss > 1.5 x the EMA of the losses before it). FP8 is judged the
   other way round: kept if >= 1.2x bf16 tokens/s, loss no worse than bf16 by more
   than the noise, and no more spikes. A missing arm (failed, refused by the budget,
   no seed re-run) keeps the simpler option, and the summary says so.
5. results/ab/summary.md (every run: config diff, final val loss, bits per byte,
   tokens/s, spikes, decision) and results/ab/winners.toml (the overrides for the
   full run; load_config(path, tomllib.load(winners)) takes it as is).

Money (the box is billed per hour; the credit is not refundable):
- Spend is the BOX's, from the shared ledger quipu/spend.py (results/spend.json, or
  $QUIPU_SPEND_LEDGER; outside every --out dir, shared with the M9 launcher): box
  time since the box session started x its rate, over every box session, plus named
  adjustments. --budget-usd caps that total (so it includes setup, shard building
  and earlier invocations). --usd-per-hour is required to train: it starts the box
  session on first use (`python -m quipu.spend start` at box start is better: it
  also counts the setup before this runs). The ledger is ticked at start, every
  60 s (a background thread) and after every run, so a SIGKILL loses <= 60 s. A
  corrupt or unreadable ledger stops the orchestrator (exit 2), never reads as $0.
- --spent-usd X (spend the ledger cannot see, e.g. an earlier box) is stored as the
  adjustment named --spent-key (default "spent-usd"): passing the same flag again
  replaces it, never adds it twice. Use another key for another amount.
- Before each run (and each retry) its cost is estimated from the slowest measured
  tokens/s of the runs so far (tokens / wall clock of a first, not resumed attempt,
  overhead included; after a restart, from the cached runs), or from
  --tokens-per-second + --overhead-seconds before any run is measured; a run that
  would take spend past --budget-usd is refused and nothing after it starts.
- During a run, once spend + STOP_GRACE_S of box time reaches the budget, the child
  gets SIGINT (Windows: CTRL_BREAK) and writes its interrupt checkpoint; after
  STOP_GRACE_S it is terminated, then killed. The run is "stopped_budget": not
  cached, its checkpoint kept, so the next invocation (with more budget) resumes it.
  The orchestrator then exits 4. The runs keep train.budget_usd 0 (the orchestrator
  guards); train.budget_usd / usd_per_hour and the per-run bookkeeping keys
  (RESERVED_OVERRIDES) are refused as --override.
- Resume: each finished run is cached as results/ab/cache/<key>.json, key = hash of
  the config file, the --override flags and the run's settings and tokens. A re-run
  of the orchestrator skips every cached run. A crashed run (exit 1) is retried once
  (from its checkpoint if it has one), then cached as failed; a CUDA out-of-memory
  run is not retried (failed_oom: the same settings run out again); --retry-failed
  tries failed and failed_oom runs again.
- Early stop: a non-finite loss, the trainer's non-finite stop (exit 3), or a train
  loss above DIVERGE_FACTOR x its baseline's at any printed step from DIVERGE_AT of
  the run kills the run and marks it diverged (a diverged other-option loses).
- Every run checkpoints half way (train.ckpt_every = ceil(steps / 2)) so a crash
  late in a run resumes rather than restarts. Checkpoints of finished runs are
  deleted (~12 bytes per parameter: the A/B model's is ~6.2 GB); --keep-checkpoints
  keeps them and is refused when they would not fit on the disk.
- Evals: eval_every is the divisor of the run's steps nearest the config's value
  with at most MAX_EVALS evals per run, so the last step is an eval step.
- Compile: the children share a persistent inductor cache (<out>/inductor-cache,
  FX graph + autograd caches on), so arms with the same shapes compile once. The
  summary reports each run's startup (to its first step line) and total overhead
  (wall - tokens / steady tokens/s) against the PRIOR_OVERHEAD_S prior.

--dry-run prints the run list (worst case, every seed re-run distinct), the token
total and the estimated cost at --tokens-per-second, and trains nothing.

Exit codes: 0 done (whatever the decisions), 2 usage error, 4 stopped by the budget
(summary and winners still written), 130 interrupted.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quipu.config import Config, load_config, parse_overrides
from quipu.fsio import write_text_atomic
from quipu.spend import TICK_S, Ledger, LedgerError, Ticker

ROOT = Path(__file__).resolve().parents[1]

ADAMW_LRS = (3e-4, 6e-4, 1.2e-3)
MUON_LRS = (0.01, 0.02, 0.04)
SWEEP_TOKENS = 100_000_000
ARM_TOKENS = 200_000_000
ATTNRES_ON = 4                  # four blocks of two layers on the 8-layer scale-down
DEFAULT_PAIRS = ("optimizer", "attnres", "activation")

SPIKE_FACTOR = 1.5
SPIKE_EMA_BETA = 0.98
ATTNRES_MAX_COST = 0.10         # AttnRes may cost at most 10% tokens/s
FP8_MIN_SPEEDUP = 1.2
DIVERGE_FACTOR = 2.0
DIVERGE_AT = 0.25               # fraction of the run's steps
DIVERGE_WINDOW = 5              # baseline losses averaged over the steps up to it
MAX_EVALS = 20                  # evals per run at most (eval_every is clamped)

PRIOR_TOKENS_PER_S = 100_000.0  # conservative until a run has been measured
PRIOR_OVERHEAD_S = 180.0        # startup, compile, evals and checkpoints per run
MAX_ATTEMPTS = 2                # a crashed run is retried once
STOP_GRACE_S = 300.0            # an interrupted child's time to write its checkpoint
KILL_WAIT_S = 30.0              # after terminate() before kill()
BUDGET_POLL_S = 5.0             # how often a running child's deadline is checked
CKPT_BYTES_PER_PARAM = 12       # fp32 weights + two AdamW moments (Muon holds less)
SPENT_KEY = "spent-usd"         # the ledger adjustment --spent-usd sets

# Keys the orchestrator owns: refused as --override (a run's budget belongs to the
# box ledger; the others are set per run).
RESERVED_OVERRIDES = {
    "train.budget_usd": "the orchestrator enforces the budget: use --budget-usd",
    "train.usd_per_hour": "the orchestrator bills the box: use --usd-per-hour",
    "train.total_tokens": "the orchestrator sets each run's tokens: use --sweep-tokens/--arm-tokens",
    "train.ckpt_dir": "the orchestrator gives each run its own checkpoint dir under --out",
    "train.milestones": "the orchestrator sets milestones = [] for A/B runs",
    "train.eval_every": "the orchestrator aligns eval_every to each run's last step",
    "train.ckpt_every": "the orchestrator checkpoints each run half way",
}
OOM_MARKERS = ("out of memory", "outofmemoryerror")

EXIT_BUDGET = 4
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
# The trainer's exit codes (quipu.train): 0 ok, 1 crash, 2 usage, 3 non-finite stop,
# 130 interrupt; Windows reports a Ctrl+C'd child as STATUS_CONTROL_C_EXIT.
TRAIN_USAGE, TRAIN_NONFINITE = 2, 3
TRAIN_INTERRUPT_CODES = frozenset({130, -1073741510, 3221225786})


# ---- settings and pairs ----------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Pair:
    number: int
    name: str
    key: str            # the dotted setting the pair changes
    simple: Any
    label: str          # the other option, in run names


PAIRS = (
    Pair(1, "optimizer", "train.optimizer", "adamw", "muon"),
    Pair(2, "attnres", "model.attnres_blocks", 0, "attnres"),
    Pair(3, "activation", "model.activation", "swiglu", "situ_glu"),
    Pair(4, "precision", "train.precision", "bf16", "fp8"),
)
PAIR_BY_NAME = {p.name: p for p in PAIRS}


class Pending:
    """A value only known once earlier runs finish (dry-run listing only)."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __str__(self) -> str:
        return f"<{self.label}>"


def fmt_value(value: Any) -> str:
    """A setting as the text of an --override flag (parse_overrides reads it back)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def fmt_lr(value: float) -> str:
    """6e-4 -> "6e-04", 1.2e-3 -> "1.2e-03" (run names)."""
    mantissa, exponent = f"{value:.6e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"


@dataclasses.dataclass
class RunSpec:
    name: str
    phase: str                      # "sweep" | "pair" | "seed"
    settings: dict[str, Any]        # dotted key -> value (the A/B-controlled settings)
    tokens: int
    baseline: str | None = None     # cache key of the run whose losses it is held to


@dataclasses.dataclass
class RunContext:
    config: Path
    run_id: str
    run_dir: Path
    ckpt_dir: Path
    log_path: Path
    overrides: list[str]
    steps: int
    baseline_losses: list[float] | None
    # The run's deadline: True once the box ledger says the budget is (about to be)
    # spent; the runner then interrupts the child. None = no deadline.
    budget_check: Callable[[], bool] | None = None
    inductor_cache: Path | None = None      # TORCHINDUCTOR_CACHE_DIR for the child


@dataclasses.dataclass
class RunResult:
    # completed | diverged | failed | failed_oom | stopped_budget
    # (| crashed, usage, interrupted from a runner)
    status: str
    final_val_loss: float | None
    bpb: float | None
    tokens_per_s: float | None      # steady-state, from the trainer's step lines
    effective_tokens_per_s: float | None   # tokens / wall clock, overhead included
    wall_s: float
    spikes: int
    train_losses: list[float]
    reason: str = ""
    attempts: int = 1
    resumed: bool = False           # this attempt resumed from a checkpoint
    skipped: int | None = None      # non-finite steps the trainer skipped (run log)
    startup_s: float | None = None  # process start to the first step line
    overhead_s: float | None = None # wall - tokens / steady tokens/s

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        """Fields missing from an older cache entry take their defaults."""
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d})


def usable(r: RunResult | None) -> bool:
    return (r is not None and r.status == "completed" and r.final_val_loss is not None
            and math.isfinite(r.final_val_loss))


def _tps(r: RunResult) -> float:
    return r.tokens_per_s or r.effective_tokens_per_s or 0.0


# ---- decision rules ------------------------------------------------------------------------

def count_spikes(losses: list[float], factor: float = SPIKE_FACTOR,
                 beta: float = SPIKE_EMA_BETA) -> int:
    """Steps whose train loss exceeds factor x the EMA of the losses before it."""
    ema = None
    spikes = 0
    for x in losses:
        if x is None or not math.isfinite(x):
            continue
        if ema is not None and x > factor * ema:
            spikes += 1
        ema = x if ema is None else beta * ema + (1 - beta) * x
    return spikes


def decide_sweep(results: dict[float, RunResult | None], default: float) -> tuple[float, str]:
    """The value with the lowest final val loss among finished runs, or the config's
    value if none finished."""
    done = {v: r.final_val_loss for v, r in results.items() if usable(r)}
    if not done:
        return default, f"no sweep run finished; the config value {default:g} is used"
    best = min(done, key=done.get)
    return best, f"lowest final val loss of {len(done)} finished ({done[best]:.4f})"


@dataclasses.dataclass
class Decision:
    pair: str
    keep: str                   # "simple" | "complex"
    prelim: str | None          # the arm whose seed re-run measures the noise
    noise: float | None
    reason: str


def preliminary(pair: str, simple: RunResult, other: RunResult) -> tuple[str, str]:
    """Which arm wins on every rule except the noise rule, and why the simpler one
    did if it did. The seed re-run is of this arm."""
    if pair == "precision":
        speed = _tps(other) / _tps(simple) if _tps(simple) else 0.0
        if speed < FP8_MIN_SPEEDUP:
            return "simple", (f"fp8 ran at {speed:.2f}x bf16 tokens/s, below the "
                              f"{FP8_MIN_SPEEDUP}x rule")
        if other.spikes > simple.spikes:
            return "simple", f"fp8 had more loss spikes ({other.spikes} vs {simple.spikes})"
        return "complex", f"fp8 at {speed:.2f}x bf16 tokens/s"
    if other.final_val_loss >= simple.final_val_loss:
        return "simple", (f"the simpler option's loss was equal or lower "
                          f"({simple.final_val_loss:.4f} vs {other.final_val_loss:.4f})")
    if pair == "attnres" and _tps(simple):
        cost = 1 - _tps(other) / _tps(simple)
        if cost > ATTNRES_MAX_COST:
            return "simple", (f"AttnRes throughput cost {cost:.1%} is over the "
                              f"{ATTNRES_MAX_COST:.0%} limit")
    if pair == "activation" and other.spikes > simple.spikes:
        return "simple", f"SiTU-GLU had more loss spikes ({other.spikes} vs {simple.spikes})"
    return "complex", "lower loss"


def decide_pair(pair: str, simple: RunResult | None, other: RunResult | None,
                reseed: RunResult | None) -> Decision:
    """The pair's decision. `reseed` is the seed re-run of the preliminary winner
    (None if it was not run)."""
    if not usable(simple):
        if simple is not None and simple.status == "diverged" and usable(other):
            return Decision(pair, "complex", "complex", None,
                            "the simpler arm diverged; the other option is kept")
        state = "not run" if simple is None else simple.status
        return Decision(pair, "simple", None, None,
                        f"simpler arm missing ({state}); the simpler option is kept by default")
    if other is not None and other.status == "diverged":
        return Decision(pair, "simple", "simple", None,
                        f"the other arm diverged ({other.reason or 'early stop'})")
    if not usable(other):
        state = "not run" if other is None else other.status
        return Decision(pair, "simple", None, None,
                        f"other arm missing ({state}); the simpler option is kept by default")
    prelim, why = preliminary(pair, simple, other)
    winner = simple if prelim == "simple" else other
    noise = (abs(winner.final_val_loss - reseed.final_val_loss) if usable(reseed) else None)
    if prelim == "simple":
        return Decision(pair, "simple", "simple", noise, why)
    if noise is None:
        return Decision(pair, "simple", "complex", None,
                        f"{why}, but the winner's seed re-run is missing: noise unknown, "
                        "the simpler option is kept")
    if pair == "precision":
        worse = other.final_val_loss - simple.final_val_loss
        if worse <= noise:
            return Decision(pair, "complex", "complex", noise,
                            f"{why}, loss within noise ({worse:+.4f} vs noise {noise:.4f})")
        return Decision(pair, "simple", "complex", noise,
                        f"{why}, but loss worse by {worse:.4f}, beyond noise {noise:.4f}")
    delta = simple.final_val_loss - other.final_val_loss
    if delta > noise:
        return Decision(pair, "complex", "complex", noise,
                        f"lower loss by {delta:.4f}, beyond noise {noise:.4f}")
    return Decision(pair, "simple", "complex", noise,
                    f"lower loss by only {delta:.4f}, within noise {noise:.4f}; "
                    "the simpler option is kept")


# ---- spend guard ----------------------------------------------------------------------------

class SpendGuard:
    """spend = the box ledger's spend (every box session x its rate + adjustments).
    usd_per_hour is the current box session's rate, or the one given until a
    session exists (dry run)."""

    def __init__(self, budget_usd: float | None, ledger: Ledger, usd_per_hour: float,
                 prior_tokens_per_s: float = PRIOR_TOKENS_PER_S,
                 overhead_s: float = PRIOR_OVERHEAD_S) -> None:
        self.budget_usd = budget_usd
        self.ledger = ledger
        self._rate = usd_per_hour
        self.prior_tokens_per_s = prior_tokens_per_s
        self.overhead_s = overhead_s
        self.measured: list[float] = []      # effective tokens/s of measured runs

    @property
    def usd_per_hour(self) -> float:
        rate = self.ledger.usd_per_hour
        return self._rate if rate is None else rate

    def spent(self) -> float:
        return self.ledger.spent_usd()

    def deadline_reached(self, reserve_s: float = STOP_GRACE_S) -> bool:
        """True once spend plus `reserve_s` of box time reaches the budget."""
        if self.budget_usd is None:
            return False
        return self.spent() + reserve_s / 3600 * self.usd_per_hour >= self.budget_usd

    def observe(self, tokens: int, wall_s: float) -> None:
        if wall_s > 0:
            self.measured.append(tokens / wall_s)

    def estimate_s(self, tokens: int) -> float:
        if self.measured:
            return tokens / min(self.measured)       # the slowest run: conservative
        return tokens / self.prior_tokens_per_s + self.overhead_s

    def estimate_usd(self, tokens: int) -> float:
        return self.estimate_s(tokens) / 3600 * self.usd_per_hour

    def allows(self, cost_usd: float) -> bool:
        return self.budget_usd is None or self.spent() + cost_usd <= self.budget_usd


# ---- the orchestrator -----------------------------------------------------------------------

def aligned_eval_every(steps: int, want: int, max_evals: int = MAX_EVALS) -> int:
    """The divisor of steps closest to `want` (log scale) among those giving at most
    max_evals evals, so the last step is an eval step ("final val loss" is measured
    at the end of the run) and evals cannot eat the run (steps itself: one eval)."""
    divisors = [d for d in range(1, steps + 1) if steps % d == 0 and steps // d <= max_evals]
    return min(divisors, key=lambda d: (abs(math.log(d / want)), -d))


class Orchestrator:
    def __init__(
        self,
        config: str | Path,
        out: str | Path,
        runner: Callable[[RunSpec, RunContext], RunResult],
        *,
        clock: Callable[[], float] = time.monotonic,
        budget_usd: float | None = None,
        usd_per_hour: float | None = None,
        tokens_per_second: float = PRIOR_TOKENS_PER_S,
        overhead_s: float = PRIOR_OVERHEAD_S,
        spent_usd: float = 0.0,
        spent_key: str = SPENT_KEY,
        ledger: Ledger | None = None,
        tick_every: float | None = None,
        with_fp8: bool = False,
        skip_sweeps: bool = False,
        pairs: tuple[str, ...] = DEFAULT_PAIRS,
        seed_reruns: bool = True,
        sweep_tokens: int = SWEEP_TOKENS,
        arm_tokens: int = ARM_TOKENS,
        attnres_on: int = ATTNRES_ON,
        extra_overrides: dict[str, Any] | None = None,
        retry_failed: bool = False,
        keep_checkpoints: bool = False,
        bytes_per_token: float | None | str = "auto",
        echo: Callable[[str], None] | None = None,
    ) -> None:
        self.config_path = Path(config).resolve()
        self.out = Path(out).resolve()
        self.runner = runner
        self.clock = clock
        self.echo = echo or (lambda s: print(s, flush=True))
        pairs = tuple(pairs) + (("precision",) if with_fp8 and "precision" not in pairs else ())
        unknown = [p for p in pairs if p not in PAIR_BY_NAME]
        if unknown:
            raise ValueError(f"unknown pair(s) {unknown}; choose from {list(PAIR_BY_NAME)}")
        self.pairs = tuple(p.name for p in PAIRS if p.name in pairs)
        self.skip_sweeps = skip_sweeps
        self.seed_reruns = seed_reruns
        self.sweep_tokens = sweep_tokens
        self.arm_tokens = arm_tokens
        self.attnres_on = attnres_on
        self.retry_failed = retry_failed
        self.keep_checkpoints = keep_checkpoints
        self.extra = {k: fmt_value(v) for k, v in (extra_overrides or {}).items()}
        refused = [k for k in self.extra if k in RESERVED_OVERRIDES]
        if refused:
            raise ValueError("; ".join(f"--override {k} is not allowed: {RESERVED_OVERRIDES[k]}"
                                       for k in refused) + " (orchestrator-owned key)")
        self.cfg: Config = load_config(self.config_path, self._parse(self.extra))
        self.config_sha = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        # The box ledger (never written until run() begins: a dry run changes nothing).
        self.ledger = ledger if ledger is not None else Ledger.load()
        self.usd_per_hour_arg = usd_per_hour
        self.spent_usd = spent_usd
        self.spent_key = spent_key
        self.tick_every = tick_every
        rate = usd_per_hour
        if rate is None:
            rate = self.ledger.usd_per_hour or self.cfg.train.usd_per_hour
        self.guard = SpendGuard(budget_usd, self.ledger, rate, tokens_per_second, overhead_s)
        self.guard.measured = self._cached_throughput()
        self.spent_at_start: float | None = None
        self._bpt = bytes_per_token
        self.stopped: str | None = None
        self.records: list[dict[str, Any]] = []      # one per run the procedure asked for
        self.results: dict[str, RunResult | None] = {}
        self.labels: dict[str, list[str]] = {}
        self.decisions: list[tuple[Pair, Decision, dict[str, Any]]] = []
        self.lr_notes: list[str] = []
        self.final_settings: dict[str, Any] = {}

    # -- plumbing --

    @staticmethod
    def _parse(flat: dict[str, str]) -> dict[str, Any]:
        return parse_overrides([f"{k}={v}" for k, v in flat.items()])

    def begin(self) -> None:
        """Anchor spend to the box: start the ledger's box session on first use
        (at --usd-per-hour), record --spent-usd as its named adjustment, tick."""
        if self.spent_at_start is not None:
            return
        if self.usd_per_hour_arg is None:
            raise ValueError("--usd-per-hour is required to train (the box's rate)")
        if self.ledger.ensure_session(self.usd_per_hour_arg):
            self.echo(f"[ab] started a box session in {self.ledger.path} at "
                      f"${self.usd_per_hour_arg:.2f}/h (run `python -m quipu.spend start` "
                      "when the box starts to count the setup too)")
        if self.spent_usd:
            self.ledger.adjust(self.spent_key, self.spent_usd)
        self.ledger.tick()
        self.spent_at_start = self.ledger.spent_usd()
        self.echo(f"[ab] box spend so far ${self.spent_at_start:.2f} (ledger {self.ledger.path})")

    def projected_spent(self) -> float:
        """The ledger's spend with --spent-usd applied (for the dry run, which
        writes nothing)."""
        spent = self.ledger.spent_usd()
        if self.spent_usd:
            spent += self.spent_usd - self.ledger.adjustments.get(self.spent_key, 0.0)
        return spent

    def _tick(self) -> None:
        try:
            self.ledger.tick()
        except (OSError, LedgerError) as exc:
            self.echo(f"[ab] warning: spend ledger tick failed ({exc})")

    def _deadline(self) -> bool:
        """A running child's deadline (called from the runner's watcher thread)."""
        if self.spent_at_start is not None:           # the box session exists
            try:
                self.ledger.tick_if_due(TICK_S)
            except (OSError, LedgerError) as exc:
                self.echo(f"[ab] warning: spend ledger tick failed ({exc})")
        return self.guard.deadline_reached()

    def _cached_throughput(self) -> list[float]:
        """Effective tokens/s of this config's cached runs that completed on their
        first, not resumed attempt: an orchestrator restarted after a crash or a
        budget stop estimates from them rather than from the prior."""
        out = []
        for path in sorted((self.out / "cache").glob("*.json")):
            try:
                d = json.loads(path.read_text("utf-8"))
                if d.get("config_sha256") != self.config_sha:
                    continue
                r = RunResult.from_dict(d["result"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (r.status == "completed" and r.attempts == 1 and not r.resumed
                    and r.effective_tokens_per_s):
                out.append(float(r.effective_tokens_per_s))
        return out

    def key(self, spec: RunSpec) -> str:
        blob = json.dumps({"config": self.config_sha, "extra": self.extra,
                           "settings": spec.settings, "tokens": spec.tokens}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def steps_for(self, tokens: int) -> int:
        return tokens // self.cfg.train.batch_tokens

    def overrides_for(self, spec: RunSpec, ckpt_dir: Path | str) -> list[str]:
        """Every --override flag of a run: the global extras, the run's settings, then
        the bookkeeping (tokens, its own checkpoint dir, no milestones, at most
        MAX_EVALS evals ending on the last step, a checkpoint half way, warmup
        shortened only if it would not fit)."""
        steps = self.steps_for(spec.tokens)
        flat = dict(self.extra)
        flat.update({k: fmt_value(v) for k, v in spec.settings.items()})
        flat["train.total_tokens"] = str(spec.tokens)
        flat["train.ckpt_dir"] = str(ckpt_dir)
        flat["train.milestones"] = "[]"
        if steps >= 1:
            flat["train.eval_every"] = str(aligned_eval_every(steps, self.cfg.train.eval_every))
            flat["train.ckpt_every"] = str(max(1, math.ceil(steps / 2)))
            if self.cfg.train.warmup_steps >= steps:
                flat["train.warmup_steps"] = str(max(1, steps // 10))
        return [f"{k}={v}" for k, v in flat.items()]

    def context_for(self, spec: RunSpec, baseline_losses: list[float] | None) -> RunContext:
        key = self.key(spec)
        run_id = f"ab-{spec.name}-{key[:8]}"
        ckpt_dir = self.out / "ckpt" / key
        return RunContext(
            config=self.config_path, run_id=run_id, run_dir=self.out / "runs",
            ckpt_dir=ckpt_dir, log_path=self.out / "logs" / f"{run_id}.log",
            overrides=self.overrides_for(spec, ckpt_dir), steps=self.steps_for(spec.tokens),
            baseline_losses=baseline_losses, budget_check=self._deadline,
            inductor_cache=self.out / "inductor-cache",
        )

    def bytes_per_token(self) -> float | None:
        if self._bpt == "auto":
            self._bpt = _val_bytes_per_token(self.cfg)
        return self._bpt

    def _cache_path(self, key: str) -> Path:
        return self.out / "cache" / f"{key}.json"

    def _load_cache(self, key: str) -> tuple[str, RunResult] | None:
        try:
            d = json.loads(self._cache_path(key).read_text("utf-8"))
            return d["name"], RunResult.from_dict(d["result"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_cache(self, key: str, spec: RunSpec, result: RunResult) -> None:
        write_text_atomic(self._cache_path(key), json.dumps({
            "key": key, "name": spec.name, "phase": spec.phase, "tokens": spec.tokens,
            "settings": spec.settings, "extra": self.extra, "config": str(self.config_path),
            "config_sha256": self.config_sha, "result": result.to_dict(),
        }, indent=2))

    def _label(self, key: str, text: str) -> None:
        labels = self.labels.setdefault(key, [])
        if text not in labels:
            labels.append(text)

    # -- one run --

    def execute(self, spec: RunSpec) -> RunResult | None:
        key = self.key(spec)
        record = {"spec": spec, "key": key, "note": ""}
        self.records.append(record)
        cached = self._load_cache(key)
        if cached and (cached[1].status not in ("failed", "failed_oom") or not self.retry_failed):
            name, result = cached
            record["note"] = "cached" if name == spec.name else f"same run as {name} (cached)"
            self.results[key] = result
            self.echo(f"[ab] {spec.name}: cached ({result.status})")
            return result
        if self.stopped:
            record["note"] = f"not run ({self.stopped})"
            self.results.setdefault(key, None)
            return None

        base = self.results.get(spec.baseline) if spec.baseline else None
        ctx = self.context_for(spec, base.train_losses if usable(base) else None)
        try:
            load_config(self.config_path, parse_overrides(ctx.overrides))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result = RunResult("failed", None, None, None, None, 0.0, 0, [],
                               f"config refused: {exc}", 0)
            return self._finish(spec, key, record, result, ctx)

        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            cost = self.guard.estimate_usd(spec.tokens)
            if not self.guard.allows(cost):
                self.stopped = "budget"
                self.echo(
                    f"[ab] refused {spec.name}: estimated ${cost:.2f} would take spend from "
                    f"${self.guard.spent():.2f} past the ${self.guard.budget_usd:.2f} budget; "
                    "no further runs start"
                )
                if result is None:
                    record["note"] = "not run (budget)"
                    self.results.setdefault(key, None)
                    return None
                result.status = "failed"
                result.reason += "; retry refused by the budget"
                break
            self.echo(f"[ab] start {spec.name} ({spec.tokens:,} tokens, attempt {attempt}, "
                      f"est ${cost:.2f}, spent ${self.guard.spent():.2f})")
            t0 = self.clock()
            result = dataclasses.replace(self.runner(spec, ctx))   # never mutate the runner's object
            wall = self.clock() - t0
            result.attempts = attempt
            result.wall_s = wall
            # A resumed attempt trained only part of the tokens: its tokens / wall
            # would overstate the throughput, so it is neither reported nor observed.
            result.effective_tokens_per_s = (spec.tokens / wall if wall > 0 and not result.resumed
                                             else None)
            if result.status == "interrupted":
                raise KeyboardInterrupt
            if result.status == "stopped_budget":
                return self._stopped_by_budget(spec, key, record, result)
            if result.status == "completed" and not result.resumed:
                self.guard.observe(spec.tokens, wall)
            if result.status in ("completed", "diverged", "failed_oom"):
                break             # failed_oom: the same settings would run out again
            if result.status == "crashed" and attempt < MAX_ATTEMPTS:
                self.echo(f"[ab] {spec.name} crashed ({result.reason}); retrying once")
                continue
            result.status = "failed"
            break
        return self._finish(spec, key, record, result, ctx)

    def _stopped_by_budget(self, spec: RunSpec, key: str, record: dict[str, Any],
                           result: RunResult) -> RunResult:
        """Not cached and its checkpoint kept: the next invocation resumes it."""
        self.stopped = "budget"
        record["note"] = "stopped by the budget mid-run; checkpoint kept, resumes next time"
        self.results[key] = result
        self._tick()
        self.echo(f"[ab] {spec.name}: stopped by the budget mid-run (checkpoint kept); "
                  f"spent ${self.guard.spent():.2f}{self._of_budget()}; no further runs start")
        return result

    def _finish(self, spec: RunSpec, key: str, record: dict[str, Any], result: RunResult,
                ctx: RunContext) -> RunResult:
        if usable(result) and result.bpb is None:
            bpt = self.bytes_per_token()
            if bpt:
                result.bpb = result.final_val_loss / (math.log(2) * bpt)
        self._save_cache(key, spec, result)
        self.results[key] = result
        if not self.keep_checkpoints and ctx.ckpt_dir.exists():
            shutil.rmtree(ctx.ckpt_dir, ignore_errors=True)
        self._tick()
        val = "-" if result.final_val_loss is None else f"{result.final_val_loss:.4f}"
        self.echo(f"[ab] {spec.name}: {result.status}, val {val}"
                  f"{' (' + result.reason + ')' if result.reason else ''}; "
                  f"spent ${self.guard.spent():.2f}{self._of_budget()}")
        return result

    def _of_budget(self) -> str:
        return "" if self.guard.budget_usd is None else f" of ${self.guard.budget_usd:.2f}"

    # -- the procedure --

    def base_settings(self) -> dict[str, Any]:
        """The config's values, with every pair under test at its simpler option."""
        t, m = self.cfg.train, self.cfg.model
        s = {"train.optimizer": t.optimizer, "train.lr": t.lr, "train.muon_lr": t.muon_lr,
             "model.attnres_blocks": m.attnres_blocks, "model.activation": m.activation,
             "train.precision": t.precision, "train.seed": t.seed}
        for p in PAIRS:
            if p.name in self.pairs:
                s[p.key] = p.simple
        return s

    def other_option(self, pair: Pair, best_muon: Any) -> dict[str, Any]:
        return {"optimizer": {"train.optimizer": "muon", "train.muon_lr": best_muon},
                "attnres": {"model.attnres_blocks": self.attnres_on},
                "activation": {"model.activation": "situ_glu"},
                "precision": {"train.precision": "fp8"}}[pair.name]

    def sweep_specs(self, kind: str, base: dict[str, Any], lr: Any) -> list[RunSpec]:
        if kind == "adamw":
            grid, key, default = ADAMW_LRS, "train.lr", self.cfg.train.lr
            fixed = {"train.optimizer": "adamw"}
        else:
            grid, key, default = MUON_LRS, "train.muon_lr", self.cfg.train.muon_lr
            fixed = {"train.optimizer": "muon", "train.lr": lr}
        order = sorted(grid, key=lambda v: v != default)      # the config value first
        return [RunSpec(f"sweep-{kind}-lr{fmt_lr(v)}", "sweep", {**base, **fixed, key: v},
                        self.sweep_tokens) for v in order]

    def _sweep(self, kind: str, base: dict[str, Any], lr: Any) -> float:
        specs = self.sweep_specs(kind, base, lr)
        results: dict[float, RunResult | None] = {}
        key_name = "train.lr" if kind == "adamw" else "train.muon_lr"
        default = self.cfg.train.lr if kind == "adamw" else self.cfg.train.muon_lr
        for i, spec in enumerate(specs):
            if i:
                spec.baseline = self.key(specs[0])
            results[spec.settings[key_name]] = self.execute(spec)
        best, note = decide_sweep(results, default)
        label = "AdamW lr" if kind == "adamw" else "Muon lr"
        self.lr_notes.append(f"{label}: {best:g} ({note})")
        for spec in specs:
            if spec.settings[key_name] == best and usable(results[best]):
                self._label(self.key(spec), f"best {label}")
        return best

    def _seed_rerun(self, winner: RunSpec) -> RunResult | None:
        if not self.seed_reruns:
            return None
        seed = winner.settings["train.seed"] + 1
        spec = RunSpec(f"{winner.name}-seed{seed}", "seed",
                       {**winner.settings, "train.seed": seed}, winner.tokens,
                       baseline=self.key(winner))
        result = self.execute(spec)
        self._label(self.key(spec), f"seed re-run of {winner.name}")
        return result

    def _pair(self, pair: Pair, simple: RunSpec, other: RunSpec) -> RunSpec:
        r_simple = self.results.get(self.key(simple))
        if simple.name not in {r["spec"].name for r in self.records}:
            r_simple = self.execute(simple)
        if usable(r_simple):
            other.baseline = self.key(simple)
            r_other = self.execute(other)
        else:
            r_other = None
            self.records.append({"spec": other, "key": self.key(other),
                                 "note": "not run (baseline missing)"})
            self.results.setdefault(self.key(other), None)
        reseed = None
        if usable(r_simple) and (usable(r_other) or (r_other and r_other.status == "diverged")):
            prelim = "simple" if not usable(r_other) else preliminary(pair.name, r_simple, r_other)[0]
            reseed = self._seed_rerun(simple if prelim == "simple" else other)
        d = decide_pair(pair.name, r_simple, r_other, reseed)
        kept = simple if d.keep == "simple" else other
        note = ""
        if reseed is not None and d.prelim == "simple":
            # The noise rule only ever overturns a win of the other option.
            note = ("the seed re-run cannot change this decision (the simpler option "
                    "won on the other rules); it only measures the noise")
        self.decisions.append((pair, d, {"simple": simple.name, "other": other.name,
                                         "kept": fmt_value(kept.settings[pair.key]),
                                         "note": note}))
        self._label(self.key(kept), f"pair {pair.number}: kept")
        loser = other if d.keep == "simple" else simple
        if usable(self.results.get(self.key(loser))):
            self._label(self.key(loser), f"pair {pair.number}: not kept")
        self.echo(f"[ab] pair {pair.number} ({pair.name}): keep "
                  f"{kept.settings[pair.key]} ({d.reason})")
        return kept

    def run(self) -> int:
        """The whole procedure. Returns 0, or EXIT_BUDGET if the budget stopped it.
        summary.md, winners.toml and the spend ledger are written even when it is
        interrupted. The ledger is ticked every tick_every seconds meanwhile (a
        background thread; None = only at start, after each run and at the end)."""
        self.begin()
        ticker = (Ticker(self.ledger, self.tick_every) if self.tick_every
                  else contextlib.nullcontext())
        try:
            with ticker:
                self._procedure()
        finally:
            self.write_outputs()
        return EXIT_BUDGET if self.stopped == "budget" else 0

    def _procedure(self) -> None:
        base = self.base_settings()
        if self.skip_sweeps:
            best_lr, best_muon = self.cfg.train.lr, self.cfg.train.muon_lr
            self.lr_notes.append(f"sweeps skipped: config lr {best_lr:g}, muon_lr {best_muon:g}")
        else:
            best_lr = self._sweep("adamw", base, None)
            best_muon = self._sweep("muon", base, best_lr)
        base = {**base, "train.lr": best_lr, "train.muon_lr": best_muon}
        self.final_settings = dict(base)
        if "optimizer" in self.pairs:
            p1 = PAIR_BY_NAME["optimizer"]
            simple = RunSpec("p1-adamw", "pair", {**base, "train.optimizer": "adamw"},
                             self.arm_tokens)
            other = RunSpec("p1-muon", "pair", {**base, **self.other_option(p1, best_muon)},
                            self.arm_tokens)
            winner = self._pair(p1, simple, other)
        else:
            winner = RunSpec("p1-base", "pair", base, self.arm_tokens)
            self.execute(winner)
        self.final_settings = dict(winner.settings)
        for pair in PAIRS[1:]:
            if pair.name not in self.pairs:
                continue
            other = RunSpec(f"p{pair.number}-{pair.label}", "pair",
                            {**winner.settings, **self.other_option(pair, best_muon)},
                            self.arm_tokens)
            kept = self._pair(pair, winner, other)
            self.final_settings[pair.key] = kept.settings[pair.key]

    # -- dry run --

    def outline(self) -> list[RunSpec]:
        """The worst-case run list (every seed re-run distinct), with values that
        depend on earlier runs shown as <placeholders>."""
        base = self.base_settings()
        runs: list[RunSpec] = []
        if self.skip_sweeps:
            best_lr, best_muon = self.cfg.train.lr, self.cfg.train.muon_lr
        else:
            runs += self.sweep_specs("adamw", base, None)
            best_lr = Pending("best AdamW lr")
            runs += self.sweep_specs("muon", base, best_lr)
            best_muon = Pending("best Muon lr")
        base = {**base, "train.lr": best_lr, "train.muon_lr": best_muon}
        seed = self.cfg.train.seed + 1
        if "optimizer" in self.pairs:
            runs.append(RunSpec("p1-adamw", "pair", {**base, "train.optimizer": "adamw"},
                                self.arm_tokens))
            runs.append(RunSpec("p1-muon", "pair",
                                {**base, **self.other_option(PAIRS[0], best_muon)},
                                self.arm_tokens))
            winner: dict[str, Any] = {"<pair-1 winner settings>": ""}
        else:
            runs.append(RunSpec("p1-base", "pair", base, self.arm_tokens))
            winner = base
        for pair in PAIRS[1:]:
            if pair.name in self.pairs:
                runs.append(RunSpec(f"p{pair.number}-{pair.label}", "pair",
                                    {**winner, **self.other_option(pair, best_muon)},
                                    self.arm_tokens))
        if self.seed_reruns:
            for pair in PAIRS:
                if pair.name in self.pairs:
                    runs.append(RunSpec(
                        f"p{pair.number}-winner-seed{seed}", "seed",
                        {f"<pair-{pair.number} winner settings>": "", "train.seed": seed},
                        self.arm_tokens))
        return runs

    # -- outputs --

    def config_diff(self, spec: RunSpec) -> str:
        t, m = self.cfg.train, self.cfg.model
        now = {"train.optimizer": t.optimizer, "train.lr": t.lr, "train.muon_lr": t.muon_lr,
               "model.attnres_blocks": m.attnres_blocks, "model.activation": m.activation,
               "train.precision": t.precision, "train.seed": t.seed}
        parts = [f"{k.split('.', 1)[1]}={fmt_value(v)}" for k, v in spec.settings.items()
                 if now.get(k) != v]
        return ", ".join(parts) or "(config)"

    def winners(self) -> dict[str, dict[str, Any]]:
        s = dict(self.final_settings or self.base_settings())
        s.pop("train.seed", None)
        if s.get("train.optimizer") != "muon":
            s.pop("train.muon_lr", None)
        out: dict[str, dict[str, Any]] = {}
        for k, v in sorted(s.items()):
            section, field = k.split(".", 1)
            out.setdefault(section, {})[field] = v
        return out

    def write_outputs(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.spent_at_start is not None:
            self._tick()
        write_text_atomic(self.out / "winners.toml", self._winners_toml())
        write_text_atomic(self.out / "summary.md", self._summary_md())

    def _winners_toml(self) -> str:
        lines = [
            "# A/B winners from scripts/ab_runs.py: overrides for the full run",
            f"# config {self.config_path.name} (sha256 {self.config_sha[:12]}), "
            f"written {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "# load_config(path, tomllib.load(this file)) applies them.",
        ]
        lines += [f"# {n}" for n in self.lr_notes]
        for pair, d, names in self.decisions:
            lines.append(f"# pair {pair.number} {pair.name}: {names['kept']} - {d.reason}")
        if self.stopped:
            lines.append(f"# stopped early ({self.stopped}): missing arms kept the simpler option")
        for section, values in self.winners().items():
            lines += ["", f"[{section}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in values.items()]
        return "\n".join(lines) + "\n"

    def _summary_md(self) -> str:
        g = self.guard
        budget = "no budget" if g.budget_usd is None else f"budget ${g.budget_usd:.2f}"
        at_start = ("" if self.spent_at_start is None
                    else f"; ${self.spent_at_start:.2f} when this invocation started")
        lines = [
            f"# A/B runs: {self.cfg.name}", "",
            f"- Config: `{self.config_path.name}` (sha256 {self.config_sha[:12]})"
            + (f", overrides {', '.join(f'{k}={v}' for k, v in self.extra.items())}"
               if self.extra else ""),
            f"- Box spend: ${g.spent():.2f} ({budget}, ${g.usd_per_hour:.2f}/h{at_start}; "
            f"ledger `{self.ledger.path}`)",
            f"- Pairs: {', '.join(self.pairs)}; seed re-runs "
            f"{'on' if self.seed_reruns else 'off'}",
        ]
        if self.stopped:
            lines.append(f"- **Stopped early ({self.stopped})**: runs after the refused or "
                         "stopped one were not started; missing arms keep the simpler option.")
        lines += ["", "## Runs", "",
                  "| # | run | config diff | tokens | status | final val loss | bpb | "
                  "tokens/s | spikes | skipped | startup s | overhead s | decision |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        seen: set[tuple[str, str]] = set()
        n = 0
        for rec in self.records:
            spec, key = rec["spec"], rec["key"]
            if (spec.name, key) in seen:
                continue
            seen.add((spec.name, key))
            n += 1
            r = self.results.get(key)
            status = rec["note"] if r is None else r.status
            if r is not None and r.reason:
                status += f" ({r.reason})"
            if r is not None and rec["note"]:
                status += f"; {rec['note']}"
            lines.append("| " + " | ".join([
                str(n), spec.name, self.config_diff(spec), f"{spec.tokens:,}", status,
                _f(r.final_val_loss if r else None), _f(r.bpb if r else None, 4),
                "-" if r is None or not _tps(r) else f"{_tps(r):,.0f}",
                "-" if r is None else str(r.spikes),
                "-" if r is None or r.skipped is None else str(r.skipped),
                _f(r.startup_s if r else None, 0), _f(r.overhead_s if r else None, 0),
                "; ".join(self.labels.get(key, [])) or "-",
            ]) + " |")
        lines += ["", "## Overhead", "", self._overhead_note()]
        lines += ["", "## Learning rates", ""] + [f"- {n_}" for n_ in self.lr_notes]
        lines += ["", "## Decisions", "",
                  "| pair | simpler | other | kept | noise | reason |", "|---|---|---|---|---|---|"]
        for pair, d, names in self.decisions:
            run = names["simple"] if d.keep == "simple" else names["other"]
            reason = d.reason + (f". Note: {names['note']}" if names.get("note") else "")
            lines.append(f"| {pair.number} {pair.name} | {names['simple']} | {names['other']} "
                         f"| {pair.key.split('.', 1)[1]}={names['kept']} ({run}) "
                         f"| {_f(d.noise)} | {reason} |")
        lines += ["", "Rules: the other option needs lower final val loss by more than the "
                  "seed noise; AttnRes <= 10% tokens/s cost; SiTU-GLU no more spikes "
                  f"(step loss > {SPIKE_FACTOR} x EMA, beta {SPIKE_EMA_BETA}); FP8 >= "
                  f"{FP8_MIN_SPEEDUP}x tokens/s, loss within noise, no more spikes. A missing "
                  "arm keeps the simpler option (AdamW, no AttnRes, SwiGLU, bf16).",
                  "", "## winners.toml", "", "```toml", self._winners_toml().rstrip(), "```", ""]
        return "\n".join(lines)

    def _overhead_note(self) -> str:
        """The measured per-run overhead against the prior the first estimates used."""
        runs = [r for r in self.results.values() if r is not None]
        over = [r.overhead_s for r in runs if r.overhead_s is not None and not r.resumed]
        start = [r.startup_s for r in runs if r.startup_s is not None]
        prior = f"prior {self.guard.overhead_s:.0f} s (--overhead-seconds)"
        if not over:
            return f"No run measured its overhead yet; estimates use the {prior}."
        text = (f"Per-run overhead (wall - tokens / steady tokens/s: startup, compile, evals, "
                f"checkpoints): median {statistics.median(over):.0f} s over {len(over)} runs "
                f"(min {min(over):.0f}, max {max(over):.0f}) vs the {prior}.")
        if start:
            text += (f" Startup + compile to the first step line: median "
                     f"{statistics.median(start):.0f} s.")
        return text


def _f(x: float | None, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _val_bytes_per_token(cfg: Config) -> float | None:
    """Mean UTF-8 bytes per target token over the batches the trainer's eval reads
    (eval_batches x micro_batch x context from the start of <shard_dir>/val), so
    bits per byte = final val loss / (ln 2 x this), exactly: every arm evaluates the
    same tokens. None when the val shards or the tokenizer are not there."""
    try:
        val_dir = Path(cfg.data.shard_dir) / "val"
        if not val_dir.is_absolute():
            val_dir = ROOT / val_dir
        tok_path = cfg.data.tokenizer
        if tok_path != "gpt2" and not Path(tok_path).is_absolute():
            tok_path = str(ROOT / tok_path)
        if not val_dir.is_dir() or (tok_path != "gpt2" and not Path(tok_path).is_file()):
            return None
        import numpy as np

        from quipu.loader import TokenStream
        from quipu.tokenizer import make_tokenizer

        lens = np.asarray(make_tokenizer(tok_path).token_byte_lengths())
        stream = TokenStream(val_dir, cfg.train.micro_batch, cfg.model.context)
        total = count = 0
        try:
            for _ in range(cfg.train.eval_batches):
                _, y = stream.next_batch()
                total += int(lens[y.numpy()].sum())
                count += y.numel()
        finally:
            stream.close()
        return total / count if total else None
    except Exception as exc:          # bpb is a report column, never a reason to stop
        print(f"[ab] warning: no bits per byte ({exc})", file=sys.stderr, flush=True)
        return None


# ---- the training subprocess ------------------------------------------------------------------

STEP_LINE = re.compile(r"^step (\d+)/(\d+)\s+loss (\S+).*?([\d,.]+) tok/s")


class SubprocessRunner:
    """Runs one A/B arm as `python -m quipu.train` and watches it.

    - Step lines: a non-finite loss, or a loss above DIVERGE_FACTOR x the baseline's
      at any printed step from DIVERGE_AT of the run, kills it (diverged).
    - A watcher thread polls ctx.budget_check every poll_s; once it is True the
      child is interrupted (SIGINT; on Windows CTRL_BREAK, which the trainer handles
      as SIGBREAK) so it writes its interrupt checkpoint, given grace_s, then
      terminated, then killed: the run is "stopped_budget".
    - Ctrl+C here (KeyboardInterrupt): on POSIX the child got the SIGINT too, so it
      is first given grace_s to checkpoint and exit; then it is interrupted, then
      terminated, then killed (kill_wait_s between the last steps). On Windows the
      child runs in its own process group (so CTRL_BREAK can reach it alone) and
      does not see the console's Ctrl+C: it is interrupted first.
    - "out of memory" in the output with a failed exit is failed_oom (not retried).
    - --resume is added when the run's checkpoint exists (a retry after a crash, or
      a run the budget stopped); a stale run log without a checkpoint is removed so
      the retry starts clean. The result says whether it resumed.
    - The child shares a persistent inductor cache (ctx.inductor_cache).
    Output goes to ctx.log_path and to the console."""

    def __init__(self, cmd: list[str] | None = None, device: str = "auto",
                 cwd: str | Path = ROOT, echo: Callable[[str], None] | None = None,
                 grace_s: float = STOP_GRACE_S, kill_wait_s: float = KILL_WAIT_S,
                 poll_s: float = BUDGET_POLL_S) -> None:
        self.cmd = cmd or [sys.executable, "-m", "quipu.train"]
        self.device = device
        self.cwd = Path(cwd)
        self.echo = echo or (lambda s: print(s, flush=True))
        self.grace_s = grace_s
        self.kill_wait_s = kill_wait_s
        self.poll_s = poll_s

    # -- stopping a child --

    @staticmethod
    def _interrupt(proc: subprocess.Popen) -> bool:
        """SIGINT (Windows: CTRL_BREAK to the child's process group). False if it
        could not be sent (the process is gone, or no console to send it through)."""
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
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
        except KeyboardInterrupt:       # another Ctrl+C: go on to the next step
            return False

    def _stop_child(self, proc: Any, child_got_it: bool) -> None:
        """Let the child checkpoint and exit; escalate only if it does not:
        (wait grace_s if it already got the interrupt) -> interrupt -> terminate ->
        kill."""
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

    # -- one run --

    def __call__(self, spec: RunSpec, ctx: RunContext) -> RunResult:
        args = [*self.cmd, "--config", str(ctx.config), "--run-id", ctx.run_id,
                "--run-dir", str(ctx.run_dir), "--device", self.device]
        for o in ctx.overrides:
            args += ["--override", o]
        run_log = ctx.run_dir / f"{ctx.run_id}.json"
        resumed = (ctx.ckpt_dir / "latest.pt").exists()
        if resumed:
            args.append("--resume")
        elif run_log.exists():
            run_log.unlink()          # our own log of an attempt that never checkpointed
        ctx.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TORCHINDUCTOR_FX_GRAPH_CACHE="1",
                   TORCHINDUCTOR_AUTOGRAD_CACHE="1")
        if ctx.inductor_cache is not None:
            ctx.inductor_cache.mkdir(parents=True, exist_ok=True)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(ctx.inductor_cache)
        check_at = max(1, math.ceil(DIVERGE_AT * ctx.steps))
        diverged = ""
        oom = False
        first_step_at: float | None = None
        tok_s: list[float] = []
        budget_hit = threading.Event()
        done = threading.Event()
        t0 = time.monotonic()
        proc = subprocess.Popen(
            args, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)

        def watch() -> None:
            while not done.wait(self.poll_s):
                if proc.poll() is not None:
                    return
                try:
                    over = ctx.budget_check()
                except Exception as exc:        # noqa: BLE001 - a check failure is not a stop
                    self.echo(f"[ab] warning: budget check failed ({exc})")
                    continue
                if over:
                    budget_hit.set()
                    self.echo(f"[ab] budget reached during {spec.name}: interrupting it "
                              f"(checkpoint, up to {self.grace_s:.0f} s)")
                    self._stop_child(proc, child_got_it=False)
                    return

        watcher = threading.Thread(target=watch, name="budget-watch", daemon=True)
        if ctx.budget_check is not None:
            watcher.start()
        try:
            with open(ctx.log_path, "a", encoding="utf-8") as log:
                for line in proc.stdout:
                    log.write(line)
                    self.echo(line.rstrip("\n"))
                    if any(mark in line.lower() for mark in OOM_MARKERS):
                        oom = True
                    m = STEP_LINE.match(line.strip())
                    if not m or diverged:
                        continue
                    if first_step_at is None:
                        first_step_at = time.monotonic()
                    step, loss = int(m.group(1)), float(m.group(3))
                    tok_s.append(float(m.group(4).replace(",", "")))
                    if not math.isfinite(loss):
                        diverged = f"non-finite loss at step {step}"
                    elif step >= check_at and ctx.baseline_losses:
                        ref = _baseline_ref(ctx.baseline_losses, step)
                        if ref is not None and loss > DIVERGE_FACTOR * ref:
                            diverged = (f"loss {loss:.3f} > {DIVERGE_FACTOR:g} x baseline "
                                        f"{ref:.3f} at step {step}/{ctx.steps}")
                    if diverged:
                        self.echo(f"[ab] early stop {spec.name}: {diverged}")
                        proc.kill()
            code = proc.wait()
        except KeyboardInterrupt:
            done.set()
            self.echo(f"[ab] interrupted: letting {spec.name} write its checkpoint")
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
        wall = time.monotonic() - t0

        try:
            record = json.loads(run_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        steps = record.get("steps", [])
        losses = [s.get("train_loss") for s in steps]
        losses = [float(x) for x in losses if x is not None]
        skipped = steps[-1].get("skipped") if steps else None
        evals = record.get("evals") or []
        val = float(evals[-1]["val_loss"]) if evals else None
        steady = (statistics.median(tok_s[1:]) if len(tok_s) > 1
                  else (tok_s[0] if tok_s else None))
        eff = spec.tokens / wall if wall > 0 and not resumed else None
        finished = code == 0 and val is not None and math.isfinite(val)
        if budget_hit.is_set() and not finished:
            last = steps[-1].get("step") if steps else 0
            status, reason = "stopped_budget", (f"the budget was reached mid-run; "
                                                f"interrupted after step {last}/{ctx.steps}")
        elif diverged:
            status, reason = "diverged", diverged
        elif finished:
            status, reason = "completed", ""
        elif code == 0:
            status, reason = "crashed", "finished without a final val loss"
        elif code == TRAIN_NONFINITE:
            status, reason = "diverged", "the trainer's non-finite stop (exit 3)"
        elif code in TRAIN_INTERRUPT_CODES:
            status, reason = "interrupted", f"exit {code}"
        elif code == TRAIN_USAGE:
            status, reason = "usage", "usage/config error (exit 2)"
        elif oom:
            status, reason = "failed_oom", f"CUDA out of memory (exit {code}); not retried"
        else:
            status, reason = "crashed", f"exit {code}"
        return RunResult(
            status=status, final_val_loss=val if status == "completed" else None,
            bpb=None, tokens_per_s=steady or eff, effective_tokens_per_s=eff,
            wall_s=wall, spikes=count_spikes(losses), train_losses=losses, reason=reason,
            resumed=resumed, skipped=int(skipped) if isinstance(skipped, (int, float)) else None,
            startup_s=None if first_step_at is None else first_step_at - t0,
            overhead_s=(wall - spec.tokens / steady) if steady and not resumed else None,
        )


def _baseline_ref(losses: list[float], step: int) -> float | None:
    window = [x for x in losses[max(0, step - DIVERGE_WINDOW):step]
              if x is not None and math.isfinite(x)]
    return sum(window) / len(window) if window else None


# ---- CLI --------------------------------------------------------------------------------------

def _print_dry_run(o: Orchestrator, tps: float) -> None:
    runs = o.outline()
    total = sum(r.tokens for r in runs)
    n_seed = sum(r.phase == "seed" for r in runs)
    fewest = len(runs) - max(0, n_seed - 1)
    hours = (total / tps + len(runs) * o.guard.overhead_s) / 3600
    cost = hours * o.guard.usd_per_hour
    print(f"A/B plan for {o.config_path.name} -> {o.out}")
    print(f"pairs: {', '.join(o.pairs)}; batch {o.cfg.train.batch_tokens:,} tokens; "
          "every run also gets train.total_tokens, its own train.ckpt_dir, "
          f"train.milestones=[], an eval_every that ends on the last step (<= {MAX_EVALS} "
          "evals) and a checkpoint half way")
    print(f"  {'#':>2}  {'phase':<6} {'run':<24} {'tokens':>13}  overrides")
    for i, r in enumerate(runs, 1):
        flags = " ".join(f"{k}={fmt_value(v)}" if v != "" else k for k, v in r.settings.items())
        print(f"  {i:>2}  {r.phase:<6} {r.name:<24} {r.tokens:>13,}  {flags}")
    print(f"Total: {len(runs)} runs (worst case; seed re-runs shared with an earlier one are "
          f"cached, so as few as {fewest}), {total:,} tokens")
    print(f"Estimated time and cost at {tps:,.0f} tokens/s and ${o.guard.usd_per_hour:.2f}/h "
          f"(+{o.guard.overhead_s:.0f} s overhead per run): {hours:.2f} h, ${cost:.2f}")
    spent = o.projected_spent()
    print(f"Box spend so far ${spent:.2f} (ledger {o.ledger.path})")
    if o.guard.budget_usd is not None:
        verdict = "fits" if cost <= o.guard.budget_usd - spent else "does NOT fit"
        print(f"Budget ${o.guard.budget_usd:.2f} (spent ${spent:.2f}): {verdict}")


def _checkpoint_bytes(cfg: Config) -> int:
    """One resumable checkpoint of this config: parameters (counted on the meta
    device, nothing allocated) x CKPT_BYTES_PER_PARAM."""
    import torch

    from quipu.model_factory import build_model

    with torch.device("meta"):
        model = build_model(cfg.model)
    return sum(p.numel() for p in model.parameters()) * CKPT_BYTES_PER_PARAM


def _free_bytes(path: Path) -> int:
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def checkpoint_disk_check(o: Orchestrator) -> tuple[bool, str]:
    """(fits, message). A running run needs ckpt_keep + 1 checkpoints at its peak
    (the new one is written before the old is pruned); with --keep-checkpoints every
    run of the worst-case list also keeps ckpt_keep of them."""
    per = _checkpoint_bytes(o.cfg)
    keep = o.cfg.train.ckpt_keep
    peak = per * (keep + 1)
    runs = len(o.outline())
    need = peak + (runs * keep * per if o.keep_checkpoints else 0)
    free = _free_bytes(o.out)
    gb = 1e9
    what = (f"{runs} runs x {keep} kept + a running run's {keep + 1}" if o.keep_checkpoints
            else f"a running run's {keep + 1}")
    msg = (f"checkpoints: {per / gb:.1f} GB each; {what} = {need / gb:.1f} GB needed, "
           f"{free / gb:.1f} GB free on the disk of {o.out}")
    return need <= free, msg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/quipu-moe-ab.toml")
    p.add_argument("--out", default="results/ab")
    p.add_argument("--dry-run", action="store_true", help="print the run list and cost; train nothing")
    p.add_argument("--with-fp8", action="store_true", help="add pair 4, bf16 vs fp8 (~$0.40 more)")
    p.add_argument("--budget-usd", type=float,
                   help="cap on the box ledger's total spend (required to train)")
    p.add_argument("--usd-per-hour", type=float,
                   help="the box's rate (required to train; starts the box session if "
                        "`python -m quipu.spend start` was not run)")
    p.add_argument("--spent-usd", type=float, default=0.0,
                   help="spend the ledger cannot see; recorded once as the adjustment "
                        "--spent-key (the same key again replaces it, never adds)")
    p.add_argument("--spent-key", default=SPENT_KEY)
    p.add_argument("--tokens-per-second", type=float, default=PRIOR_TOKENS_PER_S,
                   help="throughput assumed before a run is measured (and for --dry-run)")
    p.add_argument("--overhead-seconds", type=float, default=PRIOR_OVERHEAD_S)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                   help=f"comma list from {','.join(PAIR_BY_NAME)}")
    p.add_argument("--skip-sweeps", action="store_true", help="use the config's lr and muon_lr")
    p.add_argument("--no-seed-reruns", action="store_true")
    p.add_argument("--sweep-tokens", type=int, default=SWEEP_TOKENS)
    p.add_argument("--arm-tokens", type=int, default=ARM_TOKENS)
    p.add_argument("--attnres-blocks", type=int, default=ATTNRES_ON)
    p.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="applied to every run (part of the cache key); repeatable")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    args = p.parse_args(argv)

    extra: dict[str, str] = {}
    for item in args.override:
        k, sep, v = item.partition("=")
        if not sep:
            print(f"error: --override {item!r} must be section.key=value", file=sys.stderr)
            return EXIT_USAGE
        extra[k.strip()] = v.strip()
    try:
        o = Orchestrator(
            args.config, args.out,
            SubprocessRunner(device=args.device),
            budget_usd=args.budget_usd, usd_per_hour=args.usd_per_hour,
            tokens_per_second=args.tokens_per_second, overhead_s=args.overhead_seconds,
            spent_usd=args.spent_usd, spent_key=args.spent_key, tick_every=TICK_S,
            with_fp8=args.with_fp8, skip_sweeps=args.skip_sweeps,
            pairs=tuple(x.strip() for x in args.pairs.split(",") if x.strip()),
            seed_reruns=not args.no_seed_reruns, sweep_tokens=args.sweep_tokens,
            arm_tokens=args.arm_tokens, attnres_on=args.attnres_blocks, extra_overrides=extra,
            retry_failed=args.retry_failed, keep_checkpoints=args.keep_checkpoints,
        )
    except (OSError, ValueError, TypeError, KeyError, LedgerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    fits, disk_msg = checkpoint_disk_check(o)
    if args.dry_run:
        _print_dry_run(o, args.tokens_per_second)
        print(("" if fits else "WARNING: ") + disk_msg)
        return 0
    if args.budget_usd is None:
        print("error: pass --budget-usd (the A/B config's budget_usd 0 means this "
              "orchestrator enforces the cap)", file=sys.stderr)
        return EXIT_USAGE
    if args.usd_per_hour is None or args.usd_per_hour <= 0 or not math.isfinite(args.usd_per_hour):
        print("error: pass --usd-per-hour > 0 (the box's rate; the spend ledger bills "
              "box time at it)", file=sys.stderr)
        return EXIT_USAGE
    if not fits:
        if args.keep_checkpoints:
            print(f"error: --keep-checkpoints would not fit on the disk: {disk_msg}",
                  file=sys.stderr)
            return EXIT_USAGE
        print(f"[ab] WARNING: a run's checkpoints may not fit on the disk: {disk_msg}",
              file=sys.stderr, flush=True)
    try:
        o.begin()
    except (OSError, ValueError, LedgerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        return o.run()
    except KeyboardInterrupt:
        print("[ab] interrupted; summary written", file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
