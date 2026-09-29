"""The A/B runs for quipu-moe (spec sections 6.1 and 12, plan Task M8).

    python scripts/ab_runs.py --config configs/quipu-moe-ab.toml --out results/ab \\
        --budget-usd 3 --usd-per-hour 0.55 [--with-fp8] [--dry-run]

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
- Spend guard: spend = earlier invocations' spend (results/ab/spend.json, plus
  --spent-usd) + wall clock since this orchestrator started x --usd-per-hour. Before
  each run (and each retry) its cost is estimated from the slowest measured
  tokens/s of the runs so far (wall clock, overhead included), or from
  --tokens-per-second + --overhead-seconds before any run has finished; a run that
  would take spend past --budget-usd is refused and nothing after it starts. The
  running spend is printed after every run. The runs themselves keep the config's
  train.budget_usd (0 in quipu-moe-ab.toml: the orchestrator guards, not each run).
- Resume: each finished run is cached as results/ab/cache/<key>.json, key = hash of
  the config file, the --override flags and the run's settings and tokens. A re-run
  of the orchestrator skips every cached run. A crashed run (exit 1) is retried once
  (from its checkpoint if it has one), then cached as failed; --retry-failed tries
  failed runs again.
- Early stop: a non-finite loss, the trainer's non-finite stop (exit 3), or a train
  loss above DIVERGE_FACTOR x its baseline's at DIVERGE_AT of the run kills the run
  and marks it diverged (a diverged other-option loses its pair).
- Checkpoints of finished runs are deleted (a 1B-parameter checkpoint with optimizer
  state is ~12 GB); --keep-checkpoints keeps them.

--dry-run prints the run list (worst case, every seed re-run distinct), the token
total and the estimated cost at --tokens-per-second, and trains nothing.

Exit codes: 0 done (whatever the decisions), 2 usage error, 4 stopped by the budget
(summary and winners still written), 130 interrupted.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quipu.config import Config, load_config, parse_overrides
from quipu.fsio import write_text_atomic

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

PRIOR_TOKENS_PER_S = 100_000.0  # conservative until a run has been measured
PRIOR_OVERHEAD_S = 180.0        # startup, compile, evals and checkpoints per run
MAX_ATTEMPTS = 2                # a crashed run is retried once

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


@dataclasses.dataclass
class RunResult:
    status: str                     # completed | diverged | failed (| crashed, usage, interrupted from a runner)
    final_val_loss: float | None
    bpb: float | None
    tokens_per_s: float | None      # steady-state, from the trainer's step lines
    effective_tokens_per_s: float | None   # tokens / wall clock, overhead included
    wall_s: float
    spikes: int
    train_losses: list[float]
    reason: str = ""
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        return cls(**{f.name: d.get(f.name) for f in dataclasses.fields(cls)})


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
    """spend = prior_usd + (clock() - start) / 3600 x usd_per_hour."""

    def __init__(self, budget_usd: float | None, usd_per_hour: float,
                 clock: Callable[[], float], prior_usd: float = 0.0,
                 prior_tokens_per_s: float = PRIOR_TOKENS_PER_S,
                 overhead_s: float = PRIOR_OVERHEAD_S) -> None:
        self.budget_usd = budget_usd
        self.usd_per_hour = usd_per_hour
        self.clock = clock
        self.prior_usd = prior_usd
        self.prior_tokens_per_s = prior_tokens_per_s
        self.overhead_s = overhead_s
        self.start = clock()
        self.measured: list[float] = []      # effective tokens/s of finished runs

    def spent(self) -> float:
        return self.prior_usd + (self.clock() - self.start) / 3600 * self.usd_per_hour

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

def aligned_eval_every(steps: int, want: int) -> int:
    """The divisor of steps closest to `want` (log scale), so the last step is an eval
    step and "final val loss" is measured at the end of the run."""
    divisors = [d for d in range(1, steps + 1) if steps % d == 0]
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
        self.cfg: Config = load_config(self.config_path, self._parse(self.extra))
        self.config_sha = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        rate = self.cfg.train.usd_per_hour if usd_per_hour is None else usd_per_hour
        self.guard = SpendGuard(budget_usd, rate, clock, self._ledger() + spent_usd,
                                tokens_per_second, overhead_s)
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

    def _ledger(self) -> float:
        try:
            return float(json.loads((self.out / "spend.json").read_text("utf-8"))["spent_usd"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0

    def _save_ledger(self) -> None:
        write_text_atomic(self.out / "spend.json", json.dumps(
            {"spent_usd": self.guard.spent(), "usd_per_hour": self.guard.usd_per_hour,
             "updated": datetime.now(timezone.utc).isoformat()}, indent=2))

    def key(self, spec: RunSpec) -> str:
        blob = json.dumps({"config": self.config_sha, "extra": self.extra,
                           "settings": spec.settings, "tokens": spec.tokens}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def steps_for(self, tokens: int) -> int:
        return tokens // self.cfg.train.batch_tokens

    def overrides_for(self, spec: RunSpec, ckpt_dir: Path | str) -> list[str]:
        """Every --override flag of a run: the global extras, the run's settings, then
        the bookkeeping (tokens, its own checkpoint dir, no milestones, an eval on the
        last step, warmup shortened only if it would not fit)."""
        steps = self.steps_for(spec.tokens)
        flat = dict(self.extra)
        flat.update({k: fmt_value(v) for k, v in spec.settings.items()})
        flat["train.total_tokens"] = str(spec.tokens)
        flat["train.ckpt_dir"] = str(ckpt_dir)
        flat["train.milestones"] = "[]"
        if steps >= 1:
            flat["train.eval_every"] = str(aligned_eval_every(steps, self.cfg.train.eval_every))
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
            baseline_losses=baseline_losses,
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
        if cached and (cached[1].status != "failed" or not self.retry_failed):
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
            if wall > 0:
                result.effective_tokens_per_s = spec.tokens / wall
            if result.status == "interrupted":
                raise KeyboardInterrupt
            if result.status == "completed":
                self.guard.observe(spec.tokens, wall)
            if result.status in ("completed", "diverged"):
                break
            if result.status == "crashed" and attempt < MAX_ATTEMPTS:
                self.echo(f"[ab] {spec.name} crashed ({result.reason}); retrying once")
                continue
            result.status = "failed"
            break
        return self._finish(spec, key, record, result, ctx)

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
        self._save_ledger()
        val = "-" if result.final_val_loss is None else f"{result.final_val_loss:.4f}"
        budget = ("" if self.guard.budget_usd is None
                  else f" of ${self.guard.budget_usd:.2f}")
        self.echo(f"[ab] {spec.name}: {result.status}, val {val}"
                  f"{' (' + result.reason + ')' if result.reason else ''}; "
                  f"spent ${self.guard.spent():.2f}{budget}")
        return result

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
        self.decisions.append((pair, d, {"simple": simple.name, "other": other.name,
                                         "kept": fmt_value(kept.settings[pair.key])}))
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
        interrupted."""
        try:
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
        self._save_ledger()
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
        lines = [
            f"# A/B runs: {self.cfg.name}", "",
            f"- Config: `{self.config_path.name}` (sha256 {self.config_sha[:12]})"
            + (f", overrides {', '.join(f'{k}={v}' for k, v in self.extra.items())}"
               if self.extra else ""),
            f"- Spend: ${g.spent():.2f} ({budget}, ${g.usd_per_hour:.2f}/h; earlier "
            f"invocations ${g.prior_usd:.2f})",
            f"- Pairs: {', '.join(self.pairs)}; seed re-runs "
            f"{'on' if self.seed_reruns else 'off'}",
        ]
        if self.stopped:
            lines.append(f"- **Stopped early ({self.stopped})**: runs after the refused one "
                         "were not started; missing arms keep the simpler option.")
        lines += ["", "## Runs", "",
                  "| # | run | config diff | tokens | status | final val loss | bpb | "
                  "tokens/s | spikes | decision |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
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
                "; ".join(self.labels.get(key, [])) or "-",
            ]) + " |")
        lines += ["", "## Learning rates", ""] + [f"- {n_}" for n_ in self.lr_notes]
        lines += ["", "## Decisions", "",
                  "| pair | simpler | other | kept | noise | reason |", "|---|---|---|---|---|---|"]
        for pair, d, names in self.decisions:
            run = names["simple"] if d.keep == "simple" else names["other"]
            lines.append(f"| {pair.number} {pair.name} | {names['simple']} | {names['other']} "
                         f"| {pair.key.split('.', 1)[1]}={names['kept']} ({run}) "
                         f"| {_f(d.noise)} | {d.reason} |")
        lines += ["", "Rules: the other option needs lower final val loss by more than the "
                  "seed noise; AttnRes <= 10% tokens/s cost; SiTU-GLU no more spikes "
                  f"(step loss > {SPIKE_FACTOR} x EMA, beta {SPIKE_EMA_BETA}); FP8 >= "
                  f"{FP8_MIN_SPEEDUP}x tokens/s, loss within noise, no more spikes. A missing "
                  "arm keeps the simpler option (AdamW, no AttnRes, SwiGLU, bf16).",
                  "", "## winners.toml", "", "```toml", self._winners_toml().rstrip(), "```", ""]
        return "\n".join(lines)


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
    """Runs one A/B arm as `python -m quipu.train` and watches its step lines:
    a non-finite loss, or a loss above DIVERGE_FACTOR x the baseline's at DIVERGE_AT
    of the run, kills it (diverged). --resume is added when the run's checkpoint
    exists (a retry after a crash); a stale run log without a checkpoint is removed
    so the retry starts clean. Output goes to ctx.log_path and to the console."""

    def __init__(self, cmd: list[str] | None = None, device: str = "auto",
                 cwd: str | Path = ROOT, echo: Callable[[str], None] | None = None) -> None:
        self.cmd = cmd or [sys.executable, "-m", "quipu.train"]
        self.device = device
        self.cwd = Path(cwd)
        self.echo = echo or (lambda s: print(s, flush=True))

    def __call__(self, spec: RunSpec, ctx: RunContext) -> RunResult:
        args = [*self.cmd, "--config", str(ctx.config), "--run-id", ctx.run_id,
                "--run-dir", str(ctx.run_dir), "--device", self.device]
        for o in ctx.overrides:
            args += ["--override", o]
        run_log = ctx.run_dir / f"{ctx.run_id}.json"
        if (ctx.ckpt_dir / "latest.pt").exists():
            args.append("--resume")
        elif run_log.exists():
            run_log.unlink()          # our own log of an attempt that never checkpointed
        ctx.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        check_at = max(1, math.ceil(DIVERGE_AT * ctx.steps))
        checked = False
        diverged = ""
        tok_s: list[float] = []
        t0 = time.monotonic()
        proc = subprocess.Popen(args, cwd=self.cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env)
        try:
            with open(ctx.log_path, "a", encoding="utf-8") as log:
                for line in proc.stdout:
                    log.write(line)
                    self.echo(line.rstrip("\n"))
                    m = STEP_LINE.match(line.strip())
                    if not m or diverged:
                        continue
                    step, loss = int(m.group(1)), float(m.group(3))
                    tok_s.append(float(m.group(4).replace(",", "")))
                    if not math.isfinite(loss):
                        diverged = f"non-finite loss at step {step}"
                    elif not checked and step >= check_at and ctx.baseline_losses:
                        checked = True
                        ref = _baseline_ref(ctx.baseline_losses, step)
                        if ref is not None and loss > DIVERGE_FACTOR * ref:
                            diverged = (f"loss {loss:.3f} > {DIVERGE_FACTOR:g} x baseline "
                                        f"{ref:.3f} at step {step}/{ctx.steps}")
                    if diverged:
                        self.echo(f"[ab] early stop {spec.name}: {diverged}")
                        proc.kill()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        wall = time.monotonic() - t0

        try:
            record = json.loads(run_log.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        losses = [s.get("train_loss") for s in record.get("steps", [])]
        losses = [float(x) for x in losses if x is not None]
        evals = record.get("evals") or []
        val = float(evals[-1]["val_loss"]) if evals else None
        steady = (statistics.median(tok_s[1:]) if len(tok_s) > 1
                  else (tok_s[0] if tok_s else None))
        eff = spec.tokens / wall if wall > 0 else None
        if diverged:
            status, reason = "diverged", diverged
        elif code == 0 and val is not None and math.isfinite(val):
            status, reason = "completed", ""
        elif code == 0:
            status, reason = "crashed", "finished without a final val loss"
        elif code == TRAIN_NONFINITE:
            status, reason = "diverged", "the trainer's non-finite stop (exit 3)"
        elif code in TRAIN_INTERRUPT_CODES:
            status, reason = "interrupted", f"exit {code}"
        elif code == TRAIN_USAGE:
            status, reason = "usage", "usage/config error (exit 2)"
        else:
            status, reason = "crashed", f"exit {code}"
        return RunResult(status=status, final_val_loss=val if status == "completed" else None,
                         bpb=None, tokens_per_s=steady or eff, effective_tokens_per_s=eff,
                         wall_s=wall, spikes=count_spikes(losses), train_losses=losses,
                         reason=reason)


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
          "train.milestones=[] and an eval_every that ends on the last step")
    print(f"  {'#':>2}  {'phase':<6} {'run':<24} {'tokens':>13}  overrides")
    for i, r in enumerate(runs, 1):
        flags = " ".join(f"{k}={fmt_value(v)}" if v != "" else k for k, v in r.settings.items())
        print(f"  {i:>2}  {r.phase:<6} {r.name:<24} {r.tokens:>13,}  {flags}")
    print(f"Total: {len(runs)} runs (worst case; seed re-runs shared with an earlier one are "
          f"cached, so as few as {fewest}), {total:,} tokens")
    print(f"Estimated time and cost at {tps:,.0f} tokens/s and ${o.guard.usd_per_hour:.2f}/h "
          f"(+{o.guard.overhead_s:.0f} s overhead per run): {hours:.2f} h, ${cost:.2f}")
    if o.guard.budget_usd is not None:
        verdict = "fits" if cost <= o.guard.budget_usd - o.guard.prior_usd else "does NOT fit"
        print(f"Budget ${o.guard.budget_usd:.2f} (spent ${o.guard.prior_usd:.2f}): {verdict}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="configs/quipu-moe-ab.toml")
    p.add_argument("--out", default="results/ab")
    p.add_argument("--dry-run", action="store_true", help="print the run list and cost; train nothing")
    p.add_argument("--with-fp8", action="store_true", help="add pair 4, bf16 vs fp8 (~$0.40 more)")
    p.add_argument("--budget-usd", type=float, help="spend cap for the A/B runs (required to train)")
    p.add_argument("--usd-per-hour", type=float, help="box rate (default: the config's usd_per_hour)")
    p.add_argument("--spent-usd", type=float, default=0.0, help="spend before this orchestrator")
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
            spent_usd=args.spent_usd, with_fp8=args.with_fp8, skip_sweeps=args.skip_sweeps,
            pairs=tuple(x.strip() for x in args.pairs.split(",") if x.strip()),
            seed_reruns=not args.no_seed_reruns, sweep_tokens=args.sweep_tokens,
            arm_tokens=args.arm_tokens, attnres_on=args.attnres_blocks, extra_overrides=extra,
            retry_failed=args.retry_failed, keep_checkpoints=args.keep_checkpoints,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.dry_run:
        _print_dry_run(o, args.tokens_per_second)
        return 0
    if args.budget_usd is None:
        print("error: pass --budget-usd (the A/B config's budget_usd 0 means this "
              "orchestrator enforces the cap)", file=sys.stderr)
        return EXIT_USAGE
    if o.guard.usd_per_hour <= 0 and args.budget_usd > 0:
        print("error: the budget needs --usd-per-hour > 0", file=sys.stderr)
        return EXIT_USAGE
    try:
        return o.run()
    except KeyboardInterrupt:
        print("[ab] interrupted; summary written", file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
