"""The training loop, gradient accumulation, and a checkpoint that resumes exactly.

The real run is ~23 hours unattended, so everything that touches the disk during
training is written to survive the ordinary failures of a Windows laptop: checkpoints
are swapped in atomically (with retry on transient antivirus/indexer locks), a failed
log write is a warning rather than a crash, and old checkpoints are pruned so the
run cannot fill the disk.

Milestones are separate from resumable checkpoints: bf16 model weights only, in
<ckpt_dir>/milestones/, never pruned and never resumed from. They exist for the
post-run evaluation to see how the model changed over the run.

Exit codes of `python -m quipu.train` (the weekend launcher decides whether to retry
from these; see run_main): 0 completed, 1 any other crash, 2 a usage/config error,
3 the non-finite stop, 130 an interrupt.

Models and optimizers (quipu-moe, M6): the model comes from build_model (dense
Quipu or QuipuMoE) and the optimizers from build_optimizers (one AdamW, or Muon +
AdamW). Every optimizer is stepped, and the LR schedule scales every group by the
same factor: group lr = group base_lr x lr_at(step) / train.lr. Checkpoints hold
every optimizer's state under "optimizers"; the pre-M6 format (one AdamW under
"optimizer") still loads.

MoE models, per optimizer step: expert counts and padded-dispatch drops are summed
over all the step's micro-batches, and the router scores of every micro-batch are
stashed (capped at BALANCE_SAMPLE_TOKENS per step) so the Quantile Balancing update
after the optimizer step sees the whole step. Every eval_every steps the per-layer
expert load (min / max as a fraction of target, coefficient of variation, dead
experts) and drop rate go to the run log under "moe". An expert outside
[MOE_HEALTH_LOW, MOE_HEALTH_HIGH] x target load for more than MOE_HEALTH_WINDOW
consecutive steps raises a health alert: a stderr line and an entry in the run
log's "moe_health" list (the flag). Evaluation always runs loop dispatch. The
interval's summed counts and the health streaks are checkpointed, so a resume
mid-interval logs the same "moe" entry the uninterrupted run would have.

train.compile wraps the training forward in torch.compile (backend COMPILE_BACKEND);
where that cannot work (no Triton on CUDA, no C++ compiler for CPU inductor on
Windows) it warns and trains eagerly. Because torch.compile is lazy, one trial step
on a synthetic batch runs at startup; a compile that fails there also falls back to
eager (see Trainer._compile). Evaluation always runs the eager module, and the
un-compiled module is what is checkpointed.

train.precision "fp8" (spec section 12) converts attention q/k/v/o and the
shared-expert linears to FP8 matmuls (quipu.fp8) right after the model is built,
before the optimizers and the compile trial. The state_dict is bf16's key for key,
so checkpoints and milestones are unchanged. FP8 on a device without FP8 tensor
cores (or on CPU) is a usage error, exit 2.
"""
from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from quipu.config import Config, ModelConfig, TrainConfig, load_config
from quipu.eval import estimate_loss
from quipu.fp8 import Fp8Unsupported, apply_precision
from quipu.fsio import replace_with_retry
from quipu.loader import TokenStream
from quipu.model_factory import build_model
from quipu.model_moe import QuipuMoE
from quipu.optim import Muon, apply_config_lrs, build_optimizers
from quipu.runlog import RunLog

LATEST = "latest.pt"
MILESTONE_DIR = "milestones"
PRINT_EVERY = 10
MAX_NONFINITE_STREAK = 3
_CKPT_NAME = re.compile(r"^step_(\d+)\.pt$")

# MoE health alert: an expert below LOW or above HIGH times its target load (k*T/n
# assignments per step) for more than WINDOW consecutive steps. Trainer attributes
# moe_health_low / _high / _window start from these (tests shrink the window).
MOE_HEALTH_LOW = 0.1
MOE_HEALTH_HIGH = 3.0
MOE_HEALTH_WINDOW = 500
# Router-score rows kept per layer per step for the balance update (spread evenly
# over the micro-batches). 65,536 x 64 experts x 4 bytes = 16 MiB a layer; the whole
# 524k-token step would be 128 MiB a layer, 2 GiB over 16 layers.
BALANCE_SAMPLE_TOKENS = 65_536
# torch.compile backend for train.compile. Tests use "eager" to run real dynamo
# without Triton or a C++ compiler.
COMPILE_BACKEND = "inductor"

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_USAGE = 2
EXIT_NONFINITE = 3
EXIT_INTERRUPTED = 130

# SetThreadExecutionState flags (winbase.h).
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class NonFiniteStop(RuntimeError):
    """The non-finite guard gave up. Retrying from the last checkpoint would replay
    the same data into the same weights and fail the same way, so it has its own
    exit code (3) that the launcher does not retry."""


class UsageError(Exception):
    """A mistake in how the trainer was invoked (config, run id, --resume). Retrying
    cannot fix it; it exits 2 with the message and no traceback."""


class RunLogUnreadable(ValueError):
    """--resume found no readable run log (RunLog raises ValueError for it)."""


def _set_thread_execution_state(flags: int) -> None:
    """Ask Windows not to sleep while training (the request a media player makes;
    no system setting changes). A no-op elsewhere, and never fatal: a failed
    keep-awake request is not a reason to stop a run."""
    if sys.platform != "win32":
        return
    try:
        previous = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (AttributeError, OSError) as exc:
        print(f"warning: keep-awake request failed: {exc}", file=sys.stderr, flush=True)
        return
    if previous == 0:          # the documented failure return
        print(f"warning: SetThreadExecutionState({flags:#x}) failed; the laptop may sleep",
              file=sys.stderr, flush=True)


def _last_logged_step(record: dict[str, Any]) -> int:
    return max((s["step"] for s in record.get("steps", [])), default=0)


def _bf16_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """CPU bf16 copy of the model's state_dict. Tensors that share storage (the tied
    embedding / lm_head) stay shared, so torch.save writes them once."""
    out: dict[str, torch.Tensor] = {}
    seen: dict[tuple[int, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
    for k, v in model.state_dict().items():
        key = (v.data_ptr(), v.dtype, tuple(v.shape))
        if key not in seen:
            v = v.detach()
            seen[key] = v.cpu().to(torch.bfloat16) if v.is_floating_point() else v.cpu()
        out[k] = seen[key]
    return out


def _compile_unavailable(device: str) -> str | None:
    """Why torch.compile cannot work here with COMPILE_BACKEND, or None if it can.
    torch.compile itself is lazy (failures surface at the first forward), so the
    known blockers are checked up front."""
    try:
        import torch._dynamo
        if not torch._dynamo.is_dynamo_supported():
            return f"dynamo does not support Python {sys.version_info.major}.{sys.version_info.minor}"
    except Exception as exc:          # a broken install is a reason, not a crash
        return f"torch._dynamo unusable: {exc}"
    if COMPILE_BACKEND != "inductor":
        return None
    if str(device).startswith("cuda"):
        if importlib.util.find_spec("triton") is None:
            return "Triton is not installed (inductor needs it on CUDA)"
    elif sys.platform == "win32" and shutil.which("cl") is None:
        return "no MSVC cl.exe on PATH (inductor needs a C++ compiler on CPU)"
    return None


def _load_optimizer_state(opt: torch.optim.Optimizer, state: dict[str, Any]) -> None:
    """load_state_dict, keeping group keys the checkpoint lacks. A pre-M6 checkpoint's
    AdamW groups have no base_lr or decay tag; load_state_dict replaces each group
    wholesale with the saved one, so the built values are put back for those keys."""
    built = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    opt.load_state_dict(state)
    for group, extra in zip(opt.param_groups, built):
        for key, value in extra.items():
            group.setdefault(key, value)


def _atomic_save(obj: Any, path: Path) -> None:
    """torch.save to a temp name beside `path`, then swap it in. A kill mid-write
    leaves at worst a stray .tmp, never a truncated checkpoint at the final path."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        shard_dir: str | Path,
        device: str,
        run_dir: str | Path,
        run_id: str,
        val_dir: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        torch.manual_seed(train_cfg.seed)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.step = 0
        self.log_failures = 0
        self.nonfinite_streak = 0
        self.skipped_steps = 0      # cumulative non-finite skips, checkpointed
        self.ckpt_dir = Path(train_cfg.ckpt_dir)

        # self.model is always the plain module: checkpoints, milestones, the
        # balancer and set_dispatch go through it. self.forward_model is what runs
        # the forward (the torch.compile wrapper when compile is on, else the same).
        # train.precision "fp8" swaps some linears for FP8 ones (quipu.fp8) here:
        # before the optimizers are built and before compile, so both see the FP8
        # model. Parameter names and objects are unchanged, so checkpoints load
        # into either precision.
        self.model = apply_precision(build_model(model_cfg).to(device),
                                     train_cfg.precision, device)
        self.forward_model: nn.Module = self.model
        self.compiled = False
        self.is_moe = isinstance(self.model, QuipuMoE)
        self.stream = TokenStream(shard_dir, train_cfg.micro_batch, model_cfg.context)
        self.val_stream = (
            TokenStream(val_dir, train_cfg.micro_batch, model_cfg.context) if val_dir else None
        )

        # optimizer "adamw" is quipu-114m's grouping exactly: weight decay on
        # matrices only (the tied embedding included, as GPT-2/nanoGPT), none on
        # RMSNorm gains. "muon" adds Muon for the hidden weight matrices; see
        # quipu.optim.groups. self.opt is the AdamW, which is always built and
        # always last (under "adamw" it is the only optimizer).
        self.optimizers = build_optimizers(self.model, train_cfg)
        self.opt = self.optimizers[-1]

        self.moe_health_low = MOE_HEALTH_LOW
        self.moe_health_high = MOE_HEALTH_HIGH
        self.moe_health_window = MOE_HEALTH_WINDOW
        self.last_step_counts: torch.Tensor | None = None     # [n_layer, n_experts], CPU
        self.last_step_dropped: torch.Tensor | None = None    # [n_layer], CPU
        if self.is_moe:
            shape = (model_cfg.n_layer, model_cfg.n_experts)
            self._health_streak = torch.zeros(shape, dtype=torch.long)
            self._interval_counts = torch.zeros(shape, dtype=torch.long)
            self._interval_dropped = torch.zeros(model_cfg.n_layer, dtype=torch.long)
            self.balance_rows = max(1, BALANCE_SAMPLE_TOKENS // train_cfg.grad_accum)
        # dataclasses.asdict copies; vars() would hand out the frozen config's live
        # __dict__. Not wrapped in _safe_log: a bad log path at startup should fail
        # loudly, before hours of compute are spent.
        config = {
            "model": dataclasses.asdict(model_cfg),
            "train": dataclasses.asdict(train_cfg),
            "derived": {"steps": train_cfg.steps, "grad_accum": train_cfg.grad_accum},
        }
        if resume and not (self.ckpt_dir / LATEST).exists():
            # Checked before RunLog, whose resume path rewrites the log on open: a
            # refused resume must leave the log exactly as it was. An unreadable
            # log is left for RunLog to report.
            try:
                record = json.loads(
                    (Path(run_dir) / f"{run_id}.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                record = None
            if isinstance(record, dict):
                self._check_restart_allowed(_last_logged_step(record))
        try:
            self.log = RunLog(run_dir, run_id, config, resume=resume)
        except ValueError as exc:
            raise RunLogUnreadable(str(exc)) from exc
        if train_cfg.compile:
            self._compile()

    def _compile(self) -> None:
        """Wrap the forward in torch.compile, or warn and stay eager when it cannot
        work here. The outcome is recorded in the run log under "compile".

        torch.compile is lazy: the real compile happens at the first forward, where a
        failure would crash the run. So one trial step runs here on a synthetic batch
        (_trial_compiled_step, which leaves no trace in RNG, gradients or the
        balancer). If it raises, dynamo is reset and training runs eagerly ("fell
        back: ..." in the run log). If it succeeds, dynamo's suppress_errors is set
        so a later recompile failure (a new shape, say) runs that frame eagerly
        instead of crashing a long run."""
        reason = _compile_unavailable(self.device)
        if reason is None:
            try:
                self.forward_model = torch.compile(self.model, backend=COMPILE_BACKEND)
                self.compiled = True
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        if reason is not None:
            print(f"warning: torch.compile unavailable ({reason}); training without it",
                  file=sys.stderr, flush=True)
            self._safe_log(self.log.note, "compile", f"skipped: {reason}")
            return
        import torch._dynamo as dynamo        # `as`: a bare import would shadow torch here
        try:
            self._trial_compiled_step()
        except Exception as exc:
            dynamo.reset()
            self.forward_model = self.model
            self.compiled = False
            first = (str(exc).strip().splitlines() or [""])[0][:500]
            failure = f"{type(exc).__name__}: {first}"
            print(f"warning: torch.compile failed at the first forward ({failure}); "
                  "falling back to eager training", file=sys.stderr, flush=True)
            self._safe_log(self.log.note, "compile", f"fell back: {failure}")
            return
        dynamo.config.suppress_errors = True
        self._safe_log(self.log.note, "compile", f"on ({COMPILE_BACKEND})")

    def _trial_compiled_step(self) -> None:
        """One forward + backward through forward_model on a synthetic batch (random
        tokens on the training device, same autocast as train_step), to force the
        lazy compile now. The data stream is not touched, and everything the step
        changed is put back: RNG (CPU and CUDA), gradients, router scores and
        last_stats. Raises whatever the compiled forward or backward raises."""
        cfg, mcfg = self.train_cfg, self.model_cfg
        use_amp = self.device.startswith("cuda")
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if use_amp else None
        was_training = self.model.training
        try:
            self.model.train()
            x = torch.randint(mcfg.vocab_size, (cfg.micro_batch, mcfg.context),
                              device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = self.forward_model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.reshape(-1))
            loss.backward()
        finally:
            self._zero_grad()
            self.model.zero_grad(set_to_none=True)
            if self.is_moe:
                self.model.clear_balance_scores()
                self.model.last_stats = [None] * mcfg.n_layer
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.model.train(was_training)

    # ---- logging -------------------------------------------------------------

    def _safe_log(self, fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        """A failed log write costs one line of history, not the run."""
        try:
            fn(*args, **kwargs)
        except OSError as exc:
            self.log_failures += 1
            print(
                f"warning: run log write failed at step {self.step} "
                f"({self.log_failures} so far): {exc}",
                file=sys.stderr, flush=True,
            )

    # ---- optimisation --------------------------------------------------------

    def lr_at(self, step: int) -> float:
        cfg = self.train_cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if step >= cfg.steps:
            return cfg.lr_min
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine

    def _zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _set_lrs(self, lr: float) -> float | None:
        """Every group of every optimizer to base_lr x the schedule factor. Returns
        the Muon lr (None without Muon), for the log. A group whose base_lr is
        train.lr gets `lr` itself: cfg.lr x (lr / cfg.lr) can differ from lr in the
        last bit, and quipu-114m's AdamW must stay bit-identical to pre-M6."""
        base = self.train_cfg.lr
        factor = lr / base
        muon_lr = None
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = lr if group["base_lr"] == base else group["base_lr"] * factor
                if muon_lr is None and isinstance(opt, Muon):
                    muon_lr = group["lr"]
        return muon_lr

    def train_step(self) -> float:
        cfg = self.train_cfg
        lr = self.lr_at(self.step)
        muon_lr = self._set_lrs(lr)

        t0 = time.perf_counter()
        # If anything escapes before opt.step() (Ctrl+C mid-step, most likely), the
        # batches read so far had their gradients thrown away; rewind so the
        # interrupt checkpoint does not skip them.
        start = self.stream.state_dict()
        stepped = False
        step_counts = step_dropped = None
        try:
            self.model.train()
            self._zero_grad()
            if self.is_moe:
                # Scores left over from an abandoned step or an eval forward must
                # not reach this step's balance update.
                self.model.clear_balance_scores()
                n_layer, n_experts = self.model_cfg.n_layer, self.model_cfg.n_experts
                step_counts = torch.zeros(n_layer, n_experts, dtype=torch.long, device=self.device)
                step_dropped = torch.zeros(n_layer, dtype=torch.long, device=self.device)
            # The step's loss stays on the device (float64, so the sum is exactly the
            # old per-micro-batch loss.item() / grad_accum sum) and is read back once
            # per step: no host sync per micro-batch.
            total_t = torch.zeros((), dtype=torch.float64, device=self.device)
            use_amp = self.device.startswith("cuda")
            for _ in range(cfg.grad_accum):
                x, y = self.stream.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.forward_model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                # Divide before backward so the accumulated gradient is the mean over
                # the whole batch, not the sum over micro-batches.
                (loss / cfg.grad_accum).backward()
                total_t += loss.detach().double() / cfg.grad_accum
                if self.is_moe:
                    # last_stats and the stashed scores cover this micro-batch only;
                    # summed here, the step's balance update and load logging see
                    # every micro-batch.
                    stats = self.model.last_stats
                    step_counts += torch.stack([s.counts for s in stats])
                    step_dropped += torch.stack([s.dropped for s in stats])
                    self.model.accumulate_balance_scores(self.balance_rows)

            total = total_t.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if not (math.isfinite(total) and torch.isfinite(grad_norm)):
                # One NaN/inf batch taken as a step would poison the weights and
                # AdamW's moments, and every later checkpoint (retention would then
                # prune the clean ones). Skip it: step and lr stay where they are.
                self._zero_grad()
                if self.is_moe:
                    self.model.clear_balance_scores()   # nor the balancer
                self.nonfinite_streak += 1
                self.skipped_steps += 1
                print(
                    f"warning: non-finite loss/grad at step {self.step} "
                    f"(loss {total}, grad norm {float(grad_norm)}); step skipped "
                    f"({self.nonfinite_streak} in a row)",
                    file=sys.stderr, flush=True,
                )
                if self.nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise NonFiniteStop(
                        f"non-finite loss/grad for {MAX_NONFINITE_STREAK} consecutive "
                        f"steps at step {self.step}; stopping without checkpointing "
                        "poisoned weights"
                    )
                return total
            self.nonfinite_streak = 0
            stepped = True
            for opt in self.optimizers:
                opt.step()
            if self.is_moe:
                # After the weights move, from the scores the forwards captured:
                # no extra forward.
                self.model.update_balance()
        except BaseException:
            # Once opt.step() has started the weights may already reflect these
            # batches, so rewinding would train on them twice; only rewind before.
            if not stepped:
                self.stream.load_state_dict(start)
            raise
        # Free the gradients now rather than at the next step: eval and checkpointing
        # run in between, and VRAM headroom at micro_batch 4 is ~480 MiB.
        self._zero_grad()
        self.step += 1
        extra = {} if muon_lr is None else {"muon_lr": muon_lr}
        # wraps > 0 means the model is re-reading data; it must be visible in the log.
        self._safe_log(
            self.log.log_step,
            step=self.step, train_loss=total, lr=lr,
            tokens=self.step * cfg.batch_tokens,
            wraps=self.stream.wraps, grad_norm=float(grad_norm),
            skipped=self.skipped_steps, **extra,
        )
        if self.is_moe:
            self._record_moe_step(step_counts.cpu(), step_dropped.cpu())
        if self.step % PRINT_EVERY == 0:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            tok_s = cfg.batch_tokens / max(time.perf_counter() - t0, 1e-9)
            line = (
                f"step {self.step}/{cfg.steps}  loss {total:.4f}  lr {lr:.2e}  "
                f"grad {float(grad_norm):.3f}  {tok_s:,.0f} tok/s"
            )
            if self.stream.wraps:
                line += f"  wraps {self.stream.wraps}"
            print(line, flush=True)
        return total

    # ---- MoE expert load -----------------------------------------------------

    def _record_moe_step(self, counts: torch.Tensor, dropped: torch.Tensor) -> None:
        """One taken step's expert counts [n_layer, n_experts] and drops [n_layer],
        summed over its micro-batches: health tracking every step, load statistics
        every eval_every steps."""
        self.last_step_counts, self.last_step_dropped = counts, dropped
        self._interval_counts += counts
        self._interval_dropped += dropped
        self._track_moe_health(counts)
        if self.step % self.train_cfg.eval_every == 0:
            layers = self._load_stats(self._interval_counts, self._interval_dropped)
            self._safe_log(self.log.log_moe, self.step, layers)
            print(f"moe step {self.step}: " + "  ".join(
                f"L{l} load {s['load_min']:.2f}-{s['load_max']:.2f} cv {s['load_cv']:.2f} "
                f"dead {s['dead']} drop {s['drop_rate']:.1%}"
                for l, s in enumerate(layers)), flush=True)
            self._interval_counts.zero_()
            self._interval_dropped.zero_()

    @staticmethod
    def _load_stats(counts: torch.Tensor, dropped: torch.Tensor) -> list[dict[str, Any]]:
        """Per layer: min / max expert load as a fraction of the target (the mean,
        k*T/n assignments), coefficient of variation (population std / mean), dead
        experts (no tokens at all), and the fraction of assignments the padded
        dispatch dropped."""
        out = []
        for c, d in zip(counts.double(), dropped.tolist()):
            total = float(c.sum())
            mean = total / c.numel()
            safe = mean if mean > 0 else 1.0
            out.append({
                "load_min": float(c.min()) / safe,
                "load_max": float(c.max()) / safe,
                "load_cv": float(c.std(unbiased=False)) / safe,
                "dead": int((c == 0).sum()),
                "drop_rate": d / total if total > 0 else 0.0,
            })
        return out

    def _track_moe_health(self, counts: torch.Tensor) -> None:
        """Count consecutive steps each expert spends outside [low, high] x target
        load; alert once per episode, on the step the streak passes the window."""
        c = counts.double()
        target = (c.sum(-1, keepdim=True) / c.shape[-1]).clamp_min(1e-12)
        load = c / target
        bad = (load < self.moe_health_low) | (load > self.moe_health_high)
        self._health_streak = torch.where(bad, self._health_streak + 1, 0)
        for l, e in (self._health_streak == self.moe_health_window + 1).nonzero().tolist():
            alert = {"step": self.step, "layer": l, "expert": e,
                     "load": float(load[l, e]), "streak": int(self._health_streak[l, e])}
            print(
                f"warning: MoE health: layer {l} expert {e} at {alert['load']:.0%} of its "
                f"target load for {alert['streak']} consecutive steps (step {self.step}; "
                f"band {self.moe_health_low:.0%}-{self.moe_health_high:.0%})",
                file=sys.stderr, flush=True,
            )
            self._safe_log(self.log.add_alert, "moe_health", alert)

    # ---- checkpoints ---------------------------------------------------------

    def save_checkpoint(self) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"step_{self.step:06d}.pt"
        _atomic_save(
            {
                "step": self.step,
                # The plain module's state_dict (no "_orig_mod." prefixes under
                # compile). It includes every Quantile Balancing bias: a buffer.
                "model": self.model.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.optimizers],
                "optimizer_kinds": [type(opt).__name__ for opt in self.optimizers],
                "moe_health_streak": self._health_streak.clone() if self.is_moe else None,
                # The expert-load interval so far (since the last eval_every step), so
                # the first "moe" entry after a resume covers the whole interval.
                "moe_interval": ((self._interval_counts.clone(), self._interval_dropped.clone())
                                 if self.is_moe else None),
                "stream": self.stream.state_dict(),
                "skipped_steps": self.skipped_steps,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )
        # The pointer holds a bare file name, resolved against ckpt_dir on load, so
        # the checkpoint directory can be moved without breaking resume.
        _atomic_save({"file": path.name}, self.ckpt_dir / LATEST)
        self._prune_checkpoints(keep_name=path.name)
        return path

    def _prune_checkpoints(self, keep_name: str) -> None:
        """Keep the newest ckpt_keep step_*.pt files (~1.4 GB each); never delete the
        one latest.pt points to."""
        # iterdir() is not recursive and the name must match step_NNNNNN.pt, so the
        # milestones/ subdirectory (and everything in it) is never a candidate.
        found = []
        for p in self.ckpt_dir.iterdir():
            m = _CKPT_NAME.match(p.name)
            if m and p.is_file():
                found.append((int(m.group(1)), p))
        found.sort(reverse=True)
        for _, p in found[self.train_cfg.ckpt_keep :]:
            if p.name == keep_name:
                continue
            try:
                p.unlink()
            except OSError as exc:
                # A locked old checkpoint is disk usage, not a reason to stop.
                print(f"warning: could not delete old checkpoint {p}: {exc}",
                      file=sys.stderr, flush=True)

    def _check_restart_allowed(self, last_step: int) -> None:
        """With no latest.pt, restarting from 0 is safe only if no step_*.pt exists
        in ckpt_dir and the run log never got past the first checkpoint interval."""
        ckpts = (
            sorted(p.name for p in self.ckpt_dir.iterdir() if _CKPT_NAME.match(p.name))
            if self.ckpt_dir.is_dir() else []
        )
        every = self.train_cfg.ckpt_every
        if ckpts or last_step > every:
            found = f"found {', '.join(ckpts)} but " if ckpts else ""
            raise UsageError(
                f"cannot resume: {found}no {LATEST} in {self.ckpt_dir}, and the run "
                f"log reaches step {last_step} (ckpt_every {every}), so a checkpoint "
                "should exist. Refusing to restart from step 0 over real progress; "
                f"check ckpt_dir in the config or restore {LATEST}"
            )

    def milestone_path(self, step: int) -> Path:
        return self.ckpt_dir / MILESTONE_DIR / f"step_{step:06d}.pt"

    def save_milestone(self) -> Path | None:
        """bf16 weights at the current step, for post-run evaluation. A file that is
        already there is left alone: a resumed run re-reaching a milestone step must
        not rewrite what the first pass wrote (returns None then)."""
        path = self.milestone_path(self.step)
        if path.exists():
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(_bf16_state_dict(self.model), path)
        return path

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        if path is None:
            pointer = torch.load(self.ckpt_dir / LATEST, weights_only=False)
            path = self.ckpt_dir / pointer["file"]
        # Load to CPU: set_rng_state needs CPU ByteTensors, and model/optimizer
        # load_state_dict move their tensors to the right device themselves.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "optimizers" in state:
            saved_opts = state["optimizers"]
            kinds = list(state.get("optimizer_kinds") or [])
        else:                                   # pre-M6 (quipu-114m): one AdamW
            saved_opts = [state["optimizer"]]
            kinds = ["AdamW"]
        mine = [type(opt).__name__ for opt in self.optimizers]
        if kinds != mine or len(saved_opts) != len(self.optimizers):
            raise UsageError(
                f"checkpoint {path} holds optimizer state for {kinds}, but train.optimizer "
                f"{self.train_cfg.optimizer!r} builds {mine}; resume with the config the "
                "run was started with"
            )
        self.step = state["step"]
        self.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
        for opt, saved in zip(self.optimizers, saved_opts):
            _load_optimizer_state(opt, saved)
        # load_state_dict brought back the checkpoint's base_lr / weight_decay; the
        # config wins (a deliberate LR change on resume must take effect).
        apply_config_lrs(self.optimizers, self.train_cfg)
        if self.is_moe and state.get("moe_health_streak") is not None:
            self._health_streak = state["moe_health_streak"].clone()
        if self.is_moe and state.get("moe_interval") is not None:
            counts, dropped = state["moe_interval"]
            self._interval_counts = counts.clone()
            self._interval_dropped = dropped.clone()
        self.stream.load_state_dict(state["stream"])
        self.skipped_steps = int(state.get("skipped_steps", 0))   # absent before it existed
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def resume_from_latest(self) -> None:
        """Load the latest checkpoint and drop log entries past it: those steps are
        about to be re-run and must not appear twice in the run log.

        A run log with no checkpoint yet (the first attempt died before its first
        save) restarts from step 0 rather than failing: the launcher resumes any run
        id that has a log, and a crash would otherwise repeat on every retry. That is
        allowed only when nothing suggests real progress exists somewhere: no
        latest.pt, no step_*.pt in ckpt_dir, and a run log that never got past the
        first checkpoint interval. Anything else (a deleted pointer, a moved or
        renamed ckpt_dir) is a UsageError and the run log is left untouched."""
        if not (self.ckpt_dir / LATEST).exists():
            last = _last_logged_step(self.log.record)
            self._check_restart_allowed(last)
            print(
                f"no checkpoint in {self.ckpt_dir} yet (run log reaches step {last}); "
                "starting again from step 0",
                flush=True,
            )
            self.log.truncate_to(0)
            return
        self.load_checkpoint()
        self.log.truncate_to(self.step)

    # ---- the loop ------------------------------------------------------------

    def run(self) -> None:
        _set_thread_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        try:
            self._run()
        finally:
            _set_thread_execution_state(ES_CONTINUOUS)

    def _run(self) -> None:
        cfg = self.train_cfg
        milestones = set(cfg.milestones)
        saved_at = None
        # Resumed from checkpoint N whose milestone never got written (a Ctrl+C
        # during the milestone write or the eval before it saves checkpoint N and
        # exits): the weights just loaded are exactly the step-N weights.
        try:
            if self.step > 0 and (self.step in milestones or self.step == cfg.steps):
                self.save_milestone()      # no-op when the file exists
            while self.step < cfg.steps:
                before = self.step
                loss = self.train_step()
                if self.step == before:
                    continue       # non-finite step skipped; nothing new to eval or save
                if self.step % cfg.eval_every == 0 and self.val_stream is not None:
                    # estimate_loss runs MoE layers with loop dispatch (no drops) and
                    # restores the training dispatch afterwards. The eager module, not
                    # the compiled one: eval mode, no_grad and loop dispatch would each
                    # force a recompile (or a compile failure) mid-run.
                    val = estimate_loss(self.model, self.val_stream,
                                        cfg.eval_batches, self.device)
                    self._safe_log(self.log.log_eval, self.step, val)
                    print(f"step {self.step:>6}  train {loss:.4f}  val {val:.4f}", flush=True)
                # Milestone before checkpoint: a resume from checkpoint N starts after
                # step N, so milestone N must already be on disk by then.
                if self.step in milestones or self.step == cfg.steps:
                    self.save_milestone()
                if self.step % cfg.ckpt_every == 0:
                    self.save_checkpoint()
                    saved_at = self.step
            if self.step == cfg.steps:
                self.save_milestone()      # no-op if the loop already wrote it
            if saved_at != self.step:
                self.save_checkpoint()
            self._safe_log(self.log.finish, "completed")
        except KeyboardInterrupt:
            self.save_checkpoint()
            self._safe_log(self.log.finish, "interrupted")
            raise
        except Exception:
            # No checkpoint here: the likeliest cause is the non-finite guard, and
            # saving would write the very weights it refused to train on.
            self._safe_log(self.log.finish, "crashed")
            raise


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    # is_available() alone is not enough: with CUDA_VISIBLE_DEVICES="" it can report
    # True with no device, and the first .to("cuda") fails.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    """Train. Raises on failure; run_main turns the outcome into an exit code."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--run-id", default="quipu-114m-001")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto = cuda if usable, else cpu")
    args = parser.parse_args(argv)

    try:
        cfg: Config = load_config(args.config)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise UsageError(f"bad config {args.config!r}: {exc}") from exc
    device = _pick_device(args.device)
    try:
        trainer = Trainer(
            model_cfg=cfg.model, train_cfg=cfg.train,
            shard_dir=Path(cfg.data.shard_dir) / "train",
            val_dir=Path(cfg.data.shard_dir) / "val",
            device=device, run_dir="results/runs", run_id=args.run_id,
            resume=args.resume,
        )
    except FileExistsError as exc:
        raise UsageError(
            f"run id {args.run_id!r} already has a log in results/runs; pass --resume "
            "to continue it or choose a new --run-id"
        ) from exc
    except RunLogUnreadable as exc:
        raise UsageError(
            f"{exc}; drop --resume to start a new run, or check the run id"
        ) from exc
    except Fp8Unsupported as exc:
        raise UsageError(f"{exc}; set train.precision = \"bf16\" to train here") from exc
    if args.resume:
        trainer.resume_from_latest()
        print(
            f"resumed at step {trainer.step}, stream position {trainer.stream.position}",
            flush=True,
        )
    trainer.run()


def run_main(argv: list[str] | None = None) -> int:
    """main() with every outcome mapped to the exit code the launcher relies on:
    0 completed, 1 any other crash, 2 usage/config error, 3 non-finite stop,
    130 interrupt. Codes 2, 3 and 130 are not worth retrying; 1 may be."""
    try:
        main(argv)
    except NonFiniteStop as exc:
        print(f"stopped: {exc}", file=sys.stderr, flush=True)
        return EXIT_NONFINITE
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted; checkpoint saved if training had started", file=sys.stderr, flush=True)
        return EXIT_INTERRUPTED
    except SystemExit as exc:          # argparse: --help is 0, a bad argument is 2
        if exc.code is None or isinstance(exc.code, int):
            return exc.code or EXIT_OK
        print(exc.code, file=sys.stderr, flush=True)
        return EXIT_CRASH
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        return EXIT_CRASH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(run_main())
