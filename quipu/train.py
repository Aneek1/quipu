"""The training loop, gradient accumulation, and a checkpoint that resumes exactly.

The real run is ~23 hours unattended, so everything that touches the disk during
training is written to survive the ordinary failures of a Windows laptop: checkpoints
are swapped in atomically (with retry on transient antivirus/indexer locks), a failed
log write is a warning rather than a crash, and old checkpoints are pruned so the
run cannot fill the disk.
"""
from __future__ import annotations

import dataclasses
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from quipu.config import Config, ModelConfig, TrainConfig, load_config
from quipu.eval import estimate_loss
from quipu.fsio import replace_with_retry
from quipu.loader import TokenStream
from quipu.model import Quipu
from quipu.runlog import RunLog

LATEST = "latest.pt"
PRINT_EVERY = 10
MAX_NONFINITE_STREAK = 3
_CKPT_NAME = re.compile(r"^step_(\d+)\.pt$")


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
        self.ckpt_dir = Path(train_cfg.ckpt_dir)

        self.model = Quipu(model_cfg).to(device)
        self.stream = TokenStream(shard_dir, train_cfg.micro_batch, model_cfg.context)
        self.val_stream = (
            TokenStream(val_dir, train_cfg.micro_batch, model_cfg.context) if val_dir else None
        )

        # Weight decay on matrices only (p.dim() >= 2). RMSNorm gains are excluded:
        # decaying them shrinks the residual scale rather than regularising anything.
        # The tied embedding IS decayed, deliberately: it is also the output
        # projection (lm_head), and decaying it matches GPT-2/nanoGPT for tied
        # embeddings.
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=train_cfg.lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )
        # dataclasses.asdict copies; vars() would hand out the frozen config's live
        # __dict__. Not wrapped in _safe_log: a bad log path at startup should fail
        # loudly, before hours of compute are spent.
        config = {
            "model": dataclasses.asdict(model_cfg),
            "train": dataclasses.asdict(train_cfg),
            "derived": {"steps": train_cfg.steps, "grad_accum": train_cfg.grad_accum},
        }
        self.log = RunLog(run_dir, run_id, config, resume=resume)

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
                file=sys.stderr,
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

    def train_step(self) -> float:
        cfg = self.train_cfg
        lr = self.lr_at(self.step)
        for group in self.opt.param_groups:
            group["lr"] = lr

        t0 = time.perf_counter()
        # If anything escapes before opt.step() (Ctrl+C mid-step, most likely), the
        # batches read so far had their gradients thrown away; rewind so the
        # interrupt checkpoint does not skip them.
        start = self.stream.state_dict()
        try:
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            total = 0.0
            use_amp = self.device.startswith("cuda")
            for _ in range(cfg.grad_accum):
                x, y = self.stream.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                # Divide before backward so the accumulated gradient is the mean over
                # the whole batch, not the sum over micro-batches.
                (loss / cfg.grad_accum).backward()
                total += loss.item() / cfg.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if not (math.isfinite(total) and torch.isfinite(grad_norm)):
                # One NaN/inf batch taken as a step would poison the weights and
                # AdamW's moments, and every later checkpoint (retention would then
                # prune the clean ones). Skip it: step and lr stay where they are.
                self.opt.zero_grad(set_to_none=True)
                self.nonfinite_streak += 1
                print(
                    f"warning: non-finite loss/grad at step {self.step} "
                    f"(loss {total}, grad norm {float(grad_norm)}); step skipped "
                    f"({self.nonfinite_streak} in a row)",
                    file=sys.stderr, flush=True,
                )
                if self.nonfinite_streak >= MAX_NONFINITE_STREAK:
                    raise RuntimeError(
                        f"non-finite loss/grad for {MAX_NONFINITE_STREAK} consecutive "
                        f"steps at step {self.step}; stopping without checkpointing "
                        "poisoned weights"
                    )
                return total
            self.nonfinite_streak = 0
            self.opt.step()
        except BaseException:
            self.stream.load_state_dict(start)
            raise
        # Free the gradients now rather than at the next step: eval and checkpointing
        # run in between, and VRAM headroom at micro_batch 4 is ~480 MiB.
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        # wraps > 0 means the model is re-reading data; it must be visible in the log.
        self._safe_log(
            self.log.log_step,
            step=self.step, train_loss=total, lr=lr,
            tokens=self.step * cfg.batch_tokens,
            wraps=self.stream.wraps, grad_norm=float(grad_norm),
        )
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

    # ---- checkpoints ---------------------------------------------------------

    def save_checkpoint(self) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"step_{self.step:06d}.pt"
        _atomic_save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "stream": self.stream.state_dict(),
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
        found = []
        for p in self.ckpt_dir.iterdir():
            m = _CKPT_NAME.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(reverse=True)
        for _, p in found[self.train_cfg.ckpt_keep :]:
            if p.name == keep_name:
                continue
            try:
                p.unlink()
            except OSError as exc:
                # A locked old checkpoint is disk usage, not a reason to stop.
                print(f"warning: could not delete old checkpoint {p}: {exc}", file=sys.stderr)

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        if path is None:
            pointer = torch.load(self.ckpt_dir / LATEST, weights_only=False)
            path = self.ckpt_dir / pointer["file"]
        # Load to CPU: set_rng_state needs CPU ByteTensors, and model/optimizer
        # load_state_dict move their tensors to the right device themselves.
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.step = state["step"]
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.stream.load_state_dict(state["stream"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def resume_from_latest(self) -> None:
        """Load the latest checkpoint and drop log entries past it: those steps are
        about to be re-run and must not appear twice in the run log."""
        self.load_checkpoint()
        self.log.truncate_to(self.step)

    # ---- the loop ------------------------------------------------------------

    def run(self) -> None:
        cfg = self.train_cfg
        saved_at = None
        try:
            while self.step < cfg.steps:
                before = self.step
                loss = self.train_step()
                if self.step == before:
                    continue       # non-finite step skipped; nothing new to eval or save
                if self.step % cfg.eval_every == 0 and self.val_stream is not None:
                    val = estimate_loss(self.model, self.val_stream, cfg.eval_batches, self.device)
                    self._safe_log(self.log.log_eval, self.step, val)
                    print(f"step {self.step:>6}  train {loss:.4f}  val {val:.4f}", flush=True)
                if self.step % cfg.ckpt_every == 0:
                    self.save_checkpoint()
                    saved_at = self.step
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--run-id", default="quipu-114m-001")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        trainer = Trainer(
            model_cfg=cfg.model, train_cfg=cfg.train,
            shard_dir=Path(cfg.data.shard_dir) / "train",
            val_dir=Path(cfg.data.shard_dir) / "val",
            device=device, run_dir="results/runs", run_id=args.run_id,
            resume=args.resume,
        )
    except FileExistsError:
        sys.exit(
            f"run id {args.run_id!r} already has a log in results/runs; pass --resume "
            "to continue it or choose a new --run-id"
        )
    if args.resume:
        trainer.resume_from_latest()
        print(f"resumed at step {trainer.step}, stream position {trainer.stream.position}")
    trainer.run()


if __name__ == "__main__":
    main()
