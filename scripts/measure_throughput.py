"""Time real training steps and project the full run.

Task 1 measured a dense matmul. This measures the actual loop, including the
optimiser, the accumulation and the data, which is the number that matters. It
also times one eval and one checkpoint save, and folds them into the projection at
the config's cadence.

This laptop never raises CUDA OOM: the Windows (WDDM) driver spills VRAM into
shared system RAM and the step silently becomes tens to hundreds of times slower.
So the fit test is by memory, not by exception: after the warm-up step, if the
peak reserved memory exceeds --vram-budget-gib the run stops there and says so.
A timed step more than 3x the median is also reported as a likely spill.

Everything (run log, checkpoint) goes into a fresh temporary directory, so repeated
runs never collide on a run id and never leave files in the repo.

Run: uv run python scripts/measure_throughput.py --micro-batch 4
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

from quipu.config import load_config
from quipu.eval import estimate_loss
from quipu.train import Trainer

GIB = 2**30
SPILL_FACTOR = 3.0


def _sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--vram-budget-gib", type=float, default=7.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available; this measures the GPU loop only")

    with tempfile.TemporaryDirectory(prefix="quipu-throughput-") as tmp:
        tmp_path = Path(tmp)
        cfg = load_config(
            args.config,
            overrides={"train": {
                "micro_batch": args.micro_batch,
                "ckpt_dir": str(tmp_path / "checkpoints"),
            }},
        )
        tc = cfg.train
        shard_root = Path(cfg.data.shard_dir)
        trainer = Trainer(
            model_cfg=cfg.model, train_cfg=tc,
            shard_dir=shard_root / "train", val_dir=shard_root / "val",
            device="cuda", run_dir=tmp_path / "runs", run_id=f"mb{args.micro_batch}",
        )
        print(f"micro_batch={tc.micro_batch}  grad_accum={tc.grad_accum}  "
              f"batch_tokens={tc.batch_tokens:,}  budget={args.vram_budget_gib:.2f} GiB")

        torch.cuda.reset_peak_memory_stats()
        # Warm up: the first step includes allocator growth and kernel selection.
        for i in range(args.warmup):
            t0 = time.perf_counter()
            trainer.train_step()
            _sync()
            print(f"  warm-up step {i + 1}: {time.perf_counter() - t0:.2f} s")

        reserved = torch.cuda.max_memory_reserved() / GIB
        allocated = torch.cuda.max_memory_allocated() / GIB
        print(f"peak after warm-up: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")
        if reserved > args.vram_budget_gib:
            print(f"micro_batch {tc.micro_batch}: {reserved:.2f} GiB reserved exceeds VRAM "
                  f"budget of {args.vram_budget_gib:.2f} GiB — would spill to shared memory")
            sys.exit(2)

        times: list[float] = []
        for i in range(args.steps):
            t0 = time.perf_counter()
            trainer.train_step()
            _sync()
            times.append(time.perf_counter() - t0)
            print(f"  step {i + 1}: {times[-1]:.2f} s")

        median = statistics.median(times)
        slow = [t for t in times if t > SPILL_FACTOR * median]
        if slow:
            print(f"WARNING: {len(slow)} step(s) over {SPILL_FACTOR:.0f}x the median "
                  f"({', '.join(f'{t:.1f}s' for t in slow)}) — likely spilling to shared memory")

        # One eval and one checkpoint, timed the way run() would call them.
        t0 = time.perf_counter()
        estimate_loss(trainer.model, trainer.val_stream, tc.eval_batches, "cuda")
        _sync()
        eval_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        trainer.save_checkpoint()
        ckpt_s = time.perf_counter() - t0

        per_step = sum(times) / len(times)
        tps = tc.batch_tokens / per_step
        allocated = torch.cuda.max_memory_allocated() / GIB
        reserved = torch.cuda.max_memory_reserved() / GIB
        train_h = tc.steps * per_step / 3600
        eval_h = (tc.steps // tc.eval_every) * eval_s / 3600
        ckpt_h = (tc.steps // tc.ckpt_every + 1) * ckpt_s / 3600
        total_h = train_h + eval_h + ckpt_h

        print(f"\n{per_step:.2f} s/step (median {median:.2f})   {tps:,.0f} tokens/s")
        print(f"peak VRAM: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved "
              f"(card {torch.cuda.get_device_properties(0).total_memory / GIB:.2f} GiB)")
        print(f"one eval ({tc.eval_batches} batches): {eval_s:.2f} s, every {tc.eval_every} steps")
        print(f"one checkpoint save: {ckpt_s:.2f} s, every {tc.ckpt_every} steps")
        if reserved > args.vram_budget_gib:
            print(f"WARNING: peak reserved {reserved:.2f} GiB after eval/checkpoint exceeds the "
                  f"{args.vram_budget_gib:.2f} GiB budget — would spill to shared memory")
        print(f"projected full run ({tc.steps:,} steps): {total_h:.1f} hours "
              f"(train {train_h:.2f} + eval {eval_h:.2f} + checkpoints {ckpt_h:.2f})")
        del trainer


if __name__ == "__main__":
    main()
