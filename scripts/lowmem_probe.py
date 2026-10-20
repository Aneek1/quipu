"""Probe the low-memory training options on a GPU (spec 2026-10-20, Testing 2-5).

  parity: train a reduced quipu-moe (the 1B's width and expert layout, --layers deep)
          twice from the same seed and data, once bf16 with no memory options and once
          with the T4 settings (fp16, bf16 momentum, block checkpointing, chunked
          loss), and compare. Needs a bf16-capable GPU (the laptop's RTX 5060).
  memory: one training step at the given settings; peak memory, step time, tokens/s.

  python scripts/lowmem_probe.py parity --shards data/shards-moe/train --steps 200
  python scripts/lowmem_probe.py memory --shards data/shards-moe/train --layers 4 --micro-batch 2 --t4
  torchrun --standalone --nproc_per_node=2 scripts/lowmem_probe.py memory --shards ... --layers 16 --t4

Writes a JSON summary to --out (default results/lowmem_probe.json) and prints it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu import dist  # noqa: E402
from quipu.config import load_config  # noqa: E402
from quipu.train import Trainer  # noqa: E402

T4 = {"precision": "fp16", "optimizer_state_dtype": "bf16",
      "activation_checkpoint": "block", "loss_chunk_tokens": 4096}
REFERENCE = {"precision": "bf16", "optimizer_state_dtype": "fp32",
             "activation_checkpoint": "none", "loss_chunk_tokens": 0}


def make_trainer(args, train_over: dict, run_dir: Path, run_id: str, steps: int) -> Trainer:
    cfg = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    model = dataclasses.replace(cfg.model, n_layer=args.layers, moe_dispatch="loop")
    # The trainer logs the expert-load statistics ("moe") only at eval_every steps, so
    # eval_every = steps puts one entry, covering the whole run, at the last step.
    # Evaluation itself runs in Trainer.run, which the probe never calls.
    train = dataclasses.replace(
        cfg.train, micro_batch=args.micro_batch,
        batch_tokens=args.micro_batch * model.context * args.accum * dist.world_size(),
        compile=False, budget_usd=0.0, usd_per_hour=0.0, eval_every=steps,
        ckpt_every=10**9, ckpt_dir=str(run_dir / "ckpt"), **train_over)
    return Trainer(model_cfg=model, train_cfg=train, shard_dir=args.shards,
                   device=f"cuda:{dist.local_rank()}" if dist.world_size() > 1 else "cuda",
                   run_dir=run_dir, run_id=run_id)


def run(args, over: dict, run_dir: Path, run_id: str, steps: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t = make_trainer(args, over, run_dir, run_id, steps)
    losses, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        losses.append(t.train_step())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tokens = t.train_cfg.batch_tokens
    step_s = sorted(times)[len(times) // 2]
    moe = getattr(t.log, "record", {}).get("moe", [])    # main rank only
    out = {"settings": over, "layers": args.layers, "micro_batch": args.micro_batch,
           "losses": losses, "nonfinite": sum(not math.isfinite(x) for x in losses),
           "overflow_steps": getattr(t, "overflow_steps", 0), "steps_taken": t.step,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "median_step_s": step_s, "tokens_per_s": tokens / step_s,
           "moe": moe[-1] if moe else None}
    del t
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["parity", "memory"])
    p.add_argument("--shards", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4, help="micro-batches per step per rank")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--t4", action="store_true", help="memory mode: use the T4 settings")
    p.add_argument("--out", default=str(ROOT / "results" / "lowmem_probe.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 2
    dist.init_from_env(cuda=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.mode == "parity":
            ref = run(args, REFERENCE, tmp / "ref", "ref", args.steps)
            low = run(args, T4, tmp / "low", "low", args.steps)
            tail = lambda r: sum(r["losses"][-10:]) / 10  # noqa: E731
            summary = {"reference": ref, "t4": low,
                       "final_loss_rel_diff": abs(tail(low) - tail(ref)) / tail(ref),
                       "pass": abs(tail(low) - tail(ref)) / tail(ref) < 0.01
                       and low["nonfinite"] == low["overflow_steps"]}
        else:
            summary = run(args, T4 if args.t4 else REFERENCE, tmp / "mem", "mem",
                          max(3, min(args.steps, 5)))
    if dist.is_main():
        text = json.dumps(summary, indent=2)
        print(text)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    dist.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
