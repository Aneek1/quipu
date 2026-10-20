"""Probe the low-memory training options on a GPU (spec 2026-10-20, Testing 2-5).

  parity: train a reduced quipu-moe (the 1B's width and expert layout, --layers deep)
          twice from the same seed and data, once bf16 with no memory options and once
          with the T4 settings (fp16, bf16 momentum, block checkpointing, chunked
          loss), and compare. Needs a bf16-capable GPU (the laptop's RTX 5060).
          Both runs train until they have taken --steps optimizer steps (an fp16
          overflow skips a step, so the fp16 run may need more calls; capped at 2x).
          pass = tail loss within 1% (over the same step indices), no non-finite
          loss, and the fp16 run took at least 90% of --steps.
  memory: train until at least 2 optimizer steps were taken (so the optimizer state
          exists); peak allocated and reserved memory, step time, tokens/s.

  --force-fp16-ns runs Muon's Newton-Schulz in fp16 (what a T4 does) on a newer GPU.

  python scripts/lowmem_probe.py parity --shards data/shards-moe/train --steps 200 --force-fp16-ns
  python scripts/lowmem_probe.py memory --shards data/shards-moe/train --layers 4 --micro-batch 2 --t4
  torchrun --standalone --nproc_per_node=2 scripts/lowmem_probe.py memory --shards ... --layers 16 --t4

Writes a JSON summary to --out (default results/lowmem_probe.json) and prints it.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
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
from quipu.optim import muon as muon_module  # noqa: E402
from quipu.train import Trainer  # noqa: E402

T4 = {"precision": "fp16", "optimizer_state_dtype": "bf16",
      "activation_checkpoint": "block", "loss_chunk_tokens": 2048}
REFERENCE = {"precision": "bf16", "optimizer_state_dtype": "fp32",
             "activation_checkpoint": "none", "loss_chunk_tokens": 0}
TAIL = 10                # parity compares the mean loss of the last TAIL taken steps
MIN_STEPS_TAKEN = 0.9    # parity: the fp16 run must take this share of --steps


def make_trainer(args, train_over: dict, run_dir: Path, run_id: str) -> Trainer:
    cfg = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    model = dataclasses.replace(cfg.model, n_layer=args.layers, moe_dispatch="loop")
    # eval_every = 1: the trainer logs the expert-load statistics ("moe") at every
    # taken step, so a last entry exists however many calls were skipped. Evaluation
    # itself runs in Trainer.run, which the probe never calls.
    train = dataclasses.replace(
        cfg.train, micro_batch=args.micro_batch,
        batch_tokens=args.micro_batch * model.context * args.accum * dist.world_size(),
        compile=False, budget_usd=0.0, usd_per_hour=0.0, eval_every=1,
        ckpt_every=10**9, ckpt_dir=str(run_dir / "ckpt"), **train_over)
    return Trainer(model_cfg=model, train_cfg=train, shard_dir=args.shards,
                   device=f"cuda:{dist.local_rank()}" if dist.world_size() > 1 else "cuda",
                   run_dir=run_dir, run_id=run_id)


def run(args, over: dict, run_dir: Path, run_id: str, steps: int, max_calls: int) -> dict:
    """Call train_step until `steps` optimizer steps were taken or `max_calls` calls
    were made. t.step is the same on every rank, so all ranks stop together."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = make_trainer(args, over, run_dir, run_id)
    all_losses, taken_losses, times, calls = [], [], [], 0
    while t.step < steps and calls < max_calls:
        before = t.step
        t0 = time.perf_counter()
        loss = t.train_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        calls += 1
        all_losses.append(loss)
        if t.step > before:
            taken_losses.append(loss)
    tokens = t.train_cfg.batch_tokens
    step_s = sorted(times)[len(times) // 2]
    moe = getattr(t.log, "record", {}).get("moe", [])    # main rank only
    out = {"settings": over, "layers": args.layers, "micro_batch": args.micro_batch,
           "force_fp16_ns": args.force_fp16_ns,
           "steps_requested": steps, "calls": calls, "steps_taken": t.step,
           "losses": taken_losses, "all_losses": all_losses,
           "nonfinite": sum(not math.isfinite(x) for x in all_losses),
           "overflow_steps": t.overflow_steps, "loss_scale": t.scaler.scale,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "median_step_s": step_s, "tokens_per_s": tokens / step_s,
           "moe": moe[-1] if moe else None}
    del t
    gc.collect()
    torch.cuda.empty_cache()
    return out


def tail_mean(losses: list[float], n: int) -> float:
    """Mean of the TAIL losses ending at taken step n, so both runs are compared over
    the same step indices."""
    window = losses[max(0, n - TAIL):n]
    return sum(window) / len(window) if window else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["parity", "memory"])
    p.add_argument("--shards", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4, help="micro-batches per step per rank")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--t4", action="store_true", help="memory mode: use the T4 settings")
    p.add_argument("--force-fp16-ns", action="store_true",
                   help="run Muon's Newton-Schulz in fp16, as on a T4 (sm_75)")
    p.add_argument("--out", default=str(ROOT / "results" / "lowmem_probe.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 2
    if args.force_fp16_ns:
        # Probe only: newton_schulz looks cuda_work_dtype up in its module at call time.
        muon_module.cuda_work_dtype = lambda capability: torch.float16
    dist.init_from_env(cuda=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.mode == "parity":
            ref = run(args, REFERENCE, tmp / "ref", "ref", args.steps, 2 * args.steps)
            low = run(args, T4, tmp / "low", "low", args.steps, 2 * args.steps)
            n = min(ref["steps_taken"], low["steps_taken"])
            ref_tail, low_tail = tail_mean(ref["losses"], n), tail_mean(low["losses"], n)
            rel = abs(low_tail - ref_tail) / ref_tail if n else float("nan")
            summary = {"reference": ref, "t4": low, "compared_steps": n,
                       "final_loss_rel_diff": rel,
                       "overflow_steps": low["overflow_steps"],
                       "pass": n > 0 and rel < 0.01
                       and ref["nonfinite"] == 0 and low["nonfinite"] == 0
                       and low["steps_taken"] >= MIN_STEPS_TAKEN * args.steps}
        else:
            target = max(2, min(args.steps, 5))
            summary = run(args, T4 if args.t4 else REFERENCE, tmp / "mem", "mem",
                          target, 3 * target)
            summary["valid"] = summary["steps_taken"] > 0
    if dist.is_main():
        text = json.dumps(summary, indent=2)
        print(text)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    dist.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
