"""Prove the full training path before leaving the real run unattended for a day.

Runs the REAL model on the REAL shards on CUDA, with the shipped config shortened
to five steps: train, eval, checkpoint, prune, finish; then a resume from the
latest checkpoint and one more step. Everything is written to a temporary
directory. Needs data/shards and a GPU, so it is a script, not part of pytest.

Prints PASS/FAIL per check (SKIPPED for checks that depend on a phase that
raised) and exits non-zero on any failure.

Run: uv run python scripts/preflight.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import torch

from quipu.config import load_config
from quipu.train import LATEST, Trainer

RUN_ID = "preflight"
_CKPT = re.compile(r"^step_\d+\.pt$")


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.skipped = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        self.failed += not ok
        return ok

    def crashed(self, phase: str, exc: BaseException, dependent: list[str]) -> None:
        """A phase raised: report it as one FAIL and its dependent checks as SKIPPED."""
        self(f"{phase} raised", False, f"{type(exc).__name__}: {exc}")
        for name in dependent:
            print(f"  SKIPPED  {name}", flush=True)
            self.skipped += 1


RUN_CHECKS = [
    "status completed", "reached the final step", "train losses finite",
    "at least one eval logged", "checkpoints on disk and <= ckpt_keep",
    "oldest checkpoint pruned", "latest.pt points to an existing file",
]
RESUME_CHECKS = [
    "resumed at the checkpointed step", "step continues from the checkpoint",
    "resumed loss finite", "log has a resumes entry", "no duplicate step numbers in the log",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    # 5 steps saves at 2, 4 and 5, so pruning to ckpt_keep=2 actually runs.
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    args = parser.parse_args()

    check = Checks()
    if not check("CUDA available", torch.cuda.is_available() and torch.cuda.device_count() > 0):
        sys.exit(1)

    started = time.perf_counter()
    # Windows can briefly lock a just-written checkpoint; a failed cleanup of the
    # temp dir must not turn a passing pre-flight into a traceback.
    with tempfile.TemporaryDirectory(prefix="quipu-preflight-", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        base = load_config(args.config)
        cfg = load_config(args.config, overrides={"train": {
            "total_tokens": args.steps * base.train.batch_tokens,
            "warmup_steps": 1,   # must be < steps; the schedule shape is not under test
            "eval_every": 2, "ckpt_every": 2, "ckpt_keep": 2,
            "ckpt_dir": str(tmp_path / "checkpoints"),
        }})
        tc = cfg.train
        assert tc.steps == args.steps, (tc.steps, args.steps)
        shard_root = Path(cfg.data.shard_dir)
        run_dir = tmp_path / "runs"
        ckpt_dir = Path(tc.ckpt_dir)
        print(f"real config, micro_batch {tc.micro_batch}, grad_accum {tc.grad_accum}, "
              f"{tc.steps} steps, eval/ckpt every 2, keep 2; temp dir {tmp_path}")

        def trainer(resume: bool) -> Trainer:
            return Trainer(
                model_cfg=cfg.model, train_cfg=tc,
                shard_dir=shard_root / "train", val_dir=shard_root / "val",
                device="cuda", run_dir=run_dir, run_id=RUN_ID, resume=resume,
            )

        def read_log() -> dict:
            return json.loads((run_dir / f"{RUN_ID}.json").read_text(encoding="utf-8"))

        # ---- phase 1: a complete (short) run ---------------------------------
        print("\nrun():")
        run_ok = False
        t = None
        try:
            t = trainer(resume=False)
            t.run()
            run_ok = True
        except Exception as exc:
            check.crashed("run()", exc, RUN_CHECKS)
        if run_ok:
            try:
                record = read_log()
                losses = [s["train_loss"] for s in record["steps"]]
                check("status completed", record["status"] == "completed", record["status"])
                check("reached the final step", t.step == tc.steps,
                      f"step {t.step} of {tc.steps}")
                check("train losses finite",
                      bool(losses) and all(math.isfinite(x) for x in losses),
                      ", ".join(f"{x:.4f}" for x in losses))
                evals = record["evals"]
                check("at least one eval logged",
                      len(evals) >= 1 and all(math.isfinite(e["val_loss"]) for e in evals),
                      ", ".join(f"step {e['step']} val {e['val_loss']:.4f}" for e in evals))
                ckpts = sorted(p.name for p in ckpt_dir.iterdir() if _CKPT.match(p.name))
                check("checkpoints on disk and <= ckpt_keep",
                      1 <= len(ckpts) <= tc.ckpt_keep, f"keep {tc.ckpt_keep}: {', '.join(ckpts)}")
                check("oldest checkpoint pruned", "step_000002.pt" not in ckpts)
                pointer = torch.load(ckpt_dir / LATEST, weights_only=False)
                run_ok = check("latest.pt points to an existing file",
                               (ckpt_dir / pointer["file"]).is_file(), pointer["file"])
            except Exception as exc:
                # A missing log or checkpoint dir is a failed check, not a traceback.
                check("inspecting the run's log and checkpoints", False,
                      f"{type(exc).__name__}: {exc}")
                run_ok = False
        saved_step = t.step if t is not None else None
        del t
        torch.cuda.empty_cache()

        # ---- phase 2: resume and take one more step --------------------------
        print("\nresume:")
        if not run_ok:
            for name in RESUME_CHECKS:
                print(f"  SKIPPED  {name}  (run() did not leave a usable checkpoint)")
                check.skipped += 1
        else:
            r = None
            try:
                r = trainer(resume=True)
                r.resume_from_latest()
            except Exception as exc:
                check.crashed("Trainer(resume=True) + resume_from_latest()", exc, RESUME_CHECKS)
            else:
                check("resumed at the checkpointed step", r.step == saved_step, f"step {r.step}")
                try:
                    loss = r.train_step()
                except Exception as exc:
                    check.crashed("train_step() after resume", exc, RESUME_CHECKS[1:])
                else:
                    check("step continues from the checkpoint", r.step == saved_step + 1,
                          f"step {r.step}")
                    check("resumed loss finite", math.isfinite(loss), f"{loss:.4f}")
                    record = read_log()
                    check("log has a resumes entry", len(record.get("resumes", [])) >= 1,
                          json.dumps(record.get("resumes")))
                    step_nums = [s["step"] for s in record["steps"]]
                    check("no duplicate step numbers in the log",
                          len(step_nums) == len(set(step_nums)), str(step_nums))
            del r
            torch.cuda.empty_cache()

    failed = check.failed or check.skipped
    print(f"\n{'FAIL' if failed else 'PASS'}: {check.failed} check(s) failed, "
          f"{check.skipped} skipped, {time.perf_counter() - started:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
