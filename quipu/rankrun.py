"""quipu.train under torchrun with a per-rank record: what each rank did, written by
the rank itself, so a launcher can check every rank instead of trusting torchrun's
single exit code.

    torchrun --standalone --nproc_per_node=N -m quipu.rankrun --marker-dir DIR -- \
        --config ... --run-id ... [quipu.train arguments]

Each rank writes DIR/rank<R>.json when quipu.train.run_main returns (crash included;
a rank killed outright writes nothing, and the launcher treats a missing record as a
failure):

    rank, world        from the process group (RANK / WORLD_SIZE before it exists)
    exit               run_main's exit code (0 completed, 4 budget stop, ...)
    start_step         the step training started from (after --resume / --branch-from)
    final_step         the step it stopped at
    checkpoints        [step, ...] of every checkpoint this rank took part in
    params_match       [bool, ...] quipu.dist.params_match at each of them
    batches            [digest, ...] one per training micro-batch this rank consumed:
                       blake2b-64 of the input tokens, so two ranks (or a run and its
                       resume) that read the same data show the same digest
    step_seconds       wall time of each train_step on this rank (CUDA synchronised)
    tokens_per_step    train.batch_tokens (the global batch)

Recording is a few wrappers installed around Trainer methods for this process only;
the training itself is quipu.train's, unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _digest(x: Any) -> str:
    return hashlib.blake2b(x.numpy().tobytes(), digest_size=8).hexdigest()


def install(record: dict[str, Any]) -> None:
    """Wrap quipu.train.Trainer so this process's training fills `record`."""
    import torch

    import quipu.train as T
    from quipu import dist

    init, resume, run, step, save = (T.Trainer.__init__, T.Trainer.resume_from_latest,
                                     T.Trainer.run, T.Trainer.train_step,
                                     T.Trainer.save_checkpoint)
    load = T.Trainer.load_checkpoint
    match = dist.params_match

    def __init__(self, *a: Any, **k: Any) -> None:
        init(self, *a, **k)
        record["tokens_per_step"] = self.train_cfg.batch_tokens
        stream = self.stream
        inner = stream.next_batch

        def next_batch() -> Any:
            x, y = inner()
            record["batches"].append(_digest(x))
            return x, y
        stream.next_batch = next_batch          # instance attribute: this trainer only

    def load_checkpoint(self, *a: Any, **k: Any) -> None:
        load(self, *a, **k)
        record["start_step"] = self.step

    def resume_from_latest(self) -> None:
        resume(self)
        record["start_step"] = self.step

    def train_step(self) -> float:
        t0 = time.perf_counter()
        out = step(self)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        record["step_seconds"].append(round(time.perf_counter() - t0, 4))
        return out

    def save_checkpoint(self) -> Path:
        record["checkpoints"].append(self.step)
        return save(self)

    def run_(self) -> None:
        if record["start_step"] is None:        # not resumed or branched: 0
            record["start_step"] = self.step
        try:
            run(self)
        finally:
            record["final_step"] = self.step

    def params_match(module: Any) -> bool:
        ok = match(module)
        record["params_match"].append(bool(ok))
        return ok

    T.Trainer.__init__ = __init__
    T.Trainer.load_checkpoint = load_checkpoint
    T.Trainer.resume_from_latest = resume_from_latest
    T.Trainer.train_step = train_step
    T.Trainer.save_checkpoint = save_checkpoint
    T.Trainer.run = run_
    dist.params_match = params_match        # train.py calls it as dist.params_match


def new_record() -> dict[str, Any]:
    return {"rank": int(os.environ.get("RANK", "0")),
            "world": int(os.environ.get("WORLD_SIZE", "1")),
            "exit": None, "start_step": None, "final_step": None, "checkpoints": [],
            "params_match": [], "batches": [], "step_seconds": [], "tokens_per_step": None}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        cut = argv.index("--")
        own, train_argv = argv[:cut], argv[cut + 1:]
    else:
        own, train_argv = argv, []
    p = argparse.ArgumentParser(prog="python -m quipu.rankrun")
    p.add_argument("--marker-dir", required=True)
    args = p.parse_args(own)
    record = new_record()
    install(record)
    import quipu.train as T

    out = Path(args.marker_dir) / f"rank{record['rank']}.json"
    try:
        record["exit"] = T.run_main(train_argv)
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, out)
    return record["exit"]


if __name__ == "__main__":
    sys.exit(main())
