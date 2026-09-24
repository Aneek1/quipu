"""One JSON file per run, flushed on every step.

Flushing every step rather than at the end is deliberate: a run that is killed at
hour 19 must still have its history. The cost is one small write per step, which is
nothing beside a training step.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, out_dir: str | Path, run_id: str, config: dict[str, Any]) -> None:
        self.path = Path(out_dir) / f"{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "config": config,
            "steps": [],
            "evals": [],
        }
        self._flush()

    def log_step(self, step: int, train_loss: float, lr: float, tokens: int, **extra: Any) -> None:
        self.record["steps"].append(
            {"step": step, "train_loss": train_loss, "lr": lr, "tokens": tokens, **extra}
        )
        self._flush()

    def log_eval(self, step: int, val_loss: float) -> None:
        self.record["evals"].append({"step": step, "val_loss": val_loss})
        self._flush()

    def finish(self, status: str) -> None:
        self.record["status"] = status
        self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def _flush(self) -> None:
        # Write to a temp name first and swap it in atomically, same as
        # quipu/data.py's write_shard: a kill mid-write must never leave a
        # truncated, unparseable JSON at the final path, since that would
        # destroy the entire run history rather than just the latest step.
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
