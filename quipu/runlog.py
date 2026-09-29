"""One JSON file per run, flushed on every step.

Flushing every step rather than at the end is deliberate: a run that is killed at
hour 19 must still have its history. The cost is one small write per step, which is
nothing beside a training step.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quipu.fsio import replace_with_retry


class RunLog:
    def __init__(
        self,
        out_dir: str | Path,
        run_id: str,
        config: dict[str, Any],
        *,
        resume: bool = False,
    ) -> None:
        self.path = Path(out_dir) / f"{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if resume:
            # A resume with nothing to resume from is a mistake worth stopping
            # for, not silently starting a fresh run under the same run_id.
            try:
                record = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot resume {run_id!r}: no readable run log at {self.path}"
                ) from exc
            record["status"] = "running"
            last_step = max((s["step"] for s in record.get("steps", [])), default=0)
            record.setdefault("resumes", []).append(
                {"at": datetime.now(timezone.utc).isoformat(), "from_step": last_step}
            )
            self.record: dict[str, Any] = record
        else:
            # Starting a "new" run must never silently clobber an old one's
            # history: __init__ flushes immediately, so without this guard a
            # duplicate run_id (e.g. Task 10's --resume path constructing a
            # plain RunLog by mistake) would wipe the existing log on the spot.
            if self.path.exists():
                raise FileExistsError(
                    f"run log already exists at {self.path}; pass resume=True "
                    "to continue it or choose a different run_id"
                )
            self.record = {
                "run_id": run_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "config": config,
                "steps": [],
                "evals": [],
                "resumes": [],
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

    def log_moe(self, step: int, layers: list[dict[str, Any]]) -> None:
        """Per-layer expert-load statistics for the interval ending at `step`."""
        self.record.setdefault("moe", []).append({"step": step, "layers": layers})
        self._flush()

    def add_alert(self, key: str, alert: dict[str, Any]) -> None:
        """Append to a record-level alert list (e.g. "moe_health"); the key only
        exists once something has fired, so its presence is the flag. Each alert
        carries the "step" it fired at, for truncate_to."""
        self.record.setdefault(key, []).append(alert)
        self._flush()

    def note(self, key: str, value: Any) -> None:
        """Set one record-level value (e.g. "compile": how the model was compiled)."""
        self.record[key] = value
        self._flush()

    def truncate_to(self, step: int) -> None:
        """Drop steps/evals (and MoE entries and alerts) logged after `step`.

        Task 10 calls this right after load_checkpoint: steps logged past the
        last saved checkpoint are about to be re-run from that checkpoint and
        would otherwise show up twice in the log.
        """
        self.record["steps"] = [s for s in self.record["steps"] if s["step"] <= step]
        self.record["evals"] = [e for e in self.record["evals"] if e["step"] <= step]
        for key in ("moe", "moe_health"):
            if key in self.record:
                kept = [e for e in self.record[key] if e["step"] <= step]
                if kept:
                    self.record[key] = kept
                else:
                    del self.record[key]
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
        # replace_with_retry absorbs Windows' transient post-write file locks
        # (antivirus/indexer) instead of crashing a 23-hour run over them.
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
            replace_with_retry(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
