"""The box-level spend ledger: what the rented box has cost so far, for every tool
that runs on it (the A/B orchestrator, M8, and the long-run launcher, M9).

The box is billed per hour from the moment it starts, whatever runs on it, so the
ledger anchors spend to the box, not to a process or an --out directory:

    spend = sum over box sessions of (last_seen - box_start) x usd_per_hour
            (the current session - the last one - uses now instead of last_seen)
          + the named adjustments

- A box session starts once, when the box starts: `python -m quipu.spend start
  --usd-per-hour R` (with --box-start EPOCH to backdate it to when the box was
  rented), or on the first use by a tool (ensure_session) if nobody ran `start`.
  Start a new session on every new box; a leftover ledger without one just keeps
  counting the old session up to now (overcounts, never undercounts).
- tick() records last_seen. The tools tick at start and every TICK_S (60 s), so
  when the box dies with a process still running (SIGKILL, box stopped) at most one
  interval of spend is lost; the next session starts from a new box_start.
- Adjustments are spend the ledger cannot see (e.g. `--spent-usd`: an earlier box
  whose ledger is gone). Each is stored under a key, and setting the same key again
  replaces its amount: passing the same --spent-usd twice counts it once.
- The file (default results/spend.json in the repo, or $QUIPU_SPEND_LEDGER) lives
  outside any --out directory, is written atomically (unique temp name + replace)
  and merged on every write (sessions by box_start with the latest last_seen,
  adjustments by key), so two tools on the same box never drop each other's entries.
- A missing ledger is an empty one. An unreadable or corrupt one is a LedgerError:
  never a silent $0.

API: Ledger.load(path=None, clock=time.time), .start_session(rate), .ensure_session
(rate), .tick(), .tick_if_due(every), .adjust(key, usd), .spent_usd(now=None),
.remaining(budget, now=None), .usd_per_hour.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from quipu.fsio import replace_with_retry

ENV_VAR = "QUIPU_SPEND_LEDGER"
DEFAULT_RELATIVE = Path("results") / "spend.json"
TICK_S = 60.0
STALE_NOTE_S = 600.0     # a session not seen for this long may be a box that stopped

ROOT = Path(__file__).resolve().parents[1]


class LedgerError(RuntimeError):
    """The ledger file cannot be read or does not make sense."""


def default_path() -> Path:
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return ROOT / DEFAULT_RELATIVE


def _finite(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _parse(text: str, path: Path) -> tuple[list[dict[str, float]], dict[str, float]]:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LedgerError(f"spend ledger {path} is not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise LedgerError(f"spend ledger {path} must be a JSON object")
    sessions = data.get("sessions", [])
    adjustments = data.get("adjustments", {})
    if not isinstance(sessions, list) or not isinstance(adjustments, dict):
        raise LedgerError(f"spend ledger {path}: sessions must be a list, adjustments a table")
    out: list[dict[str, float]] = []
    for s in sessions:
        if not isinstance(s, dict) or not all(
                _finite(s.get(k)) for k in ("box_start", "usd_per_hour", "last_seen")):
            raise LedgerError(f"spend ledger {path}: bad session {s!r}")
        if s["usd_per_hour"] < 0 or s["last_seen"] < s["box_start"]:
            raise LedgerError(f"spend ledger {path}: impossible session {s!r}")
        out.append({k: float(s[k]) for k in ("box_start", "usd_per_hour", "last_seen")})
    adj: dict[str, float] = {}
    for k, v in adjustments.items():
        if not _finite(v) or v < 0:
            raise LedgerError(f"spend ledger {path}: bad adjustment {k!r} = {v!r}")
        adj[str(k)] = float(v)
    return out, adj


class Ledger:
    def __init__(self, path: Path, sessions: list[dict[str, float]],
                 adjustments: dict[str, float], clock: Callable[[], float]) -> None:
        self.path = Path(path)
        self.sessions = sessions
        self.adjustments = adjustments
        self.clock = clock
        self._lock = threading.RLock()
        self._last_tick: float | None = None

    @classmethod
    def load(cls, path: str | Path | None = None,
             clock: Callable[[], float] = time.time) -> "Ledger":
        path = Path(path) if path is not None else default_path()
        sessions, adjustments = cls._read(path)
        return cls(path, sessions, adjustments, clock)

    @staticmethod
    def _read(path: Path) -> tuple[list[dict[str, float]], dict[str, float]]:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], {}
        except OSError as exc:
            raise LedgerError(f"spend ledger {path} cannot be read ({exc})") from exc
        return _parse(text, path)

    # -- state --

    @property
    def current(self) -> dict[str, float] | None:
        return self.sessions[-1] if self.sessions else None

    @property
    def usd_per_hour(self) -> float | None:
        return self.current["usd_per_hour"] if self.current else None

    def spent_usd(self, now: float | None = None) -> float:
        with self._lock:
            now = self.clock() if now is None else now
            total = sum(self.adjustments.values())
            for i, s in enumerate(self.sessions):
                end = max(now, s["last_seen"]) if i == len(self.sessions) - 1 else s["last_seen"]
                total += (end - s["box_start"]) / 3600 * s["usd_per_hour"]
            return total

    def remaining(self, budget: float, now: float | None = None) -> float:
        return budget - self.spent_usd(now)

    # -- changes (each is written at once) --

    def start_session(self, usd_per_hour: float, box_start: float | None = None) -> None:
        """A new box session. The previous one (another box) ends at its last tick."""
        if not _finite(usd_per_hour) or usd_per_hour < 0:
            raise ValueError(f"usd_per_hour must be a finite number >= 0, got {usd_per_hour!r}")
        with self._lock:
            now = self.clock()
            start = now if box_start is None else float(box_start)
            if not math.isfinite(start) or start > now:
                raise ValueError(f"box start {box_start!r} must be a time not after now")
            if self.current and start < self.current["last_seen"]:
                raise ValueError(
                    f"box start {start} is before the last session's last tick "
                    f"({self.current['last_seen']}); sessions cannot overlap")
            self.sessions.append({"box_start": start, "usd_per_hour": float(usd_per_hour),
                                  "last_seen": now})
            self._save()
            self._last_tick = now

    def ensure_session(self, usd_per_hour: float) -> bool:
        """Start a session on first use; True if one was started. An existing
        session keeps its own rate (the box's), and a different rate is reported."""
        with self._lock:
            if self.current is None:
                self.start_session(usd_per_hour)
                return True
            cur = self.current
            if not math.isclose(cur["usd_per_hour"], usd_per_hour):
                print(f"[spend] note: the box session in {self.path} is billed at "
                      f"${cur['usd_per_hour']:.2f}/h, not ${usd_per_hour:.2f}/h; keeping the "
                      "session's rate (python -m quipu.spend start starts a new session)",
                      file=sys.stderr, flush=True)
            idle = self.clock() - cur["last_seen"]
            if idle > STALE_NOTE_S:
                print(f"[spend] note: the box session in {self.path} was last seen "
                      f"{idle / 60:.0f} min ago and is counted as running since; on a new "
                      "box run `python -m quipu.spend start --usd-per-hour R` first",
                      file=sys.stderr, flush=True)
            return False

    def tick(self) -> None:
        with self._lock:
            if self.current is None:
                raise LedgerError(f"spend ledger {self.path} has no box session to tick")
            now = self.clock()
            self.current["last_seen"] = max(self.current["last_seen"], now)
            self._save()
            self._last_tick = now

    def tick_if_due(self, every: float = TICK_S) -> bool:
        with self._lock:
            if self._last_tick is not None and self.clock() - self._last_tick < every:
                return False
            self.tick()
            return True

    def adjust(self, key: str, usd: float) -> None:
        """Set the named adjustment (replaces an earlier amount under the same key)."""
        if not key or not isinstance(key, str):
            raise ValueError("an adjustment needs a non-empty key")
        if not _finite(usd) or usd < 0:
            raise ValueError(f"adjustment {key!r} must be a finite amount >= 0, got {usd!r}")
        with self._lock:
            self.adjustments[key] = float(usd)
            self._save()

    # -- the file --

    def _save(self) -> None:
        """Merge with what is on disk now (another tool may have written), then
        write atomically. Our adjustments win for the keys we hold."""
        disk_sessions, disk_adj = self._read(self.path)
        by_start: dict[float, dict[str, float]] = {}
        for s in disk_sessions + self.sessions:
            have = by_start.get(s["box_start"])
            if have is None:
                by_start[s["box_start"]] = dict(s)
            else:
                have["last_seen"] = max(have["last_seen"], s["last_seen"])
        merged = [by_start[k] for k in sorted(by_start)]
        adjustments = {**disk_adj, **self.adjustments}
        self.sessions[:] = merged
        self.adjustments = adjustments
        payload = json.dumps({
            "sessions": merged, "adjustments": adjustments,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            replace_with_retry(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


class Ticker:
    """A daemon thread that ticks the ledger every `every` seconds until stopped.
    A failed write is reported, not raised: the next tick tries again."""

    def __init__(self, ledger: Ledger, every: float = TICK_S) -> None:
        self.ledger = ledger
        self.every = every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="spend-ticker", daemon=True)

    def __enter__(self) -> "Ticker":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.ledger.tick()
            except Exception as exc:      # noqa: BLE001 - reported, retried next tick
                print(f"[spend] warning: ledger tick failed ({exc})", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m quipu.spend",
                                description="The box-level spend ledger.")
    p.add_argument("--ledger", help=f"ledger file (default ${ENV_VAR} or {DEFAULT_RELATIVE})")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="record that a new box session started")
    s.add_argument("--usd-per-hour", type=float, required=True)
    s.add_argument("--box-start", type=float, help="epoch seconds the box started (default now)")
    a = sub.add_parser("adjust", help="set a named extra spend (same key = replaced)")
    a.add_argument("--key", required=True)
    a.add_argument("--usd", type=float, required=True)
    sub.add_parser("show", help="print the spend so far")
    sub.add_parser("tick", help="record that the box is still up")
    args = p.parse_args(argv)
    try:
        led = Ledger.load(args.ledger)
        if args.cmd == "start":
            led.start_session(args.usd_per_hour, args.box_start)
        elif args.cmd == "adjust":
            led.adjust(args.key, args.usd)
        elif args.cmd == "tick":
            led.tick()
        rate = led.usd_per_hour
        print(f"{led.path}: spent ${led.spent_usd():.2f} over {len(led.sessions)} box "
              f"session(s)" + (f", now ${rate:.2f}/h" if rate is not None else "")
              + (f"; adjustments {led.adjustments}" if led.adjustments else ""))
        return 0
    except (LedgerError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
