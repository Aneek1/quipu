"""The box-level spend ledger: what the rented box has cost so far, for every tool
that runs on it (the A/B orchestrator, M8, and the long-run launcher, M9).

The box is billed per hour from the moment it starts, whatever runs on it, so the
ledger anchors spend to the box, not to a process or an --out directory:

    spend = sum over box sessions of (last_seen - box_start) x usd_per_hour
            (the current session - the last one, unless ended - uses now instead
            of last_seen)
          + the named adjustments

- A box session starts once, when the box starts: `python -m quipu.spend start
  --usd-per-hour R` (with --box-start EPOCH to backdate it to when the box was
  rented), or on the first use by a tool (ensure_session) if nobody ran `start`.
- Idle box time counts too. `start` installs a per-minute ticker that runs whether
  or not any tool does: a tagged crontab line (`* * * * * cd <repo> && <python> -m
  quipu.spend --ledger <file> tick`), or, without crontab or a running cron daemon,
  a detached `nohup python -m quipu.spend tick --loop 60` whose PID is kept in
  <ledger>.ticker.pid. It prints what it installed. `python -m quipu.spend stop`
  removes it and ends the session (last_seen = now, marked ended): a stopped box
  costs nothing more, and a stray tick cannot revive the session.
- tick() records last_seen on the latest session (the tools tick at start and every
  TICK_S as well). What is lost when the box dies: at most one ticker interval
  (60 s) with the ticker installed. Without it (`start --no-ticker`, or a session a
  tool started with ensure_session), the box's idle time after the last tool's last
  tick is NOT counted.
- A new `start` ends the previous session at its last tick. If that session was
  never stopped and its last tick is before the new box start, the gap is not
  counted, and `start` prints a loud warning with the gap and its cost at the old
  rate: if the old box was still up and billing, add it with `python -m
  quipu.spend adjust --key <name> --usd <amount>`.
- Adjustments are spend the ledger cannot see (e.g. `--spent-usd`: an earlier box
  whose ledger is gone). Each is stored under a key, and setting the same key again
  replaces its amount: passing the same --spent-usd twice counts it once.
- The file (default results/spend.json in the repo, or $QUIPU_SPEND_LEDGER) lives
  outside any --out directory. Every change is one locked transaction: an exclusive
  lock on <ledger>.lock (flock on POSIX, msvcrt.locking on Windows; LedgerError
  after lock_timeout_s), re-read the file, apply only this change to what is on
  disk, write atomically (unique temp name + replace). So two tools on the same box
  never drop each other's entries, and a stale in-memory copy never overwrites a
  newer value on disk. A tick always goes to the latest session on disk: it never
  extends a session that a newer `start` superseded (it says so once).
- A missing ledger is an empty one. An unreadable or corrupt one is a LedgerError:
  never a silent $0. A read that meets a Windows replace in progress is retried.

API: Ledger.load(path=None, clock=time.time), .start_session(rate), .ensure_session
(rate), .tick(), .tick_if_due(every), .adjust(key, usd), .end_session(),
.spent_usd(now=None), .remaining(budget, now=None), .usd_per_hour; Ticker(ledger).
CLI: python -m quipu.spend [--ledger F] start|stop|show|adjust|tick.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from quipu.fsio import replace_with_retry

ENV_VAR = "QUIPU_SPEND_LEDGER"
DEFAULT_RELATIVE = Path("results") / "spend.json"
TICK_S = 60.0
STALE_NOTE_S = 600.0     # a session not seen for this long may be a box that stopped
LOCK_TIMEOUT_S = 30.0
READ_ATTEMPTS = 6        # a read racing a Windows replace: 0.05 s .. 1.6 s backoff
CRON_TAG = "quipu-spend-ticker"

ROOT = Path(__file__).resolve().parents[1]


class LedgerError(RuntimeError):
    """The ledger file cannot be read or does not make sense."""


class SessionEnded(LedgerError):
    """The latest box session was ended (`spend stop`): there is nothing to tick."""


def default_path() -> Path:
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return ROOT / DEFAULT_RELATIVE


def lock_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".lock")


def ticker_pid_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".ticker.pid")


def _finite(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _parse(text: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
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
    out: list[dict[str, Any]] = []
    for s in sessions:
        if not isinstance(s, dict) or not all(
                _finite(s.get(k)) for k in ("box_start", "usd_per_hour", "last_seen")):
            raise LedgerError(f"spend ledger {path}: bad session {s!r}")
        if s["usd_per_hour"] < 0 or s["last_seen"] < s["box_start"]:
            raise LedgerError(f"spend ledger {path}: impossible session {s!r}")
        if not isinstance(s.get("ended", False), bool):
            raise LedgerError(f"spend ledger {path}: bad session {s!r}")
        session: dict[str, Any] = {k: float(s[k]) for k in ("box_start", "usd_per_hour", "last_seen")}
        if s.get("ended"):
            session["ended"] = True
        out.append(session)
    adj: dict[str, float] = {}
    for k, v in adjustments.items():
        if not _finite(v) or v < 0:
            raise LedgerError(f"spend ledger {path}: bad adjustment {k!r} = {v!r}")
        adj[str(k)] = float(v)
    return out, adj


# ---- the cross-process lock ----------------------------------------------------------

if os.name == "nt":
    import msvcrt

    def _try_lock(f: Any) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f: Any) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(f: Any) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f: Any) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """An exclusive lock on `path` (created if missing, never deleted: removing a
    lock file others may hold open would let two writers in). Released on exit, and
    by the OS if the process dies holding it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        deadline = time.monotonic() + timeout
        delay = 0.002
        while True:
            try:
                _try_lock(f)
                break
            except OSError as exc:          # held by another writer
                if time.monotonic() >= deadline:
                    raise LedgerError(f"spend ledger lock {path} still held after "
                                      f"{timeout:g} s ({exc})") from exc
                time.sleep(delay)
                delay = min(delay * 2, 0.05)
        try:
            yield
        finally:
            _unlock(f)


# ---- the ledger ------------------------------------------------------------------------

class Ledger:
    def __init__(self, path: Path, sessions: list[dict[str, Any]],
                 adjustments: dict[str, float], clock: Callable[[], float]) -> None:
        self.path = Path(path)
        self.sessions = sessions
        self.adjustments = adjustments
        self.clock = clock
        self.lock_timeout_s = LOCK_TIMEOUT_S
        self._lock = threading.RLock()
        self._last_tick: float | None = None
        # The latest session's box_start as this process last saw it: a tick that
        # finds another one on disk says so (a newer `start` superseded it).
        self._known_start = sessions[-1]["box_start"] if sessions else None

    @classmethod
    def load(cls, path: str | Path | None = None,
             clock: Callable[[], float] | None = None) -> "Ledger":
        path = Path(path) if path is not None else default_path()
        sessions, adjustments = cls._read(path)
        return cls(path, sessions, adjustments, clock if clock is not None else time.time)

    @staticmethod
    def _read(path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if path.is_dir():
            raise LedgerError(f"spend ledger {path} cannot be read (it is a directory)")
        for attempt in range(READ_ATTEMPTS):
            try:
                text = path.read_text(encoding="utf-8")
                break
            except FileNotFoundError:
                return [], {}
            except PermissionError as exc:  # Windows: a replace is in progress
                if attempt == READ_ATTEMPTS - 1:
                    raise LedgerError(f"spend ledger {path} cannot be read ({exc})") from exc
                time.sleep(0.05 * 2 ** attempt)
            except OSError as exc:
                raise LedgerError(f"spend ledger {path} cannot be read ({exc})") from exc
        return _parse(text, path)

    # -- state --

    @property
    def current(self) -> dict[str, Any] | None:
        return self.sessions[-1] if self.sessions else None

    @property
    def usd_per_hour(self) -> float | None:
        return self.current["usd_per_hour"] if self.current else None

    def spent_usd(self, now: float | None = None) -> float:
        with self._lock:
            now = self.clock() if now is None else now
            total = sum(self.adjustments.values())
            for i, s in enumerate(self.sessions):
                running = i == len(self.sessions) - 1 and not s.get("ended")
                end = max(now, s["last_seen"]) if running else s["last_seen"]
                total += (end - s["box_start"]) / 3600 * s["usd_per_hour"]
            return total

    def remaining(self, budget: float, now: float | None = None) -> float:
        return budget - self.spent_usd(now)

    # -- changes (each is one locked transaction on the file) --

    def _update(self, change: Callable[[list[dict[str, Any]], dict[str, float]], Any]) -> Any:
        """Under the file lock: read what is on disk now, apply `change` to it (and
        nothing else this process holds in memory), write it, and adopt it."""
        with self._lock, _file_lock(lock_path(self.path), self.lock_timeout_s):
            sessions, adjustments = self._read(self.path)
            result = change(sessions, adjustments)
            self._write(sessions, adjustments)
            self.sessions, self.adjustments = sessions, adjustments
            self._known_start = sessions[-1]["box_start"] if sessions else None
            return result

    def refresh(self) -> None:
        """Re-read the file (another tool may have written)."""
        with self._lock:
            self.sessions, self.adjustments = self._read(self.path)
            self._known_start = self.sessions[-1]["box_start"] if self.sessions else None

    def start_session(self, usd_per_hour: float, box_start: float | None = None) -> None:
        """A new box session. The previous one (another box) ends at its last tick;
        if it was never stopped, the uncounted gap is reported loudly."""
        if not _finite(usd_per_hour) or usd_per_hour < 0:
            raise ValueError(f"usd_per_hour must be a finite number >= 0, got {usd_per_hour!r}")

        def change(sessions: list[dict[str, Any]], adj: dict[str, float]) -> float:
            now = self.clock()
            start = now if box_start is None else float(box_start)
            if not math.isfinite(start) or start > now:
                raise ValueError(f"box start {box_start!r} must be a time not after now")
            prev = sessions[-1] if sessions else None
            if prev and start < prev["last_seen"]:
                raise ValueError(
                    f"box start {start} is before the last session's last tick "
                    f"({prev['last_seen']}); sessions cannot overlap")
            if prev and not prev.get("ended"):
                gap = start - prev["last_seen"]
                if gap > 0:
                    self._warn_gap(prev, gap)
                prev["ended"] = True
            sessions.append({"box_start": start, "usd_per_hour": float(usd_per_hour),
                             "last_seen": now})
            return now

        self._last_tick = self._update(change)

    def _warn_gap(self, prev: dict[str, Any], gap: float) -> None:
        cost = gap / 3600 * prev["usd_per_hour"]
        h, m = divmod(int(round(gap / 60)), 60)
        seen = datetime.fromtimestamp(prev["last_seen"], timezone.utc).isoformat(timespec="seconds")
        key = "gap-" + datetime.fromtimestamp(prev["last_seen"], timezone.utc).strftime("%Y%m%d-%H%M")
        print(f"[spend] WARNING: the previous box session (${prev['usd_per_hour']:.2f}/h) was "
              f"last seen {seen}, {h}:{m:02d} (h:mm) before this box start, and was never "
              f"stopped. It ends at its last tick, so that {h}:{m:02d} (${cost:.2f} at its "
              "rate) is NOT counted. If that box was still up and billing, add it: "
              f"python -m quipu.spend adjust --key {key} --usd {cost:.2f}",
              file=sys.stderr, flush=True)

    def ensure_session(self, usd_per_hour: float) -> bool:
        """Start a session on first use (or after `spend stop` ended the last one);
        True if one was started. A running session keeps its own rate (the box's),
        and a different rate is reported."""
        if not _finite(usd_per_hour) or usd_per_hour < 0:
            raise ValueError(f"usd_per_hour must be a finite number >= 0, got {usd_per_hour!r}")

        def change(sessions: list[dict[str, Any]], adj: dict[str, float]) -> bool:
            # Checked and started in one transaction: two tools starting together
            # on an empty ledger make one session, not two.
            if sessions and not sessions[-1].get("ended"):
                return False
            now = self.clock()
            sessions.append({"box_start": now, "usd_per_hour": float(usd_per_hour),
                             "last_seen": now})
            self._last_tick = now
            return True

        with self._lock:
            if self._update(change):
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
        """last_seen = now on the latest session on disk. Never on a session a newer
        `start` superseded (a tick after that goes to the new one, and says so), and
        never on an ended session (SessionEnded)."""
        known = self._known_start

        def change(sessions: list[dict[str, Any]], adj: dict[str, float]) -> float:
            if not sessions:
                raise LedgerError(f"spend ledger {self.path} has no box session to tick")
            cur = sessions[-1]
            if cur.get("ended"):
                raise SessionEnded(f"spend ledger {self.path}: the box session was ended "
                                   "(spend stop); `python -m quipu.spend start` starts a new one")
            if known is not None and cur["box_start"] != known:
                print(f"[spend] note: the box session in {self.path} changed (a newer "
                      "`spend start`); ticking the new session, not extending the old one",
                      file=sys.stderr, flush=True)
            now = self.clock()
            cur["last_seen"] = max(cur["last_seen"], now)
            return now

        self._last_tick = self._update(change)

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

        def change(sessions: list[dict[str, Any]], adj: dict[str, float]) -> None:
            adj[key] = float(usd)

        self._update(change)

    def end_session(self) -> bool:
        """End the latest session now (the box is stopping). False if there was
        none running."""
        def change(sessions: list[dict[str, Any]], adj: dict[str, float]) -> bool:
            if not sessions or sessions[-1].get("ended"):
                return False
            cur = sessions[-1]
            cur["last_seen"] = max(cur["last_seen"], self.clock())
            cur["ended"] = True
            return True

        return self._update(change)

    # -- the file --

    def _write(self, sessions: list[dict[str, Any]], adjustments: dict[str, float]) -> None:
        """Atomic: a unique temp name, then replace (the caller holds the lock)."""
        payload = json.dumps({
            "sessions": sessions, "adjustments": adjustments,
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


# ---- the idle ticker: box time counts whether or not a tool runs --------------------------

class System:
    """The OS calls the idle ticker makes (tests swap SYSTEM for a fake)."""

    posix = os.name == "posix"

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(self, args: list[str], input: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(args, input=input, capture_output=True, text=True, timeout=30)

    def cron_alive(self) -> bool:
        """A cron daemon is running (containers often ship crontab without one).
        Without /proc (not Linux) it is assumed to be."""
        proc = Path("/proc")
        if not proc.is_dir():
            return True
        for d in proc.iterdir():
            if not d.name.isdigit():
                continue
            try:
                if (d / "comm").read_text().strip() in ("cron", "crond"):
                    return True
            except OSError:
                continue
        return False

    def spawn(self, args: list[str], cwd: Path) -> int:
        kw: dict[str, Any] = {"cwd": cwd, "stdin": subprocess.DEVNULL,
                              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                              "close_fds": True}
        if os.name == "nt":
            kw["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            kw["start_new_session"] = True
        return subprocess.Popen(args, **kw).pid


SYSTEM = System()


def _cron_tag(ledger: Path) -> str:
    return f"# {CRON_TAG} {ledger}"


def _crontab_lines(system: System) -> list[str] | None:
    """The current crontab's lines ([] when there is none), None if it cannot be read."""
    try:
        r = system.run(["crontab", "-l"])
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return r.stdout.splitlines()
    if "no crontab" in (r.stderr or "").lower():
        return []
    return None


def _set_crontab(system: System, lines: list[str]) -> bool:
    try:
        r = system.run(["crontab", "-"], input="".join(ln + "\n" for ln in lines))
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def install_ticker(ledger: str | Path, python: str = sys.executable, repo: Path = ROOT,
                   every: float = TICK_S, system: System | None = None) -> str:
    """Install the per-minute ticker for this ledger (idempotent); returns what was
    installed. A crontab line when crontab and a cron daemon are there, else a
    detached tick loop whose PID goes to <ledger>.ticker.pid."""
    system = system or SYSTEM
    ledger = Path(ledger).resolve()
    tag = _cron_tag(ledger)
    why = "not POSIX"
    if system.posix:
        why = "no crontab"
        if system.which("crontab"):
            why = "no cron daemon running"
            if system.cron_alive():
                why = "crontab could not be read or written"
                line = (f"* * * * * cd {shlex.quote(str(repo))} && {shlex.quote(python)} -m "
                        f"quipu.spend --ledger {shlex.quote(str(ledger))} tick "
                        f">/dev/null 2>&1 {tag}")
                current = _crontab_lines(system)
                if "%" in line:
                    why = "a % in the paths (cron would split the command there)"
                elif current is not None:
                    keep = [ln for ln in current if not ln.rstrip().endswith(tag)]
                    if _set_crontab(system, keep + [line]):
                        return f"crontab line (every minute): {line}"
    args = [python, "-m", "quipu.spend", "--ledger", str(ledger), "tick", "--loop",
            f"{every:g}"]
    if system.posix and system.which("nohup"):
        args = ["nohup", *args]
    pid = system.spawn(args, repo)
    pid_file = ticker_pid_path(ledger)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n", encoding="utf-8")
    return (f"ticker process {pid} ({why} for a crontab line): {' '.join(args)}; "
            f"its PID is in {pid_file}")


def remove_ticker(ledger: str | Path, system: System | None = None) -> list[str]:
    """Remove this ledger's crontab line and tick loop; returns what was removed.
    The loop is not signalled (its PID may have been reused): it sees its PID file
    gone and exits at its next wake, and its last tick is refused (session ended)."""
    system = system or SYSTEM
    ledger = Path(ledger).resolve()
    done: list[str] = []
    if system.posix and system.which("crontab"):
        tag = _cron_tag(ledger)
        current = _crontab_lines(system)
        if current and any(ln.rstrip().endswith(tag) for ln in current):
            if _set_crontab(system, [ln for ln in current if not ln.rstrip().endswith(tag)]):
                done.append("removed the crontab line")
            else:
                done.append("WARNING: could not rewrite the crontab; remove the line "
                            f"ending in '{tag}' by hand (crontab -e)")
    pid_file = ticker_pid_path(ledger)
    try:
        pid = pid_file.read_text(encoding="utf-8").strip()
        pid_file.unlink()
        done.append(f"tick loop {pid} stops at its next wake (within {TICK_S:g} s)")
    except FileNotFoundError:
        pass
    return done


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def tick_loop(ledger: str | Path, every: float = TICK_S,
              clock: Callable[[], float] = time.time,
              sleep: Callable[[float], None] = time.sleep) -> int:
    """`tick --loop`: tick every `every` s until the session is ended or gone, or
    <ledger>.ticker.pid no longer names this process (`stop`, or a newer `start`)."""
    ledger = Path(ledger)
    pid_file = ticker_pid_path(ledger)
    me = os.getpid()
    while True:
        try:
            Ledger.load(ledger, clock=clock).tick()
        except SessionEnded:
            return 0
        except (LedgerError, OSError) as exc:
            print(f"[spend] warning: tick failed ({exc})", file=sys.stderr, flush=True)
        sleep(every)
        if _read_pid(pid_file) != me:
            return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m quipu.spend",
                                description="The box-level spend ledger.")
    p.add_argument("--ledger", help=f"ledger file (default ${ENV_VAR} or {DEFAULT_RELATIVE})")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start", help="record that a new box session started, and install "
                                     "the per-minute ticker that counts idle box time")
    s.add_argument("--usd-per-hour", type=float, required=True)
    s.add_argument("--box-start", type=float, help="epoch seconds the box started (default now)")
    s.add_argument("--no-ticker", action="store_true",
                   help="do not install the ticker (idle box time is then not counted)")
    sub.add_parser("stop", help="the box is stopping: remove the ticker, end the session")
    a = sub.add_parser("adjust", help="set a named extra spend (same key = replaced)")
    a.add_argument("--key", required=True)
    a.add_argument("--usd", type=float, required=True)
    sub.add_parser("show", help="print the spend so far")
    t = sub.add_parser("tick", help="record that the box is still up")
    t.add_argument("--loop", type=float, metavar="SECONDS",
                   help="keep ticking every SECONDS (the ticker process `start` spawns "
                        "when there is no crontab)")
    args = p.parse_args(argv)
    try:
        if args.cmd == "tick" and args.loop:
            return tick_loop(args.ledger or default_path(), args.loop)
        led = Ledger.load(args.ledger)
        if args.cmd == "start":
            led.start_session(args.usd_per_hour, args.box_start)
            if not args.no_ticker:
                try:
                    print(f"[spend] idle ticker: {install_ticker(led.path)}")
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"[spend] WARNING: no idle ticker ({exc}); box time with no tool "
                          "running is not counted", file=sys.stderr)
        elif args.cmd == "stop":
            for what in remove_ticker(led.path):
                print(f"[spend] {what}")
            print("[spend] ended the box session" if led.end_session()
                  else "[spend] no running box session to end")
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
