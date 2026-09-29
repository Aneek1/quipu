"""How the box tools (scripts/ab_runs.py, scripts/remote/run_moe.py) start a trainer
and stop cleanly, shared so both behave the same.

- popen_kwargs(): the trainer runs in a session of its own on POSIX
  (start_new_session: a terminal hangup or the terminal's Ctrl+C never reaches it
  directly; the tool forwards the interrupt) and in a process group of its own on
  Windows (CTRL_BREAK reaches it alone).
- install_stop_signals(): SIGTERM and SIGHUP (POSIX, main thread) raise
  KeyboardInterrupt, so they take Ctrl+C's path: the running trainer is sent the
  interrupt and given its grace to write its checkpoint. Only the first one raises:
  a repeat (a hangup followed by a supervisor's SIGTERM) must not cut the child's
  checkpoint short.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Any, Callable

STOP_SIGNALS = ("SIGTERM", "SIGHUP")


def popen_kwargs(posix: bool = os.name != "nt") -> dict[str, Any]:
    """A child of its own: a new session on POSIX, a new process group on Windows."""
    if posix:
        return {"start_new_session": True}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}


def stop_signal_handler(echo: Callable[[str], None], tag: str) -> Callable[[int, Any], None]:
    """SIGTERM / SIGHUP -> KeyboardInterrupt, the first time only."""
    fired = False

    def stop(signum: int, frame: Any) -> None:
        nonlocal fired
        if fired:
            echo(f"{tag} signal {signum} again; still stopping (waiting for the running "
                 "child's checkpoint; Ctrl+C escalates)")
            return
        fired = True
        echo(f"{tag} signal {signum}: stopping like Ctrl+C (the running child checkpoints "
             "first)")
        raise KeyboardInterrupt

    return stop


def install_stop_signals(echo: Callable[[str], None], tag: str) -> dict[int, Any]:
    """POSIX, main thread only: SIGTERM and SIGHUP raise KeyboardInterrupt (once).
    Returns the previous handlers (restore them with restore_signals)."""
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        return {}
    handler = stop_signal_handler(echo, tag)
    previous: dict[int, Any] = {}
    for name in STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            previous[sig] = signal.signal(sig, handler)
    return previous


def restore_signals(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)
