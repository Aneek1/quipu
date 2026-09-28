"""The three checks that decide whether a step passed: pyflakes, pytest, npm build.

- `pyflakes`: `python -m pyflakes backend` from the project root. Fails only when
  pyflakes reports something (syntax errors, undefined names, unused imports).
- `pytest`: `python -m pytest -q tests` with cwd `backend`, so `from app import
  create_app` and `from models import Store` resolve the way they would when the
  app is run for real. The template's `tests/test_smoke.py` means there is always
  at least one test, so "no tests collected" never needs special-casing.
- `npm_build`: `npm run build` (vite build) in `frontend`, using the node_modules
  the sandbox linked in.

Python checks use the current interpreter (`sys.executable`), so they see the
same Flask and pytest as the harness, not whatever `python` is first on PATH.

Each check runs with a timeout. A timeout is a failed check whose output is
exactly "timed out after Ns", and the whole process tree is killed (`taskkill /T
/F` on Windows, where npm spawns node children that would otherwise outlive it;
the process group elsewhere). Output is the last 60 lines of stdout and stderr
together: the end is where pytest's summary and vite's error are, and it keeps
the feedback that goes back to the model short.

No network during checks is best effort only: HTTP_PROXY / HTTPS_PROXY /
ALL_PROXY point at 127.0.0.1:9 (discard port, nothing listens) and NO_PROXY is
empty, so well-behaved HTTP clients (requests, urllib, npm, node fetch with a
proxy agent) fail fast instead of reaching the internet. Raw sockets and clients
that ignore proxy variables are not stopped; this is not a sandbox for untrusted
code.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from stepbuild.harness.sandbox import Sandbox

CHECK_NAMES = ("pyflakes", "pytest", "npm_build")
OUTPUT_LINES = 60
UNREACHABLE_PROXY = "http://127.0.0.1:9"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    output: str      # last 60 lines of stdout + stderr, or "timed out after Ns"
    seconds: float


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_with_timeout(
    cmd: Sequence[str], cwd: Path, env: dict[str, str], timeout_s: float
) -> tuple[int | None, str]:
    """Run `cmd` with stdout and stderr merged. Returns (exit code, output), or
    (None, partial output) if it timed out, after killing its whole process tree."""
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out = b""  # a grandchild still holds the pipe; give up on the output
        return None, (out or b"").decode("utf-8", errors="replace")
    return proc.returncode, out.decode("utf-8", errors="replace")


def _check_env() -> dict[str, str]:
    env = os.environ.copy()
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[var] = env[var.lower()] = UNREACHABLE_PROXY
    env["NO_PROXY"] = env["no_proxy"] = ""
    env["npm_config_offline"] = "true"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env.pop("PYTHONPATH", None)  # the backend must import from its own directory
    return env


def _command(name: str, root: Path) -> tuple[list[str], Path] | str:
    """The command and working directory for a check, or an error message if the
    tool it needs is missing."""
    if name == "pyflakes":
        return [sys.executable, "-m", "pyflakes", "backend"], root
    if name == "pytest":
        return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], root / "backend"
    npm = shutil.which("npm")
    if npm is None:
        return "npm not found on PATH: install Node.js to build the frontend"
    return [npm, "run", "build"], root / "frontend"


def _tail(text: str, lines: int = OUTPUT_LINES) -> str:
    return "\n".join(text.replace("\r\n", "\n").rstrip("\n").split("\n")[-lines:])


def run_checks(
    sandbox: "Sandbox", names: Sequence[str], timeout_s: int = 180
) -> list[CheckResult]:
    """Run the named checks in order and return one result per name. Every check
    runs even if an earlier one failed, so the model sees all problems at once."""
    if isinstance(names, str):
        raise TypeError("names must be a sequence of check names, not a single str")
    unknown = [n for n in names if n not in CHECK_NAMES]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; expected some of {list(CHECK_NAMES)}")
    root = Path(sandbox.root)
    env = _check_env()
    results = []
    for name in names:
        start = time.monotonic()
        command = _command(name, root)
        if isinstance(command, str):
            results.append(CheckResult(name, False, command, 0.0))
            continue
        cmd, cwd = command
        code, output = run_with_timeout(cmd, cwd, env, timeout_s)
        seconds = time.monotonic() - start
        if code is None:
            results.append(CheckResult(name, False, f"timed out after {timeout_s}s", seconds))
        else:
            results.append(CheckResult(name, code == 0, _tail(output), seconds))
    return results
