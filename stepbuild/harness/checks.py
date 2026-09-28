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
exactly "timed out after Ns", and the whole process tree is killed. On Windows
each check runs in a Job Object with KILL_ON_JOB_CLOSE, so everything it started
dies when the check ends, timeout or not, even a detached grandchild (npm spawns
node children, and a test may start a server); `taskkill /T /F` stays as the
fallback on timeout. Elsewhere the check's process group is killed on timeout.

Output is the last 60 lines of stdout and stderr together: the end is where
pytest's summary and vite's error are, and it keeps the feedback that goes back
to the model short.

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

from stepbuild.harness.plan import CHECKS

if TYPE_CHECKING:
    from stepbuild.harness.sandbox import Sandbox

CHECK_NAMES = CHECKS
OUTPUT_LINES = 60
UNREACHABLE_PROXY = "http://127.0.0.1:9"


@dataclasses.dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    output: str      # last 60 lines of stdout + stderr, or "timed out after Ns"
    seconds: float


class _Job:
    """A Windows Job Object with KILL_ON_JOB_CLOSE: closing it kills every process
    still in it, including grandchildren that detached from the console or process
    group, which `taskkill /T` (it walks parent pids) can miss. Children inherit
    the job, so assigning the check's process is enough, as long as it happens
    before that process spawns anything: the gap between Popen returning and the
    assignment is microseconds, while Python or node take far longer to start.

    ctypes only, no new dependency. If the job cannot be created or assigned (for
    example on a very old Windows), the check runs without it and taskkill /T on
    timeout remains the fallback."""

    _LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        )
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._k32 = k32
        self._handle = k32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = self._LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
            self._handle, self._EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, "SetInformationJobObject failed")

    def assign(self, proc: subprocess.Popen) -> None:
        import ctypes

        if not self._k32.AssignProcessToJobObject(self._handle, int(proc._handle)):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def terminate(self) -> None:
        if self._handle:
            self._k32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Close the job, killing whatever is still running in it."""
        if self._handle:
            self._k32.CloseHandle(self._handle)
            self._handle = None


def _start_job(proc: subprocess.Popen) -> _Job | None:
    if sys.platform != "win32":
        return None
    try:
        job = _Job()
    except OSError:
        return None
    try:
        job.assign(proc)
    except OSError:
        job.close()
        return None
    return job


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
    job = _start_job(proc)
    try:
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if job is not None:
                job.terminate()
            _kill_tree(proc)  # fallback, and the only kill where there is no job
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                out = b""  # a grandchild still holds the pipe; give up on the output
            return None, (out or b"").decode("utf-8", errors="replace")
        return proc.returncode, out.decode("utf-8", errors="replace")
    finally:
        if job is not None:
            job.close()  # kills detached grandchildren the check left behind


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
