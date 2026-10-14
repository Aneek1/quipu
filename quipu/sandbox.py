"""Run a generated Python answer against assert tests in a separate process.

Used by the distillation CUDA arm (scripts/distill_verify.py) to check teacher
answers by execution. Isolation is basic, not a security boundary: a fresh
interpreter in isolated mode (-I: no user site-packages, no PYTHON* variables),
an empty temporary working directory that is deleted afterwards, a wall-clock
timeout, and an environment stripped to what Windows needs to start Python (no
API keys or tokens reach the child). It does not block network or file access
outside the temporary directory; only run code from models you chose.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

FENCE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\r?\n(.*?)```", re.S)
KEEP_ENV = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
RESULT_TAG = "__QUIPU_RESULT__"

# The solution and its tests run in their own namespace (named __main__, as a script
# would be): sharing the harness's globals let an answer's own `ok`, `first` or
# `TESTS` overwrite the counts.
HARNESS = r'''
def _main():
    import json, sys
    out = sys.stdout
    tests = json.load(open("tests.json", encoding="utf-8"))
    ns = {{"__name__": "__main__", "__builtins__": __builtins__}}
    try:
        exec(compile(open("solution.py", encoding="utf-8").read(), "solution.py", "exec"), ns)
    except BaseException as e:
        out.write("\n{tag}" + json.dumps({{"passed": 0, "total": len(tests), "error": "solution:" + type(e).__name__}}) + "\n")
        return
    ok, first = 0, ""
    for t in tests:
        try:
            exec(t, ns)
            ok += 1
        except BaseException as e:
            first = first or type(e).__name__
    out.write("\n{tag}" + json.dumps({{"passed": ok, "total": len(tests), "error": first}}) + "\n")
    out.flush()


_main()
'''.format(tag=RESULT_TAG)


@dataclass
class Result:
    passed: int
    total: int
    error: str  # "" when every test passed; else the first failure kind

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


def extract_code(response: str, entry_point: str | None = None) -> str | None:
    """The answer's code: the first fenced block that defines `entry_point` (a def or
    class; or the
    first fenced block when no name is given), else the whole reply when it defines
    the function itself; None when there is no code."""
    blocks = FENCE.findall(response or "")
    if entry_point:
        pat = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(entry_point)}(?!\w)", re.M)
        for b in blocks:
            if pat.search(b):
                return b
        if not blocks and pat.search(response or ""):
            return response
        return None
    return blocks[0] if blocks else None


def base_python() -> str:
    """The real interpreter, not a virtual environment's launcher: on Windows a venv's
    python.exe starts the base python.exe as a child, so killing the launcher on a
    timeout would leave the child (and the open output pipes) running."""
    return getattr(sys, "_base_executable", None) or sys.executable


def child_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env["PATH"] = str(Path(base_python()).parent)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, stdin=subprocess.DEVNULL)
    proc.kill()


def run_tests(code: str, tests: list[str], timeout: float = 10.0,
              python: str | None = None) -> Result:
    """Run `code`, then each assert in `tests`, in a fresh interpreter."""
    with tempfile.TemporaryDirectory(prefix="quipu-sbx-", ignore_cleanup_errors=True) as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, "harness.py").write_text(HARNESS, encoding="utf-8")
        Path(tmp, "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        env = child_env()
        env["TEMP"] = env["TMP"] = tmp
        proc = subprocess.Popen([python or base_python(), "-I", "harness.py"],
                                cwd=tmp, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace")
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return Result(0, len(tests), "timeout")
        for line in reversed(out.splitlines()):
            if line.startswith(RESULT_TAG):
                d = json.loads(line[len(RESULT_TAG):])
                return Result(int(d["passed"]), int(d["total"]), d.get("error") or "")
        return Result(0, len(tests), f"crash:{proc.returncode}")
