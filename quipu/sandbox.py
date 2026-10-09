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

HARNESS = r'''
import json, sys
TESTS = json.load(open("tests.json", encoding="utf-8"))
ok = 0
first = ""
try:
    exec(compile(open("solution.py", encoding="utf-8").read(), "solution.py", "exec"), globals())
except BaseException as e:
    print("{tag}" + json.dumps({{"passed": 0, "total": len(TESTS), "error": "solution:" + type(e).__name__}}))
    sys.exit(0)
for t in TESTS:
    try:
        exec(t, globals())
        ok += 1
    except BaseException as e:
        first = first or type(e).__name__
print("{tag}" + json.dumps({{"passed": ok, "total": len(TESTS), "error": first}}))
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


def child_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env["PATH"] = str(Path(sys.executable).parent)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_tests(code: str, tests: list[str], timeout: float = 10.0,
              python: str | None = None) -> Result:
    """Run `code`, then each assert in `tests`, in a fresh interpreter."""
    with tempfile.TemporaryDirectory(prefix="quipu-sbx-", ignore_cleanup_errors=True) as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, "harness.py").write_text(HARNESS, encoding="utf-8")
        Path(tmp, "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        env = child_env()
        env["TEMP"] = env["TMP"] = tmp
        try:
            proc = subprocess.run([python or sys.executable, "-I", "harness.py"],
                                  cwd=tmp, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return Result(0, len(tests), "timeout")
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(RESULT_TAG):
                d = json.loads(line[len(RESULT_TAG):])
                return Result(int(d["passed"]), int(d["total"]), d.get("error") or "")
        return Result(0, len(tests), f"crash:{proc.returncode}")
