"""scripts/remote/preflight.sh: parses, is LF, refuses bad usage, and names only test
files, scripts and configs that exist (it runs on a paid box: a typo there costs)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.test_copy_back import BASH

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote" / "preflight.sh"
needs_bash = pytest.mark.skipif(BASH is None, reason="needs bash (Git Bash on Windows)")


def _text() -> str:
    return SCRIPT.read_bytes().decode("utf-8")


def test_preflight_is_lf_and_never_touches_the_real_ledger_or_checkpoints():
    text = _text()
    assert "\r" not in text and "set -uo pipefail" in text
    assert "results/spend.json" not in text.split("set -uo pipefail", 1)[1]
    assert '--ledger "$L"' in text                       # a scratch ledger only
    assert "checkpoints/" not in text.split("set -uo pipefail", 1)[1]


def test_every_path_it_names_exists():
    text = _text()
    for rel in set(re.findall(r"(?:tests|scripts|configs)/[\w/.-]+\.(?:py|toml)", text)):
        assert (ROOT / rel).is_file(), rel


@needs_bash
def test_preflight_parses_with_bash_n():
    proc = subprocess.run([BASH, "-n"], input=_text(), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr


@needs_bash
@pytest.mark.parametrize("args", [[], ["--gpu-box"], ["--gpu-box", "--usd-per-hour", "x"],
                                  ["--cpu-box", "--probe-steps", "ten"], ["--bogus"]])
def test_usage_errors_exit_2_before_anything_runs(tmp_path, args):
    proc = subprocess.run([BASH, "-s", "--", *args], input=_text(), capture_output=True,
                          text=True, encoding="utf-8", timeout=60, cwd=tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "PASS" not in proc.stdout and "FAIL" not in proc.stdout
