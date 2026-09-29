"""scripts/remote/setup.sh: it parses, runs in strict mode, and never touches a
GitHub login (the box it runs on must not get one)."""
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "scripts" / "remote" / "setup.sh"


def _text() -> str:
    return SETUP.read_bytes().decode("utf-8")


def test_setup_is_strict_lf_and_has_no_github_credentials():
    text = _text()
    assert "set -euo pipefail" in text
    assert "\r" not in text  # CRLF breaks bash (.gitattributes keeps *.sh LF)
    lowered = text.lower()
    for word in ("gh auth", "gh_token", "github_token", "gh_enterprise_token", "hosts.yml",
                 "x-access-token", "credential.helper"):
        assert word not in lowered, word


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash on PATH")
def test_setup_parses_with_bash_n():
    # Fed on stdin, so no Windows path has to reach whichever bash is on PATH.
    proc = subprocess.run(["bash", "-n"], input=_text(), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
