"""scripts/remote/setup.sh: it parses, runs in strict mode, and never touches a
GitHub login (the box it runs on must not get one)."""
import os
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


# Every external command the script could run, as a stub that logs its arguments.
# The GPU/Node ones fail, so a run that needs them cannot pass by accident.
STUBS = {
    "uname": 'case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac',
    "dpkg": "exit 0",
    "uv": "exit 0",
    "git": ('case "$*" in *abbrev-ref*) echo main ;; *--short*) echo abc1234 ;; '
            '*log*) echo stub ;; esac'),
    "nvidia-smi": "exit 99",
    "node": "exit 99", "npm": "exit 99", "curl": "exit 99", "sudo": "exit 99",
    "apt-get": "exit 99",
}


def _run_setup(tmp_path, *args):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "calls.log"
    for name, body in STUBS.items():
        p = stubs / name
        p.write_bytes(f'#!/bin/sh\necho "{name} $*" >> "$STUB_LOG"\n{body}\n'.encode())
        p.chmod(0o755)
    workdir = tmp_path / "quipu"
    (workdir / ".git").mkdir(parents=True)  # an existing checkout: update, not clone
    env = {**os.environ, "PATH": os.pathsep.join([str(stubs),
                                                  str(Path(shutil.which("bash")).parent)]),
           "STUB_LOG": log.as_posix(), "WORKDIR": workdir.as_posix(),
           "HOME": tmp_path.as_posix()}
    proc = subprocess.run(["bash", "-s", "--", *args], input=_text(), capture_output=True,
                          text=True, encoding="utf-8", timeout=120, env=env)
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc, calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash on PATH")
def test_cpu_only_setup_never_touches_the_gpu_or_node(tmp_path):
    proc, calls = _run_setup(tmp_path, "--cpu-only")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = [c.split()[0] for c in calls]
    for gpu_or_node in ("nvidia-smi", "node", "npm", "curl", "apt-get", "sudo"):
        assert gpu_or_node not in names, calls
    uv = [c for c in calls if c.startswith("uv ")]
    assert "uv sync --frozen" in uv
    assert any("import fasttext" in c for c in uv)
    assert any("tests/test_build_shards_v2.py" in c for c in uv)
    assert not any("check_gpu" in c or "torch" in c or "test_sandbox" in c for c in uv), uv
    assert "CPU-only" in proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash on PATH")
def test_without_cpu_only_a_box_without_a_working_gpu_is_refused(tmp_path):
    proc, calls = _run_setup(tmp_path)
    assert proc.returncode != 0
    assert any(c.startswith("nvidia-smi") for c in calls)
    assert not any(c.startswith("uv ") for c in calls)
