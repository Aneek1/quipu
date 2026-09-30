"""scripts/remote/copy_back.sh: parses, dry-runs, and copies exactly the needed set from
a fake box (ssh / scp stubs over a local directory), verifying sha256, never deleting."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "remote" / "copy_back.sh"


def _bash() -> str | None:
    """Git Bash on Windows (WSL's bash cannot see Windows paths), else bash on PATH."""
    if os.name == "nt":
        found = shutil.which("bash")
        if found and "git" in found.lower():
            return found
        git = shutil.which("git")
        for parent in Path(git).parents if git else ():
            for cand in (parent / "bin" / "bash.exe", parent / "usr" / "bin" / "bash.exe"):
                if cand.is_file():
                    return str(cand)
        return None
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="needs bash (Git Bash on Windows)")


def _text() -> str:
    return SCRIPT.read_bytes().decode("utf-8")


def test_copy_back_is_lf_strict_and_never_deletes():
    text = _text()
    assert "\r" not in text and "set -euo pipefail" in text
    assert "rm " not in text and "--delete" not in text
    assert "ForwardAgent=no" in text and "BatchMode=yes" in text


@needs_bash
def test_copy_back_parses_with_bash_n():
    proc = subprocess.run([BASH, "-n"], input=_text(), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr


def _run(tmp_path, *args, env_extra=None):
    env = {**os.environ, "LOCAL_DIR": "local-copy", "HOME": tmp_path.as_posix(),
           **(env_extra or {})}
    return subprocess.run([BASH, "-s", "--", *args], input=_text(), capture_output=True,
                          text=True, encoding="utf-8", timeout=120, env=env, cwd=tmp_path)


@needs_bash
def test_dry_run_prints_the_listing_the_copies_and_the_check(tmp_path):
    proc = _run(tmp_path, "--dry-run", "1.2.3.4", "40022")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for item in ("checkpoints/quipu-moe checkpoints/quipu-moe-sft", "milestones",
                 "results/moe/run_config.toml", "data/shards-moe/manifest.json",
                 "data/shards-moe/val data/shards-moe/code_val data/shards-moe/val_lang",
                 "data/chat-sft/manifest.json", "data/chat-sft/val_by_source",
                 "inductor-cache", "sha256sum"):
        assert item in out, item
    assert "scp -P 40022" in out and "root@1.2.3.4:/workspace/quipu/F" in out
    assert "sha256sum -c --quiet copy_back.sha256" in out
    assert not (tmp_path / "local-copy").exists()


def test_usage_errors(tmp_path):
    if BASH is None:
        pytest.skip("needs bash")
    assert _run(tmp_path, "--dry-run", "onlyhost").returncode == 2
    assert _run(tmp_path, "--dry-run", "host", "notaport").returncode == 2


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _latest(step: str) -> bytes:
    # torch.save writes {"file": "step_NNNNNN.pt"} uncompressed: the name is in the bytes.
    return b"\x80\x02}q\x00X\x04\x00\x00\x00fileq\x01X\x0e\x00\x00\x00" + step.encode() + b"q\x02s."


def fake_box(root: Path, *, sft: bool = True) -> dict[str, bytes]:
    """A box tree; returns the files copy_back must bring (relative path -> bytes)."""
    want = {
        "checkpoints/quipu-moe/latest.pt": _latest("step_000900.pt"),
        "checkpoints/quipu-moe/step_000900.pt": b"final pretrain",
        "checkpoints/quipu-moe/milestones/step_000100.pt": b"m100",
        "checkpoints/quipu-moe/milestones/step_000900.pt": b"m900",
        "results/moe/run_config.toml": b"[train]\n",
        "results/moe/summary.md": b"# done\n",
        "results/ab/summary.md": b"# ab\n",
        "results/spend.json": b"{}",
        "results/sft/val_by_source.json": b"{}",
        "data/shards-moe/manifest.json": b"{}",
        "data/shards-moe/val/shard_000.bin": b"v",
        "data/shards-moe/code_val/shard_000.bin": b"c",
        "data/shards-moe/val_lang/ind_Latn/shard_000.bin": b"i",
        "data/chat-sft/manifest.json": b"{}",
        "data/chat-sft/val_by_source/aya/shard_000.bin": b"a",
    }
    if sft:
        want.update({
            "checkpoints/quipu-moe-sft/latest.pt": _latest("step_000120.pt"),
            "checkpoints/quipu-moe-sft/step_000120.pt": b"final sft",
            "checkpoints/quipu-moe-sft/milestones/step_000120.pt": b"sft m",
        })
    skip = {
        "checkpoints/quipu-moe/step_000800.pt": b"older, not the final",
        "results/ab/inductor-cache/x/y.bin": b"cache",
        "results/moe/something.pt": b"pt in results",
        "results/moe/run.json.tmp": b"tmp",
        "data/shards-moe/train/shard_000.bin": b"training shards stay",
        "data/chat-sft/train/shard_000.bin": b"chat train stays",
    }
    for rel, data in {**want, **skip}.items():
        _write(root / rel, data)
    return want


def _stubs(tmp_path) -> Path:
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    # ssh ... HOST "sh -s": run the listing script locally; scp ... HOST:SRC DEST: copy.
    (stubs / "ssh").write_bytes(b'#!/bin/sh\nexec sh -s\n')
    (stubs / "scp").write_bytes(
        b'#!/bin/sh\nfor a; do src=$dst; dst=$a; done\nsrc=${src#*@}\nsrc=${src#*:}\n'
        b'echo "$src" >> "$SCP_LOG"\ncp "$src" "$dst"\n')
    for p in stubs.iterdir():
        p.chmod(0o755)
    return stubs


def _real_run(tmp_path, box: Path):
    stubs = _stubs(tmp_path)
    key = tmp_path / "key"
    key.write_text("k")
    env = {"REMOTE_DIR": box.as_posix(), "SSH_KEY": key.as_posix(),
           "SCP_LOG": (tmp_path / "scp.log").as_posix(),
           "PATH": str(stubs) + os.pathsep + os.environ["PATH"]}
    return _run(tmp_path, "box", "22", env_extra=env)


@needs_bash
def test_copies_exactly_the_needed_set_and_verifies_it(tmp_path):
    box = tmp_path / "box"
    want = fake_box(box)
    local = tmp_path / "local-copy"
    # A local file that is not part of the set, and a stale copy of one that is.
    _write(local / "notes.txt", b"mine")
    _write(local / "results/moe/summary.md", b"old summary")
    proc = _real_run(tmp_path, box)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()}
    assert got == set(want) | {"notes.txt", "copy_back.sha256"}
    for rel, data in want.items():
        assert (local / rel).read_bytes() == data, rel
    assert (local / "notes.txt").read_bytes() == b"mine"             # never deleted
    sums = (local / "copy_back.sha256").read_text().splitlines()
    assert len(sums) == len(want)
    assert all(hashlib.sha256(want[ln.split()[1].lstrip("*")]).hexdigest() == ln.split()[0]
               for ln in sums)
    assert "all %d files match the box" % len(want) in proc.stdout
    # A second run copies nothing: every local file already matches.
    (tmp_path / "scp.log").unlink()
    proc = _real_run(tmp_path, box)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (tmp_path / "scp.log").exists()


@needs_bash
def test_a_missing_sft_checkpoint_is_reported_and_fails_the_run(tmp_path):
    box = tmp_path / "box"
    fake_box(box, sft=False)
    proc = _real_run(tmp_path, box)
    assert proc.returncode == 1
    assert "MISSING checkpoints/quipu-moe-sft/latest.pt" in proc.stdout
    assert "do not destroy the box yet" in proc.stdout
    # Everything that was there was still copied.
    assert (tmp_path / "local-copy/checkpoints/quipu-moe/step_000900.pt").is_file()
