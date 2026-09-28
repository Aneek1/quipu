"""Build tiny git repos for the mining and build tests.

Author, committer and dates are fixed, so two repos built from the same commit
list have the same SHAs (as a mirror of a GitHub repo does). Files are written as
bytes and autocrlf is off, so contents reach git exactly as given.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

_ENV = {
    "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str, n: int = 0) -> str:
    date = f"2024-01-01T00:{n // 60:02d}:{n % 60:02d}+00:00"
    env = {**os.environ, **_ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
    return proc.stdout.decode("utf-8", "replace").strip()


def init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "commit.gpgsign", "false")
    return repo


def commit(repo: Path, message: str, files: Mapping[str, str | bytes | None], n: int) -> str:
    """Write (str/bytes) or delete (None) the files, commit everything, return the SHA."""
    for rel, content in files.items():
        path = repo / rel
        if content is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", message, n=n)
    return git(repo, "rev-parse", "HEAD")


def make_repo(
    repo: Path, commits: Sequence[tuple[str, Mapping[str, str | bytes | None]]]
) -> list[str]:
    init_repo(repo)
    return [commit(repo, msg, files, n) for n, (msg, files) in enumerate(commits)]
