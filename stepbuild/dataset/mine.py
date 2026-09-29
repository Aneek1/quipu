"""Turn a repo's git history into Commit records, plus the context a step prompt
shows (spec §3.1, `mine.py`).

Cloning. `clone` makes a full bare clone (history is the point; a bare clone has
no working tree to waste disk on) into `<repos_dir>/<owner>__<name>`. It clones
into a ".partial" directory and renames it into place only when git succeeds, so
an existing directory is always a finished clone and a rerun skips it. git runs
with GIT_TERMINAL_PROMPT=0 (a deleted or private repo fails instead of asking for
a password) and a timeout.

History. `git log --first-parent --reverse --no-renames --numstat -z` walks the
default branch oldest to newest in ONE process: the commit, its parents, its full
message and per-file line counts. --no-renames matters: a rename then reads as a
delete plus an add, which drop_reason drops as `deleted_file`, instead of passing
as a one-file edit whose reply silently leaves the old file behind.

Commit records. Merge and root commits come out with no changes (drop_reason
drops them on their parent count before it looks at changes). Binary files (numstat
"-") are left out of `changes`, as are files whose contents are not UTF-8 text.
File contents come from one long-lived `git cat-file --batch` process, reading
`<sha>^:<path>` (before; missing = added in this commit) and `<sha>:<path>` (after;
missing = deleted), which is `git show <rev>:<path>` without a process per file.
Two bounds keep a huge commit or blob from costing memory for nothing, without
changing which reason drops it:
- a commit touching more than `max_load_files` files (well above drop_reason's
  max_files) is recorded with no contents (before and after None); drop_reason
  still sees its line counts, so it is reported as no_change or too_many_files.
  (A whitespace-only change to that many files is reported as too_many_files.)
- a blob over MAX_BLOB_BYTES is not read; its contents are replaced by
  OVERSIZE_LINES lines naming the blob, which drop_reason reports as
  too_long_file (no source file that size is under 400 lines).

Context (for kept commits only; it is the expensive part). The pre-commit
contents of the changed files that existed, then up to CONTEXT_BM25 more files
from the parent commit's tree, ranked by BM25 (quipu.memory.bm25, GPT-2 token
ids as in stepbuild.harness.retrieve) against the commit message. Candidates are
source files (filters' extensions), not generated or vendored, not ".env*",
at most MAX_CANDIDATE_BYTES, and once read: UTF-8 text under `max_file_lines`
lines, no over-long lines, no FILE-marker lines, nothing contains_secret flags.
Token ids are cached per blob id, since most files are unchanged from one commit
to the next.

Tree. The parent commit's files (no directory entries), as the harness shows
its PROJECT TREE: the harness's own visibility rule (no node_modules/,
__pycache__/, acceptance tests, lockfiles), minus generated or vendored paths,
at most TREE_DEPTH directories deep, and at most MAX_TREE_FILES entries (the
shallowest first), listed sorted. This is only the candidate list: the formatter
renders it with the harness's render_tree, which applies the TREE_MAX_FILES cap
and ordering (the one implementation of the tree text).
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from quipu.fsio import replace_with_retry
from quipu.memory.bm25 import BM25Index
from stepbuild.dataset.discover import valid_repo
from stepbuild.dataset.filters import (
    Commit,
    FileChange,
    _has_long_lines,
    _has_marker,
    _is_excluded,
    _is_source,
    _line_count,
    contains_secret,
)
from stepbuild.harness.prompt import _in_tree  # the harness's own PROJECT TREE rule

MAX_LOAD_FILES = 20
MAX_BLOB_BYTES = 512 * 1024
OVERSIZE_LINES = 10_000
CONTEXT_BM25 = 2
MAX_CANDIDATE_BYTES = 64 * 1024
MAX_CANDIDATES = 2000
TREE_DEPTH = 2
MAX_TREE_FILES = 200
TOKEN_CACHE = 5000
GIT_TIMEOUT = 600
CLONE_TIMEOUT = 600

_RS, _US, _GS = "\x1e", "\x1f", "\x1d"


class MineError(RuntimeError):
    """A git step failed for this repo; the build logs it and moves on."""


def repo_dir_name(repo: str) -> str:
    if not valid_repo(repo):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    return repo.replace("/", "__")


def _rmtree(path: Path) -> None:
    def onexc(func, p, exc):  # git marks pack files read-only; Windows refuses to delete them
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.exists():
        shutil.rmtree(path, onexc=onexc)


def clone(
    repo: str,
    repos_dir: Path,
    runner: Callable[..., Any] = subprocess.run,
    timeout: int = CLONE_TIMEOUT,
) -> Path:
    """The path of a full bare clone of `repo`, cloning it unless already there."""
    dest = Path(repos_dir) / repo_dir_name(repo)
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".partial")
    _rmtree(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    cmd = ["git", "clone", "--bare", "--quiet", f"https://github.com/{repo}.git", str(tmp)]
    try:
        proc = runner(cmd, capture_output=True, text=True, encoding="utf-8",
                      errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        _rmtree(tmp)
        raise MineError(f"git clone {repo} timed out after {timeout} s") from None
    if proc.returncode != 0:
        _rmtree(tmp)
        last = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit {proc.returncode}"]
        raise MineError(f"git clone {repo} failed: {last[0]}")
    replace_with_retry(tmp, dest)
    return dest


def _git(repo_dir: Path, *args: str, timeout: int = GIT_TIMEOUT) -> bytes:
    cmd = ["git", "-C", str(repo_dir), "-c", "core.quotepath=off", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise MineError(f"git {args[0]} timed out after {timeout} s") from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["?"]
        raise MineError(f"git {args[0]} failed: {err[0]}")
    return proc.stdout


class _Oversize:
    def __init__(self, oid: str) -> None:
        self.oid = oid

    def text(self) -> str:
        return f"[blob {self.oid} over {MAX_BLOB_BYTES} bytes, not read]\n" * OVERSIZE_LINES


class BlobReader:
    """One `git cat-file --batch` process answering "<rev>:<path>" lookups."""

    def __init__(self, repo_dir: Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo_dir), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
                self._proc.wait()
        if self._proc.stdout:
            self._proc.stdout.close()

    def read(self, spec: str) -> bytes | _Oversize | None:
        """The blob's bytes, _Oversize for a blob over MAX_BLOB_BYTES, or None when
        `spec` names nothing (or not a blob)."""
        if "\n" in spec or "\r" in spec:
            return None
        self._proc.stdin.write(spec.encode("utf-8") + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline()
        if not header:
            raise MineError("git cat-file --batch exited unexpectedly")
        # "<spec> missing" / "<spec> ambiguous" echo the spec, which may contain
        # spaces; a found object is always "<oid> <type> <size>".
        if header.rstrip(b"\n").endswith((b" missing", b" ambiguous")):
            return None
        parts = header.split()
        if len(parts) != 3 or not parts[2].isdigit():
            raise MineError(f"git cat-file --batch: unexpected header {header[:200]!r}")
        oid, kind, size = parts[0].decode(), parts[1], int(parts[2])
        if kind != b"blob" or size > MAX_BLOB_BYTES:
            left = size + 1
            while left:
                left -= len(self._proc.stdout.read(min(left, 1 << 20)))
            return _Oversize(oid) if kind == b"blob" else None
        data = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # the LF after the contents
        return data


def _decode(data: bytes | _Oversize | None) -> str | None | bool:
    """Text, None for a missing file, or False for contents that are not UTF-8 text."""
    if data is None:
        return None
    if isinstance(data, _Oversize):
        return data.text()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return False if "\x00" in text else text


def _parse_log(raw: bytes) -> Iterator[tuple[str, list[str], str, list[tuple[str, str, str]]]]:
    text = raw.decode("utf-8", "replace")
    for record in text.split(_RS)[1:]:
        head, _, stats = record.partition(_GS)
        ids, _, message = head.partition(_US)
        sha, *parents = ids.split()
        entries = []
        for entry in stats.lstrip("\x00\n").split("\x00"):
            entry = entry.strip("\n")
            if not entry:
                continue
            added, removed, path = entry.split("\t", 2)
            entries.append((added, removed, path))
        yield sha, parents, message, entries


class Miner:
    """Reads one repo: `commits()` walks its history, `context(commit)` and
    `tree(commit)` give the prompt parts for a kept commit. Use as a context
    manager, so the cat-file process is always closed."""

    def __init__(self, repo_dir: Path, max_load_files: int = MAX_LOAD_FILES,
                 max_file_lines: int = 400) -> None:
        self.repo_dir = Path(repo_dir)
        self.max_load_files = max_load_files
        self.max_file_lines = max_file_lines
        self.first_sha: str | None = None
        self.last_sha: str | None = None
        self.walked = 0
        self._blobs = BlobReader(self.repo_dir)
        self._tokens: dict[str, np.ndarray | None] = {}
        self._tree_rev: str | None = None
        self._tree: list[tuple[str, str, int, str]] = []
        self._tok = None

    def __enter__(self) -> "Miner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._blobs.close()

    # ------------------------------------------------------------ history

    def commits(self) -> Iterator[Commit]:
        raw = _git(
            self.repo_dir, "-c", "i18n.logOutputEncoding=UTF-8", "log", "--first-parent",
            "--reverse", "--no-renames", "--numstat", "-z",
            f"--format={_RS}%H %P{_US}%B{_GS}",
        )
        for sha, parents, message, entries in _parse_log(raw):
            if self.first_sha is None:
                self.first_sha = sha
            self.last_sha = sha
            self.walked += 1
            changes: tuple[FileChange, ...] = ()
            if len(parents) == 1:
                changes = self._changes(sha, entries)
            yield Commit(sha=sha, message=message, parents=len(parents), changes=changes)

    def _changes(self, sha: str, entries) -> tuple[FileChange, ...]:
        text_entries = [(int(a), int(r), p) for a, r, p in entries if a != "-" and r != "-"]
        if len(text_entries) > self.max_load_files:
            return tuple(FileChange(p, None, None, a, r) for a, r, p in text_entries)
        changes = []
        for added, removed, path in text_entries:
            before = _decode(self._blobs.read(f"{sha}^:{path}"))
            after = _decode(self._blobs.read(f"{sha}:{path}"))
            if before is False or after is False:
                continue  # not UTF-8 text: treated like a binary file
            changes.append(FileChange(path, before, after, added, removed))
        return tuple(changes)

    # ------------------------------------------------------------ prompt parts

    def _parent_tree(self, commit: Commit) -> list[tuple[str, str, int, str]]:
        """(type, oid, size, path) of every entry in the parent commit's tree."""
        rev = f"{commit.sha}^"
        if self._tree_rev != rev:
            raw = _git(self.repo_dir, "ls-tree", "-r", "-l", "-z", rev).decode("utf-8", "replace")
            tree = []
            for entry in raw.split("\x00"):
                if not entry:
                    continue
                meta, _, path = entry.partition("\t")
                mode, kind, oid, size = meta.split()
                if kind != "blob" or mode == "120000":  # skip submodules and symlinks
                    continue
                tree.append((kind, oid, int(size) if size.isdigit() else 0, path))
            self._tree_rev, self._tree = rev, tree
        return self._tree

    def tree(self, commit: Commit) -> list[str]:
        paths = [
            p for _, _, _, p in self._parent_tree(commit)
            if p.count("/") <= TREE_DEPTH and _in_tree(p) and not _is_excluded(p)
        ]
        paths.sort(key=lambda p: (p.count("/"), p))
        return sorted(paths[:MAX_TREE_FILES])

    def _tokenizer(self):
        if self._tok is None:
            from quipu.tokenizer import Tokenizer
            self._tok = Tokenizer("gpt2")
        return self._tok

    def _candidate(self, oid: str, path: str) -> np.ndarray | None:
        if oid in self._tokens:
            return self._tokens[oid]
        text = _decode(self._blobs.read(oid))
        tokens = None
        if (
            isinstance(text, str)
            and text.strip()
            and _line_count(text) < self.max_file_lines
            and not _has_long_lines(text)
            and not _has_marker(path, text)
            and not contains_secret(text)
        ):
            tokens = np.asarray(self._tokenizer().encode(text), dtype=np.uint32)
            if tokens.size == 0:
                tokens = None
        if len(self._tokens) >= TOKEN_CACHE:
            self._tokens.clear()
        self._tokens[oid] = tokens
        return tokens

    def _text(self, oid: str) -> str:
        text = _decode(self._blobs.read(oid))
        assert isinstance(text, str)
        return text

    def context(self, commit: Commit) -> dict[str, str]:
        ctx = {c.path: c.before for c in commit.changes if c.before is not None}
        changed = {c.path.casefold() for c in commit.changes}
        pool = [
            (oid, p) for _, oid, size, p in self._parent_tree(commit)
            if p.casefold() not in changed
            and size <= MAX_CANDIDATE_BYTES
            and _is_source(p)
            and not _is_excluded(p)
            and not any(seg.startswith(".env") for seg in p.split("/"))
        ]
        pool.sort(key=lambda e: (e[1].count("/"), e[1]))
        docs = []
        for oid, p in pool[:MAX_CANDIDATES]:
            tokens = self._candidate(oid, p)
            if tokens is not None:
                docs.append((oid, p, tokens))
        query = self._tokenizer().encode(commit.message)
        if docs and query:
            index = BM25Index([t for _, _, t in docs])
            for i in index.search(query, CONTEXT_BM25):
                oid, p, _ = docs[i]
                ctx[p] = self._text(oid)
        return ctx


def mine_repo(repo_dir: Path, max_load_files: int = MAX_LOAD_FILES) -> Iterator[Commit]:
    """Every first-parent commit of the repo, oldest first (see Miner.commits)."""
    with Miner(repo_dir, max_load_files=max_load_files) as miner:
        yield from miner.commits()
