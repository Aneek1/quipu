"""Which commits become step examples (spec §3.1, "Commit filters").

A kept commit must read as one small, self-described build step whose result can be
written as whole files in FILE blocks. `drop_reason` returns None to keep a commit,
or the first rule it breaks as a short key; the build report counts drops per key,
so the keys are stable names, listed in REASONS.

Rules, in the order they are checked (cheap structural checks first, so a merge is
reported as a merge rather than as whatever its diff happens to trip):

- merge / root: a merge's diff is against one parent only and says nothing about a
  step; the root commit is a project dump, not a step.
- no_change: no files, or files with no changed lines (renames, mode or binary
  changes). There is nothing to learn from.
- too_many_files: more than `max_files` files.
- unsafe_path: a path FileBlock refuses (absolute, `..`, backslash, device names,
  ...), or two paths that are one file on NTFS. The reply could not carry it, and
  the harness would reject the model for imitating it.
- excluded_path: generated or vendored files (node_modules/, dist/, build/, vendor/,
  .venv/, migrations/versions/ as whole path segments; *.min.*, *.map, lockfiles).
  Checked before file_type so a lockfile is counted as what it is.
- file_type: anything but .py .js .jsx .ts .tsx .css .html .sql, requirements.txt
  and package.json.
- deleted_file: a whole file removed. The full-file format has no way to say
  "delete this", so such a commit cannot be expressed.
- too_many_lines: added + removed over `max_lines`.
- too_long_file: a file over `max_file_lines` lines before or after the commit;
  the reply repeats it in full and the context shows its old version in full.
- marker_in_content: a file with a line that is a FILE marker; render_blocks
  refuses it because the reply could not be parsed back.
- low_info_message: the stripped message is under 12 characters; or its subject
  line (first line, trailing punctuation aside) is a low-information word (wip,
  fix, update, ...), a revert, a merge line (Merge branch/pull request/...), or a
  GitHub web-editor default ("Update app.py", "Add files via upload") that names a
  file and nothing else. The message is the STEP instruction, so it must say what
  to build.
"""
from __future__ import annotations

import dataclasses
import re

from stepbuild.harness.blocks import BlockError, FileBlock, render_blocks

REASONS = (
    "merge",
    "root",
    "no_change",
    "too_many_files",
    "unsafe_path",
    "excluded_path",
    "file_type",
    "deleted_file",
    "too_many_lines",
    "too_long_file",
    "marker_in_content",
    "low_info_message",
)

SOURCE_EXTENSIONS = (".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".sql")
SOURCE_NAMES = frozenset({"requirements.txt", "package.json"})
EXCLUDED_DIRS = ("node_modules", "dist", "build", "vendor", ".venv")
EXCLUDED_NESTED = (("migrations", "versions"),)
LOCKFILES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "uv.lock",
        "bun.lockb",
        "composer.lock",
        "gemfile.lock",
        "cargo.lock",
    }
)
LOW_INFO = frozenset(
    {"wip", "fix", "fixes", "update", "updates", "changes", "minor", "cleanup", "typo",
     "commit", "test", "."}
)
MIN_MESSAGE_CHARS = 12
_MERGE_LINE = re.compile(
    r"^merge (branch|pull request|remote-tracking branch|tag|commit|['\"])", re.IGNORECASE
)
_REVERT_LINE = re.compile(r"^revert\b", re.IGNORECASE)
_WEB_EDITOR = re.compile(
    r"^((update|create|delete|rename|add) \S*[./]\S*|add files via upload)$", re.IGNORECASE
)


@dataclasses.dataclass(frozen=True)
class FileChange:
    path: str           # POSIX path relative to the repo root
    before: str | None  # contents at the parent; None when the commit adds the file
    after: str | None   # contents at the commit; None when the commit deletes the file
    added: int          # lines added (git --numstat)
    removed: int        # lines removed

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError(f"path must be a str, got {type(self.path).__name__}")
        for name in ("before", "after"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None, got {type(value).__name__}")
        for name in ("added", "removed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")


@dataclasses.dataclass(frozen=True)
class Commit:
    sha: str
    message: str
    parents: int
    changes: tuple[FileChange, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sha, str) or not self.sha:
            raise ValueError("sha must be a non-empty str")
        if not isinstance(self.message, str):
            raise TypeError(f"message must be a str, got {type(self.message).__name__}")
        if not isinstance(self.parents, int) or isinstance(self.parents, bool) or self.parents < 0:
            raise ValueError(f"parents must be a non-negative int, got {self.parents!r}")
        if not isinstance(self.changes, tuple) or not all(
            isinstance(c, FileChange) for c in self.changes
        ):
            raise TypeError("changes must be a tuple of FileChange")


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _is_unsafe(paths: list[str]) -> bool:
    try:
        for p in paths:
            FileBlock(p, "")
    except BlockError:
        return True
    return len({p.casefold() for p in paths}) != len(paths)


def _is_excluded(path: str) -> bool:
    segments = path.lower().split("/")
    dirs, name = segments[:-1], segments[-1]
    if any(d in EXCLUDED_DIRS for d in dirs):
        return True
    for nested in EXCLUDED_NESTED:
        n = len(nested)
        if any(tuple(dirs[i:i + n]) == nested for i in range(len(dirs) - n + 1)):
            return True
    return ".min." in name or name.endswith(".map") or name in LOCKFILES


def _is_source(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name in SOURCE_NAMES or name.lower().endswith(SOURCE_EXTENSIONS)


def _has_marker(path: str, text: str) -> bool:
    """True when render_blocks refuses `text`: it has a line that is a FILE marker."""
    try:
        render_blocks([FileBlock(path, text)])
    except BlockError:
        return True
    return False


def is_low_info_message(message: str) -> bool:
    text = message.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) < MIN_MESSAGE_CHARS:
        return True
    subject = text.split("\n", 1)[0].strip()
    word = subject.rstrip(".!:;,").strip().lower() or subject
    return (
        word in LOW_INFO
        or bool(_REVERT_LINE.match(subject))
        or bool(_MERGE_LINE.match(subject))
        or bool(_WEB_EDITOR.match(subject))
    )


def drop_reason(
    commit: Commit, max_files: int = 3, max_lines: int = 200, max_file_lines: int = 400
) -> str | None:
    """None to keep `commit`, else the key (from REASONS) of the first rule it breaks."""
    if not isinstance(commit, Commit):
        raise TypeError(f"commit must be a Commit, got {type(commit).__name__}")
    if commit.parents > 1:
        return "merge"
    if commit.parents == 0:
        return "root"
    changes = commit.changes
    if not changes or sum(c.added + c.removed for c in changes) == 0:
        return "no_change"
    if len(changes) > max_files:
        return "too_many_files"
    paths = [c.path for c in changes]
    if _is_unsafe(paths):
        return "unsafe_path"
    if any(_is_excluded(p) for p in paths):
        return "excluded_path"
    if not all(_is_source(p) for p in paths):
        return "file_type"
    if any(c.after is None for c in changes):
        return "deleted_file"
    if sum(c.added + c.removed for c in changes) > max_lines:
        return "too_many_lines"
    texts = [(c.path, t) for c in changes for t in (c.before, c.after) if t is not None]
    if any(_line_count(t) > max_file_lines for _, t in texts):
        return "too_long_file"
    if any(_has_marker(p, t) for p, t in texts):
        return "marker_in_content"
    if is_low_info_message(commit.message):
        return "low_info_message"
    return None
