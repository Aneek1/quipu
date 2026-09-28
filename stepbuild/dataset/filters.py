"""Which commits become step examples (spec §3.1, "Commit filters").

A kept commit must read as one small, self-described build step whose result can be
written as whole files in FILE blocks. `drop_reason` returns None to keep a commit,
or the first rule it breaks as a short key; the build report counts drops per key,
so the keys are stable names, listed in REASONS in the order they are checked
(structural checks first, so a merge is reported as a merge rather than as
whatever its diff happens to trip):

- merge / root: a merge's diff is against one parent only and says nothing about a
  step; the root commit is a project dump, not a step.
- no_change: no files, or files with no changed lines (renames, mode or binary
  changes). There is nothing to learn from.
- whitespace_only: every file is the same before and after once line endings are
  read as LF, trailing whitespace is stripped per line and trailing blank lines
  dropped (CRLF conversions, trailing-space cleanups). The reply would teach the
  model to copy its input.
- too_many_files: more than `max_files` files.
- unsafe_path: a path FileBlock refuses (absolute, `..`, backslash, device names,
  ...), or two paths that are one file on NTFS. The reply could not carry it, and
  the harness would reject the model for imitating it.
- excluded_path: generated or vendored files (node_modules/, dist/, build/, vendor/,
  .venv/, .next/, coverage/, __pycache__/, migrations/versions/ as whole path
  segments; *.min.*, *.bundle.js, *.map, lockfiles). Checked before file_type so a
  lockfile is counted as what it is.
- file_type: anything but .py .js .jsx .ts .tsx .css .html .sql, requirements.txt
  and package.json (names compared case-insensitively).
- deleted_file: a whole file removed. The full-file format has no way to say
  "delete this", so such a commit cannot be expressed.
- too_many_lines: added + removed over `max_lines`.
- too_long_file: a file over `max_file_lines` lines before or after the commit;
  the reply repeats it in full and the context shows its old version in full.
- long_line: a line over MAX_LINE_CHARS, or a file whose average line is over
  MAX_AVG_LINE_CHARS (minified or generated code, embedded data). The line limit
  alone would let a 400-line file of 999-char lines through, so the reply size is
  bounded here and, finally, by format_example's token caps.
- marker_in_content: a file with a line that is a FILE marker; render_blocks
  refuses it because the reply could not be parsed back.
- secret: a changed file, before or after, that `contains_secret` flags. Such a
  commit would publish a credential in the dataset and teach the model to write
  one inline.
- unsafe_message: a message containing "\\n\\nCONTEXT FILES:" or a FILE-marker
  line. The step is read back as the text up to the first "\\n\\nCONTEXT FILES:",
  and a marker line would read as a file block in the prompt.
- dependency_only: every changed file is a dependency manifest (requirements.txt,
  package.json), or the subject is a dependency bot's ("Bump x from ...",
  "Update x requirement from ...", "chore(deps): ...", "[pre-commit.ci] ...",
  "Update dependency x to v2"). Version pins are not a build step. A dependency
  added together with code that uses it stays.
- low_info_message: the message is the STEP instruction, so it must say what to
  build. Dropped when the stripped message is under 12 characters; when the
  subject is a merge line (always); when it is a revert (subject `Revert "`, or
  a body saying "This reverts commit"); and, only when the body (after the first
  blank line) is under 40 characters, when the subject is a low-information word,
  a GitHub web-editor default ("Update app.py", "Add files via upload"), or made
  only of generic words ("Initial commit", "Updated code", "Minor bug fixes"). A
  substantive body rescues a bare subject: "fix" followed by a paragraph saying
  what was fixed is a good instruction.
"""
from __future__ import annotations

import dataclasses
import re

from stepbuild.harness.blocks import BlockError, FileBlock, render_blocks

REASONS = (
    "merge",
    "root",
    "no_change",
    "whitespace_only",
    "too_many_files",
    "unsafe_path",
    "excluded_path",
    "file_type",
    "deleted_file",
    "too_many_lines",
    "too_long_file",
    "long_line",
    "marker_in_content",
    "secret",
    "unsafe_message",
    "dependency_only",
    "low_info_message",
)

SOURCE_EXTENSIONS = (".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".sql")
DEPENDENCY_MANIFESTS = frozenset({"requirements.txt", "package.json"})
SOURCE_NAMES = DEPENDENCY_MANIFESTS
EXCLUDED_DIRS = frozenset(
    {"node_modules", "dist", "build", "vendor", ".venv", ".next", "coverage", "__pycache__"}
)
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
MAX_LINE_CHARS = 1000
MAX_AVG_LINE_CHARS = 200

MIN_MESSAGE_CHARS = 12
MIN_BODY_CHARS = 40
LOW_INFO = frozenset(
    {"wip", "fix", "fixes", "update", "updates", "changes", "minor", "cleanup", "typo",
     "commit", "test", "."}
)
GENERIC_WORDS = frozenset(
    {"update", "updated", "updates", "fix", "fixed", "fixes", "change", "changes", "changed",
     "made", "some", "small", "minor", "code", "bug", "bugs", "files", "stuff", "initial",
     "first", "commit", "init", "refactor", "more", "misc", "wip"}
)
_MERGE_LINE = re.compile(
    r"^merge (branch|pull request|remote-tracking branch|tag|commit|['\"])", re.IGNORECASE
)
_REVERT_SUBJECT = re.compile(r'^Revert "')
_REVERT_BODY = "This reverts commit"
_WEB_EDITOR = re.compile(
    r"^((update|create|delete|rename|add) \S*[./]\S*|add files via upload)$", re.IGNORECASE
)
_DEPENDENCY_BOT = re.compile(
    r"^((bump|update) \S+ (from|requirement)"
    r"|(build|chore|fix|ci|deps)\(deps(-dev)?\)"
    r"|\[pre-commit\.ci\]|pre-commit autoupdate"
    r"|update dependency |lock file maintenance|\[snyk\])",
    re.IGNORECASE,
)
_STEP_END = "\n\nCONTEXT FILES:"  # retrieve.py reads the step up to this

_SECRET_TOKENS = re.compile(
    r"AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|gh[pousr]_[A-Za-z0-9]{36}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|xox[baprs]-"
)
_SECRET_NAME = r"\w*(?:password|passwd|secret|api_key|apikey|token)\w*"
# NAME = '...', NAME: "...", and config['NAME'] = '...' (the optional quote + "]").
_SECRET_ASSIGNMENT = re.compile(
    rf"(?i){_SECRET_NAME}(?:['\"]\])?\s*[:=]\s*(['\"])(?P<value>[^'\"]{{6,}})\1"
)
# os.environ.get("NAME", "fallback") / os.getenv('NAME', 'fallback'): a hardcoded
# fallback ships the secret whenever the variable is unset.
_SECRET_FALLBACK = re.compile(
    rf"(?i)(?:environ\.get|getenv)\(\s*['\"]{_SECRET_NAME}['\"]\s*,\s*(['\"])"
    rf"(?P<value>[^'\"]{{6,}})\1"
)
_PLACEHOLDER_PREFIXES = (
    "changeme", "your-", "your_", "<your", "<", "$", "%(", "{{", "***", "process.env", "os.environ"
)
_PLACEHOLDER_WORDS = ("xxx", "example", "placeholder", "dummy", "replace", "redacted")


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


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _lines(text: str) -> list[str]:
    """The lines of `text` (any line endings), without a phantom last line."""
    lines = _to_lf(text).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _line_count(text: str) -> int:
    return len(_lines(text))


def _has_long_lines(text: str) -> bool:
    lines = _lines(text)
    if not lines:
        return False
    if any(len(line) > MAX_LINE_CHARS for line in lines):
        return True
    return sum(len(line) for line in lines) / len(lines) > MAX_AVG_LINE_CHARS


def _whitespace_normalised(text: str) -> str:
    return "\n".join(line.rstrip() for line in _lines(text)).rstrip("\n")


def _is_whitespace_only(changes: tuple[FileChange, ...]) -> bool:
    return all(
        c.before is not None
        and c.after is not None
        and _whitespace_normalised(c.before) == _whitespace_normalised(c.after)
        for c in changes
    )


def _is_unsafe(paths: list[str]) -> bool:
    try:
        for p in paths:
            FileBlock(p, "")
    except BlockError:
        return True
    return len({p.casefold() for p in paths}) != len(paths)


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1].lower()


def _is_excluded(path: str) -> bool:
    segments = path.lower().split("/")
    dirs, name = segments[:-1], segments[-1]
    if any(d in EXCLUDED_DIRS for d in dirs):
        return True
    for nested in EXCLUDED_NESTED:
        n = len(nested)
        if any(tuple(dirs[i:i + n]) == nested for i in range(len(dirs) - n + 1)):
            return True
    return (
        ".min." in name
        or name.endswith((".map", ".bundle.js"))
        or name in LOCKFILES
    )


def _is_source(path: str) -> bool:
    name = _name(path)
    return name in SOURCE_NAMES or name.endswith(SOURCE_EXTENSIONS)


def _has_marker(path: str, text: str) -> bool:
    """True when render_blocks refuses `text`: it has a line that is a FILE marker."""
    try:
        render_blocks([FileBlock(path, text)])
    except BlockError:
        return True
    return False


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith(_PLACEHOLDER_PREFIXES) or any(w in v for w in _PLACEHOLDER_WORDS)


def contains_secret(text: str) -> bool:
    """True when `text` looks like it holds a credential: a known token shape (AWS
    key id, private key header, GitHub/OpenAI/Slack token), or a quoted value of 6+
    characters that is not an obvious placeholder ("changeme", "your-...",
    "your_...", "xxx", "<...>", "${...}", "$VAR") given to a name containing
    password/passwd/secret/api_key/apikey/token (SECRET_KEY, DB_PASSWORD,
    api_token), either assigned (`NAME = '...'`, `NAME: "..."`,
    `app.config['NAME'] = '...'`) or as the fallback of an environment lookup
    (`os.environ.get("NAME", "...")`). A JSON body key (`{"password": "..."}`)
    is not an assignment. Tuned to keep false negatives low: a dropped good commit
    costs one example, a published secret costs far more."""
    if _SECRET_TOKENS.search(text):
        return True
    return any(
        not _is_placeholder(m.group("value"))
        for pattern in (_SECRET_ASSIGNMENT, _SECRET_FALLBACK)
        for m in pattern.finditer(text)
    )


def _is_unsafe_message(message: str) -> bool:
    text = _to_lf(message)
    return _STEP_END in text or _has_marker("message", text)


def _split_message(message: str) -> tuple[str, str]:
    """(subject, body): the first line, and everything after the first blank line."""
    text = _to_lf(message).strip()
    parts = text.split("\n\n", 1)
    return text.split("\n", 1)[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def _is_dependency_only(commit: Commit) -> bool:
    if all(_name(c.path) in DEPENDENCY_MANIFESTS for c in commit.changes):
        return True
    subject, _ = _split_message(commit.message)
    return bool(_DEPENDENCY_BOT.match(subject))


def _words(subject: str) -> list[str]:
    words = (re.sub(r"[^\w]", "", w).lower() for w in subject.split())
    return [w for w in words if w]


def is_low_info_message(message: str) -> bool:
    text = _to_lf(message).strip()
    if len(text) < MIN_MESSAGE_CHARS:
        return True
    subject, body = _split_message(text)
    if _MERGE_LINE.match(subject):
        return True
    if _REVERT_SUBJECT.match(subject) or _REVERT_BODY in body:
        return True
    if len(body) >= MIN_BODY_CHARS:
        return False
    word = subject.rstrip(".!:;,").strip().lower() or subject
    words = _words(subject)
    return (
        word in LOW_INFO
        or bool(_WEB_EDITOR.match(subject))
        or (bool(words) and all(w in GENERIC_WORDS for w in words))
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
    if _is_whitespace_only(changes):
        return "whitespace_only"
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
    if any(_has_long_lines(t) for _, t in texts):
        return "long_line"
    if any(_has_marker(p, t) for p, t in texts):
        return "marker_in_content"
    if any(contains_secret(t) for _, t in texts):
        return "secret"
    if _is_unsafe_message(commit.message):
        return "unsafe_message"
    if _is_dependency_only(commit):
        return "dependency_only"
    if is_low_info_message(commit.message):
        return "low_info_message"
    return None
