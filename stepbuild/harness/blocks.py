"""FILE blocks: the only channel from model output to the project on disk.

A model reply carries whole files in this form (text outside blocks is ignored,
because models chat around their output):

    === FILE: backend/models.py ===
    <the complete file contents>
    === END FILE ===

This module is the safety boundary between that text and the filesystem, so it is
strict rather than forgiving, and every rejection says what to do instead (the
message is fed back to the model as the reason its attempt failed):

- A marker is a whole line (trailing spaces/tabs aside). `===` anywhere else,
  including a line that merely starts with `=== FILE:`, is content.
- A `=== FILE:` marker inside an open block means the model forgot `=== END FILE ===`;
  that is an error, not content, so a truncated file is never written silently.
  `render_blocks` refuses content containing marker lines for the same reason, which
  is what makes parse(render(x)) == x hold for everything render accepts.
- Paths must be relative, POSIX-style and canonical: no absolute or drive paths, no
  `..`, no `.` or empty segments, no backslashes (rejected, not normalised, so the
  model learns the format), no `:` (drive-relative paths, NTFS streams), and no
  segment Windows would silently rewrite or treat as a device (trailing dot/space,
  CON/NUL/COM1...). Validation lives in FileBlock itself, so an unsafe FileBlock
  cannot exist, whoever builds it.
- CRLF and lone CR line endings (common in model output) are read as LF, both in
  a reply and in FileBlock content, so a CRLF FileBlock equals its LF version and
  parsed content always uses LF.
- Duplicate paths are detected case-insensitively: App.jsx and app.jsx are the same
  file on NTFS, and the second would silently overwrite the first.
- Content keeps its exact text except the end: trailing newlines are normalised to
  exactly one (an all-blank file becomes empty), so a model's stray blank lines at
  the end neither change the file nor break equality.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Collection, Sequence

_START = "=== FILE:"
_TAIL = "==="
END_MARKER = "=== END FILE ==="

_DRIVE = re.compile(r"^[A-Za-z]:")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{d}{i}" for d in ("COM", "LPT") for i in range(1, 10)}


class BlockError(ValueError):
    """The model's reply cannot be applied; the message says why and what to do."""


def _validate_path(path: str) -> None:
    if not path:
        raise BlockError("empty path in FILE marker: write '=== FILE: <relative/path> ==='")
    if "\\" in path:
        raise BlockError(f"backslash in path {path!r}: use '/' to separate directories")
    if path.startswith("/") or _DRIVE.match(path):
        raise BlockError(f"absolute path {path!r}: paths must be relative to the project root")
    if ":" in path:
        raise BlockError(f"':' in path {path!r}: paths must be plain relative paths")
    if any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise BlockError(f"control character in path {path!r}")
    for seg in path.split("/"):
        if seg == "..":
            raise BlockError(f"'..' in path {path!r}: files must stay inside the project")
        if seg in ("", "."):
            raise BlockError(
                f"non-canonical path {path!r}: no empty or '.' segments, no trailing '/'"
            )
        if seg != seg.rstrip(". ") or seg != seg.lstrip(" "):
            raise BlockError(f"path segment {seg!r} in {path!r} starts or ends with a space or dot")
        if seg.split(".")[0].upper() in _RESERVED:
            raise BlockError(f"path segment {seg!r} in {path!r} is a reserved device name")


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalise_content(content: str) -> str:
    body = _to_lf(content).rstrip("\n")
    return body + "\n" if body else ""


@dataclasses.dataclass(frozen=True)
class FileBlock:
    path: str      # POSIX-style relative path, validated
    content: str   # exact file contents, trailing newline normalised to exactly one

    def __post_init__(self) -> None:
        _validate_path(self.path)
        object.__setattr__(self, "content", _normalise_content(self.content))


def _start_path(line: str) -> str | None:
    """The path if `line` is a start marker, else None. (A line that starts with
    `=== FILE:` and ends with `===` is always long enough for the two not to overlap,
    and is never the END marker, so no further checks are needed.)"""
    if not line.startswith(_START) or not line.endswith(_TAIL):
        return None
    return line[len(_START):-len(_TAIL)].strip()


def _is_marker(line: str) -> bool:
    line = line.rstrip(" \t")
    return line == END_MARKER or _start_path(line) is not None


def render_blocks(blocks: Sequence[FileBlock]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for b in blocks:
        if b.path.casefold() in seen:
            raise BlockError(f"file {b.path!r} appears more than once (paths ignore case)")
        seen.add(b.path.casefold())
        if any(_is_marker(line) for line in b.content.split("\n")):
            raise BlockError(f"content of {b.path!r} contains a FILE marker line")
        out.append(f"{_START} {b.path} {_TAIL}\n{b.content}{END_MARKER}\n")
    return "".join(out)


def parse_blocks(text: str, allowed: Collection[str] | None = None) -> list[FileBlock]:
    if isinstance(allowed, str):
        # `in` on a str is a substring test: "a.py" in "backend/a.py" would pass.
        raise TypeError("allowed must be a collection of paths, not a single str")
    allowed_set = None if allowed is None else frozenset(allowed)
    lines = _to_lf(text).split("\n")
    blocks: list[FileBlock] = []
    seen: set[str] = set()
    current: str | None = None
    body: list[str] = []
    for raw in lines:
        line = raw.rstrip(" \t")
        start = _start_path(line)
        if current is None:
            if start is None:
                continue  # chatter outside blocks, including a stray END marker
            _validate_path(start)
            if start.casefold() in seen:
                raise BlockError(
                    f"file {start!r} appears more than once (paths ignore case); "
                    "send each file once"
                )
            if allowed_set is not None and start not in allowed_set:
                if not allowed_set:
                    raise BlockError(
                        f"file {start!r} is not allowed: no files may be written in this step"
                    )
                raise BlockError(
                    f"file {start!r} is not allowed in this step; allowed files: "
                    + ", ".join(sorted(allowed_set))
                )
            current, body = start, []
        elif line == END_MARKER:
            seen.add(current.casefold())
            blocks.append(FileBlock(current, "\n".join(body)))
            current = None
        elif start is not None:
            raise BlockError(
                f"unterminated block for {current!r}: end it with '{END_MARKER}' "
                f"before starting {start!r}"
            )
        else:
            body.append(raw)
    if current is not None:
        raise BlockError(f"unterminated block for {current!r}: end it with '{END_MARKER}'")
    if not blocks:
        raise BlockError(
            f"no FILE block found: reply with '{_START} <path> {_TAIL}', the full file, "
            f"then '{END_MARKER}'"
        )
    return blocks
