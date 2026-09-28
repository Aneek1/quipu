"""Render a kept commit as one chat example (spec §3.1, "Example format").

The row is read back by the harness (stepbuild.harness.retrieve.ExampleLibrary)
and a model fine-tuned on it is later prompted by stepbuild.harness.prompt, so the
pieces are built with the harness's own parts rather than copies of them:

- the system message is prompt.SYSTEM_PROMPT verbatim;
- the user message is "STEP: <message>", a blank line, "CONTEXT FILES:" with the
  context as FILE blocks (render_blocks; "(none yet)" when there is none, as in the
  harness), a blank line, and "PROJECT TREE:" with one path per line. That is the
  same framing and separators the harness uses for those sections;
- the assistant message is render_blocks of every changed file's post-commit
  contents, in the commit's order.

The message is the whole commit message, stripped and with CRLF read as LF, so a
multi-paragraph message keeps its body: retrieve takes the step as everything
between "STEP: " and "\\n\\nCONTEXT FILES:", which gives it back unchanged.

Context files whose path FileBlock refuses, or whose contents render_blocks
refuses, are left out: they are optional extras chosen by the miner, and the
harness would never show such a file either. The changed files themselves are not
allowed to be unsafe: format_example refuses a commit `drop_reason` would drop, so
the caller cannot skip the filters by mistake.

Size cap: the user message may be at most `max_user_tokens`, counted with the
harness's default counter (ceil(chars / 3), pessimistic for code), so an example
that fits here also fits the harness's budget arithmetic. An example over the cap
gives None; files are never cut, because an example with half a file teaches the
model to write half files.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from stepbuild.dataset.filters import Commit, drop_reason
from stepbuild.dataset.split import assign_split
from stepbuild.harness.blocks import BlockError, FileBlock, render_blocks
from stepbuild.harness.prompt import SYSTEM_PROMPT, default_count_tokens

TAGS = frozenset({"fullstack", "flask", "react"})
NO_CONTEXT = "(none yet)\n"


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _context_blocks(context: Mapping[str, str]) -> list[FileBlock]:
    blocks: list[FileBlock] = []
    for path, text in context.items():
        try:
            block = FileBlock(path, text)
            render_blocks(blocks + [block])  # refuses marker lines and case duplicates
        except BlockError:
            continue
        blocks.append(block)
    return blocks


def format_example(
    repo: str,
    licence: str,
    tag: str,
    commit: Commit,
    context: Mapping[str, str],
    tree: Sequence[str],
    max_user_tokens: int = 6000,
) -> dict | None:
    """The spec §3.1 JSON row for `commit`, or None if the user message is over
    `max_user_tokens`. `context` maps path -> pre-commit contents, shown in the given
    order; `tree` is the project tree, one path per entry."""
    if not isinstance(repo, str) or repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    if not isinstance(licence, str) or not licence:
        raise ValueError("licence must be a non-empty SPDX id")
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {sorted(TAGS)}, got {tag!r}")
    if isinstance(tree, str) or not all(isinstance(p, str) for p in tree):
        raise TypeError("tree must be a sequence of path strings, not a single str")
    if not isinstance(context, Mapping):
        raise TypeError("context must map path -> contents")
    reason = drop_reason(commit)
    if reason is not None:
        raise ValueError(f"commit {commit.sha} is dropped by the filters ({reason})")

    step = _to_lf(commit.message).strip()
    ctx = _context_blocks(context)
    user = (
        f"STEP: {step}\n\n"
        f"CONTEXT FILES:\n{render_blocks(ctx) if ctx else NO_CONTEXT}\n"
        "PROJECT TREE:\n" + "".join(p + "\n" for p in tree)
    )
    if default_count_tokens(user) > max_user_tokens:
        return None
    reply = render_blocks([FileBlock(c.path, c.after) for c in commit.changes])
    return {
        "repo": repo,
        "licence": licence,
        "tag": tag,
        "commit": commit.sha,
        "split": assign_split(repo),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ],
    }
