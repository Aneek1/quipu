"""Pack retrieved chunks and the prompt into one context window.

Rules, each for a reason:
- The prompt goes last and is never cut: the model continues from its final token.
- Chunks are taken in rank order and the list is truncated at the first chunk that
  does not fit, so what gets dropped is always the least relevant, never an earlier
  pick (and a shorter low-ranked chunk can't jump the queue).
- The chosen chunks are laid out in document order, not rank order: text reads as it
  was written, and neighbouring chunks rejoin into one unbroken passage.
- A separator goes between chunks that were not adjacent in the document, and
  before the prompt, so the model sees a break instead of a mid-sentence splice.
- The total never exceeds `budget`. Callers set budget = context - new tokens to
  generate, so generation never has to crop the window from the left.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class PackedWindow:
    tokens: tuple[int, ...]
    chunk_ids: tuple[int, ...]   # included chunks, document order


def _layout(ids: Sequence[int], chunks: Sequence[Sequence[int]], prompt: Sequence[int],
            separator: Sequence[int]) -> list[int]:
    out: list[int] = []
    prev: int | None = None
    for i in sorted(ids):
        if prev is not None and i != prev + 1:
            out.extend(separator)
        out.extend(int(t) for t in chunks[i])
        prev = i
    if ids:
        out.extend(separator)
    out.extend(int(t) for t in prompt)
    return out


def _layout_len(ids: Sequence[int], chunks: Sequence[Sequence[int]], prompt_len: int,
                sep_len: int) -> int:
    s = sorted(ids)
    gaps = sum(1 for a, b in zip(s, s[1:]) if b != a + 1)
    return sum(len(chunks[i]) for i in s) + sep_len * (gaps + (1 if s else 0)) + prompt_len


def pack_window(
    ranking: Sequence[int],
    chunks: Sequence[Sequence[int]],
    prompt: Sequence[int],
    budget: int,
    separator: Sequence[int] = (),
) -> PackedWindow:
    """`ranking` is chunk ids, best first; `chunks[i]` is chunk i's tokens, with ids
    in document order."""
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if len(prompt) > budget:
        raise ValueError(f"prompt ({len(prompt)} tokens) exceeds the budget ({budget})")
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"ranking lists a chunk twice: {list(ranking)!r}")
    for i in ranking:
        if not 0 <= i < len(chunks):
            raise ValueError(f"chunk id {i} out of range for {len(chunks)} chunks")

    chosen: list[int] = []
    for i in ranking:
        if _layout_len(chosen + [i], chunks, len(prompt), len(separator)) > budget:
            break
        chosen.append(i)
    tokens = _layout(chosen, chunks, prompt, separator)
    assert len(tokens) <= budget
    return PackedWindow(tokens=tuple(tokens), chunk_ids=tuple(sorted(chosen)))
