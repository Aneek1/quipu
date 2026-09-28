"""Pack retrieved chunks and the prompt into one context window.

Rules, each for a reason:
- The prompt goes last and is never cut: the model continues from its final token.
- Chunks are taken in rank order and the list is truncated at the first chunk that
  does not fit, so what gets dropped is always the least relevant, never an earlier
  pick (and a shorter low-ranked chunk can't jump the queue).
- Layout order is `order`:
  - "document" (the default, spec §5.1): chunks in document order, so text reads as
    it was written and neighbouring chunks rejoin into one unbroken passage.
  - "rank" (an experiment): reverse rank order, best chunk last, immediately before
    the prompt. A 114M model copies far better from nearby tokens, so this tests
    whether distance, not retrieval, is what loses the answer.
- A separator goes between chunks that were not adjacent in the document, and
  before the prompt, so the model sees a break instead of a mid-sentence splice.
  It is skipped when the preceding tokens already end in a break: a needle already
  ends in "\\n", and "fact\\n\\nprompt" measurably wrecks copying (the model reads
  the blank line as a topic change), so the prompt always follows exactly one break.
- The total never exceeds `budget`. Callers set budget = context - new tokens to
  generate, so generation never has to crop the window from the left.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

ORDERS = ("document", "rank")


@dataclasses.dataclass(frozen=True)
class PackedWindow:
    tokens: tuple[int, ...]
    chunk_ids: tuple[int, ...]   # included chunks, in the order they appear in the window


def _ends_in_break(tokens: Sequence[int], separator: Sequence[int],
                   ends_with_break: Callable[[int], bool] | None) -> bool:
    if not tokens:
        return False
    if ends_with_break is not None:
        return bool(ends_with_break(int(tokens[-1])))
    n = len(separator)
    return n > 0 and len(tokens) >= n and [int(t) for t in tokens[-n:]] == [int(t) for t in separator]


def join_with_break(
    head: Sequence[int], tail: Sequence[int], separator: Sequence[int],
    ends_with_break: Callable[[int], bool] | None = None,
) -> list[int]:
    """head + separator + tail, except the separator is left out when head is empty
    or already ends in a break. `ends_with_break(last_token)` decides what counts as
    a break (e.g. any token that decodes to text ending in a newline); without it,
    only a literal copy of `separator` does."""
    out = [int(t) for t in head]
    if out and not _ends_in_break(out, separator, ends_with_break):
        out.extend(int(t) for t in separator)
    out.extend(int(t) for t in tail)
    return out


def _layout(ids: Sequence[int], chunks: Sequence[Sequence[int]], prompt: Sequence[int],
            separator: Sequence[int], ends_with_break: Callable[[int], bool] | None) -> list[int]:
    out: list[int] = []
    prev: int | None = None
    for i in ids:
        if prev is not None and i != prev + 1:
            out = join_with_break(out, chunks[i], separator, ends_with_break)
        else:
            out.extend(int(t) for t in chunks[i])
        prev = i
    return join_with_break(out, prompt, separator, ends_with_break)


def pack_window(
    ranking: Sequence[int],
    chunks: Sequence[Sequence[int]],
    prompt: Sequence[int],
    budget: int,
    separator: Sequence[int] = (),
    order: str = "document",
    ends_with_break: Callable[[int], bool] | None = None,
) -> PackedWindow:
    """`ranking` is chunk ids, best first; `chunks[i]` is chunk i's tokens, with ids
    in document order."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if len(prompt) > budget:
        raise ValueError(f"prompt ({len(prompt)} tokens) exceeds the budget ({budget})")
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"ranking lists a chunk twice: {list(ranking)!r}")
    for i in ranking:
        if not 0 <= i < len(chunks):
            raise ValueError(f"chunk id {i} out of range for {len(chunks)} chunks")

    def arrange(ids: list[int]) -> list[int]:
        return sorted(ids) if order == "document" else list(reversed(ids))

    chosen: list[int] = []
    tokens = _layout([], chunks, prompt, separator, ends_with_break)
    for i in ranking:
        # Laying out for real (a few hundred tokens) is the only length that can't
        # disagree with the output, separators and skipped separators included.
        candidate = _layout(arrange(chosen + [i]), chunks, prompt, separator, ends_with_break)
        if len(candidate) > budget:
            break
        chosen.append(i)
        tokens = candidate
    assert len(tokens) <= budget
    return PackedWindow(tokens=tuple(tokens), chunk_ids=tuple(arrange(chosen)))
