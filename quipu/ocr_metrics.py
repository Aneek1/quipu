"""OCR metrics: character and word error rate (Levenshtein), exact-line match.

Rates are corpus-level: total edits over total reference length, not a mean of
per-line rates, so long lines weigh more."""
from __future__ import annotations

from typing import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def _rate(refs: list[Sequence], hyps: list[Sequence]) -> float:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    total = sum(len(r) for r in refs)
    if total == 0:
        raise ValueError("no reference characters to score against")
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / total


def cer(refs: list[str], hyps: list[str]) -> float:
    return _rate(refs, hyps)


def wer(refs: list[str], hyps: list[str]) -> float:
    return _rate([r.split() for r in refs], [h.split() for h in hyps])


def exact_match(refs: list[str], hyps: list[str]) -> float:
    if not refs:
        raise ValueError("no references")
    return sum(r == h for r, h in zip(refs, hyps)) / len(refs)
