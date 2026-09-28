"""Reciprocal-rank fusion.

BM25 scores and cosine similarities live on unrelated scales, so they are combined
by rank, not by score: each list gives an item 1 / (k + rank), rank starting at 1.
k = 60 is the value from Cormack et al. (2009); it damps the top ranks enough that
one retriever's first place can't simply override the other's consensus.
"""
from __future__ import annotations

from typing import Sequence

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Every id that appears in any ranking, by descending fused score. Ties go to
    the id with the better single best rank, then the smaller id, so the output is
    deterministic."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    score: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"a ranking lists the same id twice: {list(ranking)!r}")
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(score, key=lambda i: (-score[i], best[i], i))
