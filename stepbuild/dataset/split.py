"""Train/validation/test split by repo (spec §3.1).

Commits of one repo share code, names and style, so a repo whose commits fell on
both sides would put near-answers to test examples into training. The split is
therefore a function of the repo name alone: every commit of a repo lands in the
same split, whatever order the builder meets them in, and a rebuild or an extended
build never moves a repo.

The name is casefolded first (GitHub names are case-insensitive, so Owner/Name and
owner/name are one repo), hashed with SHA-256 (stable across processes and Python
versions, unlike hash()), and the first 8 bytes read as a fraction in [0, 1) that
is placed on the cumulative ratios.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

SPLITS = ("train", "validation", "test")


def _fraction(key: str) -> float:
    digest = hashlib.sha256(key.casefold().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(repo: str, ratios: Sequence[float] = (0.90, 0.05, 0.05)) -> str:
    if not isinstance(repo, str) or not repo:
        raise ValueError("repo must be a non-empty 'owner/name' string")
    if isinstance(ratios, str) or len(ratios) != len(SPLITS):
        raise ValueError(f"ratios must be three numbers for {SPLITS}, got {ratios!r}")
    if any(not isinstance(r, (int, float)) or r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative numbers, got {ratios!r}")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1, got {ratios!r} (sum {sum(ratios)})")
    x = _fraction(repo)
    edge = 0.0
    for name, ratio in zip(SPLITS, ratios):
        edge += ratio
        if x < edge:
            return name
    # Float rounding can leave the edge a hair under 1: the last non-empty split.
    return next(n for n, r in reversed(list(zip(SPLITS, ratios))) if r > 0)
