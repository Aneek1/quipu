"""Fixed-size chunking of a token sequence.

Chunks are cut on token counts, not on sentences or lines: the retrievers work on
token ids (BM25) or on decoded text (dense), and a fixed size makes the packing
arithmetic in window.py exact. 256 tokens means three chunks plus a prompt fit in
the 1,024-token context.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

CHUNK_TOKENS = 256


@dataclasses.dataclass(frozen=True)
class Chunk:
    """Half-open span [start, end) of the source sequence; `index` is its position
    in document order, which is also the id every index uses for it."""

    index: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def chunk_spans(n_tokens: int, size: int = CHUNK_TOKENS) -> list[Chunk]:
    """Spans that tile [0, n_tokens) exactly: every chunk is `size` long except a
    shorter last one, and there is never an empty chunk."""
    for name, value in (("n_tokens", n_tokens), ("size", size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    return [
        Chunk(index=i, start=start, end=min(start + size, n_tokens))
        for i, start in enumerate(range(0, n_tokens, size))
    ]


def split_chunks(tokens: Sequence[int] | np.ndarray, size: int = CHUNK_TOKENS) -> list[np.ndarray]:
    """The token arrays for chunk_spans(len(tokens), size), as views (no copies)."""
    arr = np.asarray(tokens)
    return [arr[c.start:c.end] for c in chunk_spans(len(arr), size)]
