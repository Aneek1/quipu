"""BM25 over chunk token ids, written out by hand.

Why token ids and not words: the model's tokenizer already splits text, and an
identifier like AMBER_VAULT_CODE maps to the same ids wherever it occurs, so exact
matching needs no second tokenizer. Why by hand: the formula is ten lines, and
owning it lets a single chunk be patched in place (needle_eval changes one chunk
per trial over a 4,000-chunk index) without rebuilding anything.

Scoring (Robertson/Sparck Jones with the Lucene idf, which never goes negative):

    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = sum over unique t in q of
                  idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _term_counts(tokens: Iterable[int]) -> dict[int, int]:
    ids, counts = np.unique(np.asarray(tokens, dtype=np.int64), return_counts=True)
    return {int(t): int(c) for t, c in zip(ids, counts)}


class BM25Index:
    def __init__(self, chunks: Sequence[Sequence[int]], k1: float = 1.2, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.k1 = k1
        self.b = b
        self.n = len(chunks)
        self._tf: list[dict[int, int]] = []
        # postings[token] = {chunk index: term frequency}
        self._postings: dict[int, dict[int, int]] = {}
        self._len = np.zeros(self.n, dtype=np.float64)
        for i, chunk in enumerate(chunks):
            self._add(i, chunk)

    def _add(self, i: int, tokens: Sequence[int]) -> None:
        counts = _term_counts(tokens)
        if i < len(self._tf):
            self._tf[i] = counts
        else:
            self._tf.append(counts)
        for t, c in counts.items():
            self._postings.setdefault(t, {})[i] = c
        self._len[i] = sum(counts.values())

    def _remove(self, i: int) -> None:
        for t in self._tf[i]:
            post = self._postings[t]
            del post[i]
            if not post:
                del self._postings[t]
        self._tf[i] = {}
        self._len[i] = 0

    def patch(self, i: int, tokens: Sequence[int]) -> None:
        """Replace chunk i's contents; df, lengths and avgdl follow automatically."""
        if not 0 <= i < self.n:
            raise IndexError(f"chunk {i} out of range for {self.n} chunks")
        self._remove(i)
        self._add(i, tokens)

    @property
    def avgdl(self) -> float:
        return float(self._len.mean())

    def idf(self, token: int) -> float:
        df = len(self._postings.get(int(token), ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: Sequence[int]) -> np.ndarray:
        """BM25 score of every chunk for `query` (repeated query tokens count once)."""
        out = np.zeros(self.n, dtype=np.float64)
        avgdl = self.avgdl or 1.0
        norm = self.k1 * (1.0 - self.b + self.b * self._len / avgdl)
        for t in {int(x) for x in query}:
            post = self._postings.get(t)
            if not post:
                continue
            ids = np.fromiter(post.keys(), dtype=np.int64, count=len(post))
            tf = np.fromiter(post.values(), dtype=np.float64, count=len(post))
            out[ids] += self.idf(t) * tf * (self.k1 + 1.0) / (tf + norm[ids])
        return out

    def search(self, query: Sequence[int], k: int) -> list[int]:
        """Chunk ids by descending score, at most k, only chunks that match at all.
        Ties go to the earlier chunk, so the ranking is deterministic."""
        s = self.scores(query)
        hits = np.flatnonzero(s > 0)
        if hits.size == 0:
            return []
        order = hits[np.lexsort((hits, -s[hits]))]
        return [int(i) for i in order[:k]]
