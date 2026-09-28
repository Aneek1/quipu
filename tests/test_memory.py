"""quipu/memory: chunker, BM25, dense (with an injected fake embedder), fusion, window.

CPU only, no network: the dense tests never touch the real e5 model.
"""
from __future__ import annotations

import numpy as np
import pytest

from quipu.memory.bm25 import BM25Index
from quipu.memory.chunker import Chunk, chunk_spans, split_chunks
from quipu.memory.dense import DenseIndex
from quipu.memory.fusion import reciprocal_rank_fusion
from quipu.memory.window import pack_window


# --------------------------------------------------------------------- chunker


def test_chunk_spans_cover_the_sequence_exactly():
    spans = chunk_spans(1000, 256)
    assert spans[0].start == 0
    assert spans[-1].end == 1000
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start
    assert [s.index for s in spans] == list(range(len(spans)))
    assert [s.end - s.start for s in spans] == [256, 256, 256, 232]


def test_chunk_spans_exact_multiple_has_no_empty_tail():
    spans = chunk_spans(512, 256)
    assert [(s.start, s.end) for s in spans] == [(0, 256), (256, 512)]


def test_chunk_spans_shorter_than_one_chunk():
    assert chunk_spans(10, 256) == [Chunk(index=0, start=0, end=10)]


def test_chunk_spans_rejects_bad_input():
    with pytest.raises(ValueError):
        chunk_spans(0, 256)
    with pytest.raises(ValueError):
        chunk_spans(10, 0)


def test_split_chunks_matches_spans():
    tokens = np.arange(600, dtype=np.int64)
    chunks = split_chunks(tokens, 256)
    assert len(chunks) == 3
    assert np.array_equal(np.concatenate(chunks), tokens)
    assert chunks[-1].tolist() == list(range(512, 600))


# ------------------------------------------------------------------------ bm25


def _corpus_with_common(n: int, common: int = 7) -> list[list[int]]:
    # Every chunk holds the common token and some filler unique to it.
    return [[common, 1000 + i, 2000 + i, common] for i in range(n)]


def test_bm25_rare_token_ranks_its_chunk_first():
    chunks = _corpus_with_common(10)
    chunks[6] = chunks[6] + [42]          # the only chunk with token 42
    index = BM25Index(chunks)
    ranked = index.search([7, 42], k=10)
    assert ranked[0] == 6


def test_bm25_idf_common_contributes_less_than_rare():
    chunks = _corpus_with_common(10)
    chunks[3] = chunks[3] + [42]
    index = BM25Index(chunks)
    assert index.idf(7) < index.idf(42)
    assert index.idf(7) > 0                # Lucene-style idf never goes negative


def test_bm25_idf_beats_raw_term_frequency():
    # Chunk 0 repeats the common token many times but lacks the rare one; chunk 1
    # has only the rare token. Without idf chunk 0's tf wins (about 1.95 vs 1.24);
    # with idf the common token is worth almost nothing (about 0.29 vs 2.47).
    chunks = _corpus_with_common(10)
    chunks[0] = [7] * 40
    chunks[1] = [42, 3000, 3001, 3002]
    index = BM25Index(chunks)
    assert index.search([7, 42], k=2)[0] == 1


def test_bm25_only_returns_matching_chunks():
    index = BM25Index([[1, 2], [3, 4], [5, 6]])
    assert index.search([4], k=10) == [1]
    assert index.search([99], k=10) == []


def test_bm25_patch_matches_a_fresh_build():
    chunks = _corpus_with_common(8)
    index = BM25Index(chunks)
    patched = [list(c) for c in chunks]
    patched[5] = patched[5] + [42, 43, 43]
    index.patch(5, patched[5])
    fresh = BM25Index(patched)
    q = [7, 42, 43, 1003]
    assert np.allclose(index.scores(q), fresh.scores(q))
    # and patching back restores the original exactly
    index.patch(5, chunks[5])
    assert np.allclose(index.scores(q), BM25Index(chunks).scores(q))


# ----------------------------------------------------------------------- dense


class FakeEmbedder:
    """Looks each text up in a fixed table: no model, no download. Unknown texts
    get [0, 0, 1]."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.passage_calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        return self.table.get(text, [0.0, 0.0, 1.0])

    def embed_passages(self, texts):
        self.passage_calls.append(list(texts))
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_queries(self, texts):
        return np.array([self._vec(t) for t in texts], dtype=np.float32)


def test_dense_cosine_top_k_ignores_vector_length():
    emb = FakeEmbedder({
        "a": [1.0, 0.0, 0.0],
        "b": [100.0, 100.0, 0.0],    # large norm, 45 degrees off the query
        "c": [0.9, 0.1, 0.0],
        "q": [1.0, 0.0, 0.0],
    })
    index = DenseIndex(["a", "b", "c"], emb)
    assert index.search("q", k=2) == [0, 2]
    assert index.search("q", k=10) == [0, 2, 1]


def test_dense_patch_reembeds_only_that_chunk():
    emb = FakeEmbedder({"a": [1, 0, 0], "b": [0, 1, 0], "needle": [0.1, 0.0, 1.0],
                        "q": [0.0, 0.0, 1.0], "zzz": [0.0, 1.0, 0.2]})
    index = DenseIndex(["a", "b", "zzz"], emb)
    assert index.search("q", k=1) == [2]
    emb.passage_calls.clear()
    index.patch(0, "needle")
    assert emb.passage_calls == [["needle"]]
    assert index.search("q", k=2) == [0, 2]
    index.patch(0, "a")
    assert index.search("q", k=1) == [2]


def test_dense_rejects_bad_patch_index():
    index = DenseIndex(["a"], FakeEmbedder({}))
    with pytest.raises(IndexError):
        index.patch(3, "x")


# ---------------------------------------------------------------------- fusion


def test_rrf_orders_by_summed_reciprocal_rank():
    bm25 = [3, 1, 2]
    dense = [1, 4, 3]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    # 1: 1/62 + 1/61 ; 3: 1/61 + 1/63 ; 4: 1/62 ; 2: 1/63
    assert fused == [1, 3, 4, 2]


def test_rrf_single_list_is_identity_and_empty_is_fine():
    assert reciprocal_rank_fusion([[5, 2, 9]]) == [5, 2, 9]
    assert reciprocal_rank_fusion([[], [4]]) == [4]
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_duplicates_within_a_ranking():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1, 1]])


# ---------------------------------------------------------------------- window


def _chunks():
    # chunk i is five copies of 100+i, so every token says where it came from
    return [[100 + i] * 5 for i in range(6)]


def test_window_prompt_is_last_and_intact():
    prompt = [1, 2, 3]
    packed = pack_window([4, 0], _chunks(), prompt, budget=50, separator=[9])
    assert packed.tokens[-3:] == (1, 2, 3)
    assert packed.tokens[-4] == 9          # separator before the prompt


def test_window_chunks_in_document_order_not_rank_order():
    packed = pack_window([4, 0, 2], _chunks(), [1], budget=50, separator=[9])
    assert packed.chunk_ids == (0, 2, 4)
    body = [t for t in packed.tokens if t >= 100]
    assert body == [100] * 5 + [102] * 5 + [104] * 5


def test_window_adjacent_chunks_need_no_separator():
    packed = pack_window([3, 2], _chunks(), [1], budget=50, separator=[9])
    assert packed.tokens == tuple([102] * 5 + [103] * 5 + [9, 1])


def test_window_never_exceeds_budget_and_truncates_by_rank():
    # budget 13 fits two non-adjacent chunks (5+1+5) + sep + prompt(1) = 13
    packed = pack_window([5, 1, 3], _chunks(), [1], budget=13, separator=[9])
    assert packed.chunk_ids == (1, 5)     # rank 3 (chunk 3) dropped, not rank 1/2
    assert len(packed.tokens) <= 13
    for budget in range(2, 40):
        p = pack_window([5, 1, 3, 0, 2], _chunks(), [1, 1], budget=budget, separator=[9])
        assert len(p.tokens) <= budget
        assert p.tokens[-2:] == (1, 1)


def test_window_stops_at_first_chunk_that_does_not_fit():
    chunks = [[100] * 10, [101] * 2, [102] * 2]
    packed = pack_window([1, 0, 2], chunks, [1], budget=6, separator=[9])
    # rank 2 (chunk 0, 10 tokens) does not fit, so rank 3 is not considered either
    assert packed.chunk_ids == (1,)


def test_window_prompt_too_long_is_an_error():
    with pytest.raises(ValueError):
        pack_window([0], _chunks(), [1] * 20, budget=10)


def test_window_rejects_unknown_or_duplicate_chunk_ids():
    with pytest.raises(ValueError):
        pack_window([0, 0], _chunks(), [1], budget=50)
    with pytest.raises(ValueError):
        pack_window([17], _chunks(), [1], budget=50)
