"""scripts/needle_eval.py: CPU only, a tiny random-init Quipu, tiny fake haystacks
written with quipu.data.write_shard, and a fake embedder (no model download).

A random model almost never copies the value, so these tests check planting,
retrieval bookkeeping, scoring arithmetic and output structure, not accuracy.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import ModelConfig
from quipu.data import write_shard
from quipu.memory.chunker import chunk_spans
from quipu.model import Quipu
from quipu.tokenizer import Tokenizer

_SPEC = importlib.util.spec_from_file_location(
    "needle_eval", Path(__file__).resolve().parent.parent / "scripts" / "needle_eval.py"
)
needle_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["needle_eval"] = needle_eval
_SPEC.loader.exec_module(needle_eval)

VOCAB = 50257


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer()


def _find(hay: list[int], needle: list[int]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(hay) - n + 1) if hay[i:i + n] == needle]


# ------------------------------------------------------------------ planting


@pytest.mark.parametrize("kind", ["text", "code"])
@pytest.mark.parametrize("depth", [0, 25, 50, 75, 100])
def test_needle_planted_once_at_depth_and_value_recoverable(tok, kind, depth):
    rng = np.random.RandomState(0)
    base = rng.randint(1000, 1100, 400).astype(np.uint16)   # ids that can't spell the needle
    needle = needle_eval.make_needle(kind, random.Random(5))
    ids = needle_eval.needle_token_ids(needle, tok)
    planted, pos = needle_eval.plant(base, ids, depth)

    assert pos == round(depth / 100 * len(base))
    assert len(planted) == len(base) + len(ids)
    assert _find(planted.tolist(), ids) == [pos]
    assert needle.value in tok.decode(planted[pos:pos + len(ids)].tolist())
    # the haystack either side is untouched
    assert planted[:pos].tolist() == base[:pos].tolist()
    assert planted[pos + len(ids):].tolist() == base[pos:].tolist()


def test_needle_shapes(tok):
    t = needle_eval.make_needle("text", random.Random(1))
    assert t.text == f"The access code for the {t.adjective} vault is {t.value}."
    assert t.prompt == f"The access code for the {t.adjective} vault is"
    c = needle_eval.make_needle("code", random.Random(1))
    assert c.text == f"{c.adjective.upper()}_VAULT_CODE = {c.value}"
    assert c.prompt == f"{c.adjective.upper()}_VAULT_CODE ="
    for n in (t, c):
        assert len(n.value) == 5 and 10000 <= int(n.value) <= 99999
        assert n.adjective in needle_eval.ADJECTIVES
        ids = needle_eval.needle_token_ids(n, tok)
        assert ids[0] == ids[-1] == tok.encode("\n")[0]
        assert len(ids) <= needle_eval.NEEDLE_RESERVE


def test_needles_are_seeded_and_reproducible():
    a = [needle_eval.make_needle("text", needle_eval.trial_rng(7, "text", 1000, 50, t)) for t in range(20)]
    b = [needle_eval.make_needle("text", needle_eval.trial_rng(7, "text", 1000, 50, t)) for t in range(20)]
    assert a == b
    assert len({n.value for n in a}) > 1                    # trials differ from each other
    c = needle_eval.make_needle("text", needle_eval.trial_rng(8, "text", 1000, 50, 0))
    assert c != a[0] or c.value != a[0].value               # the seed matters


def test_planted_chunks_rejoin_into_the_planted_document(tok):
    base = np.arange(1000, 1100, dtype=np.uint16)            # 100 tokens
    ids = needle_eval.needle_token_ids(needle_eval.make_needle("code", random.Random(2)), tok)
    for depth in (0, 25, 50, 75, 100):
        planted, pos = needle_eval.plant(base, ids, depth)
        spans = chunk_spans(len(base), 16)
        ci = needle_eval.needle_chunk_index(pos, spans)
        chunk = needle_eval.planted_chunk(base, spans[ci], pos, ids)
        chunks = [base[s.start:s.end].tolist() for s in spans]
        chunks[ci] = chunk
        assert sum(chunks, []) == planted.tolist()
        # only that one chunk changed
        assert _find(chunk, ids) != []
    assert needle_eval.needle_chunk_index(100, chunk_spans(100, 16)) == 6   # depth 100: last chunk


# ------------------------------------------------------------------- scoring


def _rec(cond, hit, passed, depth=0):
    return {"condition": cond, "hit": hit, "pass": passed, "depth": depth,
            "latency_ms": 1.0}


def test_aggregate_separates_hit_rate_from_copy_accuracy():
    records = [
        _rec("bm25", True, True, 0),
        _rec("bm25", True, False, 0),
        _rec("bm25", False, False, 50),
        _rec("bm25", False, False, 50),
    ]
    agg = needle_eval.aggregate(records)["bm25"]
    assert agg["trials"] == 4
    assert agg["hit_rate"] == 0.5
    assert agg["copy_given_hit"] == 0.5
    assert agg["accuracy"] == 0.25
    assert agg["by_depth"]["0"]["accuracy"] == 0.5
    assert agg["by_depth"]["50"]["hit_rate"] == 0.0


def test_aggregate_copy_given_hit_is_none_without_hits_and_counts_lucky_passes():
    agg = needle_eval.aggregate([_rec("off", False, False), _rec("off", False, True)])["off"]
    assert agg["hit_rate"] == 0.0
    assert agg["copy_given_hit"] is None
    assert agg["accuracy"] == 0.5
    assert agg["passes_without_hit"] == 1


def test_parse_size():
    assert needle_eval.parse_size("1k") == 1_000
    assert needle_eval.parse_size("128k") == 128_000
    assert needle_eval.parse_size("1M") == 1_000_000
    assert needle_eval.parse_size("4096") == 4096
    assert needle_eval.format_size(32_000) == "32k"
    assert needle_eval.format_size(1_000_000) == "1M"
    with pytest.raises(ValueError):
        needle_eval.parse_size("-3k")


# ---------------------------------------------------------------- end to end


class HashEmbedder:
    """Bag of words hashed into 64 dims: deterministic, no network."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for w in text.split():
            v[sum(w.encode()) % 64] += 1.0
        v[0] += 1e-3
        return v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


def _tiny_model() -> Quipu:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=16, n_layer=1, n_head=2, n_kv_head=1,
                      ffn_hidden=32, context=128, rope_base=10000.0, norm_eps=1e-6)
    return Quipu(cfg).eval()


def _write_haystacks(shard_dir: Path) -> None:
    rng = np.random.RandomState(0)
    write_shard(shard_dir / "val" / "shard_000.bin", rng.randint(0, 50000, 300).astype(np.uint16))
    write_shard(shard_dir / "code_val" / "shard_000.bin", rng.randint(0, 50000, 300).astype(np.uint16))


def test_end_to_end_outputs_and_off_hit_inside_context(tmp_path, tok):
    shard_dir = tmp_path / "shards"
    _write_haystacks(shard_dir)
    haystacks = {k: needle_eval.load_haystack(shard_dir, k) for k in ("text", "code")}
    out = tmp_path / "needle"
    needle_eval.run(
        model=_tiny_model(), tok=tok, haystacks=haystacks, sizes=[64, 200],
        depths=[0, 50, 100], trials=2, conditions=list(needle_eval.CONDITIONS),
        device="cpu", out_dir=out, seed=3, embedder_factory=HashEmbedder, chunk_tokens=16,
    )
    for kind in ("text", "code"):
        for size in ("64", "200"):
            data = json.loads((out / f"{kind}_{size}.json").read_text(encoding="utf-8"))
            assert data["haystack"] == kind
            assert set(data["aggregates"]) == set(needle_eval.CONDITIONS)
            assert len(data["trials"]) == 3 * 2 * len(needle_eval.CONDITIONS)
            for cond, agg in data["aggregates"].items():
                assert agg["trials"] == 6
                assert 0.0 <= agg["hit_rate"] <= 1.0
                assert set(agg["by_depth"]) == {"0", "50", "100"}
                assert "latency_ms_median" in agg and "peak_vram_mb" in agg
            for rec in data["trials"]:
                assert rec["window_tokens"] <= 128 - needle_eval.MAX_NEW_TOKENS
                assert len(rec["value"]) == 5
            assert data["index_build_s"]["bm25"] >= 0
            assert data["index_build_s"]["dense"] >= 0

    small = json.loads((out / "text_64.json").read_text(encoding="utf-8"))
    # 64 tokens fit in the 120-token window: memory off sees every needle.
    assert small["aggregates"]["off"]["hit_rate"] == 1.0
    # and so does memory on, since every chunk fits too
    assert small["aggregates"]["bm25"]["hit_rate"] == 1.0

    big = json.loads((out / "code_200.json").read_text(encoding="utf-8"))
    off0 = [r for r in big["trials"] if r["condition"] == "off" and r["depth"] == 0]
    assert off0 and not any(r["hit"] for r in off0)          # depth 0 falls off the left
    off100 = [r for r in big["trials"] if r["condition"] == "off" and r["depth"] == 100]
    assert all(r["hit"] for r in off100)
    # BM25 finds the needle: its identifier tokens are nowhere else in random ids
    assert big["aggregates"]["bm25"]["hit_rate"] == 1.0

    summary = (out / "summary.md").read_text(encoding="utf-8")
    for cond in needle_eval.CONDITIONS:
        assert f"| {cond} |" in summary
    assert "| code | 200 |" in summary


def test_needle_hit_uses_needle_chunk_not_value(tmp_path, tok):
    """A trial is a hit iff the needle's chunk is in the window; the record says which
    chunk that was and where it ranked."""
    shard_dir = tmp_path / "shards"
    _write_haystacks(shard_dir)
    haystacks = {"text": needle_eval.load_haystack(shard_dir, "text")}
    out = tmp_path / "needle"
    needle_eval.run(
        model=_tiny_model(), tok=tok, haystacks=haystacks, sizes=[200], depths=[50],
        trials=1, conditions=["bm25"], device="cpu", out_dir=out, seed=0,
        embedder_factory=None, chunk_tokens=16,
    )
    rec = json.loads((out / "text_200.json").read_text(encoding="utf-8"))["trials"][0]
    assert rec["hit"] == (rec["needle_chunk"] in rec["window_chunks"])
    assert rec["needle_rank"] == 0


def test_dense_condition_without_embedder_is_an_error(tmp_path, tok):
    with pytest.raises(ValueError):
        needle_eval.run(
            model=_tiny_model(), tok=tok, haystacks={"text": np.zeros(300, np.uint16)},
            sizes=[64], depths=[0], trials=1, conditions=["dense"], device="cpu",
            out_dir=tmp_path, seed=0, embedder_factory=None, chunk_tokens=16,
        )


def test_haystack_too_short_is_an_error(tmp_path, tok):
    with pytest.raises(ValueError):
        needle_eval.run(
            model=_tiny_model(), tok=tok, haystacks={"text": np.zeros(50, np.uint16)},
            sizes=[64], depths=[0], trials=1, conditions=["off"], device="cpu",
            out_dir=tmp_path, seed=0, embedder_factory=None, chunk_tokens=16,
        )


_TOML = """
name = "tiny"
[model]
vocab_size = 50257
d_model = 16
n_layer = 1
n_head = 2
n_kv_head = 1
ffn_hidden = 32
context = 128
rope_base = 10000.0
norm_eps = 1e-6
[data]
dataset = "x"
subset = "x"
shard_dir = "{shard_dir}"
shard_tokens = 1000
val_tokens = 1000
code_dataset = "x"
code_share = 0.1
code_languages = ["Python"]
code_licenses = ["mit"]
html_cap = 0.1
code_val_tokens = 1000
code_heldout_first_file = 1
code_files_total = 2
[train]
total_tokens = 2560
batch_tokens = 256
micro_batch = 2
lr = 1e-3
lr_min = 1e-4
warmup_steps = 2
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
seed = 7
ckpt_dir = "{ckpt_dir}"
ckpt_every = 1000
ckpt_keep = 3
eval_every = 1000
eval_batches = 1
"""


def test_main_cli_with_latest_pointer(tmp_path):
    shard_dir = tmp_path / "shards"
    ckpt_dir = tmp_path / "ckpt"
    _write_haystacks(shard_dir)
    ckpt_dir.mkdir()
    torch.save({"step": 1, "model": _tiny_model().state_dict()}, ckpt_dir / "step_000001.pt")
    torch.save({"file": "step_000001.pt"}, ckpt_dir / "latest.pt")
    cfg_path = tmp_path / "tiny.toml"
    cfg_path.write_text(
        _TOML.format(shard_dir=shard_dir.as_posix(), ckpt_dir=ckpt_dir.as_posix()),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    code = needle_eval.main([
        "--config", str(cfg_path), "--sizes", "64", "--depths", "0,100", "--trials", "1",
        "--conditions", "off,bm25", "--device", "cpu", "--out", str(out),
        "--chunk-tokens", "16",
    ])
    assert code == 0
    assert (out / "text_64.json").is_file() and (out / "code_64.json").is_file()
    assert (out / "summary.md").is_file()
    assert not list(out.glob("*.tmp"))


# ------------------------------------------- one newline before the prompt


@pytest.mark.parametrize("kind", ["text", "code"])
def test_off_window_has_exactly_one_newline_before_prompt_at_depth_100(tok, kind):
    base = np.arange(1000, 1400, dtype=np.uint16)
    needle = needle_eval.make_needle(kind, random.Random(4))
    ids = needle_eval.needle_token_ids(needle, tok)
    planted, pos = needle_eval.plant(base, ids, 100)
    prompt = tok.encode(needle.prompt)
    window, hit = needle_eval.off_window(planted, pos, prompt, 120, tok.encode("\n"),
                                         needle_eval.newline_predicate(tok))
    assert hit
    assert window[-len(prompt):] == prompt
    before = tok.decode(window[:-len(prompt)])
    assert before.endswith(needle.text + "\n")
    assert not before.endswith("\n\n")
    # and a tail that does not end in a newline still gets exactly one
    window, _ = needle_eval.off_window(base.copy(), 0, prompt, 120, tok.encode("\n"),
                                       needle_eval.newline_predicate(tok))
    assert window[-len(prompt) - 1] == tok.encode("\n")[0]


def test_packed_needle_window_has_one_newline_before_prompt(tok):
    base = np.arange(1000, 1100, dtype=np.uint16)
    needle = needle_eval.make_needle("code", random.Random(9))
    ids = needle_eval.needle_token_ids(needle, tok)
    planted, pos = needle_eval.plant(base, ids, 100)
    spans = chunk_spans(len(base), 16)
    ci = needle_eval.needle_chunk_index(pos, spans)
    chunks = [base[s.start:s.end].tolist() for s in spans]
    chunks[ci] = needle_eval.planted_chunk(base, spans[ci], pos, ids)
    prompt = tok.encode(needle.prompt)
    from quipu.memory.window import pack_window
    packed = pack_window([ci], chunks, prompt, 120, separator=tok.encode("\n"),
                         ends_with_break=needle_eval.newline_predicate(tok))
    text = tok.decode(list(packed.tokens))
    assert text.endswith(needle.text + "\n" + needle.prompt)


def test_newline_predicate(tok):
    pred = needle_eval.newline_predicate(tok)
    assert pred(tok.encode("\n")[0])
    assert pred(tok.encode("\n\n")[0])
    assert not pred(tok.encode(" vault")[0])


# ---------------------------------------------------------- order option


def test_condition_labels():
    assert needle_eval.condition_labels(["off", "bm25", "fused"], ["document"]) == [
        ("off", "document", "off"), ("bm25", "document", "bm25"), ("fused", "document", "fused")]
    assert needle_eval.condition_labels(["off", "bm25"], ["document", "rank"]) == [
        ("off", "document", "off"), ("bm25", "document", "bm25"), ("bm25", "rank", "bm25+rank")]
    assert needle_eval.condition_labels(["bm25"], ["rank"]) == [("bm25", "rank", "bm25+rank")]


def test_end_to_end_with_both_orders(tmp_path, tok):
    shard_dir = tmp_path / "shards"
    _write_haystacks(shard_dir)
    haystacks = {"code": needle_eval.load_haystack(shard_dir, "code")}
    out = tmp_path / "needle"
    needle_eval.run(
        model=_tiny_model(), tok=tok, haystacks=haystacks, sizes=[200], depths=[0, 100],
        trials=1, conditions=["off", "bm25", "dense"], device="cpu", out_dir=out, seed=0,
        embedder_factory=HashEmbedder, chunk_tokens=16, orders=["document", "rank"],
    )
    data = json.loads((out / "code_200.json").read_text(encoding="utf-8"))
    assert list(data["aggregates"]) == ["off", "bm25", "bm25+rank", "dense", "dense+rank"]
    ranked = [r for r in data["trials"] if r["condition"] == "bm25+rank"]
    assert all(r["window_chunks"][-1] == r["needle_chunk"] for r in ranked)   # BM25 rank 0
    for rec in data["trials"]:
        assert rec["window_tokens"] <= 128 - needle_eval.MAX_NEW_TOKENS
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| bm25+rank |" in summary


# -------------------------------------------------- review follow-ups


class BatchSensitiveEmbedder(HashEmbedder):
    """Output depends on batch composition, as a padded GPU batch does in practice:
    re-embedding one text alone never reproduces the row it got inside a batch."""

    def embed_passages(self, texts):
        return super().embed_passages(texts) + 1e-4 * len(texts)


class FailingPatchEmbedder(HashEmbedder):
    """Builds fine (many texts at once) but fails whenever a single chunk is patched."""

    def embed_passages(self, texts):
        if len(texts) == 1:
            raise RuntimeError("embedder exploded mid-trial")
        return super().embed_passages(texts)


def _bm25_state(bm25):
    return ({t: dict(p) for t, p in bm25._postings.items()}, bm25._len.copy().tolist(),
            [dict(tf) for tf in bm25._tf])


def _prepared(tok, embedder, size=200, chunk_tokens=16, conditions=("bm25", "dense")):
    rng = np.random.RandomState(0)
    hay = rng.randint(0, 50000, 300).astype(np.uint16)
    base, spans, base_chunks = needle_eval.prepare_base(hay, size, chunk_tokens)
    indexes = needle_eval.build_indexes(base_chunks, tok, list(conditions), embedder)
    return hay, indexes


def test_run_one_leaves_both_indexes_bit_identical(tok):
    hay, indexes = _prepared(tok, BatchSensitiveEmbedder())
    dense_before = indexes.dense.matrix()
    bm25_before = _bm25_state(indexes.bm25)
    needle_eval.run_one(
        model=_tiny_model(), tok=tok, kind="code", haystack=hay, size=200,
        depths=[0, 50, 100], trials=2, conditions=list(needle_eval.CONDITIONS),
        device="cpu", seed=0, embedder=indexes.dense.embedder, chunk_tokens=16,
        log=lambda m: None, indexes=indexes,
    )
    assert np.array_equal(indexes.dense.matrix(), dense_before)
    assert _bm25_state(indexes.bm25) == bm25_before


def test_failed_patch_leaves_no_index_patched(tok):
    hay, indexes = _prepared(tok, FailingPatchEmbedder())
    dense_before = indexes.dense.matrix()
    bm25_before = _bm25_state(indexes.bm25)
    with pytest.raises(RuntimeError, match="exploded"):
        needle_eval.run_one(
            model=_tiny_model(), tok=tok, kind="code", haystack=hay, size=200,
            depths=[50], trials=1, conditions=["bm25", "dense"], device="cpu", seed=0,
            embedder=indexes.dense.embedder, chunk_tokens=16, log=lambda m: None,
            indexes=indexes,
        )
    assert _bm25_state(indexes.bm25) == bm25_before
    assert np.array_equal(indexes.dense.matrix(), dense_before)


def test_needle_on_a_chunk_boundary_and_at_the_end(tok):
    base = np.arange(1000, 1064, dtype=np.uint16)            # 64 tokens = 4 chunks of 16
    spans = chunk_spans(len(base), 16)
    ids = needle_eval.needle_token_ids(needle_eval.make_needle("text", random.Random(3)), tok)
    for depth, want_chunk in ((25, 1), (50, 2), (75, 3), (100, 3), (0, 0)):
        planted, pos = needle_eval.plant(base, ids, depth)
        assert pos % 16 == 0
        ci = needle_eval.needle_chunk_index(pos, spans)
        assert ci == want_chunk
        chunks = [base[s.start:s.end].tolist() for s in spans]
        chunks[ci] = needle_eval.planted_chunk(base, spans[ci], pos, ids)
        assert sum(chunks, []) == planted.tolist()
    with pytest.raises(ValueError):
        needle_eval.needle_chunk_index(65, spans)
    with pytest.raises(ValueError):
        needle_eval.needle_chunk_index(-1, spans)


class NeedleLastEmbedder:
    """Dense retrieval that is wrong on purpose: any passage mentioning VAULT points
    away from the query, so the needle chunk ranks last."""

    def _vec(self, text: str) -> np.ndarray:
        if "VAULT" in text and not text.startswith("__query__"):
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        h = (sum(text.encode()) % 97) / 1000.0
        return np.array([1.0, 0.0, h], dtype=np.float32)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._vec("__query__" + t) for t in texts])


def test_fused_ranking_is_the_rrf_of_the_recorded_rankings(tmp_path, tok):
    from quipu.memory.fusion import reciprocal_rank_fusion
    shard_dir = tmp_path / "shards"
    _write_haystacks(shard_dir)
    out = tmp_path / "needle"
    needle_eval.run(
        model=_tiny_model(), tok=tok,
        haystacks={"code": needle_eval.load_haystack(shard_dir, "code")},
        sizes=[200], depths=[0, 50, 100], trials=2, conditions=["bm25", "dense", "fused"],
        device="cpu", out_dir=out, seed=0, embedder_factory=NeedleLastEmbedder,
        chunk_tokens=16,
    )
    data = json.loads((out / "code_200.json").read_text(encoding="utf-8"))
    by = {}
    for r in data["trials"]:
        by.setdefault((r["depth"], r["trial"]), {})[r["condition"]] = r
    n_chunks = data["chunks"]
    for trial in by.values():
        assert trial["bm25"]["needle_rank"] == 0
        assert trial["dense"]["needle_rank"] == n_chunks - 1
        expected = reciprocal_rank_fusion([trial["bm25"]["ranking"], trial["dense"]["ranking"]])
        assert trial["fused"]["ranking"] == expected
        assert trial["fused"]["needle_rank"] == expected.index(trial["fused"]["needle_chunk"])


class ReportingEmbedder(HashEmbedder):
    model_name = "fake/e5"
    revision = "abc123"

    def __init__(self):
        self.truncated = 0

    def embed_passages(self, texts):
        self.truncated += 1
        return super().embed_passages(texts)


def test_embedder_identity_and_truncations_are_recorded(tmp_path, tok):
    shard_dir = tmp_path / "shards"
    _write_haystacks(shard_dir)
    out = tmp_path / "needle"
    needle_eval.run(
        model=_tiny_model(), tok=tok,
        haystacks={"text": needle_eval.load_haystack(shard_dir, "text")},
        sizes=[64], depths=[0], trials=1, conditions=["dense"], device="cpu", out_dir=out,
        seed=0, embedder_factory=ReportingEmbedder, chunk_tokens=16,
    )
    info = json.loads((out / "text_64.json").read_text(encoding="utf-8"))["embedder"]
    assert info["model"] == "fake/e5" and info["revision"] == "abc123"
    assert info["truncated_passages"] >= 1


def test_e5_embed_empty_input_returns_empty_matrix():
    from quipu.memory.dense import E5Embedder
    e = E5Embedder.__new__(E5Embedder)       # no model load: the empty path never needs it
    e.dim = 384
    assert e._embed([]).shape == (0, 384)


def test_e5_revision_is_pinned():
    from quipu.memory import dense
    assert len(dense.E5_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in dense.E5_REVISION)


def _fake_result(**over):
    d = {"haystack": "text", "size": 1000, "size_label": "1k", "chunk_tokens": 256,
         "window_budget": 1016, "seed": 1, "trials_per_depth": 2,
         "aggregates": {"off": {"trials": 2, "hit_rate": 1.0, "copy_given_hit": 0.5,
                                "accuracy": 0.5, "by_depth": {"0": {"accuracy": 0.5}},
                                "latency_ms_median": None}}}
    d.update(over)
    return d


def test_summary_warns_on_mixed_settings_and_logs_skipped_files(tmp_path):
    (tmp_path / "text_1k.json").write_text(json.dumps(_fake_result()), encoding="utf-8")
    (tmp_path / "code_1k.json").write_text(
        json.dumps(_fake_result(haystack="code", chunk_tokens=128, seed=2)), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    logged = []
    needle_eval.write_summary(tmp_path, log=logged.append)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "mixed settings" in summary.lower()
    assert "chunk_tokens" in summary and "128" in summary
    assert any("broken.json" in m for m in logged)


def test_summary_single_settings_has_no_warning(tmp_path):
    (tmp_path / "text_1k.json").write_text(json.dumps(_fake_result()), encoding="utf-8")
    needle_eval.write_summary(tmp_path, log=lambda m: None)
    assert "mixed settings" not in (tmp_path / "summary.md").read_text(encoding="utf-8").lower()
