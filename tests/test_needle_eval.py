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
