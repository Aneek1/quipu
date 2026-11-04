"""scripts/ocr/build_text.py: cutting lines, random lines, the held-out split, sampling."""
import importlib.util
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_text", ROOT / "scripts" / "ocr" / "build_text.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_cut_lines_respects_lengths_and_word_boundaries():
    doc = "The quick brown fox jumps over the lazy dog. " * 5
    lines = bt.cut_lines(doc, random.Random(0), min_chars=5, max_chars=20)
    assert lines
    for s in lines:
        assert 5 <= len(s) <= 20
        assert s == s.strip()
        assert all(w in doc for w in s.split())


def test_cut_lines_drops_characters_outside_the_set():
    lines = bt.cut_lines("café crème naïve plain words here", random.Random(0), 5, 60)
    assert all("é" not in s and "ï" not in s for s in lines)


def test_random_line_is_in_charset_and_bounded():
    from quipu.ocr_ctc import in_charset
    rng = random.Random(1)
    for _ in range(500):
        s = bt.random_line(rng, max_chars=40)
        assert 5 <= len(s) <= 40 and in_charset(s) and s == s.strip()


def test_is_held_is_stable_and_about_one_in_twenty():
    held = [bt.is_held(f"shard_000.bin:{i}") for i in range(20000)]
    assert held == [bt.is_held(f"shard_000.bin:{i}") for i in range(20000)]
    assert 0.03 < sum(held) / len(held) < 0.07


def test_build_writes_disjoint_lf_files(tmp_path):
    docs = [(f"s:{i}", f"document number {i} has some ordinary words in it for lines")
            for i in range(400)]
    bt.build(docs, tmp_path, random_frac=0.15, seed=0)
    train = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    held = (tmp_path / "held.txt").read_text(encoding="utf-8").splitlines()
    assert train and held
    assert not {s for s in train if len(s) >= 12} & set(held)
    raw = (tmp_path / "train.txt").read_bytes() + (tmp_path / "held.txt").read_bytes()
    assert b"\r" not in raw


def test_build_keeps_short_lines_that_also_occur_in_held(tmp_path):
    docs = [(f"s:{i}", "return") for i in range(300)]
    docs += [(f"t:{i}", "return the long value") for i in range(300)]
    bt.build(docs, tmp_path, random_frac=0.0, seed=0)
    train = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    held = (tmp_path / "held.txt").read_text(encoding="utf-8").splitlines()
    assert "return" in held and "return" in train            # short: not leakage
    assert "return the long value" in held and "return the long value" not in train


def test_random_frac_must_be_below_one():
    for bad in ("1.0", "-0.1", "2"):
        with pytest.raises(SystemExit):
            bt.main(["--random-frac", bad, "--out", "unused"])


class _FakeTok:
    eot = 0

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def _fake_shards(tmp_path, monkeypatch, n_shards, docs_per_shard):
    import numpy as np
    import quipu.data
    names = [f"shard_{k:03d}.bin" for k in range(n_shards)]
    for n in names:
        (tmp_path / n).write_bytes(b"")
    one = np.array([1, 2, 3, 0, 0], dtype="<u2")      # a document, EOT, then an empty one
    arrays = {n: np.tile(one, docs_per_shard) for n in names}
    monkeypatch.setattr(quipu.data, "read_shard", lambda path: arrays[Path(path).name])


def test_shard_documents_splits_on_eot_and_skips_empty_docs(tmp_path, monkeypatch):
    _fake_shards(tmp_path, monkeypatch, n_shards=2, docs_per_shard=10)
    docs = list(bt.shard_documents(tmp_path, "x", tok=_FakeTok()))
    assert len(docs) == 20
    assert all(text == "w1 w2 w3" for _, text in docs)
    assert {k.split(":")[0] for k, _ in docs} == {"shard_000.bin", "shard_001.bin"}
    assert len({k for k, _ in docs}) == 20


def test_shard_documents_max_shards_and_random_order(tmp_path, monkeypatch):
    _fake_shards(tmp_path, monkeypatch, n_shards=4, docs_per_shard=5)
    docs = list(bt.shard_documents(tmp_path, "x", max_shards=1, tok=_FakeTok()))
    assert len({k.split(":")[0] for k, _ in docs}) == 1
    firsts = {next(bt.shard_documents(tmp_path, "x", seed=s, tok=_FakeTok()))[0].split(":")[0]
              for s in range(12)}
    assert len(firsts) > 1                                   # not always shard_000


def test_shard_documents_keeps_about_doc_keep_with_stable_keys(tmp_path, monkeypatch):
    _fake_shards(tmp_path, monkeypatch, n_shards=4, docs_per_shard=500)
    kept = list(bt.shard_documents(tmp_path, "x", seed=1, doc_keep=0.12, tok=_FakeTok()))
    assert 0.08 < len(kept) / 2000 < 0.16
    every = {k for k, _ in bt.shard_documents(tmp_path, "x", seed=1, doc_keep=1.0, tok=_FakeTok())}
    assert {k for k, _ in kept} <= every and len(every) == 2000


def test_main_writes_a_manifest(tmp_path, monkeypatch):
    import hashlib
    import json
    import quipu.tokenizer
    shards = tmp_path / "shards"
    shards.mkdir()
    _fake_shards(shards, monkeypatch, n_shards=3, docs_per_shard=50)
    monkeypatch.setattr(quipu.tokenizer, "make_tokenizer", lambda path: _FakeTok())
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    out = tmp_path / "text"
    assert bt.main(["--shards", str(shards), "--tokenizer", str(tok), "--out", str(out),
                    "--doc-keep", "0.5", "--seed", "3"]) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["tokenizer_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert sorted(m["shards_read"]) == ["shard_000.bin", "shard_001.bin", "shard_002.bin"]
    assert (m["seed"], m["doc_keep"], m["random_frac"]) == (3, 0.5, 0.15)
    for name in ("train", "held"):
        f = out / f"{name}.txt"
        assert m[f"{name}_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert m[f"{name}_lines"] == len(f.read_text(encoding="utf-8").splitlines())
    assert m["shards"] and m["tokenizer"]
