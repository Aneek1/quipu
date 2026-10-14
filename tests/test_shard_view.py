"""quipu.shard_view: the trainer's shard layout built from links into the data v3 pool,
with the pool's tokenizer checked against the one the trainer will load."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from quipu.data import write_shard
from quipu.loader import TokenStream
from quipu.shard_view import ShardViewError, make_shard_view
from quipu.train import UsageError, check_shard_tokenizer

SHA = "ab" * 32


def make_pool(root: Path, n: int = 3, tokens: int = 1000, sha: str = SHA, **manifest) -> Path:
    mixed = root / "mixed"
    shards = []
    rng = np.random.default_rng(0)
    for i in range(n):
        write_shard(mixed / f"shard_{i:05d}.bin", rng.integers(0, 500, tokens, dtype=np.uint16))
        shards.append({"file": f"shard_{i:05d}.bin", "chunk": 0, "tokens": tokens})
    (mixed / "shard_00000.ends").write_bytes(b"\0" * 8)          # not a shard: never linked
    m = {"format": "data v3 pool (quipu/pool.py)", "version": 1, "dtype": "uint16 little-endian",
         "shard_dir": "mixed", "shards": shards, "total_tokens": n * tokens,
         "provenance": {"tokenizer": {"path": "artifacts/tokenizer/tokenizer.json",
                                      "sha256": sha, "vocab_size": 49152, "eot": 0}},
         **manifest}
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return root


def make_val(root: Path, tokens: int = 300) -> Path:
    write_shard(root / "shard_000.bin", np.arange(tokens, dtype=np.uint16))
    return root


def test_the_view_has_the_trainers_layout_and_reads_the_pool_in_order(tmp_path):
    pool, val = make_pool(tmp_path / "pool"), make_val(tmp_path / "val")
    m = make_shard_view(tmp_path / "view", pool, val, SHA)
    view = tmp_path / "view"
    assert sorted(p.name for p in (view / "train").iterdir()) == [
        "shard_00000.bin", "shard_00001.bin", "shard_00002.bin"]
    assert [p.name for p in (view / "val").iterdir()] == ["shard_000.bin"]
    assert m["train"]["tokens"] == 3000 and m["val"]["tokens"] == 300
    assert m["links"] and set(m["links"]) <= {"symlink", "hardlink"}
    on_disk = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["tokenizer"]["sha256"] == SHA and on_disk["eot"] == 0
    # what the trainer reads through the view is the pool's tokens, shard by shard
    s = TokenStream(view / "train", 1, 999)
    want = np.concatenate([np.fromfile(pool / "mixed" / f"shard_{i:05d}.bin", dtype="<u2")
                           for i in range(3)])
    x, _ = s.next_batch()
    assert np.array_equal(x.numpy().ravel(), want[:999].astype(np.int64))
    s.close()


def test_the_pool_is_never_written(tmp_path):
    pool, val = make_pool(tmp_path / "pool"), make_val(tmp_path / "val")
    before = {p: p.read_bytes() for p in pool.rglob("*") if p.is_file()}
    make_shard_view(tmp_path / "view", pool, val, SHA)
    make_shard_view(tmp_path / "view", pool, val, SHA)          # rebuilt from scratch
    assert {p: p.read_bytes() for p in pool.rglob("*") if p.is_file()} == before


def test_the_trainers_tokenizer_check_passes_only_for_the_pools_tokenizer(tmp_path):
    tok = tmp_path / "tokenizer.json"
    tok.write_text('{"model": "fake"}', encoding="utf-8")
    sha = hashlib.sha256(tok.read_bytes()).hexdigest()
    pool, val = make_pool(tmp_path / "pool", sha=sha), make_val(tmp_path / "val")
    make_shard_view(tmp_path / "view", pool, val, sha)
    check_shard_tokenizer(tmp_path / "view", str(tok))           # no error
    other = tmp_path / "other.json"
    other.write_text('{"model": "other"}', encoding="utf-8")
    with pytest.raises(UsageError, match="another tokenizer"):
        check_shard_tokenizer(tmp_path / "view", str(other))


def test_a_pool_built_with_another_tokenizer_is_refused(tmp_path):
    pool, val = make_pool(tmp_path / "pool", sha="cd" * 32), make_val(tmp_path / "val")
    with pytest.raises(ShardViewError, match="tokenizer sha256"):
        make_shard_view(tmp_path / "view", pool, val, SHA)
    assert not (tmp_path / "view").exists()


@pytest.mark.parametrize("break_it, match", [
    (lambda pool, val: (pool / "mixed" / "shard_00001.bin").unlink(), "missing"),
    (lambda pool, val: (val / "shard_000.bin").unlink(), "no validation"),
    (lambda pool, val: (pool / "manifest.json").unlink(), "no readable pool manifest"),
    (lambda pool, val: (pool / "manifest.json").write_text(
        json.dumps({"format": "data v2", "provenance": {}}), encoding="utf-8"), "format"),
])
def test_a_broken_pool_or_val_is_refused(tmp_path, break_it, match):
    pool, val = make_pool(tmp_path / "pool"), make_val(tmp_path / "val")
    break_it(pool, val)
    with pytest.raises(ShardViewError, match=match):
        make_shard_view(tmp_path / "view", pool, val, SHA)


def test_a_truncated_shard_is_refused(tmp_path):
    pool, val = make_pool(tmp_path / "pool"), make_val(tmp_path / "val")
    with open(pool / "mixed" / "shard_00002.bin", "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="odd byte count"):
        make_shard_view(tmp_path / "view", pool, val, SHA)


def test_max_train_shards_keeps_the_first_shards(tmp_path):
    pool, val = make_pool(tmp_path / "pool", n=4), make_val(tmp_path / "val")
    m = make_shard_view(tmp_path / "view", pool, val, SHA, max_train_shards=2)
    assert sorted(p.name for p in (tmp_path / "view" / "train").iterdir()) == [
        "shard_00000.bin", "shard_00001.bin"]
    assert m["train"]["last"] == "shard_00001.bin"
