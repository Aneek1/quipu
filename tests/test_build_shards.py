"""scripts/build_shards.py against fake in-memory streams; no network."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quipu.data import read_shard

_spec = importlib.util.spec_from_file_location(
    "build_shards", Path(__file__).resolve().parents[1] / "scripts" / "build_shards.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

EOT = 0


class FakeTok:
    """Document "d<i>:<n>" encodes to n ids unique to document i, so any token
    other than EOT identifies the document it came from."""
    eot = EOT

    def encode(self, text):
        i, n = text[1:].split(":")
        base = 1 + int(i) * 100
        return list(range(base, base + int(n)))


def docs(count, length=9):
    return [{"text": f"d{i}:{length}"} for i in range(count)]


class CountingIter:
    def __init__(self, rows):
        self._it = iter(rows)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        row = next(self._it)
        self.pulled += 1
        return row


def split_tokens(d):
    return np.concatenate([read_shard(p) for p in sorted(Path(d).glob("shard_*.bin"))])


def test_exact_count_names_and_short_last_shard(tmp_path):
    out = tmp_path / "train"
    res = bs.build(out, 250, 100, iter(docs(40)), FakeTok())
    names = sorted(p.name for p in out.glob("shard_*.bin"))
    assert names == ["shard_000.bin", "shard_001.bin", "shard_002.bin"]
    assert [len(read_shard(out / n)) for n in names] == [100, 100, 50]
    assert res["tokens"] == 250
    assert [s["tokens"] for s in res["shards"]] == [100, 100, 50]


def test_tokens_are_the_documents_in_order(tmp_path):
    out = tmp_path / "val"
    bs.build(out, 25, 100, iter(docs(5)), FakeTok())
    expected = []
    for i in range(3):
        expected += list(range(1 + i * 100, 10 + i * 100)) + [EOT]
    assert split_tokens(out).tolist() == expected[:25]


def test_exact_multiple_has_no_empty_tail_shard(tmp_path):
    out = tmp_path / "train"
    bs.build(out, 200, 100, iter(docs(40)), FakeTok())
    assert [len(read_shard(p)) for p in sorted(out.glob("shard_*.bin"))] == [100, 100]


def test_stops_consuming_once_target_reached(tmp_path):
    it = CountingIter(docs(100))  # 10 tokens per doc including EOT
    res = bs.build(tmp_path / "v", 35, 100, it, FakeTok())
    assert it.pulled == 4  # the 4th doc completes the target mid-document
    assert res["rows_consumed"] == 4


def test_blank_documents_are_skipped(tmp_path):
    rows = [{"text": ""}, {"text": "   \n"}, {"text": None}, *docs(3)]
    res = bs.build(tmp_path / "v", 30, 100, iter(rows), FakeTok())
    assert res["documents"] == 3 and res["rows_consumed"] == 6
    assert split_tokens(tmp_path / "v").tolist().count(EOT) == 3


def test_raises_if_stream_runs_out_before_target(tmp_path):
    with pytest.raises(RuntimeError, match="ran out"):
        bs.build(tmp_path / "v", 1000, 100, iter(docs(5)), FakeTok())


def test_stale_shards_and_tmp_files_are_removed(tmp_path):
    out = tmp_path / "train"
    bs.build(out, 500, 100, iter(docs(60)), FakeTok())
    (out / "shard_009.bin.tmp").write_bytes(b"xx")
    bs.build(out, 150, 100, iter(docs(60)), FakeTok())
    assert sorted(p.name for p in out.iterdir()) == ["shard_000.bin", "shard_001.bin"]
    assert len(split_tokens(out)) == 150


def _run_all(root, stream, val=95, train=300, shard=100):
    return bs.build_all(root, stream, val_tokens=val, train_tokens=train,
                        shard_tokens=shard, tok=FakeTok(), dataset="fake/ds",
                        subset="tiny", revision="abc123")


def test_val_and_train_share_no_document(tmp_path):
    # A plain list restarts on every iter(), like a streaming HF dataset. val's
    # target (95) ends mid-document, so that document's tail must not leak.
    _run_all(tmp_path, docs(100))
    val = set(split_tokens(tmp_path / "val").tolist()) - {EOT}
    train = set(split_tokens(tmp_path / "train").tolist()) - {EOT}
    val_docs = {(t - 1) // 100 for t in val}
    train_docs = {(t - 1) // 100 for t in train}
    assert val_docs == set(range(10))
    assert not val_docs & train_docs
    assert min(train_docs) == 10  # train continues right where val stopped


def test_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text('{"stale": true}')
    m = _run_all(tmp_path, docs(100))
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == m
    assert m["dataset"] == "fake/ds" and m["subset"] == "tiny"
    assert m["dataset_revision"] == "abc123"
    assert m["tokenizer"] == "gpt2" and m["eot"] == EOT
    assert m["splits"]["val"]["tokens"] == 95
    assert m["splits"]["train"]["tokens"] == 300
    assert m["total_tokens"] == 395
    assert [s["file"] for s in m["splits"]["train"]["shards"]] == [
        "shard_000.bin", "shard_001.bin", "shard_002.bin"]
    assert m["splits"]["val"]["rows_consumed"] == 10
    assert m["splits"]["train"]["rows_consumed"] == 30
    assert m["build_started"] and m["build_finished"]
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_revision_parsing():
    class Ex:
        kwargs = {"files": ["hf://datasets/o/r@" + "a" * 40 + "/x/000.parquet"]}

    class S:
        _ex_iterable = Ex()

    assert bs.dataset_revision(S()) == "a" * 40
    assert bs.dataset_revision(object()) is None


class WideTok:
    """Like FakeTok but with room for documents up to 999 tokens; ids stay < 65536."""
    eot = EOT

    def encode(self, text):
        i, n = text[1:].split(":")
        base = 1 + int(i) * 1000
        return list(range(base, base + int(n)))


def _expected(lens, target):
    out = []
    for i, n in enumerate(lens):
        out += list(range(1 + i * 1000, 1 + i * 1000 + n)) + [EOT]
    return out[:target]


@pytest.mark.parametrize("lens,target,shard", [
    ([250, 3, 7, 400, 1, 60], 537, 64),  # docs longer than a shard, straddles, ragged target
    ([5] * 50, 299, 7),
    ([999], 500, 100),
    ([1] * 60, 120, 120),                 # target == shard size
    ([33, 2, 90], 10, 3),                 # tiny shards
])
def test_content_survives_shard_boundaries(tmp_path, lens, target, shard):
    rows = [{"text": f"d{i}:{n}"} for i, n in enumerate(lens)]
    res = bs.build(tmp_path / "s", target, shard, iter(rows), WideTok())
    assert split_tokens(tmp_path / "s").tolist() == _expected(lens, target)
    sizes = [s["tokens"] for s in res["shards"]]
    assert all(s == shard for s in sizes[:-1]) and sum(sizes) == target
