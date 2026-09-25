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


# ---------------------------------------------------------------- code mix

TEXT_BASE, CODE_BASE = 1, 30_001


class MixTok:
    """"T<k>:<n>" -> n copies of 1+k; "C<k>:<n>" -> n copies of 30001+k.

    Every token other than EOT names its source and document, so a shard can be
    audited: equal content always yields equal tokens, distinct documents never do.
    """
    eot = EOT

    def encode(self, text):
        kind, rest = text[0], text[1:]
        k, n = rest.split(":")
        base = TEXT_BASE if kind == "T" else CODE_BASE
        assert int(k) < 30_000
        return [base + int(k)] * int(n)


def text_rows(count, seed=0):
    rng = np.random.default_rng(seed)
    return [{"text": f"T{i}:{int(n)}"} for i, n in enumerate(rng.integers(20, 300, count))]


KEPT_LANGS = ("Python", "JavaScript", "HTML", "CSS", "GO")
KEPT_LICENSES = ("mit", "apache-2.0")


def code_rows(count, *, seed=1, start=0, html_every=0, langs=("Python", "JavaScript", "CSS"),
              license="mit"):
    """Code rows C<start>..; every html_every-th row is HTML (0 = none)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, n in enumerate(rng.integers(20, 600, count)):
        lang = "HTML" if html_every and j % html_every == 0 else langs[j % len(langs)]
        rows.append({"code": f"C{start + j}:{int(n)}", "language": lang, "license": license})
    return rows


def spec(train_rows, val_rows, *, share=0.2, html_cap=0.1, warmup=2_000, val_tokens=20_000):
    return bs.CodeSpec(share=share, languages=KEPT_LANGS, licenses=KEPT_LICENSES,
                       html_cap=html_cap, val_tokens=val_tokens, train_rows=train_rows,
                       val_rows=lambda: iter(val_rows), heldout_first_file=8,
                       files_total=10, html_warmup=warmup)


def run_mix(root, code, *, text=None, val=5_000, train=400_000, shard=50_000):
    return bs.build_all(root, text if text is not None else text_rows(6_000), val_tokens=val,
                        train_tokens=train, shard_tokens=shard, tok=MixTok(),
                        dataset="fake/ds", subset="tiny", revision="abc", code=code)


def is_code(tokens):
    """Per-token source mask; an EOT belongs to the document it ends."""
    mask = tokens >= CODE_BASE
    eot = np.flatnonzero(tokens == EOT)
    eot = eot[eot > 0]
    mask[eot] = tokens[eot - 1] >= CODE_BASE
    return mask


def test_mixed_train_hits_the_code_share_throughout(tmp_path):
    m = run_mix(tmp_path, spec(code_rows(4_000), code_rows(200, start=20_000)),
                text=text_rows(20_000), train=2_000_000, shard=200_000)
    tr = m["splits"]["train"]
    assert tr["tokens"] == 2_000_000
    assert abs(tr["achieved_code_share"] - 0.2) <= 0.005
    assert m["code"]["achieved_shares"]["code"] == tr["achieved_code_share"]
    # The manifest's per-source counts are the shards' actual contents.
    tokens = split_tokens(tmp_path / "train")
    code_mask = is_code(tokens)
    assert tr["sources"]["code"]["tokens"] == int(code_mask.sum())
    assert tr["sources"]["text"]["tokens"] == len(tokens) - int(code_mask.sum())
    assert sum(v["tokens"] for v in tr["sources"]["code"]["by_language"].values()) \
        == tr["sources"]["code"]["tokens"]
    # Uniform across the run, not "all text then all code": every tenth is on target.
    for part in np.array_split(code_mask, 10):
        assert abs(part.mean() - 0.2) < 0.02


def test_mixed_build_keeps_the_shared_text_iterator_rule(tmp_path):
    run_mix(tmp_path, spec(code_rows(2_000), code_rows(100, start=20_000)), val=5_000)
    val = split_tokens(tmp_path / "val")
    assert not (val >= CODE_BASE).any()  # FineWeb val is text only
    val_docs = set(val.tolist()) - {EOT}
    train = split_tokens(tmp_path / "train")
    train_text = set(train[(train > EOT) & (train < CODE_BASE)].tolist())
    assert not val_docs & train_text
    assert min(train_text) == max(val_docs) + 1  # train text continues right after val


def test_html_cap_holds_in_train_and_code_val(tmp_path):
    # Natural HTML share here is ~1/3 of rows; the cap must pull it down to 10%.
    m = run_mix(tmp_path, spec(code_rows(4_000, html_every=3),
                               code_rows(1_500, start=20_000, seed=7, html_every=3),
                               val_tokens=100_000),
                text=text_rows(20_000), train=1_500_000, shard=200_000)
    for split in (m["splits"]["train"]["sources"]["code"], m["splits"]["code_val"]):
        assert split["skipped_html_cap"] > 0
        assert 0.09 <= split["html_share_of_code"] <= 0.1 + 0.002
        assert split["by_language"]["HTML"]["share_of_code"] == split["html_share_of_code"]
    assert m["code"]["html_achieved_share_of_code"] == \
        m["splits"]["train"]["sources"]["code"]["html_share_of_code"]


def test_html_is_admitted_during_the_warmup():
    rows = [{"code": f"C{i}:100", "language": "HTML", "license": "mit"} for i in range(30)]
    docs = bs.CodeDocs(iter(rows), MixTok(), languages=KEPT_LANGS, licenses=KEPT_LICENSES,
                       html_cap=0.1, html_warmup=1_000)
    got = list(docs)
    # 101 tokens each (with EOT): admitted until 1,000 code tokens, then capped.
    assert len(got) == 10 and docs.stats["skipped_html_cap"] == 20


def test_language_and_license_filters():
    rows = [
        {"code": "C0:5", "language": "Python", "license": "mit"},        # kept
        {"code": "C1:5", "language": "C", "license": "mit"},             # language
        {"code": "C2:5", "language": "Ruby", "license": "apache-2.0"},   # language
        {"code": "C3:5", "language": "go", "license": "mit"},            # case matters: GO
        {"code": "C4:5", "language": "GO", "license": "mit"},            # kept
        {"code": "C5:5", "language": "Python", "license": "gpl-3.0"},    # licence
        {"code": "C6:5", "language": "CSS", "license": None},            # licence
        {"code": "C7:5", "language": "JavaScript", "license": "MIT"},    # licence (case)
        {"code": "   ", "language": "Python", "license": "mit"},         # blank
        {"code": "C9:5", "language": "JavaScript", "license": "apache-2.0"},  # kept
    ]
    docs = bs.CodeDocs(iter(rows), MixTok(), languages=KEPT_LANGS, licenses=KEPT_LICENSES,
                       html_cap=0.1)
    got = [(int(a[0]) - CODE_BASE, label) for a, label in docs]
    assert got == [(0, "code:Python"), (4, "code:GO"), (9, "code:JavaScript")]
    assert docs.stats == {"rows_scanned": 10, "dropped_language": 3, "dropped_license": 3,
                          "skipped_blank": 1}


def test_filters_hold_end_to_end(tmp_path):
    rows = code_rows(6_000, langs=("Python", "C", "Ruby", "CSS"))
    for j, r in enumerate(rows):
        if j % 5 == 0:
            r["license"] = "gpl-3.0"
    m = run_mix(tmp_path, spec(rows, code_rows(300, start=20_000)))
    train = split_tokens(tmp_path / "train")
    written = {int(t) - CODE_BASE for t in train[train >= CODE_BASE]}
    by_id = {int(r["code"][1:].split(":")[0]): r for r in rows}
    assert written
    assert all(by_id[k]["language"] in KEPT_LANGS for k in written)
    assert all(by_id[k]["license"] in KEPT_LICENSES for k in written)
    code = m["splits"]["train"]["sources"]["code"]
    assert set(code["by_language"]) == {"Python", "CSS"}
    assert code["dropped_language"] > 0 and code["dropped_license"] > 0


def test_code_val_is_deduplicated_against_code_train(tmp_path):
    train_rows = code_rows(3_000)
    fresh = code_rows(150, start=20_000, seed=5)
    # Copies of train documents: early ones (certainly written to train) and late
    # ones (never reached by train, so they must NOT be dropped).
    early = [dict(r) for r in train_rows[:40]]
    late = [dict(r) for r in train_rows[-40:]]
    val_rows = []
    for i, r in enumerate(fresh):
        val_rows.append(r)
        if i < 40:
            val_rows += [early[i], late[i]]
    m = run_mix(tmp_path, spec(train_rows, val_rows, val_tokens=10_000_000))
    train = split_tokens(tmp_path / "train")
    train_code = set(train[train >= CODE_BASE].tolist())
    code_val = split_tokens(tmp_path / "code_val")
    val_code = set(code_val[code_val >= CODE_BASE].tolist())
    late_ids = {CODE_BASE + int(r["code"][1:].split(":")[0]) for r in late}
    assert late_ids.isdisjoint(train_code)  # the fixture's premise
    assert not val_code & train_code  # no train document in code val
    assert late_ids <= val_code  # unseen copies are kept
    cv = m["splits"]["code_val"]
    assert cv["dropped_as_duplicate"] == 40 == m["code"]["dropped_as_duplicate"]
    assert cv["short_of_target"] is True  # the fake held-out set is small
    assert cv["tokens"] == len(code_val) and cv["documents"] == 150 + 40


def test_code_val_stops_at_its_target(tmp_path):
    m = run_mix(tmp_path, spec(code_rows(3_000), code_rows(500, start=20_000),
                               val_tokens=12_345))
    cv = m["splits"]["code_val"]
    assert cv["tokens"] == 12_345 == len(split_tokens(tmp_path / "code_val"))
    assert cv["short_of_target"] is False


def _build_bytes(root):
    run_mix(root, spec(code_rows(3_000, html_every=4),
                       code_rows(400, start=20_000, html_every=4)))
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("shard_*.bin"))}


def test_interleave_is_deterministic(tmp_path):
    a, b = _build_bytes(tmp_path / "a"), _build_bytes(tmp_path / "b")
    assert a.keys() == b.keys() and any(k.startswith("code_val/") for k in a)
    assert a == b
    ma = json.loads((tmp_path / "a" / "manifest.json").read_text())
    mb = json.loads((tmp_path / "b" / "manifest.json").read_text())
    for m in (ma, mb):
        for key in ("build_started", "build_finished", "build_seconds"):
            del m[key]
    assert ma == mb


def test_interleave_picks_the_source_furthest_below_target():
    text = iter([(np.ones(10, np.uint16), "text")] * 100)
    code = iter([(np.ones(10, np.uint16), "code:Python")] * 100)
    labels = [label for _, (_, label) in zip(range(10), bs.interleave(text, code, 0.2))]
    # Ties go to text; code is taken whenever its deficit is the larger.
    assert labels == ["text", "code:Python", "text", "text", "text",
                      "text", "code:Python", "text", "text", "text"]


def test_a_code_stream_that_runs_out_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="code stream ran out"):
        run_mix(tmp_path, spec(code_rows(20), code_rows(10, start=20_000)))


def test_manifest_records_the_mix(tmp_path):
    m = run_mix(tmp_path, spec(code_rows(3_000), code_rows(400, start=20_000)))
    assert json.loads((tmp_path / "manifest.json").read_text()) == m
    c = m["code"]
    assert c["languages"] == list(KEPT_LANGS) and c["licenses"] == list(KEPT_LICENSES)
    assert c["train_files"] == [0, 7] and c["heldout_files"] == [8, 9]
    assert c["target_shares"] == {"text": 0.8, "code": 0.2}
    assert c["html_cap"] == 0.1 and c["html_cap_warmup_tokens"] == 2_000
    assert m["total_tokens"] == 5_000 + 400_000
    tr = m["splits"]["train"]
    assert tr["sources"]["text"]["tokens"] + tr["sources"]["code"]["tokens"] == 400_000
    assert tr["sources"]["text"]["documents"] + tr["sources"]["code"]["documents"] \
        == tr["documents"]
    assert [s["tokens"] for s in tr["shards"]] == [50_000] * 8
    for split in ("train", "val", "code_val"):
        files = sorted(p.name for p in (tmp_path / split).glob("shard_*.bin"))
        assert files == [s["file"] for s in m["splits"][split]["shards"]]


def test_text_only_build_clears_an_old_code_val(tmp_path):
    run_mix(tmp_path, spec(code_rows(3_000), code_rows(400, start=20_000)))
    assert list((tmp_path / "code_val").glob("shard_*.bin"))
    _run_all(tmp_path, docs(100))
    assert not list((tmp_path / "code_val").glob("shard_*.bin"))


def test_prefetch_preserves_order_and_raises_producer_errors():
    assert list(bs.flatten(bs.prefetch(bs.batched(range(1000), 7), depth=2))) \
        == list(range(1000))

    def boom():
        yield 1
        raise OSError("network")

    it = bs.prefetch(boom(), depth=4)
    assert next(it) == 1
    with pytest.raises(OSError, match="network"):
        next(it)


def test_lower_priority_sets_below_normal():
    calls = []

    class K32:
        def GetCurrentProcess(self):
            return "me"

        def SetPriorityClass(self, handle, cls):
            calls.append((handle, cls))
            return 1

    assert bs.lower_priority(K32()) is True
    assert calls == [("me", 0x4000)]
