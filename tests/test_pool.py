"""Data v3 pool (quipu/pool.py): config, chunk allocation with the 4-epoch rule, the
per-source builder (dedup, decontamination, resume) and the mixer. No network: the
sources are in-memory rows and the tokenizer gives one token per character."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from quipu import pool

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data-v3.toml"


# ------------------------------------------------------------------ config

def test_the_v3_config_has_the_spec_shares_and_pins_every_source():
    cfg = pool.load_pool_config(CONFIG)
    by_slice = {}
    for s in cfg.sources.values():
        by_slice[s.slice] = by_slice.get(s.slice, 0.0) + s.share
    assert {k: round(v * 100, 6) for k, v in by_slice.items()} == {
        "code": 45, "code_old": 5, "code_web": 3, "english": 20, "qa": 5, "maths": 7,
        "reference": 4, "languages": 10, "verified": 1}
    assert sum(s.share for s in cfg.sources.values()) == pytest.approx(1.0)
    for s in cfg.sources.values():
        if s.kind.startswith("hf_") or s.kind == "http_jsonl":
            assert re.fullmatch(r"[0-9a-f]{40}", s.opts["revision"]), s.name
    # the code mix inside starcoderdata follows quipu-moe's language weights
    sc = {s.opts["language"]: s.share for s in cfg.sources.values()
          if s.name.startswith("starcoder-")}
    assert len(sc) == 14
    assert sc["Python"] == pytest.approx(0.45 * 0.325)
    assert sc["Dockerfile"] == pytest.approx(0.45 * 0.003)
    # Urdu: native script at 1/3 of the Urdu share; each of the nine languages 1/9
    u = cfg.sources["fw2-urd_Arab"].share
    assert u == pytest.approx(0.10 / 9 / 3)
    assert cfg.sources["fw2-urd_Latn"].share == pytest.approx(2 * u)
    assert cfg.sources["fw2-jpn_Jpan"].share == pytest.approx(0.10 / 9)
    assert cfg.chunk_tokens == 10_000_000_000 and cfg.shard_tokens == 100_000_000
    assert cfg.max_epochs == 4


def _write(tmp_path, text):
    p = tmp_path / "pool.toml"
    p.write_text(text, encoding="utf-8")
    return p


BASE = """
[pool]
tokenizer = "t.json"
shard_tokens = 100
chunk_tokens = 1000
max_epochs = 4
humaneval_revision = "%s"
mbpp_revision = "%s"
stackexchange_cutoff = "2024-04-30"
[slices]
a = 60
b = 40
[sources.x]
kind = "memory"
slice = "a"
weight = 1
[sources.y]
kind = "memory"
slice = "b"
weight = 3
[sources.z]
kind = "memory"
slice = "b"
weight = 1
""" % ("0" * 40, "1" * 40)


def test_shares_are_slice_share_times_weight_within_the_slice(tmp_path):
    cfg = pool.load_pool_config(_write(tmp_path, BASE))
    assert cfg.shares() == pytest.approx({"x": 0.6, "y": 0.3, "z": 0.1})


def test_slices_must_sum_to_100(tmp_path):
    with pytest.raises(ValueError, match="sum to 100"):
        pool.load_pool_config(_write(tmp_path, BASE.replace("b = 40", "b = 30")))


def test_an_unpinned_hub_source_is_refused(tmp_path):
    text = BASE + '[sources.h]\nkind = "hf_parquet"\nslice = "a"\nweight = 1\nrepo = "r/r"\n' \
                  'revision = "main"\nfiles = ["*.parquet"]\n'
    with pytest.raises(ValueError, match="40-hex"):
        pool.load_pool_config(_write(tmp_path, text))


def test_a_stack_exchange_dump_after_the_cutoff_is_refused(tmp_path):
    text = BASE + ('[sources.se]\nkind = "http_jsonl"\nslice = "a"\nweight = 1\n'
                   'url = "https://example/x.jsonl"\nrevision = "%s"\ndump_date = "2024-06"\n'
                   'last_modified = "2024-07-01"\nrender = "stackexchange_redpajama"\n'
                   % ("2" * 40))
    with pytest.raises(ValueError, match="after the Stack Exchange cutoff"):
        pool.load_pool_config(_write(tmp_path, text))
    ok = text.replace('"2024-06"', '"2024-04"').replace('"2024-07-01"', '"2024-04-11"')
    assert pool.load_pool_config(_write(tmp_path, ok)).sources["se"].opts["dump_date"] == "2024-04"


def test_defaults_tables_fill_a_source(tmp_path):
    text = BASE.replace('[sources.z]\nkind = "memory"', '[defaults.d]\nkind = "memory"\n'
                        'licence = "L"\n[sources.z]\ndefaults = "d"')
    cfg = pool.load_pool_config(_write(tmp_path, text))
    assert cfg.sources["z"].kind == "memory" and cfg.sources["z"].opts["licence"] == "L"


# ------------------------------------------------------------------ allocation

def test_allocation_follows_the_shares_when_nothing_is_exhausted():
    want, short = pool.chunk_allocation(1000, {"a": 0.5, "b": 0.3, "c": 0.2}, {})
    assert want == {"a": 500, "b": 300, "c": 200} and short == 0


def test_allocation_is_integer_and_sums_to_the_total():
    want, short = pool.chunk_allocation(1001, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {})
    assert sum(want.values()) == 1001 and short == 0
    assert all(isinstance(v, int) for v in want.values())


def test_an_exhausted_source_repeats_up_to_max_epochs_then_its_share_moves_on():
    # c has 10 unique tokens, none used yet: 4 epochs = 40 tokens available
    budget = pool.epoch_budget(unique=10, consumed=0, max_epochs=4)
    assert budget == 40
    want, short = pool.chunk_allocation(1000, {"a": 0.5, "b": 0.3, "c": 0.2}, {"c": budget})
    assert want["c"] == 40
    # the other 160 tokens of c's share go to a and b in proportion 5:3
    assert want["a"] == 600 and want["b"] == 360 and short == 0


def test_epoch_budget_counts_what_earlier_chunks_used():
    assert pool.epoch_budget(unique=10, consumed=25, max_epochs=4) == 15
    assert pool.epoch_budget(unique=10, consumed=40, max_epochs=4) == 0
    assert pool.epoch_budget(unique=10, consumed=55, max_epochs=4) == 0


def test_when_every_source_is_exhausted_the_shortfall_is_reported():
    want, short = pool.chunk_allocation(100, {"a": 0.5, "b": 0.5}, {"a": 10, "b": 20})
    assert want == {"a": 10, "b": 20} and short == 70


def test_build_targets_ask_each_live_source_for_what_it_still_lacks():
    shares = {"a": 0.5, "b": 0.3, "c": 0.2}
    states = {"a": pool.SourceTotals(unique=120, consumed=20, exhausted=False),
              "b": pool.SourceTotals(unique=0, consumed=0, exhausted=False),
              "c": pool.SourceTotals(unique=10, consumed=0, exhausted=True)}
    targets = pool.build_targets(1000, shares, states, max_epochs=4)
    # a wants 600 and has 100 unused; b wants 360; c is exhausted (nothing to build)
    assert targets == {"a": 500, "b": 360}


# ------------------------------------------------------------------ per-source builder

EOT = 0
OFFSET = 7


class CharTok:
    """One token per character (id = code point + OFFSET), EOT 0."""
    eot = EOT
    vocab_size = 65536

    def encode(self, text):
        return [ord(c) + OFFSET for c in text]


def char_tok():
    return CharTok()


class FakeDecontam:
    """A document containing "LEAK" holds HumanEval/0."""

    def find(self, text):
        return ("humaneval", "HumanEval/0", "substring") if "LEAK" in text else None

    def fingerprint(self):
        return "fake"


class FakeLid:
    """"@label rest" -> label (prob 0.99); untagged -> the source language."""

    def __init__(self, default):
        self.default = default

    def predict(self, text):
        label = text[1:].split(" ", 1)[0] if text.startswith("@") else self.default
        return label, 0.99, {label: 0.99}


def memory_rows(texts, metas=None):
    """open_rows over a list: rows from cursor["row"] on; cursor = the next row."""
    def open_rows(cursor):
        start = (cursor or {}).get("row", 0)
        for i in range(start, len(texts)):
            meta = dict(metas[i]) if metas else {"i": i}
            yield pool.Row(texts[i], meta, {"row": i + 1})
    return open_rows


def decode_parts(src_dir):
    """Every document of a source's parts, in order, as text."""
    st = pool.SourceState.load(src_dir)
    docs = []
    for part in st.parts:
        arr = np.fromfile(src_dir / part["file"], dtype="<u2")
        cur = []
        for t in arr.tolist():
            if t == EOT:
                docs.append("".join(chr(x - OFFSET) for x in cur))
                cur = []
            else:
                cur.append(t)
        assert not cur, "a part ends mid-document"
    return docs


def setup(**kw):
    return pool.BuildSetup(tokenizer=char_tok, decontam=kw.get("decontam", FakeDecontam()),
                           lid=kw.get("lid"), max_doc_tokens=kw.get("max_doc_tokens", 10_000))


def spec(name="s", **opts):
    return pool.SourceSpec(name, "memory", "a", 1.0, 1.0, opts)


TEXTS = [f"doc {i} " + "x" * (10 + i % 7) for i in range(40)]


def test_build_writes_unique_documents_until_the_target(tmp_path):
    out = tmp_path / "s"
    texts = TEXTS[:10] + [TEXTS[3], "   ", "has a LEAK inside it"] + TEXTS[10:]
    res = pool.build_source(out, spec(), 400, setup=setup(), open_rows=memory_rows(texts),
                            shard_tokens=10_000)
    docs = decode_parts(out)
    assert docs == TEXTS[:len(docs)]           # in order, the duplicate / blank / leak gone
    st = pool.SourceState.load(out)
    assert st.unique_tokens >= 400 and st.unique_tokens - len(TEXTS[len(docs) - 1]) - 1 < 400
    assert st.stats["dropped_duplicate"] == 1 and st.stats["dropped_blank"] == 1
    assert st.stats["dropped_contamination"] == 1
    assert st.stats["dropped_contamination_as"]["humaneval"] == 1
    assert res["tokens"] == st.unique_tokens and not st.exhausted
    assert st.documents == len(docs)


def test_a_second_build_continues_the_stream_where_the_first_stopped(tmp_path):
    out = tmp_path / "s"
    pool.build_source(out, spec(), 100, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=10_000)
    first = decode_parts(out)
    pool.build_source(out, spec(), 100, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=10_000)
    both = decode_parts(out)
    assert both[:len(first)] == first and both == TEXTS[:len(both)]
    assert len(both) > len(first)
    st = pool.SourceState.load(out)
    assert [p["file"] for p in st.parts] == ["part_00000.bin", "part_00001.bin"]
    assert len(st.builds) == 2 and st.stats.get("dropped_duplicate", 0) == 0


def test_dedup_survives_a_resume(tmp_path):
    out = tmp_path / "s"
    texts = TEXTS[:5] + TEXTS[:5] + TEXTS[5:10]
    pool.build_source(out, spec(), 40, setup=setup(), open_rows=memory_rows(texts),
                      shard_tokens=10_000)
    pool.build_source(out, spec(), 10_000, setup=setup(), open_rows=memory_rows(texts),
                      shard_tokens=10_000)
    assert decode_parts(out) == TEXTS[:10]
    assert pool.SourceState.load(out).stats["dropped_duplicate"] == 5


def test_a_stream_that_ends_marks_the_source_exhausted(tmp_path):
    out = tmp_path / "s"
    res = pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=memory_rows(TEXTS),
                            shard_tokens=10_000)
    st = pool.SourceState.load(out)
    assert st.exhausted and res["exhausted"] and decode_parts(out) == TEXTS
    again = pool.build_source(out, spec(), 100, setup=setup(), open_rows=memory_rows(TEXTS),
                              shard_tokens=10_000)
    assert again["tokens"] == 0 and len(pool.SourceState.load(out).parts) == 1


def test_parts_are_cut_at_document_ends_near_shard_tokens(tmp_path):
    out = tmp_path / "s"
    pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=100)
    st = pool.SourceState.load(out)
    assert len(st.parts) > 3
    for p in st.parts[:-1]:
        assert 100 <= p["tokens"] < 100 + 30
    assert sum(p["tokens"] for p in st.parts) == st.unique_tokens
    assert decode_parts(out) == TEXTS


def test_a_killed_build_resumes_from_its_last_checkpoint(tmp_path):
    out = tmp_path / "s"

    def dying(cursor):
        for row in memory_rows(TEXTS)(cursor):
            if row.cursor["row"] == 25:
                raise KeyboardInterrupt
            yield row

    with pytest.raises(KeyboardInterrupt):
        pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=dying, shard_tokens=100)
    # junk a crash could leave: a part that was never recorded, unsynced hashes
    (out / "part_00099.bin").write_bytes(b"\x01\x00")
    with open(out / "hashes.u64", "ab") as f:
        f.write(b"\x00" * 16)
    pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=100)
    assert decode_parts(out) == TEXTS
    assert not (out / "part_00099.bin").exists()
    st = pool.SourceState.load(out)
    assert st.stats.get("dropped_duplicate", 0) == 0 and st.documents == len(TEXTS)


def test_a_changed_source_identity_refuses_to_resume(tmp_path):
    out = tmp_path / "s"
    pool.build_source(out, spec(), 50, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=10_000, identity={"revision": "a" * 40})
    with pytest.raises(pool.ResumeError, match="revision"):
        pool.build_source(out, spec(), 50, setup=setup(), open_rows=memory_rows(TEXTS),
                          shard_tokens=10_000, identity={"revision": "b" * 40})


def test_adapter_drops_lid_and_code_filters_are_counted(tmp_path):
    out = tmp_path / "s"
    texts = ["@jpn_Jpan kana text here", "plain ind text", "keep this one too",
             "minified one", "toolong " * 50, None]
    metas = [{}, {}, {}, {"path": "dist/app.min.js"}, {}, {"_drop": "dropped_license"}]
    s = spec(lid="ind_Latn", code=True, path_field="path")
    pool.build_source(out, s, 10**9, setup=setup(lid=FakeLid("ind_Latn"), max_doc_tokens=100),
                      open_rows=memory_rows(texts, metas), shard_tokens=10_000)
    st = pool.SourceState.load(out)
    assert decode_parts(out) == ["plain ind text", "keep this one too"]
    assert st.stats["dropped_lid"] == 1 and st.stats["dropped_lid_as"] == {"jpn_Jpan": 1}
    assert st.stats["skipped_minified"] == 1 and st.stats["dropped_too_long"] == 1
    assert st.stats["dropped_license"] == 1 and st.stats["rows"] == 6


def test_provenance_is_kept_per_document(tmp_path):
    import gzip
    out = tmp_path / "s"
    metas = [{"url": f"https://x/{i}", "_attrib": "CC BY-SA"} for i in range(len(TEXTS))]
    pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=memory_rows(TEXTS, metas),
                      shard_tokens=10_000)
    st = pool.SourceState.load(out)
    lines = gzip.decompress((out / st.parts[0]["prov"]).read_bytes()).decode().splitlines()
    assert [json.loads(x)["url"] for x in lines] == [m["url"] for m in metas]
    assert st.attribution == {"CC BY-SA": len(TEXTS)}


def test_limit_docs_stops_early_without_exhausting(tmp_path):
    out = tmp_path / "s"
    pool.build_source(out, spec(), 10**9, setup=setup(), open_rows=memory_rows(TEXTS),
                      shard_tokens=10_000, limit_docs=7)
    st = pool.SourceState.load(out)
    assert st.documents == 7 and not st.exhausted


def test_html_cap_holds_html_under_its_share(tmp_path):
    out = tmp_path / "s"
    texts = [f"{'<html>' if i % 2 else 'code'} {i} " + "y" * 20 for i in range(60)]
    metas = [{"language": "HTML" if i % 2 else "Python"} for i in range(60)]
    pool.build_source(out, spec(html_cap=0.2, html_warmup=0), 10**9, setup=setup(),
                      open_rows=memory_rows(texts, metas), shard_tokens=10_000)
    st = pool.SourceState.load(out)
    assert st.stats["html_tokens"] <= 0.2 * st.unique_tokens
    assert st.stats["skipped_html_cap"] > 0


def test_worker_processes_give_the_same_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    texts = TEXTS + [TEXTS[1], "LEAK here"] + [t + "!" for t in TEXTS]
    for out, workers in ((a, 0), (b, 2)):
        pool.build_source(out, spec(), 600, setup=setup(), open_rows=memory_rows(texts),
                          shard_tokens=200, workers=workers, batch_docs=8)
    sa, sb = pool.SourceState.load(a), pool.SourceState.load(b)
    assert sa.cursor == sb.cursor and sa.stats == sb.stats
    assert [(a / p["file"]).read_bytes() for p in sa.parts] == \
        [(b / p["file"]).read_bytes() for p in sb.parts]
