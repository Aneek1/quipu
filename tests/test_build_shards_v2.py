"""Data v2 (quipu-moe) shard builder: scripts/build_shards.py build_mix and
quipu/shard_mix.py, against fake in-memory sources; no network.

The fake tokenizer gives one token per character (id = code point + OFFSET, EOT 0),
so token counts are character counts and every shard decodes back to its documents.
The fake language-ID classifier reads a tag at the start of a document
("@label:prob[,label:prob] ..."), so each test decides what the classifier says.
"""
import functools
import importlib.util
import json
import multiprocessing
import os
import random
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from quipu import shard_mix as sm
from quipu.data import read_shard

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_shards", ROOT / "scripts" / "build_shards.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

EOT = 0
OFFSET = 7
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"

CODE_W = {"Python": 0.30, "JavaScript": 0.25, "TypeScript": 0.12, "HTML": 0.08, "CSS": 0.05,
          "SQL": 0.05, "PHP": 0.03, "Java": 0.03, "GO": 0.03, "Shell": 0.02,
          "Dockerfile": 0.01, "C": 0.01, "C++": 0.01, "Rust": 0.01}
OTHER_LANGS = ("ind_Latn", "zsm_Latn", "cmn_Hani", "jpn_Jpan", "kor_Hang", "tam_Taml",
               "hin_Deva", "hin_Latn", "urd_Latn")
TEXT_W = {"eng_Latn": 0.7, **{x: 0.3 / len(OTHER_LANGS) for x in OTHER_LANGS}}
# How often each language turns up in the fake code stream (not its weight: Java is
# common and wanted little, Python the reverse, as in github-code-clean).
CODE_FREQ = {"Python": 0.12, "JavaScript": 0.2, "TypeScript": 0.1, "HTML": 0.1, "CSS": 0.06,
             "SQL": 0.06, "PHP": 0.06, "Java": 0.1, "GO": 0.05, "Shell": 0.04,
             "Dockerfile": 0.03, "C": 0.03, "C++": 0.03, "Rust": 0.02}


class CharTok:
    eot = EOT
    vocab_size = 65536

    def encode(self, text):
        return [ord(c) + OFFSET for c in text]


class FakeLid:
    """"@a:0.9,b:0.1 body" -> ("a", 0.9, {"a": 0.9, "b": 0.1}); untagged -> "other" 0.0."""

    def predict(self, text):
        if not text.startswith("@"):
            return "other", 0.0, {"other": 0.0}
        tag = text[1:].split(" ", 1)[0]
        probs = {}
        for part in tag.split(","):
            label, p = part.split(":")
            probs[label] = float(p)
        top = max(probs, key=probs.get)
        return top, probs[top], probs


def body(rng, n, alphabet="abcdefghij klmnopqrst\n"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def code_rows(n, seed=0, freq=CODE_FREQ, file_rows=500, start=0):
    rng = random.Random(seed)
    langs, probs = list(freq), list(freq.values())
    rows = []
    for i in range(start, start + n):
        lang = rng.choices(langs, probs)[0]
        r = rng.random()
        path = "node_modules/x/i.js" if r < 0.02 else f"src/m{i}.txt"
        rows.append({"code": f"<{lang} {i}> " + body(rng, rng.randint(40, 400)),
                     "language": lang, "license": "gpl-3.0" if r > 0.95 else "mit",
                     "path": path, "file": i // file_rows})
    return rows


def text_rows(lang, n, seed=0, tag=None, start=0):
    """Rows of `lang`; tag(i, rng) gives the LID tag of row i (None: untagged)."""
    rng = random.Random(f"{lang}{seed}")
    out = []
    for i in range(start, start + n):
        t = tag(i, rng) if tag else None
        head = f"@{t} " if t else ""
        out.append({"text": f"{head}<src={lang} {i}> " + body(rng, rng.randint(60, 400))})
    return out


def std_tag(lang):
    """Mostly its own label; every 7th a confident mislabel (dropped), every 11th an
    unsure mislabel (kept). cmn_Hani: a quarter Traditional."""
    def tag(i, rng):
        own = lang
        if lang == "cmn_Hani":
            own = "zho_Hant" if i % 4 == 0 else "zho_Hans"
        if i % 7 == 3:
            return "jpn_Jpan:0.92" if lang != "jpn_Jpan" else "kor_Hang:0.92"
        if i % 11 == 5:
            return ("eng_Latn:0.40,zho_Hant:0.35,zho_Hans:0.25" if lang == "cmn_Hani"
                    else "other:0.40," + own + ":0.35")
        return own + ":0.95"
    return tag


def from_file(rows):
    """A code_train source over a list: rows of files >= start (rows without "file"
    all belong to file 0), as the real source restarts at a file."""
    return lambda start=0: iter([r for r in rows if r.get("file", 0) >= start])


def text_source(rows):
    """A text_train source over a list: rows from its start_file-th file on (files in
    the order the rows give them; rows without "file" are one source, start 0 only)."""
    def src(start=0):
        if not start:
            return iter(rows)
        files = []
        for r in rows:
            if r["file"] not in files:
                files.append(r["file"])
        keep = set(files[start:])
        return iter([r for r in rows if r["file"] in keep])
    return src


def world(*, lid=False, code_n=12_000, code_freq=CODE_FREQ, text_n=None, seed=0,
          code_val_extra=()):
    tag = std_tag if lid else (lambda lang: None)
    text_n = text_n or {}
    code_train = code_rows(code_n, seed, code_freq)
    code_val = list(code_val_extra) + code_rows(300, seed + 1, CODE_FREQ, start=10**6)
    train = {"eng_Latn": text_rows("eng_Latn", text_n.get("eng_Latn", 1200), seed,
                                   tag("eng_Latn"))}
    val = {"eng_Latn": text_rows("eng_Latn", 200, seed + 99, tag("eng_Latn"), start=10**6)}
    for x in OTHER_LANGS:
        train[x] = text_rows(x, text_n.get(x, 200), seed, tag(x))
        val[x] = text_rows(x, 60, seed + 99, tag(x), start=10**6)
    return bs.MixSources(
        code_train=from_file(code_train),
        code_val=lambda: iter(code_val),
        text_train={k: text_source(v) for k, v in train.items()},
        text_val={k: (lambda v=v: iter(v)) for k, v in val.items()})


def spec(**kw):
    base = dict(train_tokens=300_000, val_tokens=5_000, shard_tokens=40_000, code_share=0.6,
                code_weights=CODE_W, text_weights=TEXT_W, licenses=("mit",), html_cap=0.08,
                max_doc_tokens=2_000, code_val_tokens=5_000, lang_val_tokens=1_000,
                slack=0.1, window=200, stall_windows=3, stall_gain=1e-4, workers=1,
                batch_docs=16)
    base.update(kw)
    return bs.MixSpec(**base)


def setup(lid=False, guard=None):
    return sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, guard=guard,
                       lid=FakeLid() if lid else None, lid_threshold=0.5)


def tokens_of(split_dir):
    paths = sorted(Path(split_dir).glob("shard_*.bin"))
    return np.concatenate([read_shard(p) for p in paths]) if paths else np.empty(0, np.uint16)


def decode_docs(split_dir):
    """The split's documents as text (the last one possibly truncated)."""
    toks = tokens_of(split_dir)
    docs, cur = [], []
    for t in toks.tolist():
        if t == EOT:
            docs.append("".join(chr(c - OFFSET) for c in cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        docs.append("".join(chr(c - OFFSET) for c in cur))
    return docs


def train_label_tokens(manifest):
    return {k: v["tokens"] for k, v in manifest["splits"]["train"]["by_label"].items()}


# ------------------------------------------------------------------ allocate

def test_allocate_proportional_when_nothing_binds():
    alloc, short = sm.allocate(100, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert alloc == pytest.approx({"a": 50, "b": 30, "c": 20})
    assert short == 0


def test_allocate_redistributes_a_short_bucket_by_weight():
    alloc, short = sm.allocate(100, {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 20})
    assert alloc["a"] == 20
    assert alloc["b"] == pytest.approx(30 + 30 * 0.3 / 0.5)
    assert alloc["c"] == pytest.approx(20 + 30 * 0.2 / 0.5)
    assert short == 0


def test_allocate_reapplies_the_cap_after_redistribution():
    # HTML is exactly at its cap before redistribution; Python's shortfall must not
    # push it over.
    w = {"Python": 0.5, "HTML": 0.1, "JS": 0.4}
    alloc, _ = sm.allocate(1000, w, {"Python": 100}, {"HTML": 0.1})
    assert alloc["HTML"] == pytest.approx(100)
    assert alloc["JS"] == pytest.approx(800)


def test_allocate_reports_shortfall():
    alloc, short = sm.allocate(100, {"a": 0.5, "b": 0.5}, {"a": 10, "b": 20})
    assert alloc == {"a": 10, "b": 20}
    assert short == pytest.approx(70)


# ------------------------------------------------------------------ collector

def test_collector_never_exhausts_a_slow_common_language():
    """Python turns up once per 60 offers and needs ~100 windows to fill; it keeps
    going (the tokenizer run's 6-window rule starved it). Dockerfile never appears
    and is exhausted after stall_windows windows."""
    col = sm.Collector({"Python": 0.5, "Java": 0.49, "Dockerfile": 0.01}, 20_000, slack=0.0,
                       window=50, stall_windows=3)
    n = 0
    while not col.done:
        lang = "Python" if n % 60 == 0 else "Java"
        if col.wants(lang):
            col.admit(lang, np.zeros(10, np.uint16))
        col.tick()
        n += 1
        assert n < 10**6
    assert "Python" not in col.exhausted
    assert list(col.exhausted) == ["Dockerfile"]
    assert col.windows > 60
    assert col.taken["Python"] >= col.base["Python"]
    assert col.base["Python"] == pytest.approx(20_000 * 0.5 / 0.99)


# ------------------------------------------------------------------ the build

@pytest.fixture(scope="module")
def full_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("mix")
    m = bs.build_mix(root, spec(), world(lid=True), setup(lid=True))
    return root, m


def test_code_language_shares_within_one_percent(full_build):
    _, m = full_build
    tokens = train_label_tokens(m)
    code = {k[len("code:"):]: v for k, v in tokens.items() if k.startswith("code:")}
    total = sum(code.values())
    for lang, w in CODE_W.items():
        assert abs(code[lang] / total - w) <= 0.01, (lang, code[lang] / total)
        assert m["code"]["by_language"][lang]["achieved_share"] == pytest.approx(
            code[lang] / total, abs=1e-6)
    assert m["code"]["html_achieved_share_of_code"] <= 0.08 + 0.005
    assert m["share_check"]["violations"] == []


def test_overall_mix_60_28_12_by_tokens(full_build):
    _, m = full_build
    tokens = train_label_tokens(m)
    total = sum(tokens.values())
    assert total == 300_000 == m["splits"]["train"]["tokens"]
    code = sum(v for k, v in tokens.items() if k.startswith("code:"))
    eng = tokens["text:eng_Latn"]
    other = total - code - eng
    assert abs(code / total - 0.60) <= 0.005
    assert abs(eng / total - 0.28) <= 0.005
    assert abs(other / total - 0.12) <= 0.005
    assert m["mix"]["achieved"]["eng_Latn"] == pytest.approx(eng / total, abs=1e-6)
    for x in OTHER_LANGS:
        got = sum(v for k, v in tokens.items()
                  if k.startswith("text:") and (k[5:] == x or (x == "cmn_Hani" and k[5:] in
                                                                sm.ZH_BUCKETS)))
        assert abs(got / total - 0.12 / 9) <= 0.002, (x, got / total)


def test_splits_and_manifest(full_build):
    root, m = full_build
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == m
    assert len(tokens_of(root / "train")) == 300_000
    assert len(tokens_of(root / "val")) == 5_000
    assert 0 < len(tokens_of(root / "code_val")) <= 5_000
    for b in ["ind_Latn", "zho_Hans", "zho_Hant", "urd_Latn"]:
        assert m["splits"]["val_lang"][b]["tokens"] > 0
        assert (root / "val_lang" / b / "shard_000.bin").is_file()
    assert not (root / bs.MIX_WORK).exists()
    assert m["eot"] == EOT


def test_held_out_rows_never_reach_train(full_build):
    root, _ = full_build
    train = decode_docs(root / "train")
    # Held-out rows are numbered from 10**6.
    assert not any(" 100000" in d.split(">", 1)[0] for d in train)
    for split in ["val", "code_val", "val_lang/ind_Latn"]:
        docs = decode_docs(root / split)
        assert docs and all(" 100" in d.split(">", 1)[0] for d in docs if ">" in d)


def test_lid_drops_confident_mislabels_and_keeps_unsure(full_build):
    root, m = full_build
    train = decode_docs(root / "train")
    # A decoded document starts with its tag.
    tags = [d.split(" ", 1)[0] for d in train if d.startswith("@")]
    assert tags and not any(t.endswith(":0.92") for t in tags)
    assert any(t.startswith("@other:0.40") for t in tags)  # unsure mislabel: kept
    for x in ("eng_Latn", "ind_Latn", "hin_Latn", "urd_Latn", "cmn_Hani"):
        s = m["text"]["by_language"][x]
        assert s["lid_dropped"] > 0 and s["lid_kept"] > 0
        assert s["lid_dropped_as"] == {"jpn_Jpan": s["lid_dropped"]}
        assert s["lid_kept"] + s["lid_dropped"] == s["offers"]


def test_lid_splits_chinese_into_equal_simplified_and_traditional(full_build):
    root, m = full_build
    tokens = train_label_tokens(m)
    hans, hant = tokens["text:zho_Hans"], tokens["text:zho_Hant"]
    assert abs(hans - hant) <= 500  # within one document
    buckets = m["text"]["by_language"]["cmn_Hani"]["buckets"]
    assert set(buckets) == {"zho_Hans", "zho_Hant"}
    docs = decode_docs(root / "train")
    hant_docs = [d for d in docs if d.startswith("@zho_Hant")]
    assert hant_docs  # Traditional is a quarter of the stream but half of train Chinese
    # The unsure cmn document goes to the likelier script (Traditional, 0.35 > 0.25).
    assert m["text"]["by_language"]["cmn_Hani"]["lid_dropped_as"] == {
        "jpn_Jpan": m["text"]["by_language"]["cmn_Hani"]["lid_dropped"]}


def test_workers_1_and_3_write_byte_identical_shards(tmp_path):
    guard = PlantedGuard()
    src = world(lid=True, code_val_extra=[])
    m1 = bs.build_mix(tmp_path / "w1", spec(batch_docs=8), src, setup(lid=True, guard=guard))
    m3 = bs.build_mix(tmp_path / "w3", spec(batch_docs=8, workers=3), src,
                      setup(lid=True, guard=guard))
    files1 = sorted(p.relative_to(tmp_path / "w1") for p in (tmp_path / "w1").rglob("*.bin"))
    files3 = sorted(p.relative_to(tmp_path / "w3") for p in (tmp_path / "w3").rglob("*.bin"))
    assert files1 == files3 and len(files1) > 10
    for rel in files1:
        assert (tmp_path / "w1" / rel).read_bytes() == (tmp_path / "w3" / rel).read_bytes(), rel
    strip = ("build_started", "build_finished", "build_seconds", "workers")
    assert ({k: v for k, v in m1.items() if k not in strip}
            == {k: v for k, v in m3.items() if k not in strip})


class PlantedGuard:
    """find_file flags documents containing "LEAKED" (picklable, for the worker test)."""
    reference_files = 1

    def find_file(self, content):
        return "planted" if "LEAKED" in content else None


# ------------------------------------------------------------------ leakage guard

def _reference_file():
    from stepbuild.bench.acceptance import list_apps, load_reference
    from stepbuild.harness.blocks import parse_blocks

    app = list_apps()[0]
    for block in parse_blocks(load_reference(app)[1]):
        if block.path.endswith(".py") and len(block.content) > 400:
            return block
    raise AssertionError("no reference .py file")


def test_leakage_guard_find_file_ignores_whitespace():
    from stepbuild.bench.run import LeakageGuard

    guard = LeakageGuard()
    block = _reference_file()
    reindented = "\n".join("    " + line for line in block.content.splitlines()) + "\n\n"
    assert guard.find_file(block.content) is not None
    assert guard.find_file(reindented) is not None
    assert guard.find_file(block.content + "\nx = 1\n") is None
    assert guard.reference_files > 10


def test_leakage_guard_drops_a_planted_reference_file(tmp_path):
    from stepbuild.bench.run import LeakageGuard

    block = _reference_file()
    planted = "\n".join("  " + line for line in block.content.splitlines())
    leak_train = {"code": planted, "language": "Python", "license": "mit", "path": "a/app.py",
                  "file": 0}
    leak_val = dict(leak_train, file=900)
    src = world(code_val_extra=[leak_val])
    rows = [leak_train] + list(src.code_train())
    src = src._replace(code_train=from_file(rows))
    m = bs.build_mix(tmp_path, spec(), src, setup(guard=LeakageGuard()))
    leak = m["code"]["leakage"]
    assert leak["dropped_train"] == 1 and leak["dropped_code_val"] == 1
    assert leak["reference_files"] > 10
    for split in ["train", "code_val"]:
        assert not any(planted[:200] in d for d in decode_docs(tmp_path / split))


# ------------------------------------------------------------------ sampling edge cases

STARVE_FREQ = {k: v for k, v in dict(CODE_FREQ, Python=0.01, Dockerfile=0.0).items() if v > 0}


def test_fake_stream_starves_python_under_the_old_window_rule():
    """The same stream through the tokenizer run's QuotaSampler (exhausted when not
    filled within 6 windows) starves Python, as happened on the rented box."""
    spec_ = importlib.util.spec_from_file_location(
        "train_tokenizer", ROOT / "scripts" / "train_tokenizer.py")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        tt = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(tt)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    sampler = tt.QuotaSampler(CODE_W, 60_000, window=300, max_windows=6)
    for row in code_rows(24_000, 0, STARVE_FREQ):
        if sampler.done:
            break
        sampler.offer(row["language"], len(row["code"]))
    assert "Python" in sampler.exhausted
    assert sampler.summary()["by_language"]["Python"]["achieved_share"] < 0.15


def test_rare_common_language_is_not_starved_and_absent_one_is_redistributed(tmp_path):
    """Python is 30% of code but 1% of the stream: filling it takes 15+ windows (the
    old rule gave up after 6, see above). Dockerfile is absent: exhausted on a stall
    and redistributed. HTML stays capped."""
    m = bs.build_mix(tmp_path, spec(train_tokens=100_000, window=300, stall_windows=5),
                     world(code_n=24_000, code_freq=STARVE_FREQ), setup())
    coll = m["code"]["collection"]
    assert list(coll["exhausted"]) == ["Dockerfile"]
    assert coll["windows"] >= 15  # the old rule gave up after 6
    langs = m["code"]["by_language"]
    assert abs(langs["Python"]["achieved_share"] - 0.30) <= 0.01
    assert langs["Dockerfile"]["tokens"] == 0 and langs["Dockerfile"]["exhausted"]
    # HTML's allocation is exactly its cap even though Dockerfile's 1% was shared out.
    assert langs["HTML"]["allocated_tokens"] <= 0.08 * 60_000 + 0.1
    assert langs["HTML"]["achieved_share"] <= 0.08 + 0.004
    assert m["share_check"]["violations"] == []


def test_off_target_share_fails_loudly_before_train_is_written(tmp_path):
    with pytest.raises(sm.ShareError, match=r"Python: .* of code, target 30\.00%"):
        # 4,000 rows is the "file cap": about 37 Python documents, well short of 30%.
        bs.build_mix(tmp_path, spec(train_tokens=100_000, window=300, stall_windows=5),
                     world(code_n=4_000, code_freq=STARVE_FREQ), setup())
    report = json.loads((tmp_path / bs.COLLECT_REPORT).read_text(encoding="utf-8"))
    assert any(v.startswith("Python") for v in report["violations"])
    assert not (tmp_path / "train").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_small_language_short_is_absorbed_without_failing(tmp_path):
    # urd_Latn has a tenth of what it needs: the other eight cover it and the group
    # still makes 12%; its weight (1/30 of text) is under the 5% check.
    src = world(text_n={"urd_Latn": 3})
    order = ("urd_Latn",) + tuple(x for x in OTHER_LANGS if x != "urd_Latn")
    m = bs.build_mix(tmp_path, spec(text_order=order), src, setup())
    tokens = train_label_tokens(m)
    total = sum(tokens.values())
    other = sum(v for k, v in tokens.items() if k.startswith("text:") and k != "text:eng_Latn")
    assert abs(other / total - 0.12) <= 0.005
    assert tokens["text:urd_Latn"] < 0.12 / 9 * total / 2


def test_read_time_filter_counts_are_committed_with_the_rows(tmp_path):
    rows = code_rows(12_000)
    rows[0] = dict(rows[0], **{bs.FILTERED: {"rows_scanned": 5, "dropped_language": 5}})
    src = world()._replace(code_train=from_file(rows))
    m = bs.build_mix(tmp_path, spec(train_tokens=100_000), src, setup())
    st = m["code"]["stats"]
    drops = sum(v for k, v in st.items()
                if k.startswith(("dropped_language", "dropped_license", "skipped_")))
    assert st["dropped_language"] >= 5
    # Every row counted reached a decision or was filtered on the way to one.
    assert st["rows_scanned"] == st["offers"] + drops - st.get("skipped_quota", 0) \
        - st.get("skipped_too_long", 0)
    assert m["code"]["last_file_read"] == rows[st["rows_scanned"] - 6]["file"]


class CopyFetcher:
    """A download stand-in: file i is a fresh copy of one local parquet (the reader
    deletes every file it has consumed)."""

    def __init__(self, src, out):
        self.src, self.out, self.fetched = Path(src), Path(out), []
        self.out.mkdir(parents=True, exist_ok=True)

    def __call__(self, index):
        import shutil
        self.fetched.append(index)
        dst = self.out / f"train-{index:05d}.parquet"
        shutil.copyfile(self.src, dst)
        return dst


def test_real_code_reader_filters_in_pyarrow_and_attaches_counts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"code": f"x{i}", "language": ["Python", "Ruby", "HTML"][i % 3],
             "license": "mit" if i % 4 else "gpl-3.0", "path": f"a{i}.py"} for i in range(40)]
    rows[3]["language"] = None
    for r in rows[30:]:  # the file's last row group is dropped whole
        r["language"] = "Ruby"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "f.parquet", row_group_size=10)
    fetch = CopyFetcher(tmp_path / "f.parquet", tmp_path / "dl")
    source = bs.mix_code_rows(fetch, range(0, 3), ["Python", "HTML"], ["mit"], ahead=2)
    got = list(source(1))  # a resumed build starts at a later file
    assert fetch.fetched == [1, 2]
    assert not list((tmp_path / "dl").iterdir())  # every consumed file is deleted
    marks = [r for r in got if bs.FILE_END in r]
    got = [r for r in got if bs.FILE_END not in r]
    kept = [r for r in rows if r["language"] in ("Python", "HTML") and r["license"] == "mit"]
    assert [r["code"] for r in got] == [r["code"] for r in kept] * 2
    assert [r["file"] for r in got] == [1] * len(kept) + [2] * len(kept)
    # Each file's trailing dropped rows travel with its own end marker, never into
    # the next file (a checkpoint at a file boundary must count them).
    assert [m[bs.FILE_END] for m in marks] == [1, 2]
    assert all(m[bs.FILTERED] == {"rows_scanned": 10, "dropped_language": 10,
                                  "dropped_license": 0} for m in marks)
    counts = Counter()
    for r in got + marks:
        counts.update(r.get(bs.FILTERED) or {})
    dropped_lang = sum(r["language"] not in ("Python", "HTML") for r in rows)
    assert counts == {"rows_scanned": 2 * (40 - len(kept)),
                      "dropped_language": 2 * dropped_lang,
                      "dropped_license": 2 * (40 - len(kept) - dropped_lang)}
    # code_offers adds the attached counts to its own: every row is counted once,
    # and it turns each file end into a MARK offer.
    offers = list(bs.code_offers(iter(list(source(1))), ["Python", "HTML"], ["mit"]))
    total = Counter()
    for o in offers:
        total.update(o.meta["counts"])
    assert total["rows_scanned"] == 80
    assert [o.meta["file_done"] for o in offers if o.kind == sm.MARK] == [1, 2]


def test_main_exits_nonzero_on_share_error(monkeypatch):
    def boom(cfg, args):
        raise sm.ShareError("Python: 14.00% of code, target 30.00%")

    monkeypatch.setattr(bs, "run_mix", boom)
    with pytest.raises(SystemExit) as exc:
        bs.main(["--config", str(ROOT / "configs" / "quipu-moe-smoke.toml"),
                 "--no-low-priority"])
    assert exc.value.code == 2


def test_only_configs_with_code_weights_take_the_v2_build():
    from quipu.config import load_config

    assert not bs.uses_mix(load_config(ROOT / "configs" / "quipu-114m.toml").data)
    assert bs.uses_mix(load_config(ROOT / "configs" / "quipu-moe-smoke.toml").data)


# ------------------------------------------------------------------ LID wrapper

class _FakeFastText:
    def __init__(self):
        self.seen = []

    def get_labels(self):
        return ["__label__eng_Latn", "__label__zho_Hans", "__label__zho_Hant"]

    def predict(self, text, k=1):
        self.seen.append(text)
        return ("__label__zho_Hant", "__label__zho_Hans"), np.array([0.7, 0.3])


def test_fasttext_wrapper_normalises_one_line_and_strips_label_prefix():
    lid = sm.FastTextLid("unused.ftz", max_chars=10)
    lid._model = _FakeFastText()
    label, prob, probs = lid.predict("ｈｉ\n\n  there\tfriend and more")
    assert (label, prob) == ("zho_Hant", pytest.approx(0.7))
    assert probs == {"zho_Hant": pytest.approx(0.7), "zho_Hans": pytest.approx(0.3)}
    assert lid._model.seen == ["hi ther"]  # first 10 chars, NFKC, whitespace collapsed
    assert lid.labels() == ["eng_Latn", "zho_Hans", "zho_Hant"]
    import pickle
    assert pickle.loads(pickle.dumps(lid))._model is None


def test_lid_decision_rules():
    d = sm.lid_decision
    assert d("hin_Latn", "hin_Latn", 0.9, {}, 0.5) == ("hin_Latn", True)
    assert d("hin_Latn", "urd_Latn", 0.9, {}, 0.5) == ("hin_Latn", False)
    assert d("urd_Latn", "hin_Latn", 0.4, {}, 0.5) == ("urd_Latn", True)
    assert d("cmn_Hani", "zho_Hant", 0.9, {}, 0.5) == ("zho_Hant", True)
    assert d("cmn_Hani", "jpn_Jpan", 0.4, {"zho_Hans": 0.3, "zho_Hant": 0.2}, 0.5) == (
        "zho_Hans", True)
    assert sm.lid_labels_needed(TEXT_W) == set(TEXT_W) - {"cmn_Hani"} | {"zho_Hans", "zho_Hant"}


# ------------------------------------------------------------------ real tokenizer

MULTILINGUAL = {
    "eng_Latn": "The quick brown fox jumps over the lazy dog, twice.",
    "cmn_Hani": "学习语言模型需要大量的数据。",
    "jpn_Jpan": "東京は日本の首都です。ひらがなとカタカナ。",
    "kor_Hang": "안녕하세요, 만나서 반갑습니다.",
    "tam_Taml": "தமிழ் ஒரு பழமையான மொழி.",
    "hin_Deva": "भारत एक विशाल देश है।",
    "hin_Latn": "mujhe yeh kitaab bahut pasand hai",
}


@pytest.mark.skipif(not TOKENIZER_JSON.is_file(), reason="artifacts/tokenizer/tokenizer.json absent")
def test_real_tokenizer_build_round_trips_multilingual_documents(tmp_path):
    from quipu.tokenizer import make_tokenizer

    factory = functools.partial(make_tokenizer, str(TOKENIZER_JSON))
    tok = factory()
    assert tok.vocab_size <= 65536
    for text in MULTILINGUAL.values():
        assert tok.decode(tok.encode(text)) == text

    def rows(lang, n, start=0):
        return [{"text": f"{MULTILINGUAL.get(lang, MULTILINGUAL['eng_Latn'])} #{i}"}
                for i in range(start, start + n)]

    code = [{"code": f"def f{i}(x):\n    return x + {i}\n", "language": lang, "license": "mit",
             "path": f"m{i}.py"} for i in range(9000) for lang in [list(CODE_W)[i % 14]]]
    src = bs.MixSources(
        code_train=from_file(code[:8000]), code_val=lambda: iter(code[8000:]),
        text_train={x: text_source(rows(x, 800)) for x in TEXT_W},
        text_val={x: (lambda x=x: iter(rows(x, 100, 10**6))) for x in TEXT_W})
    s = sm.DocSetup(tokenizer=factory, max_doc_tokens=2_000)
    m = bs.build_mix(tmp_path, spec(train_tokens=20_000, val_tokens=500, code_val_tokens=500,
                                    lang_val_tokens=100, window=100), src, s)
    toks = tokens_of(tmp_path / "train")
    assert len(toks) == 20_000 and m["eot"] == tok.eot
    pieces = np.split(toks, np.flatnonzero(toks == tok.eot) + 1)
    known = {r["text"] for x in TEXT_W for r in rows(x, 800)} | {c["code"] for c in code}
    whole = [p for p in pieces if len(p) and p[-1] == tok.eot]
    assert len(whole) > 100
    for p in whole:
        assert tok.decode(p[:-1].tolist()) in known


# ------------------------------------------------------------------ worker failures (C1)

class ExitTok(CharTok):
    """Kills its process on a planted document, but only in a worker process."""

    def encode(self, text):
        if "KILLME" in text and multiprocessing.parent_process() is not None:
            os._exit(3)
        return super().encode(text)


def worker_only_failing_tok():
    if multiprocessing.parent_process() is not None:
        raise RuntimeError("tokenizer.json unreadable in the worker")
    return CharTok()


def _in_thread(fn, timeout=90):
    """fn() in a daemon thread: (finished in time, exception or None, seconds)."""
    out = {}

    def run():
        try:
            fn()
            out["exc"] = None
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            out["exc"] = exc

    t0 = time.monotonic()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    return "exc" in out, out.get("exc"), time.monotonic() - t0


def test_a_dead_worker_fails_the_build_in_seconds_instead_of_hanging(tmp_path):
    rows = code_rows(12_000)
    rows.insert(300, {"code": "KILLME now", "language": "Python", "license": "mit",
                      "path": "k.py", "file": 0})
    src = world()._replace(code_train=from_file(rows))
    s = sm.DocSetup(tokenizer=ExitTok, max_doc_tokens=2_000)
    done, exc, secs = _in_thread(lambda: bs.build_mix(tmp_path, spec(workers=2), src, s))
    assert done, "the build hung on a dead worker"
    assert isinstance(exc, sm.WorkerDied), exc
    assert "rerun the same command to resume" in str(exc)
    assert secs < 60


def test_a_worker_that_cannot_start_raises_at_once():
    s = sm.DocSetup(tokenizer=worker_only_failing_tok, max_doc_tokens=100)
    done, exc, _ = _in_thread(lambda: sm.Runner(s, 2).close())
    assert done and isinstance(exc, sm.WorkerDied), exc


def test_runner_sets_single_thread_env_for_workers_and_restores_it(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with sm.Runner(sm.DocSetup(tokenizer=CharTok, max_doc_tokens=100), 2):
        assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_default_workers_is_at_most_eight():
    args = bs.build_parser().parse_args([])
    assert 1 <= args.workers <= 8
    assert args.download_workers == 6
    assert args.lid_threshold == 0.0 == sm.DEFAULT_LID_THRESHOLD


# ------------------------------------------------------------------ bucket store

def test_bucket_store_names_files_after_buckets_and_restores_a_snapshot(tmp_path):
    st = sm.BucketStore(tmp_path)
    st.append("C++", np.arange(5, dtype=np.uint16))
    st.append("eng_Latn", np.arange(3, dtype=np.uint16))
    snap = st.snapshot()
    st.append("C++", np.arange(4, dtype=np.uint16))   # after the checkpoint
    st.append("Rust", np.arange(2, dtype=np.uint16))  # a bucket born after it
    st.close()
    assert {p.name for p in tmp_path.iterdir()} == {
        "C%2B%2B.bin", "C%2B%2B.ends", "eng_Latn.bin", "eng_Latn.ends", "Rust.bin", "Rust.ends"}
    again = sm.BucketStore(tmp_path, fresh=False)  # a resumed build keeps the files
    with pytest.raises(RuntimeError, match="restore"):
        again.append("C++", np.arange(1, dtype=np.uint16))
    again.restore(snap)
    assert not (tmp_path / "Rust.bin").exists()
    again.append("C++", np.array([9], dtype=np.uint16))
    again.close()
    assert [d.tolist() for d in again.docs("C++")] == [[0, 1, 2, 3, 4], [9]]
    assert again.documents("C++") == 2 and again.tokens["eng_Latn"] == 3
    with pytest.raises(sm.ResumeError, match="--fresh"):
        (tmp_path / "eng_Latn.bin").write_bytes(b"")
        sm.BucketStore(tmp_path, fresh=False).restore(snap)
    sm.BucketStore(tmp_path)  # fresh (the default, and --fresh) clears the directory
    assert not list(tmp_path.glob("*.bin"))


def test_collector_snapshot_round_trips_through_json():
    col = sm.Collector({"a": 0.6, "b": 0.4}, 1000, window=3, stall_windows=2)
    for i in range(20):
        if col.wants("a") and i % 2:
            col.admit("a", np.zeros(7, np.uint16))
        col.offered["a"] += 1
        col.tick()
    again = sm.Collector.from_snapshot(json.loads(json.dumps(col.snapshot())))
    assert again.snapshot() == json.loads(json.dumps(col.snapshot()))
    assert (again.base, again.limit, again.exhausted) == (col.base, col.limit, col.exhausted)


# ------------------------------------------------------------------ resume (C2)

class Crash(RuntimeError):
    """Stands in for a preempted box / a killed build."""


def crash_code_at(source, at_file):
    """A code source that dies (once) on reaching file at_file; records every start."""
    log = {"starts": [], "armed": True}

    def src(start=0):
        log["starts"].append(start)
        for r in source(start):
            if log["armed"] and r.get("file", 0) >= at_file:
                log["armed"] = False
                raise Crash(f"killed at file {at_file}")
            yield r
    return src, log


def crash_rows_after(source, n):
    """A text source that dies (once) after n rows; counts its calls."""
    log = {"calls": 0, "armed": True}

    def src(*a):
        log["calls"] += 1
        log.setdefault("starts", []).append(a[0] if a else 0)
        for i, r in enumerate(source(*a)):
            if log["armed"] and i == n:
                log["armed"] = False
                raise Crash(f"killed after {n} rows")
            yield r
    return src, log


def counted(source, log, key):
    def src(*a):
        log[key] = log.get(key, 0) + 1
        return source(*a)
    return src


def _strip(m):
    drop = {"build_started", "build_finished", "build_seconds", "workers", "resume"}
    return {k: v for k, v in m.items() if k not in drop}


def same_output(a, b):
    fa = sorted(p.relative_to(a) for p in Path(a).rglob("*.bin"))
    fb = sorted(p.relative_to(b) for p in Path(b).rglob("*.bin"))
    assert fa == fb and len(fa) > 10
    for rel in fa:
        assert (Path(a) / rel).read_bytes() == (Path(b) / rel).read_bytes(), rel


RESUME_SPEC = dict(code_files=24, projection_min_files=2)


@pytest.fixture(scope="module")
def reference_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("ref")
    m = bs.build_mix(root, spec(**RESUME_SPEC), world(lid=True), setup(lid=True))
    return root, m


def _state(root):
    return json.loads((Path(root) / bs.MIX_WORK / bs.STATE).read_text(encoding="utf-8"))


@pytest.mark.parametrize("workers", [1, 2])
def test_a_build_killed_in_code_resumes_to_byte_identical_shards(tmp_path, reference_build,
                                                                  workers):
    ref_root, ref = reference_build
    last = ref["code"]["last_file_read"]
    assert last >= 4  # the fake stream spans several code files
    src = world(lid=True)
    code, log = crash_code_at(src.code_train, at_file=last - 1)
    src = src._replace(code_train=code)
    s = spec(workers=workers, **RESUME_SPEC)
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, s, src, setup(lid=True))
    ckpt = _state(tmp_path)["phases"]["code"]
    assert not ckpt["done"] and 1 <= ckpt["next_file"] <= last - 1
    m = bs.build_mix(tmp_path, s, src, setup(lid=True))
    assert log["starts"] == [0, ckpt["next_file"]]  # files before it are not read again
    same_output(ref_root, tmp_path)
    assert _strip(m) == _strip(ref)
    assert m["resume"]["resumed"] and m["resume"]["code_from_file"] == ckpt["next_file"]


def test_a_build_killed_in_text_redoes_only_the_unfinished_language(tmp_path,
                                                                     reference_build):
    ref_root, ref = reference_build
    src = world(lid=True)
    calls: dict = {}
    jpn, log = crash_rows_after(src.text_train["jpn_Jpan"], 10)
    text_train = {x: counted(f, calls, x) for x, f in src.text_train.items()}
    text_train["jpn_Jpan"] = jpn
    text_val = dict(src.text_val, eng_Latn=counted(src.text_val["eng_Latn"], calls, "val"))
    src = src._replace(code_train=counted(src.code_train, calls, "code"),
                       text_train=text_train, text_val=text_val)
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), src, setup(lid=True))
    done = _state(tmp_path)["phases"]["text"]["done"]
    assert done[0] == "eng_Latn" and "jpn_Jpan" not in done
    before = dict(calls)
    m = bs.build_mix(tmp_path, spec(**RESUME_SPEC), src, setup(lid=True))
    # Code, the val split and every finished language come from _work, not the source.
    assert calls["code"] == before["code"] == 1 and calls["val"] == before["val"] == 1
    for x in done:
        assert calls[x] == before[x] == 1, x
    assert log["calls"] == 2
    same_output(ref_root, tmp_path)
    assert _strip(m) == _strip(ref)


def test_resume_refuses_a_different_build_unless_fresh(tmp_path, reference_build):
    ref_root, _ = reference_build
    src = world(lid=True)
    code, _ = crash_code_at(src.code_train, at_file=2)
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), src._replace(code_train=code),
                     setup(lid=True))
    with pytest.raises(sm.ResumeError, match="--fresh") as exc:
        bs.build_mix(tmp_path, spec(slack=0.2, **RESUME_SPEC), src, setup(lid=True))
    assert "slack" in str(exc.value)
    with pytest.raises(sm.ResumeError, match="tokenizer"):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), src, setup(lid=True),
                     {"tokenizer": {"sha256": "another tokenizer"}})
    with pytest.raises(sm.ResumeError, match="--from-work"):  # the mix, not the data
        bs.build_mix(tmp_path, spec(train_tokens=200_000, **RESUME_SPEC), src, setup(lid=True))
    # --fresh throws the old work away; the result is the uninterrupted build.
    bs.build_mix(tmp_path, spec(**RESUME_SPEC), src, setup(lid=True), fresh=True)
    same_output(ref_root, tmp_path)


def _no_downloads(src):
    def refuse(*a):
        raise AssertionError("--from-work must not read a train source")
    return src._replace(code_train=refuse, text_train={x: refuse for x in src.text_train})


def test_from_work_reallocates_after_a_late_share_error_without_downloading(tmp_path):
    src = world(text_n={"eng_Latn": 300})
    with pytest.raises(sm.ShareError, match="eng_Latn"):  # after every text language
        bs.build_mix(tmp_path, spec(), src, setup())
    state = _state(tmp_path)["phases"]
    assert state["code"]["done"] and state["text"]["done"] == ["eng_Latn", *OTHER_LANGS]
    assert (tmp_path / bs.COLLECT_REPORT).is_file() and not (tmp_path / "train").exists()
    with pytest.raises(sm.ResumeError, match="--from-work"):
        bs.build_mix(tmp_path, spec(train_tokens=200_000), _no_downloads(src), setup())
    m = bs.build_mix(tmp_path, spec(train_tokens=200_000), _no_downloads(src), setup(),
                     from_work=True)
    assert m["splits"]["train"]["tokens"] == 200_000
    assert m["share_check"]["violations"] == []
    assert m["resume"]["from_work"]


def test_from_work_needs_a_finished_code_collection(tmp_path):
    with pytest.raises(sm.ResumeError, match="state"):
        bs.build_mix(tmp_path, spec(), world(), setup(), from_work=True)


# ------------------------------------------------------------------ fail early (I2)

def test_code_share_is_checked_before_any_text_is_read(tmp_path):
    src = world(code_n=4_000, code_freq=STARVE_FREQ)
    src = src._replace(text_train={x: _no_downloads(src).text_train[x] for x in src.text_train})
    with pytest.raises(sm.ShareError, match="before any text"):
        bs.build_mix(tmp_path, spec(train_tokens=100_000, window=300, stall_windows=5), src,
                     setup())
    report = json.loads((tmp_path / bs.COLLECT_REPORT).read_text(encoding="utf-8"))
    assert any(v.startswith("Python") for v in report["violations"])


def test_projection_stops_a_hopeless_code_collection_early(tmp_path):
    rows = code_rows(40 * 500, 0, STARVE_FREQ)  # 40 files; Python needs ~64
    read = []

    def source(start=0):
        for r in rows:
            if r["file"] >= start:
                read.append(r["file"])
                yield r
    src = world()._replace(code_train=source)
    with pytest.raises(sm.ShareError, match=r"(?s)projected.*Python") as exc:
        bs.build_mix(tmp_path, spec(train_tokens=400_000, window=300, stall_windows=5,
                                    code_files=40, projection_min_files=4), src, setup())
    # Off target from file 4 on; the stop needs 5 file ends in a row: 4, 5, 6, 7, 8.
    assert "file 8 of 40" in str(exc.value)
    assert max(read) <= 9  # stopped after about 8 of 40 files, not at the cap
    msg = str(exc.value)
    assert "--fresh" in msg and "--projection-min-files" in msg and "8 files" in msg


def test_projection_is_quiet_when_the_files_suffice(reference_build):
    _, m = reference_build  # projection on from file 2; the build went through
    assert m["share_check"]["violations"] == []


# ------------------------------------------------------------------ preflight (I2)

def test_preflight_report_counts_files_needed_and_flags_a_short_language():
    w = {"Python": 0.5, "Go": 0.3, "HTML": 0.2}
    per_file = [{"Python": 380.0, "Go": 378.0, "HTML": 1000.0} for _ in range(10)]
    # 1,000 tokens of code: quotas 500 / 300 / 200 (HTML capped at 0.2).
    r = sm.preflight_report(per_file, w, 1_000, caps={"HTML": 0.2}, chars_per_token=3.79,
                            file_bytes=[100] * 10)
    assert r["by_language"]["Python"]["files_to_fill"] == 5   # 100 tokens a file
    assert r["by_language"]["Go"]["files_to_fill"] == 4       # just under 100 a file
    assert r["by_language"]["HTML"]["files_to_fill"] == 1
    assert (r["files_read"], r["limiting_languages"], r["download_bytes"]) == (5, ["Python"],
                                                                               500)
    assert r["ok"]
    short = [{"Python": 3.79, "Go": 379.0, "HTML": 1000.0} for _ in range(10)]
    r = sm.preflight_report(short, w, 1_000, caps={"HTML": 0.2}, chars_per_token=3.79)
    assert not r["ok"] and r["by_language"]["Python"]["files_to_fill"] is None
    assert r["files_read"] == 10 and r["limiting_languages"] == ["Python"]
    text = sm.format_preflight(r)
    assert "verdict: FAIL" in text and "Python" in text and "never" in text


def test_code_file_counts_apply_the_build_filters():
    import pyarrow as pa

    table = pa.Table.from_pylist([
        {"language": "Python", "license": "mit", "size": 100, "path": "a.py"},
        {"language": "Python", "license": "gpl-3.0", "size": 100, "path": "b.py"},
        {"language": "Ruby", "license": "mit", "size": 100, "path": "c.rb"},
        {"language": "Python", "license": "mit", "size": 10**9, "path": "big.py"},
        {"language": "HTML", "license": "mit", "size": 50, "path": "node_modules/x.html"},
        {"language": "HTML", "license": "mit", "size": 70, "path": "site/i.html"},
        {"language": None, "license": None, "size": None, "path": None},
    ])
    got = bs.code_file_counts(table, ["Python", "HTML"], ["mit"], max_bytes=60_000)
    assert got == {"Python": 100.0, "HTML": 70.0}


def test_preflight_exits_nonzero_when_the_mix_cannot_be_met(capsys):
    w = {"Python": 0.5, "Go": 0.3, "HTML": 0.2}

    def counts(i):
        return {"Python": 3.79, "Go": 379.0, "HTML": 1000.0}
    code = bs.preflight(counts, range(10), w, 1_000, caps={"HTML": 0.2}, chars_per_token=3.79,
                        workers=3, header="test")
    out = capsys.readouterr().out
    assert code != 0 and "verdict: FAIL" in out
    code = bs.preflight(lambda i: {"Python": 1e4, "Go": 1e4, "HTML": 1e4}, range(10), w, 1_000,
                        caps={"HTML": 0.2}, workers=3, header="test")
    assert code == 0 and "verdict: OK" in capsys.readouterr().out


# ------------------------------------------------------------------ downloads (I1)

class FakeFetch:
    """Writes file i into `out`; fails `fail[i]` times first; tracks disk use."""

    def __init__(self, out, fail=None, delay=0.0):
        self.out, self.fail, self.delay = Path(out), dict(fail or {}), delay
        self.out.mkdir(parents=True, exist_ok=True)
        self.calls = Counter()
        self.lock = threading.Lock()

    def __call__(self, i):
        with self.lock:
            self.calls[i] += 1
            if self.fail.get(i, 0) > 0:
                self.fail[i] -= 1
                raise OSError(f"network blip on {i}")
        time.sleep(self.delay)
        p = self.out / f"{i}.parquet"
        p.write_bytes(bytes([i % 256]) * 10)
        return p


def test_fetch_ahead_keeps_file_order_bounds_disk_and_deletes_consumed_files(tmp_path):
    fetch = FakeFetch(tmp_path / "dl", delay=0.01)
    seen, most = [], 0
    for i, path in bs.fetch_ahead(range(3, 20), fetch, ahead=4, sleep=lambda s: None):
        seen.append(i)
        assert path.read_bytes() == bytes([i]) * 10
        most = max(most, len(list((tmp_path / "dl").iterdir())))
    assert seen == list(range(3, 20))
    assert most <= 4 + 1
    assert not list((tmp_path / "dl").iterdir())


def test_fetch_ahead_retries_with_capped_backoff_then_gives_up(tmp_path):
    waits = []
    fetch = FakeFetch(tmp_path / "dl", fail={1: 3})
    got = [i for i, _ in bs.fetch_ahead(range(3), fetch, ahead=2, sleep=waits.append)]
    assert got == [0, 1, 2] and fetch.calls[1] == 4
    assert waits == [5, 10, 20]
    waits.clear()
    fetch = FakeFetch(tmp_path / "dl2", fail={1: 99})
    it = bs.fetch_ahead(range(3), fetch, ahead=2, attempts=12, sleep=waits.append)
    assert next(it)[0] == 0
    with pytest.raises(RuntimeError, match="file 1 .* 12 attempts"):
        next(it)
    assert len(waits) == 11 and max(waits) == 300
    # A download still running when the reader failed is deleted when it finishes.
    deadline = time.monotonic() + 5
    while list((tmp_path / "dl2").iterdir()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not list((tmp_path / "dl2").iterdir())


def test_closing_the_reader_stops_downloads_and_cleans_up(tmp_path):
    fetch = FakeFetch(tmp_path / "dl", delay=0.05)
    it = bs.fetch_ahead(range(50), fetch, ahead=3, sleep=lambda s: None)
    next(it)
    it.close()
    time.sleep(0.5)
    assert sum(fetch.calls.values()) <= 5
    assert not list((tmp_path / "dl").iterdir())


def test_prefetch_thread_stops_when_its_consumer_closes():
    produced = []

    def items():
        for i in range(10**6):
            produced.append(i)
            yield i
    it = bs.prefetch(items(), depth=2)
    assert next(it) == 0
    it.close()
    time.sleep(1.0)
    n = len(produced)
    time.sleep(0.5)
    assert len(produced) == n < 100


# ------------------------------------------------------------------ LID (I3)

def test_lid_default_threshold_drops_every_mismatch(tmp_path):
    s = sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, lid=FakeLid())
    assert s.lid_threshold == 0.0
    m = bs.build_mix(tmp_path, spec(), world(lid=True), s)
    for x in ("eng_Latn", "ind_Latn", "hin_Latn"):
        st = m["text"]["by_language"][x]
        assert st["lid_kept_unsure"] == 0
        assert set(st["lid_dropped_as"]) == {"jpn_Jpan", "other"}
    assert m["lid"]["threshold"] == 0.0 and m["lid"]["max_chars"] == sm.LID_MAX_CHARS
    by = m["lid"]["by_language"]["hin_Latn"]
    assert by["kept"] + by["dropped"] == m["text"]["by_language"]["hin_Latn"]["offers"]
    decoded = decode_docs(tmp_path / "train")
    assert not any(d.startswith("@other:0.40") for d in decoded)


def test_lid_counts_documents_kept_while_unsure(full_build):
    _, m = full_build  # threshold 0.5: the "other:0.40" mislabels are kept
    for x in ("eng_Latn", "ind_Latn"):
        by = m["lid"]["by_language"][x]
        assert by["kept_while_unsure"] > 0
        assert by["kept_while_unsure"] == m["text"]["by_language"][x]["lid_kept_unsure"]


def test_heavy_lid_loss_in_a_languages_first_file_is_a_warning(tmp_path, capsys):
    src = world(lid=True)
    rows = text_rows("zsm_Latn", 200, 0, lambda i, rng: "ind_Latn:0.9" if i % 2 else
                     "zsm_Latn:0.95")
    for i, r in enumerate(rows):
        r["file"] = f"zsm/{i // 20}.parquet"
    src.text_train["zsm_Latn"] = text_source(rows)
    m = bs.build_mix(tmp_path, spec(), src, setup(lid=True))
    warn = m["text"]["by_language"]["zsm_Latn"]["lid_warning"]
    assert "zsm_Latn" in warn and "50%" in warn
    assert "lid_warning" not in m["text"]["by_language"]["ind_Latn"]
    assert "WARNING" in capsys.readouterr().err


# ------------------------------------------------------------------ startup checks (I4, M3)

@pytest.mark.skipif(not TOKENIZER_JSON.is_file(), reason="artifacts/tokenizer/tokenizer.json absent")
def test_vocab_size_mismatch_fails_at_startup(monkeypatch):
    import dataclasses

    from quipu.config import load_config

    monkeypatch.chdir(ROOT)
    cfg = load_config(ROOT / "configs" / "quipu-moe.toml")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, vocab_size=50_000))
    calls = fake_hub(monkeypatch)
    with pytest.raises(SystemExit, match="vocab"):
        bs.run_mix(cfg, bs.build_parser().parse_args([]))
    assert calls == []


@pytest.mark.skipif(not TOKENIZER_JSON.is_file(), reason="artifacts/tokenizer/tokenizer.json absent")
def test_a_broken_fasttext_fails_before_any_dataset_is_touched(monkeypatch, tmp_path):
    from quipu.config import load_config

    monkeypatch.chdir(ROOT)
    cfg = load_config(ROOT / "configs" / "quipu-moe.toml")
    calls = fake_hub(monkeypatch, lid_path=tmp_path / "lid.ftz")

    def broken(self):
        raise RuntimeError("--lid-filter needs the fastText bindings")
    monkeypatch.setattr(sm.FastTextLid, "model", property(broken))
    with pytest.raises(RuntimeError, match="fastText"):
        bs.run_mix(cfg, bs.build_parser().parse_args(["--lid-filter"]))
    assert all(c[0] in ("model_info", "hf_hub_download") for c in calls), calls
    assert not any(c[0] == "hf_hub_download" and "datasets" in str(c) for c in calls)


def fake_hub(monkeypatch, lid_path=None):
    """Replace huggingface_hub's entry points with recorders (no network)."""
    import types

    import huggingface_hub

    calls = []

    class Api:
        def model_info(self, repo, **kw):
            calls.append(("model_info", repo))
            return types.SimpleNamespace(sha="0" * 40)

        def dataset_info(self, repo, **kw):
            calls.append(("dataset_info", repo))
            raise AssertionError("no dataset may be touched")

    class FS:
        def __getattr__(self, name):
            calls.append(("fs", name))
            raise AssertionError("no dataset may be touched")

    def download(repo, filename, **kw):
        calls.append(("hf_hub_download", repo, filename, kw.get("repo_type")))
        if lid_path is None:
            raise AssertionError("unexpected download")
        Path(lid_path).write_bytes(b"not a real model")
        return str(lid_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FS)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return calls


# ------------------------------------------------------------------ re-review fixes (M7)

def _http_error(cls, status):
    import httpx
    from huggingface_hub import errors

    req = httpx.Request("GET", "https://huggingface.co/datasets/x")
    return getattr(errors, cls)(f"{status} error", response=httpx.Response(status, request=req))


def _classifier_cases():
    import errno

    import httpx
    import pyarrow as pa
    from huggingface_hub import errors

    req = httpx.Request("GET", "https://huggingface.co/x")
    return [
        ("RepositoryNotFoundError", lambda: _http_error("RepositoryNotFoundError", 401), True),
        ("RevisionNotFoundError", lambda: _http_error("RevisionNotFoundError", 404), True),
        ("RemoteEntryNotFoundError", lambda: _http_error("RemoteEntryNotFoundError", 404), True),
        ("EntryNotFoundError", lambda: errors.EntryNotFoundError("no such file"), True),
        ("GatedRepoError", lambda: _http_error("GatedRepoError", 403), True),
        ("HfHubHTTPError 401", lambda: _http_error("HfHubHTTPError", 401), True),
        ("HfHubHTTPError 403", lambda: _http_error("HfHubHTTPError", 403), True),
        ("HfHubHTTPError 404", lambda: _http_error("HfHubHTTPError", 404), True),
        ("FileNotFoundError", lambda: FileNotFoundError("gone"), True),
        ("PermissionError", lambda: PermissionError("denied"), True),
        ("ENOSPC", lambda: OSError(errno.ENOSPC, "No space left on device"), True),
        ("ArrowInvalid", lambda: pa.lib.ArrowInvalid("Parquet magic bytes not found"), True),
        ("HfHubHTTPError 429", lambda: _http_error("HfHubHTTPError", 429), False),
        ("HfHubHTTPError 500", lambda: _http_error("HfHubHTTPError", 500), False),
        ("HfHubHTTPError 503", lambda: _http_error("HfHubHTTPError", 503), False),
        ("TimeoutError", lambda: TimeoutError("read timed out"), False),
        ("ConnectionError", lambda: ConnectionResetError("reset by peer"), False),
        ("httpx.ConnectTimeout", lambda: httpx.ConnectTimeout("timeout", request=req), False),
        ("httpx.ReadError", lambda: httpx.ReadError("read", request=req), False),
        ("OSError blip", lambda: OSError("network blip"), False),
        # hf_hub_download's "no connection and nothing cached": a connection error.
        ("LocalEntryNotFoundError", lambda: errors.LocalEntryNotFoundError("offline"), False),
        ("TransientError", lambda: bs.TransientError("truncated download"), False),
    ]


@pytest.mark.parametrize("name, make, permanent", _classifier_cases(),
                         ids=[c[0] for c in _classifier_cases()])
def test_download_errors_are_classified_permanent_or_transient(name, make, permanent):
    assert bs.is_permanent(make()) is permanent, name


def test_a_404_download_fails_at_once_without_sleeping():
    slept, calls = [], []

    def fetch(i):
        calls.append(i)
        raise _http_error("HfHubHTTPError", 404)
    t0 = time.monotonic()
    it = bs.fetch_ahead([0], fetch, 1, sleep=slept.append)
    with pytest.raises(RuntimeError, match="cannot succeed"):
        next(it)
    assert calls == [0] and slept == []
    assert time.monotonic() - t0 < 1


class _FailingFS:
    def __init__(self, exc, then=None):
        self.exc, self.then, self.opens = exc, then, 0

    def open(self, path, mode="rb", **kw):
        self.opens += 1
        if self.then is not None and self.opens > 1:
            return open(self.then, mode)
        raise self.exc


def test_text_reader_fails_at_once_on_a_missing_file_and_makes_12_attempts_on_a_blip():
    slept = []
    fs = _FailingFS(FileNotFoundError("gone"))
    with pytest.raises(FileNotFoundError):
        list(bs.iter_text_files(fs, ["x.parquet"], sleep=slept.append))
    assert fs.opens == 1 and slept == []
    fs = _FailingFS(OSError("network blip"))
    with pytest.raises(OSError, match="blip"):
        list(bs.iter_text_files(fs, ["x.parquet"], sleep=slept.append))
    assert fs.opens == bs.DOWNLOAD_ATTEMPTS == 12 and len(slept) == 11


def test_preflight_read_fails_at_once_on_403_and_retries_a_timeout(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{"language": "Python", "license": "mit", "size": 100,
                                   "path": "a.py"}])
    pq.write_table(table, tmp_path / "f.parquet")
    slept = []
    fs = _FailingFS(_http_error("HfHubHTTPError", 403))
    with pytest.raises(Exception, match="403"):
        bs.read_code_file_counts(fs, "x", ["Python"], ["mit"], 1e6, sleep=slept.append)
    assert fs.opens == 1 and slept == []
    fs = _FailingFS(TimeoutError("slow"), then=tmp_path / "f.parquet")
    got = bs.read_code_file_counts(fs, "x", ["Python"], ["mit"], 1e6, sleep=slept.append)
    assert got == {"Python": 100.0} and fs.opens == 2 and slept == [5]


def test_a_truncated_code_download_is_transient_and_deleted(tmp_path, monkeypatch):
    import huggingface_hub

    def download(repo, filename, **kw):
        p = Path(kw["local_dir"]) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PAR1 truncated")
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    fetch = bs.hf_code_fetcher("r", "0" * 40, 880, tmp_path)
    with pytest.raises(bs.TransientError):
        fetch(3)
    assert not list(tmp_path.rglob("*.parquet"))


def test_download_threads_are_daemons_so_a_failed_build_exits(tmp_path):
    release = threading.Event()

    def fetch(i):
        if i == 0:
            raise _http_error("HfHubHTTPError", 404)
        release.wait(10)  # a download that hangs
        p = tmp_path / f"{i}.parquet"
        p.write_bytes(b"x")
        return p
    it = bs.fetch_ahead([0, 1, 2], fetch, 3, sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        next(it)
    it.close()
    workers = [t for t in threading.enumerate() if t.name.startswith("download")]
    assert workers and all(t.daemon for t in workers)
    release.set()
    for t in workers:
        t.join(5)
    assert not list(tmp_path.glob("*.parquet"))  # finished late: deleted


def _no_sources(src):
    def refuse(*a):
        raise AssertionError("a finished build must not read any source")
    return src._replace(code_train=refuse, code_val=refuse,
                        text_train={x: refuse for x in src.text_train},
                        text_val={x: refuse for x in src.text_val})


def test_rerunning_a_finished_build_is_a_no_op(tmp_path, reference_build, capsys):
    import shutil

    ref_root, ref = reference_build
    root = tmp_path / "done"
    shutil.copytree(ref_root, root)
    before = (root / "manifest.json").read_bytes()
    capsys.readouterr()
    m = bs.build_mix(root, spec(**RESUME_SPEC), _no_sources(world(lid=True)), setup(lid=True))
    assert m == ref
    assert (root / "manifest.json").read_bytes() == before
    assert "already built" in capsys.readouterr().out
    with pytest.raises(sm.ResumeError, match="--fresh"):
        bs.build_mix(root, spec(**RESUME_SPEC), _no_sources(world(lid=True)), setup(lid=True),
                     from_work=True)
    assert (root / "manifest.json").read_bytes() == before
    same_output(ref_root, root)


def test_a_crash_while_deleting_work_leaves_a_finished_build(tmp_path, monkeypatch,
                                                             reference_build):
    import shutil

    ref_root, _ = reference_build
    real = shutil.rmtree

    def crashing_rmtree(path, *a, **kw):
        if Path(path).name == bs.MIX_WORK:
            for p in (Path(path) / "code").glob("*.bin"):
                p.unlink()
            raise Crash("killed while deleting _work")
        return real(path, *a, **kw)
    monkeypatch.setattr(shutil, "rmtree", crashing_rmtree)
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), world(lid=True), setup(lid=True))
    monkeypatch.setattr(shutil, "rmtree", real)
    before = (tmp_path / "manifest.json").read_bytes()
    bs.build_mix(tmp_path, spec(**RESUME_SPEC), _no_sources(world(lid=True)), setup(lid=True))
    assert (tmp_path / "manifest.json").read_bytes() == before
    same_output(ref_root, tmp_path)


def test_damaged_work_next_to_a_manifest_is_an_error_that_keeps_the_manifest(tmp_path):
    bs.build_mix(tmp_path, spec(keep_work=True, **RESUME_SPEC), world(lid=True),
                 setup(lid=True))
    before = (tmp_path / "manifest.json").read_bytes()
    # A finished build kept with --keep-work is a no-op to rerun.
    bs.build_mix(tmp_path, spec(keep_work=True, **RESUME_SPEC), _no_sources(world(lid=True)),
                 setup(lid=True))
    # state.json there but not marked finished (a crash before the mark), buckets gone.
    st = _state(tmp_path)
    st.pop("finished", None)
    (tmp_path / bs.MIX_WORK / bs.STATE).write_text(json.dumps(st), encoding="utf-8")
    for p in (tmp_path / bs.MIX_WORK / "code").glob("*.bin"):
        p.unlink()
    with pytest.raises(sm.ResumeError, match="manifest"):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), _no_sources(world(lid=True)),
                     setup(lid=True))
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_a_resumed_build_sweeps_partial_downloads(tmp_path):
    src = world(lid=True)
    code, _ = crash_code_at(src.code_train, at_file=3)
    s = spec(keep_work=True, **RESUME_SPEC)
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, s, src._replace(code_train=code), setup(lid=True))
    part = tmp_path / bs.MIX_WORK / "dl" / ".cache" / "huggingface" / "download" / "data"
    part.mkdir(parents=True)
    (part / "train-00003-of-00880.parquet.abc.incomplete").write_bytes(b"x" * 100)
    bs.build_mix(tmp_path, s, src, setup(lid=True))
    assert not list((tmp_path / bs.MIX_WORK).rglob("*.incomplete"))


def test_projection_margin_is_wider_before_file_60():
    assert bs.projection_max_off(20) == pytest.approx(bs.SHARE_CHECK_MAX_OFF + 0.01)
    assert bs.projection_max_off(59) == pytest.approx(bs.SHARE_CHECK_MAX_OFF + 0.01)
    assert bs.projection_max_off(60) == pytest.approx(bs.SHARE_CHECK_MAX_OFF)
    assert bs.PROJECTION_STREAK == 5


def test_noisy_early_projection_does_not_stop_the_build(tmp_path, monkeypatch):
    """Python is missing from the first files, so the first projections are off
    target; it is common later. A single bad projection must not stop the build."""
    rows = code_rows(12_000)
    for r in rows:
        if r["file"] < 3 and r["language"] == "Python":
            r["language"] = "Java"
    src = world()._replace(code_train=from_file(rows))
    # stall_windows high enough that Python's absence is not a stall (7.5 windows).
    s = spec(code_files=24, projection_min_files=2, stall_windows=20)
    m = bs.build_mix(tmp_path / "a", s, src, setup())
    assert m["share_check"]["violations"] == []
    monkeypatch.setattr(bs, "PROJECTION_STREAK", 1)  # the old rule: stop at the first
    monkeypatch.setattr(bs, "PROJECTION_EARLY_MARGIN", 0.0)
    with pytest.raises(sm.ShareError, match="projected"):
        bs.build_mix(tmp_path / "b", s, src, setup())


def eng_files_world():
    src = world(lid=True)
    rows = text_rows("eng_Latn", 1200, 0, std_tag("eng_Latn"))
    for i, r in enumerate(rows):
        r["file"] = f"eng/{i // 40:03d}.parquet"
    src.text_train["eng_Latn"] = text_source(rows)
    return src


@pytest.mark.parametrize("workers", [1, 2])
def test_a_build_killed_inside_english_resumes_from_its_next_file(tmp_path, workers):
    s = spec(workers=workers, **RESUME_SPEC)
    ref = bs.build_mix(tmp_path / "ref", s, eng_files_world(), setup(lid=True))
    src = eng_files_world()
    eng, log = crash_rows_after(src.text_train["eng_Latn"], 300)
    src.text_train["eng_Latn"] = eng
    with pytest.raises(Crash):
        bs.build_mix(tmp_path / "b", s, src, setup(lid=True))
    partial = _state(tmp_path / "b")["phases"]["text"]["partial"]
    assert partial["lang"] == "eng_Latn" and partial["files"] >= 2
    m = bs.build_mix(tmp_path / "b", s, src, setup(lid=True))
    assert log["starts"] == [0, partial["files"]]
    same_output(tmp_path / "ref", tmp_path / "b")
    assert _strip(m) == _strip(ref)


# ------------------------------------------------------------------ decontamination

from quipu import decontam as dc  # noqa: E402

HE_PROMPT = '''from typing import List


def rolling_window_peaks(readings: List[float], width: int) -> List[float]:
    """ Return the largest reading in every window of `width` consecutive readings.
    >>> rolling_window_peaks([1.0, 3.0, 2.0, 5.0], 2)
    [3.0, 3.0, 5.0]
    """
'''
HE_SOLUTION = '''    peaks_found = []
    for start_index in range(len(readings) - width + 1):
        window_slice = readings[start_index:start_index + width]
        peaks_found.append(max(window_slice))
    return peaks_found
'''
MBPP_TEXT = "Write a function to merge overlapping calendar intervals and count the merged blocks."
MBPP_CODE = '''def merge_calendar_blocks(intervals):
    ordered_blocks = sorted(intervals, key=lambda pair: pair[0])
    merged_blocks = [list(ordered_blocks[0])]
    for block_start, block_end in ordered_blocks[1:]:
        if block_start <= merged_blocks[-1][1]:
            merged_blocks[-1][1] = max(merged_blocks[-1][1], block_end)
        else:
            merged_blocks.append([block_start, block_end])
    return len(merged_blocks), merged_blocks'''
SHORT_MBPP = "def add_two(a, b):\n    return a + b"


def planted_problems():
    return [dc.Problem(dc.HUMANEVAL, "HumanEval/900", (HE_PROMPT, HE_SOLUTION), HE_SOLUTION),
            dc.Problem(dc.MBPP, "901", (MBPP_CODE, MBPP_TEXT), MBPP_CODE),
            dc.Problem(dc.MBPP, "902", (SHORT_MBPP, "Write a function to add two numbers."),
                       SHORT_MBPP)]


def planted_decontam():
    return dc.Decontaminator(planted_problems(), revisions={dc.HUMANEVAL: "a" * 40,
                                                            dc.MBPP: "b" * 40})


def _respaced(code):
    """The same code with other whitespace: tabs, trailing spaces, blank lines."""
    lines = [line.replace("    ", "\t") + "   " for line in code.splitlines()]
    return "\n\n".join(lines)


def test_decontam_drops_a_humaneval_solution_with_other_whitespace():
    d = planted_decontam()
    doc = "import os\n\ndef helper():\n    pass\n\n" + _respaced(HE_SOLUTION) + "\n# end\n"
    assert d.find(doc) == (dc.HUMANEVAL, "HumanEval/900", "substring")
    assert d.find(HE_PROMPT.replace("\n", "\r\n") + "    pass\n")[:2] == (
        dc.HUMANEVAL, "HumanEval/900")


def test_decontam_drops_a_doc_with_most_of_an_mbpp_solutions_13_grams():
    d = planted_decontam()
    toks = dc.tokens(MBPP_CODE)
    grams = dc.ngrams(toks)
    lines = MBPP_CODE.splitlines()
    part = "\n".join(lines[:6])  # about 60% of the solution, not the whole of it
    got = len(grams & dc.ngrams(dc.tokens(part))) / len(grams)
    assert 0.5 <= got < 0.8, got
    assert dc.collapse(MBPP_CODE) not in dc.collapse(part)
    doc = "# utilities\nimport sys\n" + part + "\n    return None\n\nprint(sys.argv)\n"
    assert d.find(doc) == (dc.MBPP, "901", "ngrams")
    # Under half of the 13-grams: kept.
    few = "\n".join(lines[:3])
    assert len(grams & dc.ngrams(dc.tokens(few))) / len(grams) < 0.5
    assert d.find("# utilities\n" + few + "\n    pass\n") is None


def test_decontam_keeps_common_short_snippets_and_the_mbpp_text_is_a_needle():
    d = planted_decontam()
    assert d.find("def add(a, b):\n    return a + b\n") is None
    assert d.find(SHORT_MBPP + "\n") is None  # under 60 characters and 13 tokens
    assert d.find("x = 1\n") is None and d.find("") is None
    assert d.find(f"# {MBPP_TEXT}\ndef f():\n    pass\n") == (dc.MBPP, "901", "substring")


def test_decontam_prefilter_never_misses_a_match():
    """The anchor prefilter only skips work: with it off every problem is checked,
    and the answers are the same."""
    d = planted_decontam()
    docs = [_respaced(HE_SOLUTION), HE_PROMPT, "\n".join(MBPP_CODE.splitlines()[:6]),
            "def add(a, b): return a + b", MBPP_CODE[:200], "peaks_found = []"]
    for doc in docs:
        assert d.find(doc) == d.find(doc, prefilter=False), doc[:40]


def test_decontam_is_picklable_and_fingerprinted():
    import pickle

    d = planted_decontam()
    again = pickle.loads(pickle.dumps(d))
    assert again.fingerprint() == d.fingerprint()
    assert again.find(HE_SOLUTION) == d.find(HE_SOLUTION)
    other = dc.Decontaminator(planted_problems()[:2], revisions=d.revisions)
    assert other.fingerprint() != d.fingerprint()
    s = d.summary()
    assert s["problems"] == {dc.HUMANEVAL: 1, dc.MBPP: 2}
    assert s["revisions"] == {dc.HUMANEVAL: "a" * 40, dc.MBPP: "b" * 40}


def test_load_benchmarks_reads_humaneval_and_all_mbpp_sanitized_splits(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    he = [{"task_id": "HumanEval/0", "prompt": HE_PROMPT, "canonical_solution": HE_SOLUTION,
           "test": "", "entry_point": "rolling_window_peaks"}]
    calls = []

    def download(repo, filename, **kw):
        calls.append((repo, filename, kw.get("repo_type"), kw.get("revision")))
        p = tmp_path / repo.replace("/", "_") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if repo == dc.HUMANEVAL_DATASET:
            rows = he
        else:
            split = filename.split("/")[1].split("-")[0]
            rows = [{"source_file": "x", "task_id": 100 + len(split), "prompt": MBPP_TEXT,
                     "code": MBPP_CODE, "test_imports": [], "test_list": []}]
        pq.write_table(pa.Table.from_pylist(rows), p)
        return str(p)
    d, info = dc.load_benchmarks({dc.HUMANEVAL: "a" * 40, dc.MBPP: "b" * 40}, download)
    assert d.summary()["problems"] == {dc.HUMANEVAL: 1, dc.MBPP: 4}
    assert {c[1] for c in calls if c[0] == dc.MBPP_DATASET} == {
        f"sanitized/{s}-00000-of-00001.parquet" for s in ("prompt", "test", "train",
                                                          "validation")}
    assert all(c[2] == "dataset" for c in calls)
    assert info[dc.HUMANEVAL]["revision"] == "a" * 40 and info[dc.MBPP]["revision"] == "b" * 40
    assert info[dc.MBPP]["license"] == "CC-BY-4.0" and info[dc.HUMANEVAL]["license"] == "MIT"
    assert len(info[dc.MBPP]["sha256"]) == 4
    with pytest.raises(ValueError, match="pinned"):
        dc.load_benchmarks({dc.HUMANEVAL: "", dc.MBPP: "b" * 40}, download)


def _planted_world(extra_val=()):
    """world() with planted benchmark copies in code train, code val and English val."""
    src = world(lid=True, code_val_extra=[
        {"code": "# copied\n" + _respaced(HE_SOLUTION), "language": "Python", "license": "mit",
         "path": "v/he.py", "file": 900}])
    rows = code_rows(12_000)
    planted = [
        {"code": "import os\n" + _respaced(HE_SOLUTION) + "\n# x\n", "language": "Python",
         "license": "mit", "path": "a/he.py", "file": 0},
        {"code": "# utils\n" + "\n".join(MBPP_CODE.splitlines()[:6]) + "\n    return 0\n",
         "language": "Python", "license": "mit", "path": "a/mbpp.py", "file": 1},
        {"code": f"// {MBPP_TEXT}\nfunction f() {{ return 1; }}\n", "language": "JavaScript",
         "license": "mit", "path": "a/m.js", "file": 1},
        {"code": "def add(a, b):\n    return a + b\n", "language": "Python", "license": "mit",
         "path": "a/add.py", "file": 1},
    ]
    rows[5:5] = planted[:1]
    rows[600:600] = planted[1:]
    src = src._replace(code_train=from_file(rows))
    val = list(src.text_val["eng_Latn"]())
    val.insert(3, {"text": "@eng_Latn:0.95 A copied answer: " + HE_SOLUTION})
    src.text_val["eng_Latn"] = lambda: iter(val)
    return src


def test_the_build_drops_planted_benchmark_problems_and_counts_them(tmp_path):
    s = sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, lid=FakeLid(), lid_threshold=0.5,
                    decontam=planted_decontam())
    m = bs.build_mix(tmp_path, spec(), _planted_world(), s)
    dropped = m["decontamination"]["dropped"]
    assert dropped["train_code"] == {dc.HUMANEVAL: 1, dc.MBPP: 2}
    assert dropped["code_val"] == {dc.HUMANEVAL: 1, dc.MBPP: 0}
    assert dropped["val"] == {dc.HUMANEVAL: 1, dc.MBPP: 0}
    assert m["decontamination"]["index"]["revisions"] == {dc.HUMANEVAL: "a" * 40,
                                                          dc.MBPP: "b" * 40}
    assert m["code"]["stats"]["dropped_contamination"] == 3
    docs = [d for split in ("train", "code_val", "val") for d in decode_docs(tmp_path / split)]
    assert not any("peaks_found.append" in d or "merged_blocks" in d or MBPP_TEXT in d
                   for d in docs)
    assert any(d.startswith("def add(a, b):") for d in decode_docs(tmp_path / "train"))


def test_decontamination_is_the_same_with_1_and_3_workers(tmp_path):
    def s(workers):
        return spec(workers=workers, batch_docs=8)
    su = sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, lid=FakeLid(), lid_threshold=0.5,
                     decontam=planted_decontam())
    m1 = bs.build_mix(tmp_path / "w1", s(1), _planted_world(), su)
    m3 = bs.build_mix(tmp_path / "w3", s(3), _planted_world(), su)
    same_output(tmp_path / "w1", tmp_path / "w3")
    strip = ("build_started", "build_finished", "build_seconds", "workers")
    assert ({k: v for k, v in m1.items() if k not in strip}
            == {k: v for k, v in m3.items() if k not in strip})
    assert m1["decontamination"]["dropped"]["train_code"][dc.HUMANEVAL] == 1


@pytest.mark.skipif(not TOKENIZER_JSON.is_file(), reason="artifacts/tokenizer/tokenizer.json absent")
def test_run_mix_loads_the_benchmarks_at_the_pinned_revisions(monkeypatch):
    import huggingface_hub

    from quipu.config import load_config

    monkeypatch.chdir(ROOT)
    cfg = load_config(ROOT / "configs" / "quipu-moe.toml")
    fake_hub(monkeypatch)
    calls = []

    class Stop(Exception):
        pass

    def download(repo, filename, **kw):
        calls.append((repo, filename, kw.get("repo_type"), kw.get("revision")))
        raise Stop()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(Stop):
        bs.run_mix(cfg, bs.build_parser().parse_args([]))
    assert calls == [(dc.HUMANEVAL_DATASET, dc.BENCHMARKS[dc.HUMANEVAL]["files"][0], "dataset",
                      cfg.data.humaneval_revision)]


def test_a_changed_decontamination_index_cannot_be_resumed(tmp_path):
    src = _planted_world()
    code, _ = crash_code_at(src.code_train, at_file=2)
    s = sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, decontam=planted_decontam())
    with pytest.raises(Crash):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), src._replace(code_train=code), s)
    other = dc.Decontaminator(planted_problems()[:1], revisions={})
    with pytest.raises(sm.ResumeError, match="decontamination"):
        bs.build_mix(tmp_path, spec(**RESUME_SPEC), src,
                     sm.DocSetup(tokenizer=CharTok, max_doc_tokens=2_000, decontam=other))
