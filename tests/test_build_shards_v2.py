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
import random
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
        code_train=lambda: iter(code_train),
        code_val=lambda: iter(code_val),
        text_train={k: (lambda v=v: iter(v)) for k, v in train.items()},
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
    src = src._replace(code_train=lambda: iter(rows))
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
    src = world()._replace(code_train=lambda: iter(rows))
    m = bs.build_mix(tmp_path, spec(train_tokens=100_000), src, setup())
    st = m["code"]["stats"]
    drops = sum(v for k, v in st.items()
                if k.startswith(("dropped_language", "dropped_license", "skipped_")))
    assert st["dropped_language"] >= 5
    # Every row counted reached a decision or was filtered on the way to one.
    assert st["rows_scanned"] == st["offers"] + drops - st.get("skipped_quota", 0) \
        - st.get("skipped_too_long", 0)
    assert m["code"]["last_file_read"] == rows[st["rows_scanned"] - 6]["file"]


class LocalFS:
    """HfFileSystem stand-in: every code file path opens the same local parquet."""

    def __init__(self, path):
        self.path = path

    def open(self, path, mode="rb", block_size=None):
        return open(self.path, mode)


def test_real_code_reader_filters_in_pyarrow_and_attaches_counts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"code": f"x{i}", "language": ["Python", "Ruby", "HTML"][i % 3],
             "license": "mit" if i % 4 else "gpl-3.0", "path": f"a{i}.py"} for i in range(40)]
    rows[3]["language"] = None
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "f.parquet", row_group_size=10)
    source = bs.mix_code_rows(LocalFS(tmp_path / "f.parquet"), "repo", "rev", range(0, 2), 2,
                              ["Python", "HTML"], ["mit"])
    got = list(source())
    kept = [r for r in rows if r["language"] in ("Python", "HTML") and r["license"] == "mit"]
    assert [r["code"] for r in got] == [r["code"] for r in kept] * 2
    assert [r["file"] for r in got] == [0] * len(kept) + [1] * len(kept)
    counts = Counter()
    for r in got:
        counts.update(r.get(bs.FILTERED) or {})
    dropped_lang = sum(r["language"] not in ("Python", "HTML") for r in rows)
    assert counts == {"rows_scanned": 2 * (40 - len(kept)),
                      "dropped_language": 2 * dropped_lang,
                      "dropped_license": 2 * (40 - len(kept) - dropped_lang)}
    # code_offers adds the attached counts to its own: every row is counted once.
    offers = list(bs.code_offers(iter(got), ["Python", "HTML"], ["mit"]))
    total = Counter()
    for o in offers:
        total.update(o.meta["counts"])
    assert total["rows_scanned"] == 80


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
        code_train=lambda: iter(code[:8000]), code_val=lambda: iter(code[8000:]),
        text_train={x: (lambda x=x: iter(rows(x, 800))) for x in TEXT_W},
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
