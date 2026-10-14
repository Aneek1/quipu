"""scripts/build_pool.py end to end on local-only sources (no network): the real
tokenizer when it is present, a fake decontaminator, two JSON-lines sources."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_pool", ROOT / "scripts" / "build_pool.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)

from quipu import pool  # noqa: E402

TOKENIZER = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
needs_tokenizer = pytest.mark.skipif(not TOKENIZER.exists(), reason="no trained tokenizer")


class NoDecontam:
    def find(self, text):
        return ("mbpp", "1", "substring") if "CONTAMINATED" in text else None

    def fingerprint(self):
        return "none"

    def summary(self):
        return {"problems": {}}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def local_pool(tmp_path):
    data = tmp_path / "data"
    write_jsonl(data / "a" / "part-0.jsonl",
                [{"text": f"English document number {i}. " * 5} for i in range(300)])
    write_jsonl(data / "b" / "part-0.jsonl",
                [{"text": f"def f{i}(x):\n    return x + {i}\n"} for i in range(300)]
                + [{"text": "CONTAMINATED"}])
    cfg = tmp_path / "pool.toml"
    cfg.write_text(f"""
[pool]
tokenizer = "{TOKENIZER.as_posix()}"
shard_tokens = 2000
chunk_tokens = 6000
max_epochs = 4
humaneval_revision = "{'0' * 40}"
mbpp_revision = "{'1' * 40}"
[slices]
text = 70
code = 30
[sources.a]
kind = "local_jsonl"
slice = "text"
weight = 1
path = "{(data / 'a').as_posix()}"
files = ["part-*.jsonl"]
[sources.b]
kind = "local_jsonl"
slice = "code"
weight = 1
path = "{(data / 'b').as_posix()}"
files = ["part-*.jsonl"]
""", encoding="utf-8")
    return cfg


@needs_tokenizer
def test_run_builds_and_merges_chunks_that_continue(local_pool, tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "load_decontam", lambda cfg: (NoDecontam(), {"fake": True}))
    out = tmp_path / "pool"
    bp.main(["run", "--config", str(local_pool), "--out", str(out), "--no-lid-filter",
             "--workers", "0"])
    bp.main(["run", "--config", str(local_pool), "--out", str(out), "--no-lid-filter",
             "--workers", "0", "--chunk-tokens", "3000"])
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [c["index"] for c in m["chunks"]] == [0, 1]
    assert m["chunks"][0]["tokens"] >= 6000 and m["chunks"][1]["tokens"] >= 3000
    assert m["chunks"][1]["shards"][0] == m["chunks"][0]["shards"][1] + 1
    assert m["chunks"][0]["by_source"]["a"]["share"] == pytest.approx(0.7, abs=0.02)
    assert m["sources"]["b"]["stats"]["kept"] > 0
    assert m["provenance"]["tokenizer"]["sha256"]
    files = sorted((out / "mixed").glob("shard_*.bin"))
    assert sum(f.stat().st_size // 2 for f in files) == m["total_tokens"]


@needs_tokenizer
def test_build_one_source_then_merge_with_fit(local_pool, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bp, "load_decontam", lambda cfg: (NoDecontam(), {"fake": True}))
    out = tmp_path / "pool"
    for name in ("a", "b"):
        bp.main(["build", "--config", str(local_pool), "--out", str(out), "--source", name,
                 "--target", "1000000", "--no-lid-filter", "--workers", "0"])
    st = pool.SourceState.load(out / "sources" / "b")
    assert st.exhausted and st.stats["dropped_contamination"] == 1
    bp.main(["merge", "--config", str(local_pool), "--out", str(out), "--fit"])
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["total_tokens"] > 0


@needs_tokenizer
def test_smoke_prints_tokens_shares_and_throughput(local_pool, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bp, "load_decontam", lambda cfg: (NoDecontam(), {"fake": True}))
    out = tmp_path / "smoke"
    report = bp.main(["smoke", "--config", str(local_pool), "--out", str(out), "--docs", "40",
                      "--no-lid-filter"])
    text = capsys.readouterr().out
    assert "source" in text and "share" in text and "tok/s/core" in text
    assert set(report["sources"]) == {"a", "b"}
    for r in report["sources"].values():
        assert r["documents"] == 40 and r["tokens"] > 0 and r["resumed"]
    assert report["chunk"]["tokens"] > 0
    # the smoke run's two halves continue each other: nothing read twice
    assert report["sources"]["a"]["duplicates"] == 0


def test_plan_reports_build_targets(local_pool, tmp_path, capsys):
    out = tmp_path / "pool"
    t = bp.main(["plan", "--config", str(local_pool), "--out", str(out),
                 "--chunk-tokens", "1000"])
    assert t == {"a": 700, "b": 300}


def test_the_lid_filter_needs_fasttext_only_for_sources_that_use_it(local_pool):
    cfg = pool.load_pool_config(local_pool)
    assert bp.lid_sources(cfg) == []
    real = pool.load_pool_config(ROOT / "configs" / "data-v3.toml")
    assert "fw2-jpn_Jpan" in bp.lid_sources(real) and "fw2-urd_Arab" not in bp.lid_sources(real)
