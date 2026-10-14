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
