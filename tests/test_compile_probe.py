"""scripts/remote/compile_probe.py: parsing a probe's output and run log, and the
micro_batch / compile decisions (no GPU: hand-made outputs)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compile_probe", ROOT / "scripts" / "remote" / "compile_probe.py")
cp = importlib.util.module_from_spec(_spec)
sys.modules["compile_probe"] = cp
_spec.loader.exec_module(cp)

LOG = """compile: on (inductor)
step 10/20  loss 9.1000  lr 1.00e-04  grad 1.000  40,000 tok/s
V0930 12:00:00 torch/_dynamo/guards.py Recompiling function forward in model_moe.py:88
    triggered by the following guard failure(s):
step 20/20  loss 8.9000  lr 1.00e-04  grad 1.000  52,000 tok/s
W0930 torch._dynamo hit config.recompile_limit (8)
checkpoint step 20 saved in 41.5 s (12.0 GB)
"""


def test_parse_probe_reads_throughput_saves_recompiles_and_vram():
    run_log = {"compile": "on (inductor)",
               "compile_stats": {"step": 20, "unique_graphs": 40, "graph_breaks": 96,
                                 "recompile_limit_hits": 1}}
    p = cp.parse_probe(LOG, run_log, 0, [1000, 29000, 28000], 32607)
    assert p["ok"] and not p["oom"]
    assert p["step_tokens_per_s"] == [40000.0, 52000.0]
    assert p["tokens_per_s"] == 52000.0                  # the first step line is left out
    assert p["peak_vram_mib"] == 29000 and p["vram_share"] == pytest.approx(29000 / 32607)
    assert p["recompile_lines"] == 1 and p["recompile_limit_lines"] == 1
    assert p["max_save_s"] == 41.5
    assert p["compile"] == "on (inductor)" and p["compile_stats"]["graph_breaks"] == 96


def test_parse_probe_flags_out_of_memory_and_a_missing_run_log():
    p = cp.parse_probe("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate",
                       None, 1)
    assert p["oom"] and not p["ok"] and p["tokens_per_s"] is None
    assert p["peak_vram_mib"] is None and p["compile"] is None


def probe(tps, compile_="off", ok=True, share=0.8, oom=False, recompiles=0, limit=0):
    return {"ok": ok, "oom": oom, "tokens_per_s": tps, "vram_share": share,
            "compile": compile_, "recompile_lines": recompiles,
            "recompile_limit_lines": limit, "compile_stats": None}


def test_micro_batch_8_is_taken_only_when_it_fits_and_is_faster():
    r = {"full-mb8-c0": probe(60_000), "full-mb4-c0": probe(55_000),
         "full-mb8-c1": probe(70_000, "on (inductor)"), "full-mb4-c1": probe(60_000, "on")}
    d = cp.decide(r)
    assert d["micro_batch"] == 8 and d["compile"] is True
    assert d["run_moe_overrides"] == ["--override train.micro_batch=8"]
    assert d["full_tokens_per_s"] == 70_000
    r["full-mb8-c0"] = probe(None, ok=False, oom=True, share=None)
    d = cp.decide(r)
    assert d["micro_batch"] == 4 and any("out of memory" in n for n in d["notes"])
    r["full-mb8-c0"] = probe(60_000, share=0.97)            # over the headroom
    assert cp.decide(r)["micro_batch"] == 4
    r["full-mb8-c0"] = probe(54_000)                         # fits, but slower
    d = cp.decide(r)
    assert d["micro_batch"] == 4 and any("not faster" in n for n in d["notes"])


def test_compile_is_kept_only_for_ten_percent_and_no_storm():
    base = {"full-mb4-c0": probe(50_000), "full-mb8-c0": probe(None, ok=False, oom=True)}
    d = cp.decide({**base, "full-mb4-c1": probe(54_000, "on (inductor)")})
    assert d["compile"] is False and "needs >= 10%" in d["compile_reason"]
    assert d["run_moe_overrides"] == ["--override train.compile=false"]
    assert d["full_tokens_per_s"] == 50_000
    d = cp.decide({**base, "full-mb4-c1": probe(56_000, "on (inductor)")})
    assert d["compile"] is True and d["run_moe_overrides"] == []
    d = cp.decide({**base, "full-mb4-c1": probe(70_000, "on (inductor)", limit=2)})
    assert d["compile"] is False and "recompile_limit" in d["compile_reason"]
    d = cp.decide({**base, "full-mb4-c1": probe(70_000, "on (inductor)", recompiles=65)})
    assert d["compile"] is False and "65 recompiles" in d["compile_reason"]
    d = cp.decide({**base, "full-mb4-c1": probe(70_000, "fell back: RuntimeError: x")})
    assert d["compile"] is False and "did not stay on" in d["compile_reason"]


def test_a_storm_counts_the_run_logs_recompile_limit_hits_too():
    p = probe(1.0, "on")
    p["compile_stats"] = {"recompile_limit_hits": 3}
    assert cp.storm(p, 16) == "3 recompile_limit hit(s)"
    assert cp.storm(probe(1.0, "on", recompiles=64), 16) is None     # 4 x 16 is allowed


def test_no_eager_probe_fits_keeps_the_config_and_says_so():
    d = cp.decide({"full-mb8-c0": probe(None, ok=False), "full-mb4-c0": probe(None, ok=False)})
    assert d["micro_batch"] == 4 and d["micro_batch_ok"] is False
    assert any("do NOT start" in n for n in d["notes"])


def test_ab_throughput_attnres_cost_and_the_dry_run_command():
    d = cp.decide({"full-mb4-c0": probe(50_000), "ab-c0": probe(90_000),
                   "ab-attn-c0": probe(81_000)})
    assert d["ab_tokens_per_s"] == 90_000 and d["attnres_cost"] == pytest.approx(0.1)
    cmd = cp.ab_dry_run_command(d["ab_tokens_per_s"], 0.55)
    assert cmd.endswith("--dry-run --tokens-per-second 90000 --usd-per-hour 0.55")
    assert cp.ab_dry_run_command(None, 0.55) is None


def test_the_probe_list_covers_both_micro_batches_compile_and_attnres():
    names = {p["name"]: p for p in cp.PROBES}
    assert {(p["micro_batch"], p["compile"]) for n, p in names.items()
            if n.startswith("full")} == {(8, True), (8, False), (4, True), (4, False)}
    assert all(not p["compile"] for n, p in names.items() if n.startswith("ab"))
    assert names["ab-attn-c0"]["overrides"] == ["model.attnres_blocks=4"]


def test_usage_errors():
    assert cp.main([]) == 2
    assert cp.main(["--usd-per-hour", "0.5", "--only", "nope"]) == 2
