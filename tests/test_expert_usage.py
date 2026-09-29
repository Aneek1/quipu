"""M10: scripts/expert_usage.py on a tiny quipu-moe (CPU)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from quipu import evalsets
from tests.moe_fixtures import fresh_model, save_final, tiny_moe, write_eval_shards

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("expert_usage", ROOT / "scripts" / "expert_usage.py")
expert_usage = importlib.util.module_from_spec(_SPEC)
sys.modules["expert_usage"] = expert_usage
_SPEC.loader.exec_module(expert_usage)


def test_counts_sum_to_tokens_times_top_k_in_every_layer(tmp_path):
    cfg, tok = tiny_moe(tmp_path, moe_dispatch="padded")
    write_eval_shards(cfg, tok, tokens=300)
    model = fresh_model(cfg, 1)
    batches = evalsets.split_batches(Path(cfg.data.shard_dir) / "val", 2, cfg.model.context, 3)
    counts, tokens = expert_usage.count_usage(model, batches)
    assert counts.shape == (cfg.model.n_layer, cfg.model.n_experts)
    assert tokens == sum(x.numel() for x, _ in batches)
    assert counts.sum(1).tolist() == [tokens * cfg.model.top_k] * cfg.model.n_layer
    assert all(b.moe.dispatch == "padded" for b in model.blocks)   # restored


def test_counts_equal_a_per_batch_router_recount(tmp_path):
    cfg, tok = tiny_moe(tmp_path)
    model = fresh_model(cfg, 2).eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    counts, _ = expert_usage.count_usage(model, [(x, x), (x, x)])
    with torch.no_grad():
        model(x)
    once = torch.stack([s.counts for s in model.last_stats])
    assert torch.equal(counts, 2 * once)


def test_analyse_flags_dead_and_overloaded_experts():
    counts = torch.tensor([[0, 5, 5, 30], [10, 10, 10, 10]])   # 20 tokens, top-2
    u = expert_usage.analyse(counts, tokens=20, top_k=2)
    assert u["frequency"][1] == [0.25] * 4
    assert [(d["layer"], d["expert"]) for d in u["dead"]] == [(0, 0)]
    assert [(d["layer"], d["expert"]) for d in u["overloaded"]] == []
    u = expert_usage.analyse(torch.tensor([[1, 1, 1, 37]]), tokens=20, top_k=2)
    assert [(d["layer"], d["expert"]) for d in u["overloaded"]] == [(0, 3)]  # 3.7x


def test_main_writes_a_table_with_one_row_per_expert_and_a_heatmap(tmp_path, monkeypatch):
    cfg, tok = tiny_moe(tmp_path)
    write_eval_shards(cfg, tok, tokens=300)
    save_final(cfg, 5, fresh_model(cfg, 3))
    monkeypatch.setattr(expert_usage, "load_config", lambda _p: cfg)
    out = tmp_path / "experts"
    assert expert_usage.main(["--config", "x.toml", "--out", str(out), "--batches", "2",
                              "--device", "cpu"]) == 0
    md = (out / "summary.md").read_text(encoding="utf-8")
    for name in ["English", "Indonesian", "Chinese (Simplified)", "Tamil", "Code"]:
        assert f"## {name}" in md
    english = md.split("## English")[1].split("## ")[0]
    rows = [l for l in english.splitlines() if l.startswith("| ") and l[2].isdigit()]
    assert len(rows) == cfg.model.n_experts
    assert "| expert | L0 | L1 |" in english
    data = json.loads((out / "usage.json").read_text(encoding="utf-8"))
    for u in data["splits"].values():
        assert all(abs(sum(row) - 1.0) < 1e-9 for row in u["frequency"])
    assert (out / "heatmap.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
