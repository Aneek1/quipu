"""configs/quipu-moe-a1b.toml (spec 2026-10-14-quipu-moe-a1b-design.md) loads, and its
model is the size the spec says: counted on the meta device, so no memory is used."""
from pathlib import Path

import torch

from quipu.config import load_config
from quipu.model_factory import build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
A1B = CONFIGS / "quipu-moe-a1b.toml"
MOE = CONFIGS / "quipu-moe.toml"


def _count(model_cfg) -> tuple[int, int]:
    """(total, active): tied embedding once; routed experts at top_k / n_experts."""
    with torch.device("meta"):
        model = build_model(model_cfg)
    seen, total, routed = set(), 0, 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if ".moe.experts." in name:
            routed += p.numel()
    return total, total - routed + routed * model_cfg.top_k // model_cfg.n_experts


def test_count_matches_the_published_model():
    # The counting rule reproduces quipu-moe-1B-A149M's 998.0M / 148.7M.
    total, active = _count(load_config(MOE).model)
    assert round(total / 1e6, 1) == 998.0
    assert round(active / 1e6, 1) == 148.7


def test_a1b_loads_and_is_about_1b_active():
    cfg = load_config(A1B)
    total, active = _count(cfg.model)
    assert total == 5_040_334_848
    assert active == 1_026_385_920
    assert 0.95e9 <= active <= 1.1e9
    assert total <= 5.5e9   # the full-replica memory budget (spec, "Memory")


def test_a1b_train_settings():
    cfg = load_config(A1B)
    assert cfg.model.moe_dispatch == "grouped"
    assert cfg.train.optimizer == "muon"
    assert cfg.data.tokenizer == "artifacts/tokenizer-v3/tokenizer.json"
    assert cfg.data.shard_dir == "data/shards-v3"
    assert cfg.train.steps == 76_293
    assert cfg.train.grad_accum == 128
    # grad_accum must split over 1, 2 (the NCCL check) and 8 processes (quipu.train).
    for world in (1, 2, 4, 8):
        assert cfg.train.grad_accum % world == 0
    assert cfg.train.warmup_steps < cfg.train.milestones[0]
