"""scripts/revive_experts.py: plan and apply expert splits on a tiny MoE checkpoint."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import load_config
from quipu.data import write_shard
from quipu.model_factory import build_model
from quipu.optim.groups import build_optimizers

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("revive_experts",
                                               ROOT / "scripts" / "revive_experts.py")
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)

OVERRIDES = {"model": {"attnres_blocks": 0}}
DEAD = (3, 5)


def _cfg():
    return load_config(ROOT / "configs" / "quipu-moe-smoke.toml", OVERRIDES)


def _trained_checkpoint(cfg):
    """A training-style checkpoint after one real optimizer step (so Muon and AdamW
    hold state), with experts DEAD of layer 0 starved: router rows far negative."""
    torch.manual_seed(0)
    model = build_model(cfg.model)
    opts = build_optimizers(model, cfg.train)
    idx = torch.randint(0, cfg.model.vocab_size, (2, 64))
    model(idx).float().pow(2).mean().backward()
    for o in opts:
        o.step()
    with torch.no_grad():
        bias = torch.linspace(-0.2, 0.2, cfg.model.n_experts)
        bias[list(DEAD)] = -10.0          # scores are in [0, 1]: never in any Top-k
        model.blocks[0].moe.balancer.bias.copy_(bias)
    return {"step": 1, "model": model.state_dict(),
            "optimizers": [o.state_dict() for o in opts],
            "moe_health_streak": torch.full((cfg.model.n_layer, cfg.model.n_experts), 7),
            "moe_interval": (torch.ones(cfg.model.n_layer, cfg.model.n_experts),
                             torch.ones(cfg.model.n_layer))}


def test_plan_splits_gives_each_dead_slot_the_busiest_donor_per_copy():
    load = torch.tensor([3.0, 0.0, 1.0, 0.05, 2.0])
    # 3.0 -> slot 1 (3.0 then 1.5 per copy); next busiest per copy is 2.0 -> slot 3.
    assert rv.plan_splits(load, 0.1) == [(1, 0), (3, 4)]
    assert rv.plan_splits(torch.ones(4), 0.1) == []
    with pytest.raises(ValueError):
        rv.plan_splits(torch.zeros(3), 0.1)


def test_measure_load_sees_the_starved_experts(tmp_path):
    cfg = _cfg()
    ck = _trained_checkpoint(cfg)
    tokens = np.random.default_rng(0).integers(0, cfg.model.vocab_size, 20_000).astype(np.uint16)
    load = rv.measure_load(cfg, ck["model"], 0, tokens, seqs=8, seq_len=64, seed=0)
    assert load.shape == (cfg.model.n_experts,)
    assert math.isclose(float(load.mean()), 1.0, rel_tol=1e-6)
    for e in DEAD:
        assert load[e] == 0


def test_apply_splits_copies_weights_and_optimizer_state():
    cfg = _cfg()
    ck = _trained_checkpoint(cfg)
    before = {k: v.clone() for k, v in ck["model"].items()}
    pairs = [(3, 0), (5, 0)]
    rv.apply_splits(ck, cfg, 0, pairs, noise=0.01, seed=0)
    sd = ck["model"]
    pre = "blocks.0.moe."
    for w in ("gate", "up", "down"):
        k = pre + f"experts.{w}"
        for d, s in pairs:
            assert torch.equal(sd[k][d], before[k][s])
        keep = [e for e in range(cfg.model.n_experts) if e not in DEAD]
        assert torch.equal(sd[k][keep], before[k][keep])
    r = sd[pre + "router.weight"]
    for d, s in pairs:
        diff = (r[d] - before[pre + "router.weight"][s]).norm()
        assert 0 < diff < 0.1 * before[pre + "router.weight"][s].norm()
    assert torch.equal(r[0], before[pre + "router.weight"][0])     # donor untouched
    b = sd[pre + "balancer.bias"]
    for d, s in pairs:
        assert b[d] == b[s]                                         # copy takes donor's bias
    assert abs(float(b.mean())) < 1e-6
    # Layer 1 is not touched at all.
    for k in sd:
        if k.startswith("blocks.1."):
            assert torch.equal(sd[k], before[k]), k

    idx = rv.param_indices(cfg)
    for name in (pre + "experts.gate", pre + "router.weight"):
        o, i = idx[name]
        st = ck["optimizers"][o]["state"][i]
        tensors = [v for v in st.values() if torch.is_tensor(v) and v.dim() >= 1]
        assert tensors, name
        for v in tensors:
            for d, s in pairs:
                assert torch.equal(v[d], v[s]), name
    assert ck["moe_health_streak"][0].eq(0).all() and ck["moe_health_streak"][1].eq(7).all()
    assert ck["moe_interval"][0][0].eq(0).all() and ck["moe_interval"][0][1].eq(1).all()


def test_revived_checkpoint_loads_into_model_and_optimizers_and_revives(tmp_path):
    cfg = _cfg()
    ck = _trained_checkpoint(cfg)
    tokens = np.random.default_rng(1).integers(0, cfg.model.vocab_size, 20_000).astype(np.uint16)
    write_shard(tmp_path / "s.bin", tokens)
    src, out = tmp_path / "in.pt", tmp_path / "out.pt"
    torch.save(ck, src)
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        'inherit = "{}"\n[model]\nattnres_blocks = 0\n'.format(
            (ROOT / "configs" / "quipu-moe-smoke.toml").as_posix()), encoding="utf-8")
    argv = ["--config", str(cfg_path), "--checkpoint", str(src), "--tokens",
            str(tmp_path / "s.bin"), "--out", str(out), "--seqs", "8", "--seq-len", "64"]
    assert rv.main(argv) == 0
    new = torch.load(out, weights_only=False)
    model = build_model(cfg.model)
    model.load_state_dict(new["model"])                         # strict
    opts = build_optimizers(model, cfg.train)
    for o, s in zip(opts, new["optimizers"]):
        o.load_state_dict(s)
    after = rv.measure_load(cfg, new["model"], 0, tokens, seqs=8, seq_len=64, seed=3)
    for e in DEAD:
        assert after[e] > 0
    assert (out.parent / "out.pt.revive.json").exists()
    with pytest.raises(SystemExit):                             # never overwrites
        rv.main(argv)


def test_eval_loss_matches_a_plain_forward_of_the_model():
    cfg = _cfg()
    ck = _trained_checkpoint(cfg)
    tokens = np.random.default_rng(4).integers(0, cfg.model.vocab_size, 20_000).astype(np.uint16)
    got = rv.eval_loss(cfg, ck["model"], tokens, seqs=3, seq_len=64, seed=5)
    model = build_model(cfg.model)
    model.load_state_dict(ck["model"])
    model.eval()
    for blk in model.blocks:
        blk.moe.dispatch = "loop"
    starts = np.random.default_rng(5).integers(0, len(tokens) - 65, 3)
    want = []
    with torch.no_grad():
        for s in starts:
            row = torch.tensor(tokens[s:s + 65].astype(np.int64))
            want.append(float(torch.nn.functional.cross_entropy(
                model(row[None, :-1]).float()[0], row[1:])))
    assert math.isclose(got, sum(want) / 3, rel_tol=1e-5)


def test_report_has_loss_before_and_after_when_asked(tmp_path):
    import json
    cfg = _cfg()
    tokens = np.random.default_rng(6).integers(0, cfg.model.vocab_size, 20_000).astype(np.uint16)
    write_shard(tmp_path / "s.bin", tokens)
    torch.save(_trained_checkpoint(cfg), tmp_path / "in.pt")
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text('inherit = "{}"\n[model]\nattnres_blocks = 0\n'.format(
        (ROOT / "configs" / "quipu-moe-smoke.toml").as_posix()), encoding="utf-8")
    out = tmp_path / "out.pt"
    assert rv.main(["--config", str(cfg_path), "--checkpoint", str(tmp_path / "in.pt"),
                    "--tokens", str(tmp_path / "s.bin"), "--out", str(out), "--seqs", "8",
                    "--seq-len", "64", "--loss-seqs", "2"]) == 0
    rep = json.loads((tmp_path / "out.pt.revive.json").read_text())
    assert rep["loss_seqs"] == 2
    assert math.isfinite(rep["loss_before"]) and math.isfinite(rep["loss_after"])


# ---- scripts/remote/modal_revive.py: the in-place swap on the Volume ---------------

_mspec = importlib.util.spec_from_file_location(
    "modal_revive", ROOT / "scripts" / "remote" / "modal_revive.py")
mr = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(mr)


def _stable_dir(tmp_path):
    cfg = _cfg()
    stable = tmp_path / "stable"
    stable.mkdir()
    torch.save(_trained_checkpoint(cfg), stable / "step_000001.pt")
    torch.save({"file": "step_000001.pt"}, stable / "latest.pt")
    tokens = np.random.default_rng(2).integers(0, cfg.model.vocab_size, 200_000).astype(np.uint16)
    write_shard(tmp_path / "pool.bin", tokens)
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text('inherit = "{}"\n[model]\nattnres_blocks = 0\n'.format(
        (ROOT / "configs" / "quipu-moe-smoke.toml").as_posix()), encoding="utf-8")
    return stable, tmp_path / "pool.bin", str(cfg_path)


def test_modal_revive_copies_match_modal_train():
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "remote"))
    import modal_train
    assert mr.CONFIG == modal_train.CONFIG
    assert mr.PIP == modal_train.PIP


def test_revive_in_place_dry_run_writes_nothing(tmp_path):
    stable, pool, cfg = _stable_dir(tmp_path)
    before = sorted(p.name for p in stable.rglob("*"))
    r = mr.revive_in_place(stable, pool, dry_run=True, config=cfg, cwd=ROOT)
    assert r["written"] is False and r["pairs"]
    assert sorted(p.name for p in stable.rglob("*")) == before


def test_revive_in_place_swaps_and_keeps_the_original(tmp_path):
    stable, pool, cfg = _stable_dir(tmp_path)
    orig = (stable / "step_000001.pt").read_bytes()
    r = mr.revive_in_place(stable, pool, dry_run=False, config=cfg, cwd=ROOT)
    assert r["written"] is True
    assert (stable / "pre-revive" / "step_000001.pt").read_bytes() == orig
    assert (stable / "pre-revive" / "step_000001.pt.revive.json").exists()
    assert torch.load(stable / "latest.pt", weights_only=True) == {"file": "step_000001.pt"}
    new = torch.load(stable / "step_000001.pt", weights_only=False)
    assert not (stable / ".revive_tmp").exists()
    bias = new["model"]["blocks.0.moe.balancer.bias"]
    assert (bias[list(DEAD)] > -1).all()          # the -10 starvation bias is gone
    with pytest.raises(FileExistsError):
        mr.revive_in_place(stable, pool, dry_run=False, config=cfg, cwd=ROOT)
