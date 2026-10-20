"""Low-memory training on T4 (spec docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md)."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


def smoke_cfg(tmp_path, model=None, **train):
    over = {"data": {"shard_dir": str(tmp_path / "shards")},
            "train": {"ckpt_dir": str(tmp_path / "ckpt"), **train}}
    if model:
        over["model"] = model
    return load_config(SMOKE, over)


# ---- config ----------------------------------------------------------------------

def test_new_train_fields_default_to_todays_behaviour(tmp_path):
    t = smoke_cfg(tmp_path).train
    assert t.precision == "bf16"
    assert t.optimizer_state_dtype == "fp32"
    assert t.activation_checkpoint == "none"
    assert t.loss_chunk_tokens == 0


def test_t4_settings_load(tmp_path):
    t = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16",
                  optimizer_state_dtype="bf16", activation_checkpoint="block",
                  loss_chunk_tokens=300).train
    assert (t.precision, t.optimizer_state_dtype, t.activation_checkpoint,
            t.loss_chunk_tokens) == ("fp16", "bf16", "block", 300)


@pytest.mark.parametrize("field,value", [
    ("precision", "fp32"), ("optimizer_state_dtype", "fp16"),
    ("activation_checkpoint", "layer"), ("loss_chunk_tokens", -1),
])
def test_bad_values_are_refused(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        smoke_cfg(tmp_path, model={"attnres_blocks": 0}, **{field: value})


def test_block_checkpointing_refuses_attnres(tmp_path):
    # The smoke config has attnres_blocks = 2; AttnRes has its own checkpoint option.
    with pytest.raises(ValueError, match="activation_checkpoint"):
        smoke_cfg(tmp_path, activation_checkpoint="block")


def test_apply_precision_leaves_the_model_alone_for_fp16():
    from quipu.fp8 import apply_precision
    m = torch.nn.Linear(4, 4)
    assert apply_precision(m, "fp16", "cpu") is m


# ---- loss scaler -------------------------------------------------------------------

from quipu.lossscale import LossScaler


def test_disabled_scaler_is_identity():
    s = LossScaler(enabled=False)
    assert s.scale == 1.0
    p = torch.nn.Parameter(torch.ones(3)); p.grad = torch.full((3,), 4.0)
    s.unscale_([p])
    assert torch.equal(p.grad, torch.full((3,), 4.0))


def test_unscale_divides_grads():
    s = LossScaler(enabled=True, init_scale=8.0)
    p = torch.nn.Parameter(torch.ones(3)); p.grad = torch.full((3,), 4.0)
    q = torch.nn.Parameter(torch.ones(2))       # no grad: skipped
    s.unscale_([p, q])
    assert torch.equal(p.grad, torch.full((3,), 0.5)) and q.grad is None


def test_overflow_halves_and_resets_the_clean_count():
    s = LossScaler(enabled=True, init_scale=1024.0, growth_interval=3)
    s.good_step(); s.good_step()
    s.overflow()
    assert s.scale == 512.0 and s.clean_steps == 0


def test_growth_after_interval():
    s = LossScaler(enabled=True, init_scale=2.0, growth_interval=3)
    for _ in range(3):
        s.good_step()
    assert s.scale == 4.0 and s.clean_steps == 0


def test_scale_never_goes_below_the_floor():
    s = LossScaler(enabled=True, init_scale=2.0, floor=1.0)
    assert not s.at_floor
    s.overflow()
    assert s.scale == 1.0 and s.at_floor
    s.overflow()
    assert s.scale == 1.0


def test_state_round_trips():
    s = LossScaler(enabled=True, init_scale=64.0, growth_interval=5)
    s.overflow(); s.good_step()
    t = LossScaler(enabled=True)
    t.load_state_dict(s.state_dict())
    assert (t.scale, t.clean_steps) == (32.0, 1)


# ---- Muon ----------------------------------------------------------------------------

from quipu.optim import Muon, newton_schulz
from quipu.optim.muon import cuda_work_dtype


def test_cuda_work_dtype_by_capability():
    assert cuda_work_dtype((7, 5)) is torch.float16      # T4
    assert cuda_work_dtype((8, 0)) is torch.bfloat16     # A100
    assert cuda_work_dtype((12, 0)) is torch.bfloat16    # RTX 50xx


def test_newton_schulz_in_fp16_matches_fp32():
    g = torch.randn(48, 80, generator=torch.Generator().manual_seed(3))
    ref = newton_schulz(g, work_dtype=torch.float32)
    out = newton_schulz(g, work_dtype=torch.float16)
    assert out.dtype == g.dtype
    assert (out - ref).norm() / ref.norm() < 0.02


def _muon_run(state_dtype, steps=20):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 48))
    opt = Muon([{"params": [w], "state_dtype": state_dtype}], lr=0.02)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        w.grad = torch.randn(32, 48, generator=gen)
        opt.step()
    return w.detach().clone(), opt


def test_bf16_momentum_tracks_fp32():
    w32, _ = _muon_run("fp32")
    w16, opt = _muon_run("bf16")
    buf = next(iter(opt.state.values()))["momentum_buffer"]
    assert buf.dtype is torch.bfloat16
    assert (w16 - w32).norm() / w32.norm() < 0.01


def test_cast_state_converts_loaded_momentum():
    _, opt = _muon_run("fp32", steps=2)
    for g in opt.param_groups:
        g["state_dtype"] = "bf16"
    opt.cast_state_()
    buf = next(iter(opt.state.values()))["momentum_buffer"]
    assert buf.dtype is torch.bfloat16


def test_build_optimizers_passes_state_dtype(tmp_path):
    from quipu.model_factory import build_model
    from quipu.optim import build_optimizers
    cfg = smoke_cfg(tmp_path, optimizer_state_dtype="bf16")
    opts = build_optimizers(build_model(cfg.model), cfg.train)
    muon = [o for o in opts if isinstance(o, Muon)][0]
    assert {g["state_dtype"] for g in muon.param_groups} == {"bf16"}


@pytest.mark.parametrize("scale", [1e-6, 1e4])
def test_newton_schulz_fp16_survives_extreme_scales(scale):
    g = torch.randn(48, 80, generator=torch.Generator().manual_seed(5)) * scale
    ref = newton_schulz(g, work_dtype=torch.float32)
    out = newton_schulz(g, work_dtype=torch.float16)
    assert torch.isfinite(out).all()
    assert (out - ref).norm() / ref.norm() < 0.02


# ---- chunked loss --------------------------------------------------------------------

import torch.nn.functional as F
from quipu.lossfn import chunked_cross_entropy


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_chunked_ce_equals_full_ce_and_grads(chunk):
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(2, 30, 16, generator=gen, requires_grad=True)
    w = torch.randn(50, 16, generator=gen, requires_grad=True)
    y = torch.randint(50, (2, 30), generator=gen)
    y[0, :5] = -100                                   # ignored targets (SFT)
    full = F.cross_entropy((h @ w.T).view(-1, 50), y.view(-1), ignore_index=-100,
                           reduction="sum")
    gh, gw = torch.autograd.grad(full, (h, w))
    out = chunked_cross_entropy(h, w, y, chunk, ignore_index=-100)
    ch, cw = torch.autograd.grad(out, (h, w))
    torch.testing.assert_close(out, full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ch, gh, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cw, gw, rtol=1e-5, atol=1e-6)


def test_models_return_hidden_matching_logits(tmp_path):
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    torch.manual_seed(0)
    m = build_model(cfg.model).eval()
    x = torch.randint(cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        logits = m(x)
        h = m(x, return_hidden=True)
    torch.testing.assert_close(m.lm_head(h), logits)
