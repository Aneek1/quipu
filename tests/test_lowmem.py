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


def test_reset_growth_keeps_the_scale():
    s = LossScaler(enabled=True, init_scale=64.0, growth_interval=5)
    s.good_step(); s.good_step()
    s.reset_growth()
    assert (s.scale, s.clean_steps) == (64.0, 0)
    off = LossScaler(enabled=False)
    off.reset_growth()
    assert (off.scale, off.clean_steps) == (1.0, 0)


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


# ---- activation checkpointing --------------------------------------------------------

def _grads_after_backward(model, x):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).view(-1, model.cfg.vocab_size), x.view(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                           if p.grad is not None}


def test_moe_block_checkpointing_changes_nothing(tmp_path):
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    torch.manual_seed(0)
    plain = build_model(cfg.model).train()
    ckpt = build_model(cfg.model).train()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.checkpoint_blocks = True
    x = torch.randint(cfg.model.vocab_size, (2, 32), generator=torch.Generator().manual_seed(1))
    l1, g1 = _grads_after_backward(plain, x)
    l2, g2 = _grads_after_backward(ckpt, x)
    torch.testing.assert_close(l2, l1)
    assert g1.keys() == g2.keys()
    for n in g1:
        torch.testing.assert_close(g2[n], g1[n], rtol=1e-5, atol=1e-7, msg=n)
    for b1, b2 in zip(plain.blocks, ckpt.blocks):
        torch.testing.assert_close(b2.moe._last_scores, b1.moe._last_scores)
    for s1, s2 in zip(plain.last_stats, ckpt.last_stats):
        assert torch.equal(s1.counts, s2.counts) and torch.equal(s1.dropped, s2.dropped)


def test_dense_block_checkpointing_changes_nothing():
    from quipu.model import Quipu
    mc = dataclasses.replace(load_config(ROOT / "configs" / "quipu-114m.toml").model,
                             vocab_size=256, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                             ffn_hidden=128, context=32)
    torch.manual_seed(0)
    plain, ckpt = Quipu(mc).train(), Quipu(mc).train()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.checkpoint_blocks = True
    x = torch.randint(256, (2, 32), generator=torch.Generator().manual_seed(1))
    l1, g1 = _grads_after_backward(plain, x)
    l2, g2 = _grads_after_backward(ckpt, x)
    torch.testing.assert_close(l2, l1)
    for n in g1:
        torch.testing.assert_close(g2[n], g1[n], rtol=1e-5, atol=1e-7, msg=n)


def test_checkpointing_is_off_in_eval_and_no_grad(tmp_path, monkeypatch):
    import quipu.model_moe as mm
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    m = build_model(cfg.model)
    m.checkpoint_blocks = True
    calls = []
    monkeypatch.setattr(mm, "checkpoint", lambda *a, **k: calls.append(1) or a[0](*a[1:]))
    x = torch.randint(cfg.model.vocab_size, (1, 8))
    with torch.no_grad():
        m.train()(x)
    m.eval()(x)
    assert calls == []


def test_block_checkpointing_refuses_compile(tmp_path):
    with pytest.raises(ValueError, match="activation_checkpoint.*compile|compile.*activation_checkpoint"):
        smoke_cfg(tmp_path, model={"attnres_blocks": 0}, activation_checkpoint="block",
                  compile=True)


def _count_engagement(model, module, monkeypatch, x):
    """Wrap the real checkpoint and every block.forward with counters; run one
    training forward + backward; return (checkpoint calls, forwards after fwd, after bwd)."""
    ck, fw = [0], [0]
    real = module.checkpoint

    def counting(*a, **k):
        ck[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(module, "checkpoint", counting)
    for b in model.blocks:
        orig = b.forward

        def wrapped(*a, _orig=orig, **k):
            fw[0] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(b, "forward", wrapped)
    model.train()
    loss = F.cross_entropy(model(x).view(-1, model.cfg.vocab_size), x.view(-1))
    after_fwd = fw[0]
    loss.backward()
    return ck[0], after_fwd, fw[0]


def test_moe_checkpointing_engages_and_recomputes(tmp_path, monkeypatch):
    import quipu.model_moe as mm
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    m = build_model(cfg.model)
    m.checkpoint_blocks = True
    x = torch.randint(cfg.model.vocab_size, (2, 16))
    n = cfg.model.n_layer
    assert _count_engagement(m, mm, monkeypatch, x) == (n, n, 2 * n)


def test_dense_checkpointing_engages_and_recomputes(monkeypatch):
    import quipu.model as qm
    mc = dataclasses.replace(load_config(ROOT / "configs" / "quipu-114m.toml").model,
                             vocab_size=256, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                             ffn_hidden=128, context=32)
    m = qm.Quipu(mc)
    m.checkpoint_blocks = True
    x = torch.randint(256, (2, 16))
    n = mc.n_layer
    assert _count_engagement(m, qm, monkeypatch, x) == (n, n, 2 * n)


# ---- eval dtype ----------------------------------------------------------------------

def test_eval_functions_take_amp_dtype():
    import inspect
    from quipu.eval import estimate_loss, mean_loss
    from quipu.train import evaluate_by_source
    for fn in (estimate_loss, mean_loss, evaluate_by_source):
        p = inspect.signature(fn).parameters["amp_dtype"]
        assert p.default is torch.bfloat16


# ---- trainer ---------------------------------------------------------------------------

from quipu.data import write_mask, write_shard
from quipu.train import Trainer, UsageError, check_device_support


def make_data(tmp_path, tokens=6000, seed=0):
    d = tmp_path / "data"
    write_shard(d / "shard_000.bin",
                np.random.RandomState(seed).randint(0, 512, tokens).astype(np.uint16))
    return d


def build(tmp_path, cfg, data, run_id="t", resume=False):
    return Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data, device="cpu",
                   run_dir=tmp_path / "runs", run_id=run_id, resume=resume)


def _losses(tmp_path, name, steps=3, **train):
    cfg = smoke_cfg(tmp_path / name, model={"attnres_blocks": 0}, **train)
    t = build(tmp_path / name, cfg, make_data(tmp_path / name))
    return [t.train_step() for _ in range(steps)], t


def test_memory_options_do_not_change_training_on_cpu(tmp_path):
    base, tb = _losses(tmp_path, "base")
    low, tl = _losses(tmp_path, "low", precision="fp16", optimizer_state_dtype="fp32",
                      activation_checkpoint="block", loss_chunk_tokens=300)
    np.testing.assert_allclose(low, base, rtol=1e-5)
    for (n, p), (_, q) in zip(tb.model.named_parameters(), tl.model.named_parameters()):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6, msg=n)


def test_bf16_state_trains(tmp_path):
    losses, t = _losses(tmp_path, "bf16state", steps=5, optimizer_state_dtype="bf16")
    assert all(math.isfinite(x) for x in losses)
    muon = t.optimizers[0]
    assert all(s["momentum_buffer"].dtype is torch.bfloat16 for s in muon.state.values())


def test_fp16_overflow_halves_the_scale_and_skips_without_counting(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    t = build(tmp_path, cfg, make_data(tmp_path))
    scale0 = t.scaler.scale
    p = next(t.model.parameters())
    h = p.register_hook(lambda g: torch.full_like(g, float("inf")))
    t.train_step()
    h.remove()
    assert t.step == 0 and t.scaler.scale == scale0 / 2
    assert t.nonfinite_streak == 0 and t.overflow_steps == 1
    t.train_step()
    assert t.step == 1


def test_fp16_poison_at_the_floor_counts_and_resets_the_growth_counter(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    t = build(tmp_path, cfg, make_data(tmp_path))
    t.scaler.scale = t.scaler.floor
    t.scaler.clean_steps = 7
    p = next(t.model.parameters())
    h = p.register_hook(lambda g: torch.full_like(g, float("inf")))
    t.train_step()
    h.remove()
    assert t.step == 0 and t.nonfinite_streak == 1 and t.overflow_steps == 0
    assert t.scaler.scale == t.scaler.floor and t.scaler.clean_steps == 0


def test_fp16_nonfinite_loss_above_the_floor_keeps_the_scale(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    t = build(tmp_path, cfg, make_data(tmp_path))
    t.scaler.clean_steps = 7
    scale0 = t.scaler.scale
    assert not t.scaler.at_floor
    model = t.forward_model
    t.forward_model = lambda x, **kw: model(x, **kw) * float("nan")
    t.train_step()
    t.forward_model = model
    assert t.step == 0 and t.nonfinite_streak == 1 and t.overflow_steps == 0
    assert t.scaler.scale == scale0 and t.scaler.clean_steps == 0
    t.train_step()
    assert t.step == 1 and t.nonfinite_streak == 0


def test_fp16_overflows_are_logged_and_checkpointed(tmp_path):
    import json
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    data = make_data(tmp_path)
    t = build(tmp_path, cfg, data)
    p = next(t.model.parameters())
    h = p.register_hook(lambda g: torch.full_like(g, float("inf")))
    t.train_step()
    h.remove()
    t.train_step()
    entry = t.log.record["steps"][-1]
    assert entry["loss_scale"] == t.scaler.scale and entry["overflows"] == 1
    t.save_checkpoint()
    r = build(tmp_path, cfg, data, run_id="t", resume=True)
    r.resume_from_latest()
    assert r.overflow_steps == 1
    on_disk = json.loads(t.log.path.read_text(encoding="utf-8"))["steps"][-1]
    assert on_disk["loss_scale"] == t.scaler.scale


def test_bf16_steps_log_no_loss_scale(tmp_path):
    _, t = _losses(tmp_path, "plain", steps=1)
    assert "loss_scale" not in t.log.record["steps"][-1]


def _sft_losses(tmp_path, name, **train):
    # Minimal masked blocks: 8 blocks of the smoke context, ~1 in 4 targets assistant.
    ctx = 256
    rng = np.random.RandomState(1)
    d = tmp_path / name / "chat"
    write_shard(d / "shard_000.bin", rng.randint(0, 512, 8 * ctx).astype(np.uint16))
    mask = (rng.rand(8 * ctx) < 0.25).astype(np.uint8)
    mask[1::ctx] = 1                       # every block has an assistant target
    write_mask(d / "shard_000.mask", mask)
    cfg = smoke_cfg(tmp_path / name, model={"attnres_blocks": 0}, mode="sft",
                    milestones=[], **train)
    t = build(tmp_path / name, cfg, d)
    return [t.train_step() for _ in range(3)], t


def test_sft_chunked_loss_matches_the_full_loss(tmp_path):
    base, tb = _sft_losses(tmp_path, "full")
    chunked, tc = _sft_losses(tmp_path, "chunked", loss_chunk_tokens=300)
    np.testing.assert_allclose(chunked, base, rtol=1e-5)
    for (n, p), (_, q) in zip(tb.model.named_parameters(), tc.model.named_parameters()):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6, msg=n)


def test_loss_scale_is_checkpointed(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    data = make_data(tmp_path)
    t = build(tmp_path, cfg, data)
    t.scaler.overflow()
    t.train_step()
    t.save_checkpoint()
    r = build(tmp_path, cfg, data, run_id="t", resume=True)
    r.resume_from_latest()
    assert r.scaler.state_dict() == t.scaler.state_dict()


def test_bf16_checkpoint_resumes_into_fp16_at_the_initial_scale(tmp_path):
    data = make_data(tmp_path)
    cfg_bf16 = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    t = build(tmp_path, cfg_bf16, data)
    assert not t.scaler.enabled
    t.train_step()
    t.save_checkpoint()
    cfg_fp16 = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    r = build(tmp_path, cfg_fp16, data, run_id="t", resume=True)
    r.resume_from_latest()
    assert r.step == 1
    assert r.scaler.enabled and r.scaler.scale == 2.0 ** 16 and r.scaler.clean_steps == 0


def test_fp32_momentum_checkpoint_resumes_into_bf16_state(tmp_path):
    data = make_data(tmp_path)
    cfg32 = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    t = build(tmp_path, cfg32, data)
    t.train_step()
    path = t.save_checkpoint()
    cfg16 = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, optimizer_state_dtype="bf16")
    r = build(tmp_path, cfg16, data, run_id="u")
    r.init_weights_from(path, optimizers=True)
    muon = r.optimizers[0]
    assert muon.state
    assert all(s["momentum_buffer"].dtype is torch.bfloat16 for s in muon.state.values())


@pytest.mark.parametrize("cap,dispatch,precision,ok", [
    ((7, 5), "loop", "fp16", True),
    ((7, 5), "grouped", "fp16", False),
    ((7, 5), "loop", "bf16", False),
    ((9, 0), "grouped", "bf16", True),
    ((8, 0), "grouped", "bf16", False),
    (None, "grouped", "bf16", True),          # CPU: nothing to check
])
def test_device_guard(tmp_path, cap, dispatch, precision, ok):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0, "moe_dispatch": dispatch},
                    precision=precision)
    if ok:
        check_device_support(cfg.model, cfg.train, cap)
    else:
        with pytest.raises(UsageError):
            check_device_support(cfg.model, cfg.train, cap)


# ---- Kaggle config -------------------------------------------------------------------

def test_t4_config_is_the_cpt_model_with_low_memory_training():
    cpt = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    t4 = load_config(ROOT / "configs" / "quipu-moe-cpt-t4.toml")
    assert dataclasses.replace(t4.model, moe_dispatch=cpt.model.moe_dispatch) == cpt.model
    assert t4.model.moe_dispatch == "loop"
    t = t4.train
    assert (t.precision, t.optimizer_state_dtype, t.activation_checkpoint) == \
        ("fp16", "bf16", "block")
    assert t.loss_chunk_tokens == 2048 and t.compile is False
    assert t.batch_tokens == cpt.train.batch_tokens
    assert t.grad_accum % 2 == 0                      # splits over 2 GPUs
    assert t.ckpt_keep == 1
