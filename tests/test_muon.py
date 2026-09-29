import dataclasses
import io

import pytest
import torch
import torch.nn as nn

from quipu.config import TrainConfig, load_config
from quipu.model import Quipu
from quipu.model_factory import build_model
from quipu.model_moe import QuipuMoE
from quipu.optim import Muon, apply_config_lrs, build_optimizers, newton_schulz, orthogonalize

DENSE_CONFIG = "configs/quipu-114m.toml"
MOE_CONFIG = "configs/quipu-moe.toml"
SMOKE_CONFIG = "configs/quipu-moe-smoke.toml"


def _singular_values(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x.float())


def _train_cfg(path: str, **overrides) -> TrainConfig:
    return dataclasses.replace(load_config(path).train, **overrides)


# ---- newton_schulz -----------------------------------------------------------------

def test_ns_square_matrix_bulk_near_one():
    # A square Gaussian matrix's smallest singular value is ~1/n of its largest;
    # five steps lift a small value by at most 3.4445 ** 5 ~ 485x, so the very
    # smallest can stay below the band. Everything else lands in it.
    g = torch.randn(64, 64, generator=torch.Generator().manual_seed(0))
    s = _singular_values(newton_schulz(g, steps=5))
    assert s.max() <= 1.2
    assert (s >= 0.68).float().mean() >= 0.95, s


@pytest.mark.parametrize("shape", [(32, 96), (96, 32), (48, 64)])
def test_ns_singular_values_near_one(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    out = newton_schulz(g, steps=5)
    assert out.shape == g.shape and out.dtype == g.dtype
    s = _singular_values(out)
    # The quintic's coefficients do not converge to 1: iterated, every starting
    # value in (0, 1] ends up in the band [0.6818, 1.1344] (f(1) = 0.701). The
    # plan's [0.7, 1.2] is therefore unreachable at the low end; 0.68 is the band.
    assert s.min() >= 0.68 and s.max() <= 1.2, (s.min(), s.max())


def test_ns_tall_matrix_is_transpose_of_wide():
    g = torch.randn(96, 32, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(newton_schulz(g), newton_schulz(g.T).T)


def test_ns_bf16_input():
    g = torch.randn(48, 80, generator=torch.Generator().manual_seed(2)).bfloat16()
    out = newton_schulz(g)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    s = _singular_values(out)
    assert s.min() >= 0.6 and s.max() <= 1.3, (s.min(), s.max())


def test_ns_batched_equals_each_matrix():
    g = torch.randn(3, 16, 40, generator=torch.Generator().manual_seed(3))
    batched = newton_schulz(g)
    for i in range(3):
        torch.testing.assert_close(batched[i], newton_schulz(g[i]))


def test_ns_scale_invariant_and_zero_safe():
    g = torch.randn(16, 24, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(newton_schulz(g), newton_schulz(1e-3 * g))
    assert torch.equal(newton_schulz(torch.zeros(8, 8)), torch.zeros(8, 8))


@pytest.mark.parametrize("bad", [torch.zeros(4), torch.zeros(2, 3, 4, 5)])
def test_ns_rejects_wrong_rank(bad):
    with pytest.raises(ValueError, match="2-D or 3-D"):
        newton_schulz(bad)


def test_ns_rejects_bad_steps():
    with pytest.raises(ValueError, match="steps"):
        newton_schulz(torch.zeros(4, 4), steps=0)


# ---- orthogonalize: per-head and per-expert ----------------------------------------

def test_per_head_update_is_concatenation_of_block_ns():
    head_dim, n_head, d = 8, 4, 32
    g = torch.randn(n_head * head_dim, d, generator=torch.Generator().manual_seed(5))
    out = orthogonalize(g, steps=5, head_dim=head_dim)
    blocks = [newton_schulz(b) for b in g.split(head_dim, dim=0)]
    torch.testing.assert_close(out, torch.cat(blocks, dim=0))
    # And it is NOT the full-matrix result: the split changes the update.
    assert not torch.allclose(out, orthogonalize(g, steps=5), atol=1e-3)


def test_per_head_with_one_head_equals_full_matrix():
    g = torch.randn(24, 40, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(orthogonalize(g, head_dim=24), orthogonalize(g))


def test_full_matrix_shape_scale():
    g = torch.randn(64, 16, generator=torch.Generator().manual_seed(7))   # out 64, in 16
    torch.testing.assert_close(orthogonalize(g), newton_schulz(g) * 2.0)
    wide = torch.randn(16, 64, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(orthogonalize(wide), newton_schulz(wide))


def test_expert_tensor_orthogonalised_per_expert():
    # ExpertBank layout is [n, in, out]; each expert is its own matrix.
    n, d_in, d_out = 3, 16, 48
    g = torch.randn(n, d_in, d_out, generator=torch.Generator().manual_seed(8))
    out = orthogonalize(g, in_out=True)
    for i in range(n):
        # Scale is from the logical (out, in) shape: max(1, 48 / 16) ** 0.5.
        torch.testing.assert_close(out[i], newton_schulz(g[i]) * 3 ** 0.5)
    # Not one big NS over the flattened bank.
    assert not torch.allclose(out.reshape(n * d_in, d_out),
                              newton_schulz(g.reshape(n * d_in, d_out)) * 3 ** 0.5, atol=1e-3)


def test_orthogonalize_validation():
    with pytest.raises(ValueError, match="divide"):
        orthogonalize(torch.zeros(30, 8), head_dim=8)
    with pytest.raises(ValueError, match="head_dim"):
        orthogonalize(torch.zeros(2, 8, 8), head_dim=4)


# ---- Muon ---------------------------------------------------------------------------

def test_muon_step_matches_manual_update():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 8))
    start = p.detach().clone()
    opt = Muon([p], lr=0.1, momentum=0.9, weight_decay=0.01, ns_steps=5)
    g = torch.randn(16, 8)
    p.grad = g.clone()
    opt.step()
    buf = 0.1 * g                       # lerp from zeros by (1 - momentum)
    nesterov = g.lerp(buf, 0.9)
    expected = start * (1 - 0.1 * 0.01) - 0.1 * orthogonalize(nesterov, steps=5)
    torch.testing.assert_close(p.detach(), expected)


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_momentum_accumulates_over_steps(nesterov):
    # Hand-written reference, three steps with different gradients:
    #   buf_t = mom * buf_{t-1} + (1 - mom) * g_t          (buf_0 = 0)
    #   dir_t = (1 - mom) * g_t + mom * buf_t   (Nesterov)  or  buf_t
    #   W_t   = W_{t-1} * (1 - lr * wd) - lr * orthogonalize(dir_t)
    torch.manual_seed(3)
    lr, mom, wd = 0.05, 0.9, 0.02
    p = nn.Parameter(torch.randn(12, 20))
    w = p.detach().clone()
    buf = torch.zeros_like(w)
    opt = Muon([p], lr=lr, momentum=mom, weight_decay=wd, nesterov=nesterov)
    for _ in range(3):
        g = torch.randn(12, 20)
        p.grad = g.clone()
        opt.step()
        buf = mom * buf + (1 - mom) * g
        direction = (1 - mom) * g + mom * buf if nesterov else buf
        w = w * (1 - lr * wd) - lr * orthogonalize(direction)
        torch.testing.assert_close(p.detach(), w)
        torch.testing.assert_close(opt.state[p]["momentum_buffer"], buf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ns_bf16_on_cuda_close_to_fp32():
    g = torch.randn(256, 512, generator=torch.Generator().manual_seed(9)).cuda()
    fast = newton_schulz(g)                        # fp32 in, bf16 iteration on CUDA
    assert fast.dtype == torch.float32
    exact = newton_schulz(g.cpu())                 # fp32 iteration on CPU
    rel = (fast.cpu() - exact).norm() / exact.norm()
    assert rel < 0.05, rel
    s = _singular_values(fast)
    assert s.min() >= 0.6 and s.max() <= 1.3, (s.min(), s.max())


def test_muon_group_head_dim_splits_update():
    torch.manual_seed(1)
    per_head = nn.Parameter(torch.randn(32, 16))
    full = nn.Parameter(per_head.detach().clone())
    g = torch.randn(32, 16)
    opt_h = Muon([{"params": [per_head], "head_dim": 8}], lr=0.1, momentum=0.0)
    opt_f = Muon([full], lr=0.1, momentum=0.0)
    per_head.grad, full.grad = g.clone(), g.clone()
    opt_h.step()
    opt_f.step()
    start = full.detach() + 0.1 * orthogonalize(g)   # undo the full step
    torch.testing.assert_close(per_head.detach(), start - 0.1 * orthogonalize(g, head_dim=8))
    assert not torch.allclose(per_head, full, atol=1e-3)


def test_muon_validation():
    p = nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        Muon([p])
    q = nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="momentum"):
        Muon([q], momentum=1.0)
    with pytest.raises(ValueError, match="lr"):
        Muon([q], lr=-1.0)


def test_muon_toy_regression_converges():
    torch.manual_seed(0)
    target = torch.randn(16, 32) / 32 ** 0.5
    layer = nn.Linear(32, 16, bias=False)
    nn.init.zeros_(layer.weight)
    opt = Muon(layer.parameters(), lr=0.02, momentum=0.95)
    x = torch.randn(512, 32)
    y = x @ target.T
    first = None
    for step in range(200):
        for group in opt.param_groups:       # linear decay, as a schedule would
            group["lr"] = 0.02 * (1 - step / 200)
        loss = (layer(x) - y).pow(2).mean()
        first = loss.item() if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = (layer(x) - y).pow(2).mean().item()
    assert final < 0.01 * first, (first, final)


# ---- build_optimizers ---------------------------------------------------------------

def _meta_model(path: str) -> nn.Module:
    with torch.device("meta"):
        return build_model(load_config(path).model)


def _routing(opts):
    ids = [id(p) for opt in opts for g in opt.param_groups for p in g["params"]]
    return ids


@pytest.mark.parametrize("path", [DENSE_CONFIG, SMOKE_CONFIG, MOE_CONFIG])
@pytest.mark.parametrize("mode", ["adamw", "muon"])
def test_every_param_in_exactly_one_optimizer(path, mode):
    model = _meta_model(path)
    opts = build_optimizers(model, _train_cfg(path, optimizer=mode))
    routed = _routing(opts)
    assert len(routed) == len(set(routed)), "a parameter is in two groups"
    assert set(routed) == {id(p) for p in model.parameters()}


def test_adamw_mode_matches_train_py_grouping():
    model = _meta_model(DENSE_CONFIG)
    cfg = _train_cfg(DENSE_CONFIG, optimizer="adamw")
    (opt,) = build_optimizers(model, cfg)
    assert type(opt) is torch.optim.AdamW
    decay, no_decay = opt.param_groups
    # train.py: decay = every p.dim() >= 2 (tied embedding included), no_decay the rest.
    assert [id(p) for p in decay["params"]] == [id(p) for p in model.parameters() if p.dim() >= 2]
    assert [id(p) for p in no_decay["params"]] == [id(p) for p in model.parameters() if p.dim() < 2]
    assert decay["weight_decay"] == cfg.weight_decay and no_decay["weight_decay"] == 0.0
    for g in opt.param_groups:
        assert g["lr"] == cfg.lr and g["base_lr"] == cfg.lr
        assert g["betas"] == (cfg.beta1, cfg.beta2)


def _group_of(opts, param):
    for opt in opts:
        for g in opt.param_groups:
            if any(p is param for p in g["params"]):
                return opt, g
    raise AssertionError("parameter not routed")


def test_muon_mode_routing_moe():
    path = SMOKE_CONFIG
    model = build_model(load_config(path).model)
    assert isinstance(model, QuipuMoE)
    cfg = _train_cfg(path, optimizer="muon")
    opts = build_optimizers(model, cfg)
    block = model.blocks[0]
    hd = model.cfg.head_dim

    for p in (block.attn.q.weight, block.attn.k.weight, block.attn.v.weight):
        opt, g = _group_of(opts, p)
        assert isinstance(opt, Muon) and g["head_dim"] == hd and not g["in_out"]
    for p in (block.attn.o.weight, block.moe.shared[0].gate.weight, block.moe.shared[0].down.weight):
        opt, g = _group_of(opts, p)
        assert isinstance(opt, Muon) and g["head_dim"] is None and not g["in_out"]
    for p in (block.moe.experts.gate, block.moe.experts.up, block.moe.experts.down):
        opt, g = _group_of(opts, p)
        assert isinstance(opt, Muon) and g["in_out"] and g["head_dim"] is None
    for p in (model.embed.weight, block.moe.router.weight):
        opt, g = _group_of(opts, p)
        assert type(opt) is torch.optim.AdamW and g["weight_decay"] == cfg.weight_decay
    for g in (g for opt in opts if isinstance(opt, Muon) for g in opt.param_groups):
        assert g["lr"] == g["base_lr"] == cfg.muon_lr
        assert g["momentum"] == cfg.muon_momentum and g["ns_steps"] == cfg.muon_ns_steps
    # The balancer bias is a buffer: never handed to an optimizer.
    assert all(b is not block.moe.balancer.bias for b in
               (p for opt in opts for g in opt.param_groups for p in g["params"]))


def test_muon_groups_use_muon_weight_decay():
    model = _meta_model(SMOKE_CONFIG)
    cfg = _train_cfg(SMOKE_CONFIG, optimizer="muon", weight_decay=0.1, muon_weight_decay=0.03)
    opts = build_optimizers(model, cfg)
    muon = [g for opt in opts if isinstance(opt, Muon) for g in opt.param_groups]
    adam = [g for opt in opts if type(opt) is torch.optim.AdamW for g in opt.param_groups]
    assert muon and all(g["weight_decay"] == 0.03 for g in muon)
    assert sorted(g["weight_decay"] for g in adam) == [0.0, 0.1]


def test_expert_down_bank_gets_in_out_sqrt2_scale():
    # Smoke: d_model 128, expert_hidden 64, so down is [n, 64, 128]: logical
    # (out, in) = (128, 64), shape scale sqrt(128 / 64) = sqrt(2). Read as
    # nn.Linear's (out, in) = (64, 128) it would be 1: the routing decides it.
    torch.manual_seed(0)
    mcfg = load_config(SMOKE_CONFIG).model
    model = build_model(mcfg)
    cfg = _train_cfg(SMOKE_CONFIG, optimizer="muon")
    opts = build_optimizers(model, cfg)
    down = model.blocks[0].moe.experts.down
    assert down.shape == (mcfg.n_experts, mcfg.expert_hidden, mcfg.d_model)
    opt, g = _group_of(opts, down)
    assert isinstance(opt, Muon) and g["in_out"] is True
    start = down.detach().clone()
    x = torch.randint(0, mcfg.vocab_size, (2, 16))
    _step(model, opts, x)
    grad = down.grad.detach()
    lr, wd = g["lr"], g["weight_decay"]
    update = (start * (1 - lr * wd) - down.detach()) / lr
    # First step: the Nesterov direction is a multiple of grad, and NS is scale-invariant.
    # atol covers the fp32 rounding of recovering the update as (start - p) / lr;
    # entries are ~0.1, so the wrong scale (1 instead of sqrt 2) misses by ~0.04.
    for i in range(mcfg.n_experts):
        torch.testing.assert_close(update[i], newton_schulz(grad[i]) * 2 ** 0.5,
                                   atol=1e-3, rtol=1e-3)
    hit = [i for i in range(mcfg.n_experts) if grad[i].abs().sum() > 0]
    assert hit, "no expert got a gradient"
    for i in hit:
        ratio = update[i].pow(2).mean().sqrt() / newton_schulz(grad[i]).pow(2).mean().sqrt()
        assert abs(ratio.item() - 2 ** 0.5) < 1e-3


def test_apply_config_lrs_overrides_loaded_state():
    torch.manual_seed(0)
    mcfg = load_config(SMOKE_CONFIG).model
    old = _train_cfg(SMOKE_CONFIG, optimizer="muon")
    new = dataclasses.replace(old, lr=old.lr * 0.5, muon_lr=0.01, weight_decay=0.05,
                              muon_weight_decay=0.002)
    x = torch.randint(0, mcfg.vocab_size, (2, 16))
    a = build_model(mcfg)
    opts_a = build_optimizers(a, old)
    _step(a, opts_a, x)
    saved = [opt.state_dict() for opt in opts_a]

    b = build_model(mcfg)
    opts_b = build_optimizers(b, new)
    for opt, sd in zip(opts_b, saved):
        opt.load_state_dict(sd)
    # load_state_dict brings back the old settings: the new config is ignored...
    assert opts_b[0].param_groups[0]["base_lr"] == old.muon_lr
    apply_config_lrs(opts_b, new)
    # ...until apply_config_lrs puts the config's back.
    for opt in opts_b:
        for g in opt.param_groups:
            if isinstance(opt, Muon):
                assert g["base_lr"] == new.muon_lr and g["weight_decay"] == new.muon_weight_decay
            else:
                assert g["base_lr"] == new.lr
                assert g["weight_decay"] in (0.0, new.weight_decay)
    adam = next(opt for opt in opts_b if type(opt) is torch.optim.AdamW)
    assert sorted(g["weight_decay"] for g in adam.param_groups) == [0.0, new.weight_decay]
    # Momentum state survives.
    assert all(len(opt.state) > 0 for opt in opts_b)


def test_muon_per_head_off_gives_full_matrix_qkv():
    model = _meta_model(SMOKE_CONFIG)
    opts = build_optimizers(model, _train_cfg(SMOKE_CONFIG, optimizer="muon", muon_per_head=False))
    _, g = _group_of(opts, model.blocks[0].attn.q.weight)
    assert g["head_dim"] is None


@pytest.mark.parametrize("mode", ["adamw", "muon"])
def test_norms_and_queries_get_no_weight_decay(mode):
    model = _meta_model(SMOKE_CONFIG)
    assert model.attnres is not None
    opts = build_optimizers(model, _train_cfg(SMOKE_CONFIG, optimizer=mode))
    named = dict(model.named_parameters())
    names = model.no_decay_param_names()
    assert any(n.startswith("attnres.queries.") for n in names)
    assert any(n.endswith("norm1.weight") for n in names)
    for n in names:
        opt, g = _group_of(opts, named[n])
        assert type(opt) is torch.optim.AdamW and g["weight_decay"] == 0.0, n


def test_muon_mode_dense_routing():
    model = _meta_model(DENSE_CONFIG)
    assert isinstance(model, Quipu)
    opts = build_optimizers(model, _train_cfg(DENSE_CONFIG, optimizer="muon"))
    opt, _ = _group_of(opts, model.embed.weight)
    assert type(opt) is torch.optim.AdamW
    opt, g = _group_of(opts, model.blocks[0].ffn.up.weight)
    assert isinstance(opt, Muon) and g["head_dim"] is None


def test_unknown_optimizer_rejected():
    model = _meta_model(SMOKE_CONFIG)
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizers(model, _train_cfg(SMOKE_CONFIG, optimizer="sgd"))


# ---- state_dict round trip ----------------------------------------------------------

def _step(model, opts, x):
    loss = model(x).float().pow(2).mean()
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in opts:
        opt.step()


def test_state_dict_round_trip_gives_identical_steps():
    torch.manual_seed(0)
    mcfg = load_config(SMOKE_CONFIG).model
    cfg = _train_cfg(SMOKE_CONFIG, optimizer="muon")
    x = torch.randint(0, mcfg.vocab_size, (2, 16))
    a = build_model(mcfg)
    opts_a = build_optimizers(a, cfg)
    for _ in range(2):
        _step(a, opts_a, x)
    # Through bytes, as a checkpoint is: load_state_dict keeps same-device tensors
    # as they are, so an in-memory state_dict would share buffers between a and b.
    blob = io.BytesIO()
    torch.save({"model": a.state_dict(), "opts": [opt.state_dict() for opt in opts_a]}, blob)
    blob.seek(0)
    saved = torch.load(blob, weights_only=True)
    saved_model, saved_opts = saved["model"], saved["opts"]

    b = build_model(mcfg)
    b.load_state_dict(saved_model)
    opts_b = build_optimizers(b, cfg)
    for opt, sd in zip(opts_b, saved_opts):
        opt.load_state_dict(sd)
    assert opts_b[0].param_groups[0].keys() == opts_a[0].param_groups[0].keys()
    for _ in range(2):
        _step(a, opts_a, x)
        _step(b, opts_b, x)
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(pa, pb), n
