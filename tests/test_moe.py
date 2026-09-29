import dataclasses
import math

import pytest
import torch

from quipu.config import ModelConfig
from quipu.model import SwiGLU
from quipu.moe import (
    MoELayer,
    QuantileBalancer,
    Router,
    SiTUGLU,
    situ_glu,
)


def _cfg(**overrides) -> ModelConfig:
    # balance_update_rate is left at the ModelConfig default on purpose.
    base = dict(
        vocab_size=64, d_model=32, n_layer=4, n_head=4, n_kv_head=2, ffn_hidden=0,
        context=16, rope_base=10000.0, norm_eps=1e-6, kind="moe", n_experts=8,
        top_k=2, expert_hidden=24, shared_experts=1, shared_hidden=32,
        activation="situ_glu",
    )
    base.update(overrides)
    return ModelConfig(**base)


def _naive(layer: MoELayer, x: torch.Tensor, capacity: int | None = None) -> torch.Tensor:
    """Reference: loop over tokens and their selected experts, one matrix at a time.
    With a capacity, each expert keeps only its first `capacity` assignments in token
    order (the padded dispatch's drop rule) and the rest contribute nothing."""
    xf = x.reshape(-1, x.shape[-1])
    weights, idx, _ = layer.router(xf, layer.balancer.bias)
    bank = layer.experts
    out = torch.zeros_like(xf)
    seen = [0] * bank.n_experts
    for t in range(xf.shape[0]):
        for slot in range(idx.shape[1]):
            e = int(idx[t, slot])
            seen[e] += 1
            if capacity is not None and seen[e] > capacity:
                continue
            g = xf[t] @ bank.gate[e]
            u = xf[t] @ bank.up[e]
            out[t] += weights[t, slot] * (bank.act(g, u) @ bank.down[e])
    for shared in layer.shared:
        out = out + shared(xf)
    return out.reshape(x.shape)


def _wide(layer: MoELayer) -> MoELayer:
    """Wider expert weights so the SiTU caps are exercised, and a non-zero bias."""
    with torch.no_grad():
        layer.experts.gate.mul_(20.0)
        layer.experts.up.mul_(20.0)
        layer.balancer.bias.copy_(torch.randn(layer.n_experts) * 0.1)
    return layer


# --- SiTU-GLU -----------------------------------------------------------------

def test_situ_glu_matches_swiglu_near_zero():
    torch.manual_seed(0)
    situ = SiTUGLU(16, 32, beta_gate=4.0, beta_up=25.0)
    swi = SwiGLU(16, 32)
    swi.load_state_dict(situ.state_dict())
    x = (torch.rand(64, 16) - 0.5)          # |x| < 0.5
    a, b = situ(x), swi(x)
    # Relative to the output as a whole: single entries near zero make an
    # elementwise rtol meaningless after the down projection sums 32 terms.
    assert ((a - b).norm() / b.norm()).item() < 1e-2


def test_situ_glu_is_bounded_by_beta_product_for_large_inputs():
    g = torch.randn(4096) * 1e4
    u = torch.randn(4096) * 1e4
    h = situ_glu(g, u, 4.0, 25.0)
    assert torch.isfinite(h).all()
    assert h.abs().max() <= 4.0 * 25.0
    # The cap is actually reached: saturated gate and up give close to beta1*beta2.
    assert h.abs().max() > 0.95 * 4.0 * 25.0


def test_situ_glu_module_hidden_is_bounded():
    situ = SiTUGLU(8, 16, beta_gate=2.0, beta_up=3.0)
    x = torch.randn(32, 8) * 1e5
    h = situ.act(situ.gate(x), situ.up(x))
    assert h.abs().max() <= 2.0 * 3.0


# --- Router -------------------------------------------------------------------

def test_routing_weights_sum_to_one_per_token():
    torch.manual_seed(0)
    router = Router(32, 8, top_k=3)
    x = torch.randn(100, 32)
    weights, idx, scores = router(x, torch.zeros(8))
    assert weights.shape == idx.shape == (100, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(100), atol=1e-6)
    assert scores.dtype == torch.float32


def test_selected_experts_are_distinct_and_vary_across_tokens():
    torch.manual_seed(0)
    router = Router(32, 8, top_k=3)
    _, idx, _ = router(torch.randn(100, 32), torch.zeros(8))
    for row in idx.tolist():
        assert len(set(row)) == 3
    assert len({tuple(sorted(r)) for r in idx.tolist()}) > 1


def test_weights_come_from_unbiased_scores():
    torch.manual_seed(0)
    router = Router(32, 8, top_k=2)
    x = torch.randn(10, 32)
    bias = torch.randn(8)
    weights, idx, scores = router(x, bias)
    picked = scores.gather(-1, idx)
    assert torch.allclose(weights, picked / picked.sum(-1, keepdim=True), atol=1e-6)


# --- MoE layer: loop dispatch --------------------------------------------------

@pytest.mark.parametrize("activation", ["situ_glu", "swiglu"])
def test_moe_layer_matches_naive_per_token_loop(activation):
    torch.manual_seed(0)
    layer = _wide(MoELayer(_cfg(activation=activation)))
    x = torch.randn(3, 7, 32)
    y, stats = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, _naive(layer, x), atol=1e-5)
    assert int(stats.dropped) == 0


def test_counts_sum_to_tokens_times_k():
    layer = MoELayer(_cfg())
    _, stats = layer(torch.randn(5, 9, 32))
    assert stats.counts.shape == (8,)
    assert int(stats.counts.sum()) == 5 * 9 * 2


def test_empty_expert_path():
    torch.manual_seed(0)
    layer = MoELayer(_cfg())
    with torch.no_grad():
        layer.balancer.bias[5] = -1e4          # experts 5 and 6 can never be selected
        layer.balancer.bias[6] = -1e4
    x = torch.randn(2, 6, 32, requires_grad=True)
    y, stats = layer(x)
    assert stats.counts[5] == 0 and stats.counts[6] == 0
    assert torch.allclose(y, _naive(layer, x), atol=1e-5)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.experts.gate.grad[5].abs().max() == 0


@pytest.mark.parametrize("dispatch", ["loop", "padded"])
def test_zero_tokens_forward(dispatch):
    layer = MoELayer(_cfg(moe_dispatch=dispatch))
    y, stats = layer(torch.randn(0, 32))
    assert y.shape == (0, 32) and int(stats.counts.sum()) == 0 and int(stats.dropped) == 0


@pytest.mark.parametrize("dispatch", ["loop", "padded"])
def test_repeated_forward_backward_is_bit_identical(dispatch):
    torch.manual_seed(0)
    layer = _wide(MoELayer(_cfg(moe_dispatch=dispatch)))
    x = torch.randn(4, 16, 32)

    def run():
        layer.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        y, _ = layer(xi)
        (y * torch.arange(y.numel()).reshape(y.shape).sin()).sum().backward()
        grads = {n: p.grad.clone() for n, p in layer.named_parameters()}
        return y.detach(), xi.grad, grads

    y1, gx1, g1 = run()
    y2, gx2, g2 = run()
    assert torch.equal(y1, y2) and torch.equal(gx1, gx2)
    for name in g1:
        assert torch.equal(g1[name], g2[name]), name


# --- MoE layer: padded dispatch -------------------------------------------------

@pytest.mark.parametrize("activation", ["situ_glu", "swiglu"])
def test_padded_dispatch_equals_loop_when_nothing_drops(activation):
    torch.manual_seed(0)
    # capacity_factor = n_experts / top_k gives capacity T: nothing can overflow.
    loop = _wide(MoELayer(_cfg(activation=activation)))
    padded = MoELayer(_cfg(activation=activation, moe_dispatch="padded", capacity_factor=4.0))
    padded.load_state_dict(loop.state_dict())
    x = torch.randn(3, 11, 32, requires_grad=True)
    y_loop, s_loop = loop(x)
    y_pad, s_pad = padded(x)
    assert torch.allclose(y_pad, y_loop, atol=1e-5)
    assert torch.equal(s_pad.counts, s_loop.counts) and int(s_pad.dropped) == 0
    g_loop = torch.autograd.grad(y_loop.sum(), [x, loop.experts.down])
    g_pad = torch.autograd.grad(y_pad.sum(), [x, padded.experts.down])
    for a, b in zip(g_loop, g_pad):
        assert torch.allclose(a, b, atol=1e-5)


def test_padded_dispatch_drops_and_counts_overflow():
    torch.manual_seed(0)
    cfg = _cfg(moe_dispatch="padded", capacity_factor=1.0)
    layer = _wide(MoELayer(cfg))
    with torch.no_grad():
        layer.balancer.bias.zero_()
        layer.balancer.bias[0] = 1.0            # expert 0 takes every token
    x = torch.randn(40, 32)
    y, stats = layer(x)
    T = 40
    capacity = math.ceil(cfg.capacity_factor * T * cfg.top_k / cfg.n_experts)   # 10
    expected = int((stats.counts - capacity).clamp_min(0).sum())
    assert int(stats.counts[0]) == T
    assert expected >= T - capacity and int(stats.dropped) == expected
    # Dropped assignments contribute nothing; kept ones are the first in token order.
    assert torch.allclose(y, _naive(layer, x, capacity=capacity), atol=1e-5)


# --- Quantile Balancing ---------------------------------------------------------

def test_skewed_router_is_balanced_after_200_updates_at_the_default_rate():
    # Production layout for routing: 64 experts, top-4, 4,096 tokens.
    torch.manual_seed(0)
    n, k, T, d = 64, 4, 4096, 32
    router = Router(d, n, k)
    balancer = QuantileBalancer(n, k, ModelConfig.balance_update_rate)
    assert balancer.update_rate == 0.3
    x = torch.randn(T, d)
    x[:, 0] = x[:, 0].abs() + 4.0
    with torch.no_grad():
        router.weight.mul_(10.0)               # spread the other experts' scores
        router.weight[0].zero_()
        router.weight[0, 0] = 1.0              # expert 0 is in every token's Top-k
    _, idx, scores = router(x, balancer.bias)
    assert int((idx == 0).sum()) == T         # the skew is real before balancing
    for _ in range(200):
        balancer.update(scores)
    _, idx, _ = router(x, balancer.bias)
    load = torch.bincount(idx.reshape(-1), minlength=n).float()
    target = k * T / n
    assert ((load - target).abs() <= 0.25 * target).all(), load.tolist()


def test_starved_runner_up_expert_is_raised_by_one_update():
    # Expert 0 wins every token; expert 3 is every token's (k+1)-th choice, so it
    # gets nothing. A one-sided margin (score - (k+1)-th score) is exactly 0 for
    # expert 3: it would not move, and after re-centring it would end up LOWERED
    # (-0.015 here). The two-sided rule measures how far it sits below the k-th
    # score (0.3) and raises it.
    T = 64
    s = torch.full((T, 4), 0.05)
    s[:, 0] = 0.6
    s[:, 3] = 0.3
    bal = QuantileBalancer(4, top_k=1, update_rate=0.3)
    bal.update(s)
    assert bal.bias[3] > 0
    assert bal.bias[0] < 0


def test_balancer_update_uses_two_sided_quantile_cutoff():
    torch.manual_seed(0)
    k, n, T = 2, 5, 50
    bal = QuantileBalancer(n, top_k=k, update_rate=1.0)
    s = torch.rand(T, n)
    top = s.topk(k + 1, dim=-1)
    chosen = torch.zeros_like(s, dtype=torch.bool).scatter_(1, top.indices[:, :k], True)
    thr = torch.where(chosen, top.values[:, k:k + 1], top.values[:, k - 1:k])
    margins = s - thr
    q = round(k * T / n)                       # 20
    step = -margins.sort(dim=0, descending=True).values[q]
    bal.update(s)
    assert torch.allclose(bal.bias, step - step.mean(), atol=1e-6)


def test_balancer_bias_mean_stays_zero():
    torch.manual_seed(0)
    bal = QuantileBalancer(16, top_k=2, update_rate=0.3)
    for _ in range(100):
        s = torch.softmax(torch.randn(256, 16) + torch.linspace(0, 3, 16), dim=-1)
        bal.update(s)
        assert abs(bal.bias.mean().item()) < 1e-6
    assert bal.bias.abs().max() > 0


def test_balancer_skips_batches_too_small_to_have_a_target():
    bal = QuantileBalancer(16, top_k=2, update_rate=0.3)
    bal.update(torch.rand(7, 16))              # k*T = 14 < n = 16
    assert torch.equal(bal.bias, torch.zeros(16))


def test_update_balance_uses_the_latest_forward():
    torch.manual_seed(0)
    layer = MoELayer(_cfg())
    layer(torch.randn(64, 32))
    layer.update_balance()
    assert layer.balancer.bias.abs().max() > 0
    before = layer.balancer.bias.clone()
    layer.update_balance()                     # nothing new to learn from
    assert torch.equal(layer.balancer.bias, before)


def test_balancer_never_gets_gradients():
    layer = MoELayer(_cfg())
    assert not layer.balancer.bias.requires_grad
    assert list(layer.balancer.parameters()) == []
    y, _ = layer(torch.randn(4, 8, 32))
    y.sum().backward()
    assert layer.balancer.bias.grad is None
    assert "balancer.bias" in layer.state_dict()
    assert layer.router.weight.grad is not None


# --- precision, init, config ------------------------------------------------------

@pytest.mark.parametrize("dispatch", ["loop", "padded"])
def test_bf16_autocast_forward_is_finite(dispatch):
    torch.manual_seed(0)
    layer = MoELayer(_cfg(moe_dispatch=dispatch))
    x = torch.randn(2, 16, 32, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, _ = layer(x)
        weights, _, scores = layer.router(x.reshape(-1, 32), layer.balancer.bias)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()
    assert scores.dtype == torch.float32 and weights.dtype == torch.float32
    y.float().sum().backward()
    assert torch.isfinite(layer.experts.gate.grad).all()
    assert layer.experts.gate.grad.dtype == torch.float32
    layer.update_balance()
    assert torch.isfinite(layer.balancer.bias).all()


def test_init_scales_down_projection_by_depth():
    torch.manual_seed(0)
    layer = MoELayer(_cfg(n_layer=8, n_experts=16, expert_hidden=256))
    std_down = layer.experts.down.std().item()
    std_gate = layer.experts.gate.std().item()
    assert abs(std_gate - 0.02) < 2e-3
    assert abs(std_down - 0.02 / (2 * 8) ** 0.5) < 5e-4
    assert abs(layer.shared[0].down.weight.std().item() - 0.02 / 4) < 1e-3


def test_rejects_dense_config():
    with pytest.raises(ValueError):
        MoELayer(dataclasses.replace(_cfg(), kind="dense", n_experts=0, top_k=0,
                                     expert_hidden=0, shared_experts=0, shared_hidden=0,
                                     activation="swiglu"))
