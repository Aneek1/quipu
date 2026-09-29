import dataclasses

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
    base = dict(
        vocab_size=64, d_model=32, n_layer=4, n_head=4, n_kv_head=2, ffn_hidden=0,
        context=16, rope_base=10000.0, norm_eps=1e-6, kind="moe", n_experts=8,
        top_k=2, expert_hidden=24, shared_experts=1, shared_hidden=32,
        activation="situ_glu", balance_update_rate=1e-3,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _naive(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """Reference: loop over tokens and their selected experts, one matrix at a time."""
    xf = x.reshape(-1, x.shape[-1])
    weights, idx, _ = layer.router(xf, layer.balancer.bias)
    bank = layer.experts
    out = torch.zeros_like(xf)
    for t in range(xf.shape[0]):
        for slot in range(idx.shape[1]):
            e = int(idx[t, slot])
            g = xf[t] @ bank.gate[e]
            u = xf[t] @ bank.up[e]
            out[t] += weights[t, slot] * (bank.act(g, u) @ bank.down[e])
    for shared in layer.shared:
        out = out + shared(xf)
    return out.reshape(x.shape)


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


# --- MoE layer ----------------------------------------------------------------

@pytest.mark.parametrize("activation", ["situ_glu", "swiglu"])
def test_moe_layer_matches_naive_per_token_loop(activation):
    torch.manual_seed(0)
    layer = MoELayer(_cfg(activation=activation))
    # Wider weights so the SiTU caps are actually exercised.
    with torch.no_grad():
        layer.experts.gate.mul_(20.0)
        layer.experts.up.mul_(20.0)
        layer.balancer.bias.copy_(torch.randn(8) * 0.1)
    x = torch.randn(3, 7, 32)
    y, counts = layer(x)
    assert y.shape == x.shape
    assert torch.allclose(y, _naive(layer, x), atol=1e-5)


def test_counts_sum_to_tokens_times_k():
    layer = MoELayer(_cfg())
    y, counts = layer(torch.randn(5, 9, 32))
    assert counts.shape == (8,)
    assert int(counts.sum()) == 5 * 9 * 2


def test_empty_expert_path():
    torch.manual_seed(0)
    layer = MoELayer(_cfg())
    with torch.no_grad():
        layer.balancer.bias[5] = -1e4          # expert 5 can never be selected
        layer.balancer.bias[6] = -1e4
    x = torch.randn(2, 6, 32, requires_grad=True)
    y, counts = layer(x)
    assert counts[5] == 0 and counts[6] == 0
    assert torch.allclose(y, _naive(layer, x), atol=1e-5)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.experts.gate.grad[5].abs().max() == 0


def test_zero_tokens_forward():
    layer = MoELayer(_cfg())
    y, counts = layer(torch.randn(0, 32))
    assert y.shape == (0, 32) and int(counts.sum()) == 0


def test_skewed_router_is_balanced_after_200_updates():
    torch.manual_seed(0)
    # The shipped rate (1e-3) is tuned for thousands of steps; 0.1 lets the same
    # rule converge inside the 200 updates the test allows.
    cfg = _cfg(balance_update_rate=0.1)
    layer = MoELayer(cfg)
    T = 1024
    x = torch.randn(T, 32)
    x[:, 0] = x[:, 0].abs() + 4.0
    with torch.no_grad():
        layer.router.weight.mul_(10.0)         # spread the other experts' scores
        layer.router.weight[0].zero_()
        layer.router.weight[0, 0] = 1.0        # expert 0 is in every token's Top-k
    _, counts = layer(x)
    assert int(counts[0]) == T                # the skew is real before balancing
    for _ in range(200):
        _, counts = layer(x)
        layer.update_balance()
    target = cfg.top_k * T / cfg.n_experts
    load = counts.float()
    assert ((load - target).abs() <= 0.25 * target).all(), load.tolist()


def test_balancer_never_gets_gradients():
    layer = MoELayer(_cfg())
    assert not layer.balancer.bias.requires_grad
    assert list(layer.balancer.parameters()) == []
    y, _ = layer(torch.randn(4, 8, 32))
    y.sum().backward()
    assert layer.balancer.bias.grad is None
    assert "balancer.bias" in layer.state_dict()
    assert layer.router.weight.grad is not None


def test_balancer_update_uses_quantile_cutoff():
    # One step at rate 1 from zero bias, with scores independent across experts,
    # moves each expert's bias by minus its (q+1)-th largest margin.
    torch.manual_seed(0)
    bal = QuantileBalancer(4, top_k=1, update_rate=1.0)
    s = torch.rand(40, 4)
    alpha = s.topk(2, dim=-1).values[:, -1]
    margins = s - alpha[:, None]
    q = 10
    expected = -margins.sort(dim=0, descending=True).values[q]
    bal.update(s)
    assert torch.allclose(bal.bias, expected)


def test_bf16_autocast_forward_is_finite():
    torch.manual_seed(0)
    layer = MoELayer(_cfg())
    x = torch.randn(2, 16, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y, counts = layer(x)
        weights, _, scores = layer.router(x.reshape(-1, 32), layer.balancer.bias)
    assert torch.isfinite(y.float()).all()
    assert scores.dtype == torch.float32 and weights.dtype == torch.float32
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
