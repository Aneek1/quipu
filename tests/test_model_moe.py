import pytest
import torch

from quipu.config import ModelConfig, load_config
from quipu.model import Quipu
from quipu.model_factory import build_model
from quipu.model_moe import MoEBlock, QuipuMoE

MOE_CONFIG = "configs/quipu-moe.toml"
SMOKE_CONFIG = "configs/quipu-moe-smoke.toml"
DENSE_CONFIG = "configs/quipu-114m.toml"


def tiny(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=64, d_model=32, n_layer=4, n_head=4, n_kv_head=2, ffn_hidden=0,
        context=32, rope_base=10000.0, norm_eps=1e-6, kind="moe", n_experts=8,
        top_k=2, expert_hidden=24, shared_experts=1, shared_hidden=32,
        activation="situ_glu", attnres_blocks=2,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _randomise_queries(model: QuipuMoE, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for q in model.attnres.queries:
            q.copy_(torch.randn(q.shape, generator=g))


# ---- parameter count ------------------------------------------------------------

def _analytic_counts(cfg: ModelConfig) -> tuple[int, int]:
    """(total, active) from the spec's arithmetic. Active = embeddings + per layer
    (attention + norms + router + shared experts + top_k routed experts) + final norm."""
    d = cfg.d_model
    attn = d * cfg.n_head * cfg.head_dim * 2 + d * cfg.n_kv_head * cfg.head_dim * 2
    norms = 2 * d
    router = cfg.n_experts * d
    shared = cfg.shared_experts * 3 * d * cfg.shared_hidden
    expert = 3 * d * cfg.expert_hidden
    attnres = (cfg.n_layer + 1) * d if cfg.attnres_blocks else 0
    embed = cfg.vocab_size * d
    common = embed + d + attnres + cfg.n_layer * (attn + norms + router + shared)
    total = common + cfg.n_layer * cfg.n_experts * expert
    active = common + cfg.n_layer * cfg.top_k * expert
    return total, active


def _meta_counts(cfg: ModelConfig) -> tuple[int, int]:
    """(total, active) from a meta-device model: no memory is allocated."""
    with torch.device("meta"):
        model = QuipuMoE(cfg)
    total = sum(p.numel() for p in model.parameters())   # the tied weight counted once
    routed = sum(p.numel() for n, p in model.named_parameters() if ".moe.experts." in n)
    active = total - routed + routed * cfg.top_k // cfg.n_experts
    return total, active


@pytest.mark.parametrize("attnres_blocks", [0, 4])
def test_quipu_moe_has_the_documented_total_and_active_parameters(attnres_blocks):
    import dataclasses
    cfg = dataclasses.replace(load_config(MOE_CONFIG).model, attnres_blocks=attnres_blocks)
    total, active = _meta_counts(cfg)
    assert (total, active) == _analytic_counts(cfg)
    assert abs(total - 998.0e6) / 998.0e6 < 0.01, total
    assert abs(active - 148.7e6) / 148.7e6 < 0.01, active


# ---- structure ------------------------------------------------------------------

def test_forward_returns_logits_and_per_layer_stats():
    torch.manual_seed(0)
    cfg = tiny()
    model = QuipuMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(idx)
    assert out.shape == (2, 12, cfg.vocab_size)
    assert model.lm_head.weight is model.embed.weight
    assert len(model.last_stats) == cfg.n_layer
    for s in model.last_stats:
        assert int(s.counts.sum()) == 2 * 12 * cfg.top_k


def test_update_balance_moves_every_layers_bias_once():
    torch.manual_seed(0)
    cfg = tiny()
    model = QuipuMoE(cfg)
    model(torch.randint(0, cfg.vocab_size, (4, 32)))
    before = [b.moe.balancer.bias.clone() for b in model.blocks]
    model.update_balance()
    after = [b.moe.balancer.bias.clone() for b in model.blocks]
    assert all(not torch.equal(x, y) for x, y in zip(before, after))
    model.update_balance()   # no new forward: scores were consumed, nothing moves
    assert all(torch.equal(x, b.moe.balancer.bias) for x, b in zip(after, model.blocks))


def test_rejects_a_dense_config():
    with pytest.raises(ValueError):
        QuipuMoE(tiny(kind="dense"))


def test_attnres_blocks_zero_is_a_plain_pre_norm_residual():
    torch.manual_seed(0)
    cfg = tiny(attnres_blocks=0)
    model = QuipuMoE(cfg)
    assert model.attnres is None
    assert not any("attnres" in k for k in model.state_dict())
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        got = model(idx)
        x = model.embed(idx)
        cos, sin = model.rope_cos, model.rope_sin
        for b in model.blocks:
            x = x + b.attn(b.norm1(x), cos, sin)
            x = x + b.moe(b.norm2(x))[0]
        want = model.lm_head(model.norm(x))
    assert torch.equal(got, want)


def test_zero_pseudo_queries_feed_each_layer_the_mean_of_its_block_sources(monkeypatch):
    """At init every AttnRes weight is 1/m, so layer l's input must be the plain mean
    of the Kimi source list: [b_0 (embedding), completed block sums, partial sum]."""
    torch.manual_seed(0)
    cfg = tiny(n_layer=6, attnres_blocks=3)
    model = QuipuMoE(cfg)
    inputs, outs = [], []
    real_update = MoEBlock.update

    def recording(self, h, cos, sin):
        f, s = real_update(self, h, cos, sin)
        inputs.append(h)
        outs.append(f)
        return f, s

    monkeypatch.setattr(MoEBlock, "update", recording)
    idx = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        logits = model(idx)
        emb = model.embed(idx)
    lpb = cfg.n_layer // cfg.attnres_blocks
    for l in range(cfg.n_layer):
        n, i = divmod(l, lpb)
        srcs = [emb] + [sum(outs[b * lpb:(b + 1) * lpb]) for b in range(n)]
        if i:
            srcs.append(sum(outs[n * lpb:l]))
        torch.testing.assert_close(inputs[l], torch.stack(srcs).mean(0), atol=1e-6, rtol=0)
    blocks = [emb] + [sum(outs[b * lpb:(b + 1) * lpb]) for b in range(cfg.attnres_blocks)]
    with torch.no_grad():
        want = model.lm_head(model.norm(torch.stack(blocks).mean(0)))
    torch.testing.assert_close(logits, want, atol=1e-5, rtol=0)


# ---- gradients, causality, precision --------------------------------------------

def test_gradients_reach_every_block_and_every_used_expert():
    torch.manual_seed(0)
    cfg = tiny()
    model = QuipuMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    model(idx).float().pow(2).mean().backward()
    for name, p in model.named_parameters():
        if name == "attnres.queries.0":
            # The first layer has only b_0 to attend to: no gradient is possible.
            continue
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    for l, (block, stats) in enumerate(zip(model.blocks, model.last_stats)):
        bank = block.moe.experts
        assert int((stats.counts > 0).sum()) >= 2
        for e, c in enumerate(stats.counts.tolist()):
            used = bank.gate.grad[e].abs().sum() > 0 and bank.down.grad[e].abs().sum() > 0
            assert bool(used) == (c > 0), (l, e, c)


@pytest.mark.parametrize("attnres_blocks", [0, 2])
def test_changing_a_later_token_never_changes_earlier_logits(attnres_blocks):
    """Routing is per token, but expert matmuls run on per-expert slices whose sizes
    change when a later token routes elsewhere; a BLAS kernel may round a row
    differently at another slice height, so this compares with a tolerance far below
    what any real leak would produce."""
    torch.manual_seed(0)
    cfg = tiny(attnres_blocks=attnres_blocks)
    model = QuipuMoE(cfg).eval()
    if attnres_blocks:
        _randomise_queries(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        before = model(ids)
    for t in (3, 9, 14):
        ids2 = ids.clone()
        ids2[0, t + 1] = (int(ids2[0, t + 1]) + 1) % cfg.vocab_size
        with torch.no_grad():
            after = model(ids2)
        torch.testing.assert_close(after[0, :t + 1], before[0, :t + 1], atol=1e-6, rtol=0)
        assert (after[0, t + 1] - before[0, t + 1]).abs().max() > 1e-3


def test_smoke_config_forward_backward_is_finite_in_bf16_on_cpu():
    torch.manual_seed(0)
    cfg = load_config(SMOKE_CONFIG).model
    assert cfg.attnres_blocks and cfg.activation == "situ_glu"
    model = QuipuMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, cfg.vocab_size), idx.view(-1))
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits.float()).all() and torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


# ---- factory --------------------------------------------------------------------

def test_build_model_on_the_dense_config_is_the_unchanged_quipu():
    cfg = load_config(DENSE_CONFIG).model
    with torch.device("meta"):
        built = build_model(cfg)
        reference = Quipu(cfg)
    assert type(built) is Quipu
    assert list(built.state_dict()) == list(reference.state_dict())
    assert sum(p.numel() for p in built.parameters()) == 114_114_048


def test_build_model_on_a_moe_config_is_quipu_moe():
    with torch.device("meta"):
        assert type(build_model(tiny())) is QuipuMoE


def test_build_model_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        build_model(tiny(kind="sparse"))
