import pytest
import torch

from quipu.config import ModelConfig, load_config
from quipu.model import Quipu
from quipu.model_factory import build_model
from quipu.model_moe import QuipuMoE

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
    attnres = (2 * cfg.n_layer + 1) * d if cfg.attnres_blocks else 0   # a query per sub-layer + head
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


def test_zero_pseudo_queries_feed_each_sub_layer_the_mean_of_its_block_sources(monkeypatch):
    """At init every AttnRes weight is 1/m, so sub-layer i's input must be the plain
    mean of the Kimi source list: [b_0 (embedding), completed block sums, partial
    sum]. Sub-layers alternate attention (even i) and MoE (odd i); a block of
    n_layer / attnres_blocks layers holds twice that many sub-layers."""
    torch.manual_seed(0)
    cfg = tiny(n_layer=6, attnres_blocks=3)
    model = QuipuMoE(cfg)
    inputs, outs = [], []
    real = QuipuMoE._sublayer

    def recording(self, i, h, cos, sin):
        f = real(self, i, h, cos, sin)
        inputs.append(h)
        outs.append(f)
        return f

    monkeypatch.setattr(QuipuMoE, "_sublayer", recording)
    idx = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        logits = model(idx)
        emb = model.embed(idx)
    n_steps = 2 * cfg.n_layer
    assert len(inputs) == n_steps
    spb = n_steps // cfg.attnres_blocks
    for i in range(n_steps):
        n, j = divmod(i, spb)
        srcs = [emb] + [sum(outs[b * spb:(b + 1) * spb]) for b in range(n)]
        if j:
            srcs.append(sum(outs[n * spb:i]))
        torch.testing.assert_close(inputs[i], torch.stack(srcs).mean(0), atol=1e-6, rtol=0)
    blocks = [emb] + [sum(outs[b * spb:(b + 1) * spb]) for b in range(cfg.attnres_blocks)]
    with torch.no_grad():
        want = model.lm_head(model.norm(torch.stack(blocks).mean(0)))
    torch.testing.assert_close(logits, want, atol=1e-5, rtol=0)


def test_attnres_has_one_pseudo_query_per_sub_layer_plus_the_head():
    cfg = tiny(n_layer=4, attnres_blocks=2)
    with torch.device("meta"):
        model = QuipuMoE(cfg)
    assert len(model.attnres.queries) == 2 * cfg.n_layer + 1
    assert model.attnres.steps_per_block == 2 * cfg.n_layer // cfg.attnres_blocks


def test_attnres_blocks_must_split_whole_layers():
    # 2 * n_layer divides by 4 but n_layer (2) does not: a block would end mid-layer.
    with pytest.raises(ValueError, match="attnres_blocks"):
        QuipuMoE(tiny(n_layer=2, attnres_blocks=4))


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


# ---- sub-layer granularity (I1) ---------------------------------------------------

def _plain_residual_logits(model: QuipuMoE, idx: torch.Tensor) -> torch.Tensor:
    """The plain pre-norm residual computation on model's own weights."""
    x = model.embed(idx)
    cos, sin = model.rope_cos, model.rope_sin
    for b in model.blocks:
        x = x + b.attn(b.norm1(x), cos, sin)
        x = x + b.moe(b.norm2(x))[0]
    return model.lm_head(model.norm(x))


@pytest.mark.parametrize("n_layer, attnres_blocks", [(4, 2), (6, 3), (4, 4), (4, 1)])
def test_zero_queries_make_attnres_the_plain_residual_model(n_layer, attnres_blocks):
    """With zero pseudo-queries every step input is the mean of its sources, i.e. the
    plain residual stream / m. Every sub-layer reads it through a scale-invariant
    RMSNorm, so the model IS the plain residual model -- but only when attention and
    MoE are separate AttnRes steps (a paired step would feed MoE norm(stream/m + a)).
    norm_eps is made negligible: at init the embedding has RMS ~0.02, so stream/m has
    a mean square near the default eps (1e-6) and eps alone would break the identity."""
    torch.manual_seed(0)
    cfg = tiny(n_layer=n_layer, attnres_blocks=attnres_blocks, norm_eps=1e-12)
    model = QuipuMoE(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        got = model(idx)
        want = _plain_residual_logits(model, idx)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


# ---- AttnRes memory (I2) ------------------------------------------------------------

def _saved_bytes(model: QuipuMoE, idx: torch.Tensor) -> int:
    """Bytes of the distinct non-parameter storages autograd saves for backward."""
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen: dict[int, int] = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in params:
            seen[s.data_ptr()] = s.nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = model(idx).float().pow(2).mean()
    loss.backward()
    return sum(seen.values())


def test_attnres_adds_at_most_a_quarter_to_the_plain_models_saved_activations():
    """Every layer its own block (the most sources: up to n_layer + 1 per mix), no
    checkpoint. The mix may keep the sources themselves and per-position scalars,
    never stacked or normalised copies of the sources. Measured (16 layers, d 32,
    2 x 32 tokens): plain 4298 KiB; AttnRes 4722 KiB (x1.10); the stacking mix of
    f8f3d3e saved 6822 KiB (x1.59)."""
    ids = torch.randint(0, 64, (2, 32), generator=torch.Generator().manual_seed(0))
    sizes = {}
    for blocks in (0, 16):
        torch.manual_seed(0)
        model = QuipuMoE(tiny(n_layer=16, attnres_blocks=blocks,
                              attnres_checkpoint=False)).train()
        sizes[blocks] = _saved_bytes(model, ids)
    assert sizes[16] <= 1.25 * sizes[0], sizes


def test_attnres_checkpoint_gives_the_same_logits_and_gradients():
    idx = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(0))
    runs = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        model = QuipuMoE(tiny(attnres_checkpoint=ckpt)).train()
        _randomise_queries(model)
        out = model(idx)
        out.float().pow(2).mean().backward()
        runs.append((out.detach(), {n: p.grad for n, p in model.named_parameters()}))
    (o0, g0), (o1, g1) = runs
    assert torch.equal(o0, o1)
    for name, g in g0.items():
        if g is None:
            assert g1[name] is None or torch.count_nonzero(g1[name]) == 0, name
        else:
            torch.testing.assert_close(g1[name], g, atol=1e-6, rtol=1e-5, msg=name)


# ---- padded dispatch is batch-dependent (I3) ------------------------------------------

def _overflowing_padded_model() -> QuipuMoE:
    torch.manual_seed(0)
    cfg = tiny(moe_dispatch="padded", capacity_factor=1.0, attnres_blocks=0)
    model = QuipuMoE(cfg).eval()
    with torch.no_grad():
        for b in model.blocks:
            b.moe.balancer.bias.copy_(torch.tensor([5.0, 5.0] + [0.0] * (cfg.n_experts - 2)))
    return model


def test_padded_dispatch_keeps_earlier_positions_of_a_row_exactly_unchanged():
    """Inside one row, overflow is decided in token order, so changing later tokens
    never changes an earlier position -- even with every expert overflowing."""
    model = _overflowing_padded_model()
    ids = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        before = model(ids)
        assert all(int(s.dropped) > 0 for s in model.last_stats)
        ids2 = ids.clone()
        ids2[0, 10:] = (ids2[0, 10:] + 1) % 64
        after = model(ids2)
    assert torch.equal(after[0, :10], before[0, :10])


def test_padded_dispatch_drops_depend_on_the_rest_of_the_batch():
    """Switch-style capacity: the same row gives different logits alone and after
    another row (whose tokens fill the experts first). Use loop dispatch to evaluate."""
    model = _overflowing_padded_model()
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 64, (2, 16), generator=g)
    with torch.no_grad():
        alone = model(rows[1:])
        batched = model(rows)
    assert not torch.allclose(alone[0], batched[1])
    model.set_dispatch("loop")
    with torch.no_grad():
        alone = model(rows[1:])
        batched = model(rows)
    torch.testing.assert_close(batched[1], alone[0], atol=1e-5, rtol=0)


def test_dispatch_can_be_switched_at_runtime():
    model = _overflowing_padded_model()
    ids = torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    loop_model = QuipuMoE(tiny(attnres_blocks=0)).eval()
    loop_model.load_state_dict(model.state_dict())
    model.set_dispatch("loop")
    assert all(b.moe.dispatch == "loop" for b in model.blocks)
    with torch.no_grad():
        assert torch.equal(model(ids), loop_model(ids))
        assert all(int(s.dropped) == 0 for s in model.last_stats)
    model.blocks[0].moe.dispatch = "padded"
    assert model.blocks[0].moe.dispatch == "padded"
    with pytest.raises(ValueError):
        model.set_dispatch("scatter")
    with pytest.raises(ValueError):
        model.blocks[0].moe.dispatch = "scatter"


# ---- stats survive recompute (M3) --------------------------------------------------

@pytest.mark.parametrize("attnres_blocks", [0, 2])
def test_last_stats_has_one_entry_per_layer_after_repeated_forwards(attnres_blocks):
    torch.manual_seed(0)
    cfg = tiny(attnres_blocks=attnres_blocks)
    model = QuipuMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    for _ in range(2):
        model(idx)
        assert len(model.last_stats) == cfg.n_layer
        assert all(s is not None for s in model.last_stats)


def test_recomputing_a_sub_layer_overwrites_its_stats_instead_of_appending(monkeypatch):
    """A trainer that checkpoints sub-layers re-runs them in backward; the stats list
    must stay one entry per layer."""
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    cfg = tiny()
    model = QuipuMoE(cfg).train()
    real = QuipuMoE._sublayer
    monkeypatch.setattr(QuipuMoE, "_sublayer", lambda self, i, h, cos, sin: checkpoint(
        real, self, i, h, cos, sin, use_reentrant=False))
    model(torch.randint(0, cfg.vocab_size, (2, 8))).float().pow(2).mean().backward()
    assert len(model.last_stats) == cfg.n_layer


# ---- optimizer grouping helper -------------------------------------------------------

def test_no_decay_param_names_are_the_norms_and_pseudo_queries():
    cfg = tiny(n_layer=4, attnres_blocks=2)
    with torch.device("meta"):
        model = QuipuMoE(cfg)
    names = model.no_decay_param_names()
    params = dict(model.named_parameters())
    assert all(params[n].ndim == 1 for n in names)
    assert {n for n, p in params.items() if p.ndim == 1} == set(names)
    assert sum(n.startswith("attnres.queries.") for n in names) == 2 * cfg.n_layer + 1
    assert "norm.weight" in names and "blocks.0.norm1.weight" in names
    assert not any("experts" in n or "embed" in n for n in names)
