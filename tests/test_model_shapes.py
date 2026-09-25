import math

import pytest
import torch
import torch.nn.functional as F

from quipu.config import ModelConfig, load_config
from quipu.model import Attention, Quipu, apply_rope, build_rope_cache

CONFIG_PATH = "configs/quipu-114m.toml"


def tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=32, rope_base=10000.0, norm_eps=1e-6,
    )


def test_forward_returns_logits_over_the_vocabulary():
    cfg = tiny()
    model = Quipu(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (2, 8)))
    assert out.shape == (2, 8, cfg.vocab_size)


def test_embeddings_are_tied():
    # Tied weights save 38.6M parameters at the real scale. If the tie silently
    # breaks, the parameter count test below is the only thing that notices.
    model = Quipu(tiny())
    assert model.lm_head.weight is model.embed.weight


def test_parameter_count_is_exactly_the_documented_figure():
    cfg = load_config("configs/quipu-114m.toml").model
    total = sum(p.numel() for p in Quipu(cfg).parameters())
    assert total == 114_114_048


def test_causal_mask_does_not_leak():
    """The single most important test in this sub-project.

    A mask that lets position t see position t+1 produces a beautiful loss curve
    and a model that cannot generate. Changing the LAST token must leave every
    earlier position bit-identical.
    """
    torch.manual_seed(0)
    cfg = tiny()
    model = Quipu(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        before = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (int(ids2[0, -1]) + 1) % cfg.vocab_size
    with torch.no_grad():
        after = model(ids2)

    assert torch.equal(before[0, :-1], after[0, :-1])
    # And the changed position really did change, or the test proves nothing.
    assert not torch.equal(before[0, -1], after[0, -1])


def test_accepts_a_sequence_shorter_than_the_context():
    cfg = tiny()
    assert Quipu(cfg)(torch.randint(0, cfg.vocab_size, (1, 5))).shape == (1, 5, cfg.vocab_size)


@pytest.mark.cuda
def test_attention_runs_on_a_fused_kernel():
    """Performance guard: on this build (Windows torch 2.11 cu128, sm_120) SDPA's
    default backend picker sends enable_gqa=True to the MATH kernel even though a
    fused kernel exists and handles GQA fine once selected explicitly -- the
    picker just never tries it for this call shape. Measured directly (B=8,
    real 114M shape, bf16 autocast, fwd+bwd, no sdpa_kernel restriction, i.e.
    exactly what training hits): enable_gqa=True peaks at ~1.7 GiB here vs ~0.23
    GiB for the repeat_interleave fix. If someone reintroduces enable_gqa, this
    must fail loudly instead of silently eating ~7x the memory.

    (A guard built on `sdpa_kernel([EFFICIENT_ATTENTION, CUDNN_ATTENTION])` was
    tried first, matching the original ask, but on this torch build cuDNN
    attention accepts enable_gqa fine once explicitly forced into the allowed
    set, so that version passed even against the unfixed enable_gqa=True code --
    it exercises kernel support, not what the default picker actually chooses.
    This version measures the real, unrestricted path instead.)"""
    cfg = load_config(CONFIG_PATH).model  # the real 114M shape
    attn = Attention(cfg).cuda()
    cos, sin = build_rope_cache(cfg.context, cfg.head_dim, device="cuda")
    x = torch.randn(8, cfg.context, cfg.d_model, device="cuda")
    torch.cuda.synchronize()
    # Measure a DELTA, not an absolute peak: max_memory_allocated counts every live
    # CUDA tensor in the process, so an unrelated allocation elsewhere (e.g. a later
    # GPU test in the same session) would inflate the absolute number and falsely
    # fail this one.
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = attn(x, cos, sin)
    y.float().sum().backward()
    torch.cuda.synchronize()
    peak_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
    # MATH-kernel fallback measures ~1709 MiB here; the fused kernel ~221 MiB.
    assert peak_mib < 500, f"peak {peak_mib:.0f} MiB: attention is not on a fused kernel"


@pytest.mark.cuda
def test_attention_runs_under_efficient_attention_only():
    """Deterministic companion to the memory-delta guard above: restrict SDPA to
    EFFICIENT_ATTENTION only (no cuDNN, no math) and run a real forward+backward.
    enable_gqa=True raises "No available kernel" here, because the
    memory-efficient kernel rejects mismatched query/KV head counts outright;
    the repeat_interleave fix passes because K/V are pre-expanded to match."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cfg = load_config(CONFIG_PATH).model  # the real 114M shape
    attn = Attention(cfg).cuda()
    cos, sin = build_rope_cache(cfg.context, cfg.head_dim, device="cuda")
    x = torch.randn(2, cfg.context, cfg.d_model, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16), \
         sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        y = attn(x, cos, sin)
    y.float().sum().backward()


def test_grouped_query_attention_matches_explicit_per_head_grouping():
    """Query head h must read KV head h // rep. repeat_interleave gives that
    layout; repeat (tile) does not, so this also catches that mutation."""
    torch.manual_seed(0)
    cfg = tiny()
    attn = Attention(cfg)
    T = cfg.context
    B = 2
    cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_base)
    x = torch.randn(B, T, cfg.d_model)

    out = attn(x, cos, sin)

    with torch.no_grad():
        q = attn.q(x).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
        k = attn.k(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        v = attn.v(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :T], sin[:, :, :T])
        k = apply_rope(k, cos[:, :, :T], sin[:, :, :T])

        rep = attn.n_head // attn.n_kv_head
        causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        heads_out = []
        for h in range(attn.n_head):
            kv_h = h // rep  # the grouping the whole test exists to pin down
            scores = (q[:, h] @ k[:, kv_h].transpose(-2, -1)) / math.sqrt(attn.head_dim)
            weights = torch.softmax(scores + causal, dim=-1)
            heads_out.append(weights @ v[:, kv_h])
        y_ref = torch.stack(heads_out, dim=1)
        out_ref = attn.o(y_ref.transpose(1, 2).contiguous().view(B, T, -1))

    torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=1e-5)


def test_no_output_depends_on_a_future_input():
    torch.manual_seed(0)
    cfg = tiny()
    m = Quipu(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.context))
    emb = m.embed(ids).detach().requires_grad_(True)
    h = emb
    for b in m.blocks:
        h = b(h, m.rope_cos, m.rope_sin)
    out = m.lm_head(m.norm(h))
    for t in range(cfg.context - 1):
        g, = torch.autograd.grad(out[:, t].sum(), emb, retain_graph=True)
        assert g[:, t + 1:].abs().max() == 0, f"position {t} sees the future"


def test_rejects_a_sequence_longer_than_the_context():
    cfg = tiny()
    model = Quipu(cfg)
    with pytest.raises(AssertionError):
        model(torch.randint(0, cfg.vocab_size, (1, cfg.context + 1)))
