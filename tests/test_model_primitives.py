import torch

from quipu.model import RMSNorm, SwiGLU, apply_rope, build_rope_cache


def test_rmsnorm_gives_unit_rms_when_weight_is_one():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64) * 5.0
    rms = norm(x).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_centre():
    # RMSNorm scales but must not subtract the mean; that is LayerNorm.
    norm = RMSNorm(64)
    x = torch.randn(1, 1, 64) + 10.0
    assert norm(x).mean().abs() > 0.1


def test_rmsnorm_preserves_dtype():
    norm = RMSNorm(64)
    assert norm(torch.randn(2, 8, 64, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_rope_preserves_shape():
    cos, sin = build_rope_cache(16, 64)
    x = torch.randn(2, 4, 16, 64)
    assert apply_rope(x, cos, sin).shape == x.shape


def test_rope_preserves_vector_norm():
    # RoPE is a rotation, so it must not change magnitudes.
    cos, sin = build_rope_cache(16, 64)
    x = torch.randn(2, 4, 16, 64)
    before = x.norm(dim=-1)
    after = apply_rope(x, cos, sin).norm(dim=-1)
    assert torch.allclose(before, after, atol=1e-4)


def test_rope_is_position_dependent():
    # The same vector at two positions must come out different, or RoPE is a no-op.
    cos, sin = build_rope_cache(16, 64)
    x = torch.randn(1, 1, 1, 64).expand(1, 1, 16, 64).contiguous()
    out = apply_rope(x, cos, sin)
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 5], atol=1e-4)


def test_rope_leaves_position_zero_unrotated():
    cos, sin = build_rope_cache(16, 64)
    x = torch.randn(1, 1, 16, 64)
    assert torch.allclose(apply_rope(x, cos, sin)[0, 0, 0], x[0, 0, 0], atol=1e-5)


def test_swiglu_shape_and_parameter_count():
    ffn = SwiGLU(768, 2048)
    assert ffn(torch.randn(2, 8, 768)).shape == (2, 8, 768)
    # Three matrices, no biases.
    assert sum(p.numel() for p in ffn.parameters()) == 768 * 2048 * 3
