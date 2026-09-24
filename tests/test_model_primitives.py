import torch
import torch.nn.functional as F

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


def test_rope_scores_depend_only_on_relative_position():
    D = 64
    cos, sin = build_rope_cache(64, D)
    q, k = torch.randn(D, dtype=torch.float64), torch.randn(D, dtype=torch.float64)
    def at(v, p):
        x = v.view(1, 1, 1, D).expand(1, 1, 64, D).contiguous()
        return apply_rope(x, cos.double(), sin.double())[0, 0, p]
    ref = at(q, 7) @ at(k, 3)
    for m, n in [(11, 7), (40, 36), (63, 59)]:
        assert torch.allclose(at(q, m) @ at(k, n), ref, atol=1e-4)

def test_rope_matches_complex_reference():
    # NeoX pairing: z_j = x_j + i*x_{j+D/2}, rotated by pos * base^(-2j/D)
    D, base = 64, 10000.0
    cos, sin = build_rope_cache(8, D, base)
    x = torch.randn(1, 1, 8, D, dtype=torch.float64)
    j = torch.arange(D // 2, dtype=torch.float64)
    ang = torch.arange(8, dtype=torch.float64)[:, None] * base ** (-2 * j / D)
    z = torch.complex(x[..., :D // 2], x[..., D // 2:]) * torch.exp(1j * ang)
    assert torch.allclose(apply_rope(x, cos.double(), sin.double()),
                          torch.cat([z.real, z.imag], -1), atol=1e-5)

def test_rmsnorm_applies_weight():
    n = RMSNorm(16)
    with torch.no_grad():
        n.weight.copy_(torch.arange(16.0))
    x = torch.randn(3, 16)
    exp = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * torch.arange(16.0)
    assert torch.allclose(n(x), exp, atol=1e-5)

def test_swiglu_matches_reference():
    f = SwiGLU(8, 16)
    x = torch.randn(2, 8)
    exp = (F.silu(x @ f.gate.weight.T) * (x @ f.up.weight.T)) @ f.down.weight.T
    assert torch.allclose(f(x), exp, atol=1e-6)


def test_rope_preserves_bf16_dtype():
    cos, sin = build_rope_cache(16, 64)
    x = torch.randn(2, 4, 16, 64, dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16
