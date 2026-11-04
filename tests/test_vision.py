"""quipu/vision.py: shapes, parameter count, padding invariance."""
import pytest
import torch

from quipu.vision import LineViT, ViTConfig

TINY = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)


def test_default_config_is_about_22m_parameters():
    n = sum(p.numel() for p in LineViT(ViTConfig()).parameters())
    assert 20e6 < n < 23e6, n


def test_output_shape():
    m = LineViT(TINY).eval()
    images = torch.rand(2, 1, 64, 160)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert m(images, mask).shape == (2, 4, 10, 32)


def test_padding_does_not_change_real_columns():
    torch.manual_seed(0)
    m = LineViT(TINY).eval()
    img = torch.rand(1, 1, 64, 64)
    padded = torch.cat([img, torch.zeros(1, 1, 64, 64)], dim=-1)
    a = m(img, torch.ones(1, 4, dtype=torch.bool))
    mask = torch.tensor([[True] * 4 + [False] * 4])
    b = m(padded, mask)
    torch.testing.assert_close(a, b[:, :, :4], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape,cols,match", [
    ((1, 1, 32, 64), 4, "height"),
    ((1, 1, 64, 70), 4, "multiple of 16"),
    ((1, 1, 64, 1040), 65, "max_width"),
    ((1, 1, 64, 64), 3, "col_mask"),
])
def test_bad_inputs_raise(shape, cols, match):
    m = LineViT(TINY)
    with pytest.raises(ValueError, match=match):
        m(torch.zeros(shape), torch.ones(1, cols, dtype=torch.bool))


def test_float_mask_raises():
    m = LineViT(TINY)
    with pytest.raises(ValueError, match="col_mask must be bool"):
        m(torch.zeros(1, 1, 64, 64), torch.ones(1, 4))


def test_item_with_no_real_columns_raises():
    m = LineViT(TINY)
    mask = torch.tensor([[True] * 4, [False] * 4])
    with pytest.raises(ValueError, match="no real columns"):
        m(torch.zeros(2, 1, 64, 64), mask)


def test_linear_init_is_small_with_zero_bias():
    m = LineViT(TINY)
    lin = m.blocks[0].qkv
    assert lin.bias.abs().max().item() == 0
    assert 0.01 < lin.weight.std().item() < 0.03
    assert lin.weight.abs().max().item() <= 0.04 + 1e-6
