"""The gate: if these fail, nothing downstream is worth building.

These tests deliberately FAIL, not skip, when CUDA is missing: a silently skipped
gate would let everything downstream run on CPU. They carry the `gpu_gate` marker
(see tests/conftest.py) and are not covered by the `cuda` auto-skip.

To run the rest of the suite with the GPU hidden (e.g. while a training run owns it):

    CUDA_VISIBLE_DEVICES= python -m uv run python -m pytest -m "not gpu_gate"

The `cuda`-marked tests elsewhere then skip instead of failing.
"""
import pytest
import torch

pytestmark = pytest.mark.gpu_gate


def test_torch_is_a_cuda_build():
    # A +cpu wheel reports None here and every later task would silently run on CPU.
    assert torch.version.cuda is not None, "CPU-only torch: reinstall from the cu128 index"


def test_cuda_is_available():
    assert torch.cuda.is_available(), "CUDA not available to torch"


def test_device_is_blackwell():
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) == (12, 0), f"expected sm_120, got sm_{major}{minor}"


def test_bf16_matmul_and_backward_run_on_gpu():
    # Blackwell support has historically failed at the first real kernel, not at
    # device detection, so this exercises a matmul AND a backward pass, and checks
    # the gradient is numerically correct, not just present.
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = (x @ w).float().sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # d/dx[i,j] sum(x @ w) = sum_k w[j,k] = w.sum(dim=1)[j], the same for every row i.
    expected = w.float().sum(dim=1).unsqueeze(0).expand_as(x)
    torch.testing.assert_close(x.grad.float(), expected, rtol=2e-2, atol=2e-2)


def test_scaled_dot_product_attention_supports_gqa():
    # The model depends on enable_gqa; check it matches the manually-expanded
    # equivalent, not just that it runs and returns the right shape.
    q = torch.randn(1, 12, 64, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
    assert out.shape == (1, 12, 64, 64)

    k_expanded = k.repeat_interleave(3, dim=1)
    v_expanded = v.repeat_interleave(3, dim=1)
    out_expanded = torch.nn.functional.scaled_dot_product_attention(
        q, k_expanded, v_expanded, is_causal=True, enable_gqa=False
    )
    torch.testing.assert_close(out.float(), out_expanded.float(), rtol=2e-2, atol=2e-2)
