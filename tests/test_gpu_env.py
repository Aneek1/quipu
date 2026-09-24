"""The gate: if these fail, nothing downstream is worth building."""
import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"
)


def test_torch_is_a_cuda_build():
    # A +cpu wheel reports None here and every later task would silently run on CPU.
    assert torch.version.cuda is not None, "CPU-only torch: reinstall from the cu128 index"


@requires_cuda
def test_device_is_blackwell():
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) == (12, 0), f"expected sm_120, got sm_{major}{minor}"


@requires_cuda
def test_bf16_matmul_and_backward_run_on_gpu():
    # Blackwell support has historically failed at the first real kernel, not at
    # device detection, so this exercises a matmul AND a backward pass.
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = (x @ w).float().sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@requires_cuda
def test_scaled_dot_product_attention_supports_gqa():
    # The model depends on enable_gqa; assert it exists before building around it.
    q = torch.randn(1, 12, 64, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
    assert out.shape == (1, 12, 64, 64)
