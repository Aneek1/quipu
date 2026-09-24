"""Measure what this GPU actually sustains, so the token budget is decided on a
number rather than on my estimate of 20 TFLOPS.

Run: uv run python scripts/check_gpu.py
"""
import time

import torch


def measure_tflops(size: int = 4096, iters: int = 50) -> float:
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):          # warm up: the first kernels include compilation
        a @ b
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    flops = 2 * size**3 * iters   # one multiply and one add per output element
    return flops / elapsed / 1e12


def main() -> None:
    print(f"device:     {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"torch:      {torch.__version__}  cuda {torch.version.cuda}")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"vram:       {total:.1f} GB")

    peak = measure_tflops()
    print(f"\ndense bf16 matmul: {peak:.1f} TFLOPS")

    # Training reaches a fraction of peak matmul. 40% is a reasonable planning
    # figure for a well-implemented loop; the real number lands in Task 14.
    planning = peak * 0.40
    tokens, params = 2.5e9, 114_114_048
    hours = (6 * params * tokens) / (planning * 1e12) / 3600
    print(f"at 40% of that ({planning:.1f} TFLOPS effective):")
    print(f"  2.5B tokens at 114M params -> {hours:.1f} hours")


if __name__ == "__main__":
    main()
