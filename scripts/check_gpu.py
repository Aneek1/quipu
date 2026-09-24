"""Measure what this GPU actually sustains, so the token budget is decided on a
number rather than on my estimate of 20 TFLOPS.

Run: uv run python scripts/check_gpu.py
"""
import time

import torch

WINDOW_SECONDS = 5.0
TOTAL_SECONDS = 60.0


def measure_sustained_tflops(size: int = 4096) -> list[float]:
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(size, size, device="cuda", dtype=torch.bfloat16)

    for _ in range(10):          # warm up: the first kernels include compilation
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    flops_per_matmul = 2 * size**3  # one multiply and one add per output element
    windows: list[float] = []
    overall_started = time.perf_counter()

    while time.perf_counter() - overall_started < TOTAL_SECONDS:
        window_started = time.perf_counter()
        window_iters = 0
        while time.perf_counter() - window_started < WINDOW_SECONDS:
            torch.matmul(a, b, out=c)
            window_iters += 1
        torch.cuda.synchronize()
        window_elapsed = time.perf_counter() - window_started
        tflops = flops_per_matmul * window_iters / window_elapsed / 1e12
        windows.append(tflops)
        print(f"  window: {tflops:.1f} TFLOPS ({window_iters} iters in {window_elapsed:.2f}s)")

    return windows


def main() -> None:
    print(f"device:     {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"torch:      {torch.__version__}  cuda {torch.version.cuda}")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"vram:       {total:.1f} GiB")

    print(f"\nmeasuring sustained dense bf16 matmul over {TOTAL_SECONDS:.0f}s in {WINDOW_SECONDS:.0f}s windows:")
    windows = measure_sustained_tflops()
    minimum = min(windows)
    print(f"\nminimum sustained window: {minimum:.1f} TFLOPS (of {len(windows)} windows)")

    # Training reaches a fraction of peak matmul. 40% is a reasonable planning
    # figure for a well-implemented loop; the real number lands in Task 14.
    # Base the planning figure on the thermally-throttled minimum, not a burst.
    planning = minimum * 0.40
    tokens, params = 2.5e9, 114_114_048
    L, T, d = 12, 1024, 768

    flops_params_only = 6 * params
    flops_with_attn = 6 * params + 12 * L * T * d

    hours_params_only = (flops_params_only * tokens) / (planning * 1e12) / 3600
    hours_with_attn = (flops_with_attn * tokens) / (planning * 1e12) / 3600

    print(f"\nat 40% of the minimum sustained window ({planning:.1f} TFLOPS effective):")
    print(f"  2.5B tokens at 114M params, param-only 6N          -> {hours_params_only:.1f} hours")
    print(f"  2.5B tokens at 114M params, 6N + 12*L*T*d attention -> {hours_with_attn:.1f} hours")


if __name__ == "__main__":
    main()
