"""Measure what this GPU actually sustains, so the token budget is decided on a
number rather than on my estimate of 20 TFLOPS.

Run: uv run python scripts/check_gpu.py
"""
import time

import torch

from quipu.config import load_config
from quipu.model import Quipu

CONFIG = "configs/quipu-114m.toml"
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

    # Each window stops queueing matmuls after WINDOW_SECONDS and then waits for the
    # queue to drain, so on this card a window really lasts ~9.5 s. TFLOPS divides by
    # the real elapsed time, so the figure is unaffected.
    print(f"\nmeasuring sustained dense bf16 matmul for ~{TOTAL_SECONDS:.0f}s; each window queues "
          f"{WINDOW_SECONDS:.0f}s of work and runs longer while the queue drains:")
    windows = measure_sustained_tflops()
    minimum = min(windows)
    print(f"\nminimum sustained window: {minimum:.1f} TFLOPS (of {len(windows)} windows)")

    # Training reaches a fraction of peak matmul. 40% is a reasonable planning
    # figure for a well-implemented loop; the real number lands in Task 14.
    # Base the planning figure on the thermally-throttled minimum, not a burst.
    planning = minimum * 0.40
    cfg = load_config(CONFIG)
    with torch.device("meta"):     # count parameters without allocating them
        params = sum(p.numel() for p in Quipu(cfg.model).parameters())
    tokens = cfg.train.total_tokens
    L, T, d = cfg.model.n_layer, cfg.model.context, cfg.model.d_model

    flops_params_only = 6 * params
    flops_with_attn = 6 * params + 12 * L * T * d

    hours_params_only = (flops_params_only * tokens) / (planning * 1e12) / 3600
    hours_with_attn = (flops_with_attn * tokens) / (planning * 1e12) / 3600

    print(f"\nat 40% of the minimum sustained window ({planning:.1f} TFLOPS effective):")
    label = f"{tokens / 1e9:.1f}B tokens at {params / 1e6:.0f}M params"
    print(f"  {label}, param-only 6N          -> {hours_params_only:.1f} hours")
    print(f"  {label}, 6N + 12*L*T*d attention -> {hours_with_attn:.1f} hours")


if __name__ == "__main__":
    main()
