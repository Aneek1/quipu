"""Shared test markers.

`cuda`: the test needs a usable GPU and is skipped without one. "Usable" means
is_available() AND device_count() > 0: with CUDA_VISIBLE_DEVICES="" torch still
reports is_available() True but has no device, and the test would fail with
"Invalid device id" instead of skipping.

`gpu_gate`: tests/test_gpu_env.py, the environment gate. It deliberately FAILS
rather than skips when CUDA is missing, so it is never auto-skipped; deselect it
explicitly with -m "not gpu_gate" when the GPU is hidden on purpose.
"""
import pytest
import torch

HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a usable CUDA device; skipped without one")
    config.addinivalue_line(
        "markers", "gpu_gate: the GPU environment gate; fails (never skips) without CUDA"
    )


def pytest_collection_modifyitems(config, items):
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="needs a CUDA device (none visible)")
    for item in items:
        if item.get_closest_marker("cuda") is not None:
            item.add_marker(skip)
