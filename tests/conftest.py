"""Shared test markers.

`cuda`: the test needs a usable GPU and is skipped without one. "Usable" means
is_available() AND device_count() > 0: with CUDA_VISIBLE_DEVICES="" torch still
reports is_available() True but has no device, and the test would fail with
"Invalid device id" instead of skipping.

`gpu_gate`: tests/test_gpu_env.py, the environment gate. It deliberately FAILS
rather than skips when CUDA is missing, so it is never auto-skipped; deselect it
explicitly with -m "not gpu_gate" when the GPU is hidden on purpose.

`npm` / `git`: the test shells out to that tool and is skipped when it is not on
PATH, so the unit suite still runs on a machine without Node or git.
"""
import shutil

import pytest
import torch

HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
MISSING_TOOLS = [tool for tool in ("npm", "git") if shutil.which(tool) is None]


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a usable CUDA device; skipped without one")
    config.addinivalue_line(
        "markers", "gpu_gate: the GPU environment gate; fails (never skips) without CUDA"
    )
    config.addinivalue_line("markers", "npm: needs npm on PATH; skipped without it")
    config.addinivalue_line("markers", "git: needs git on PATH; skipped without it")


def pytest_collection_modifyitems(config, items):
    for tool in MISSING_TOOLS:
        skip_tool = pytest.mark.skip(reason=f"needs {tool} on PATH (not found)")
        for item in items:
            if item.get_closest_marker(tool) is not None:
                item.add_marker(skip_tool)
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="needs a CUDA device (none visible)")
    for item in items:
        if item.get_closest_marker("cuda") is not None:
            item.add_marker(skip)
