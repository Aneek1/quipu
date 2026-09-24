# Quipu Pretraining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A validated from-scratch pretraining pipeline that trains a 114,114,048-parameter decoder-only transformer on 2.5B tokens of FineWeb-Edu, in roughly a day, on one RTX 5060 Laptop GPU.

**Architecture:** A modern decoder-only transformer (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings) with no retrieval and no memory — those arrive in later sub-projects. Data is pre-tokenized into flat `uint16` shards; training reads them through a loader whose position is part of the checkpoint, so a 20-hour run survives the laptop sleeping.

**Tech Stack:** Python 3.13, PyTorch 2.9.1+cu128 (Blackwell sm_120), tiktoken, HuggingFace `datasets`, numpy, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-24-quipu-pretraining-pipeline-design.md`

---

## Environment facts (verified 2026-09-24)

Do not re-derive these; they were measured on the target machine.

| Fact | Value |
|---|---|
| GPU | RTX 5060 Laptop, 8151 MiB, **compute capability 12.0 (sm_120)** |
| Driver | 592.15 |
| Python | 3.13.13 |
| uv | 0.12.6 |
| Torch wheel that exists for this combo | `torch-2.9.1+cu128-cp313-cp313-win_amd64.whl` |
| Free space on C: | 366.8 GB |

**Run on native Windows, not WSL.** WSL2 `vmIdleTimeout` kills long jobs on this machine and a 20-hour run cannot sit inside it.

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, pytest config |
| `configs/quipu-114m.toml` | The single file that determines a run |
| `quipu/config.py` | Load and validate TOML into frozen dataclasses; derive step counts |
| `quipu/tokenizer.py` | Thin wrapper over tiktoken `gpt2`, so sub-project 2 can swap it |
| `quipu/data.py` | Tokenize documents, write/read `uint16` shards |
| `quipu/loader.py` | Resumable token stream over shards |
| `quipu/model.py` | RMSNorm, RoPE, SwiGLU, GQA attention, Block, Quipu |
| `quipu/train.py` | Training loop, gradient accumulation, checkpoint, resume |
| `quipu/eval.py` | Held-out loss and generation probes |
| `quipu/runlog.py` | Per-run JSON records |
| `quipu/results_table.py` | Generates RESULTS.md from run records |
| `scripts/check_gpu.py` | Task 1 gate: proves Blackwell works and measures real throughput |
| `scripts/build_shards.py` | Downloads FineWeb-Edu and writes the shards |

`quipu/loader.py` is an addition to the spec's component table. The spec assigns shard *format* to
`data.py` and this keeps the *streaming position* — which is checkpoint state — in its own unit.

### Two config values differ from the spec, deliberately

| Value | Spec | Plan | Why |
|---|---|---|---|
| warmup_steps | 700 | **200** | The spec's 700 is the nanoGPT default for a 600k-step run. This run is ~4,768 steps, where 700 would be 15% of training spent warming up. 200 is 4.2%. |
| total steps | "~5,000" | **4,768** | 2,500,000,000 / 524,288, computed rather than rounded. |

---

## Task 1: Prove Blackwell works before building anything

The spec makes this the gate: torch is not installed, sm_120 is new, and everything downstream
assumes both a working GPU and ~20 TFLOPS. Measure, don't assume.

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/check_gpu.py`
- Create: `tests/test_gpu_env.py`

- [ ] **Step 1: Create the project definition**

Create `pyproject.toml`:

```toml
[project]
name = "quipu"
version = "0.1.0"
description = "A small language model trained from scratch"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.9.1",
    "numpy>=2.1",
    "tiktoken>=0.8",
    "datasets>=3.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["quipu"]
```

- [ ] **Step 2: Install torch from the CUDA 12.8 index**

CUDA 12.8 is the first release with Blackwell (sm_120) support. The default PyPI torch wheel is
CPU-only on Windows, so the index URL is required, not optional.

Run:
```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"
```

Expected: torch 2.9.1+cu128 installs. If it resolves to a `+cpu` build, the index URL was dropped.

- [ ] **Step 3: Write the environment test**

Create `tests/test_gpu_env.py`:

```python
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
```

- [ ] **Step 4: Run the environment test**

Run: `uv run pytest tests/test_gpu_env.py -v`
Expected: 4 passed. If `test_device_is_blackwell` fails, stop and report the capability found.

- [ ] **Step 5: Write the throughput measurement**

Create `scripts/check_gpu.py`:

```python
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
```

- [ ] **Step 6: Run it and record the number**

Run: `uv run python scripts/check_gpu.py`
Expected: prints a TFLOPS figure and an hours estimate.

**This is a decision gate.** If the projected run exceeds 36 hours, stop and reduce the token
budget before building anything else. Write the measured figure into the commit message.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/check_gpu.py tests/test_gpu_env.py
git commit -m "Prove the GPU works and measure what it sustains

sm_120 with torch 2.9.1+cu128: bf16 matmul, backward and GQA attention all run.
Measured <N> TFLOPS dense bf16, projecting <H> hours for the planned budget."
```

---

## Task 2: Config loading

One file determines a run. This is success criterion 3 in the spec.

**Files:**
- Create: `quipu/__init__.py`
- Create: `quipu/config.py`
- Create: `configs/quipu-114m.toml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest

from quipu.config import load_config


CONFIG = "configs/quipu-114m.toml"


def test_loads_the_shipped_config():
    cfg = load_config(CONFIG)
    assert cfg.name == "quipu-114m"
    assert cfg.model.d_model == 768
    assert cfg.model.n_layer == 12
    assert cfg.model.n_kv_head == 4


def test_derives_step_count_from_token_budget():
    cfg = load_config(CONFIG)
    # 2,500,000,000 / 524,288 = 4768 (floor)
    assert cfg.train.steps == 4768


def test_derives_gradient_accumulation():
    cfg = load_config(CONFIG)
    # 524,288 tokens per step / (micro_batch x context)
    expected = cfg.train.batch_tokens // (cfg.train.micro_batch * cfg.model.context)
    assert cfg.train.grad_accum == expected
    assert cfg.train.grad_accum >= 1


def test_rejects_a_batch_that_does_not_divide_evenly():
    # A batch_tokens that is not a multiple of micro_batch x context would silently
    # train on a different number of tokens than the config claims.
    with pytest.raises(ValueError, match="batch_tokens"):
        load_config(CONFIG, overrides={"train": {"batch_tokens": 524_289}})


def test_config_is_frozen():
    cfg = load_config(CONFIG)
    with pytest.raises(Exception):
        cfg.model.d_model = 1024
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.config'`

- [ ] **Step 3: Write the config module**

Create `quipu/__init__.py` (empty file).

Create `quipu/config.py`:

```python
"""One TOML file fully determines a run. Everything derived is computed here, once,
so no two call sites can disagree about how many steps there are."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    ffn_hidden: int
    context: int
    rope_base: float
    norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    subset: str
    shard_dir: str
    shard_tokens: int
    val_tokens: int


@dataclass(frozen=True)
class TrainConfig:
    total_tokens: int
    batch_tokens: int
    micro_batch: int
    context: int
    lr: float
    lr_min: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    seed: int
    ckpt_dir: str
    ckpt_every: int
    eval_every: int
    eval_batches: int

    @property
    def steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def grad_accum(self) -> int:
        return self.batch_tokens // (self.micro_batch * self.context)


@dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        raw = _merge(raw, overrides)

    model = ModelConfig(**raw["model"])
    data = DataConfig(**raw["data"])
    train = TrainConfig(context=model.context, **raw["train"])

    if train.batch_tokens % (train.micro_batch * model.context) != 0:
        raise ValueError(
            f"batch_tokens ({train.batch_tokens}) must be a multiple of "
            f"micro_batch x context ({train.micro_batch} x {model.context})"
        )
    if model.d_model % model.n_head != 0:
        raise ValueError("d_model must divide evenly by n_head")
    if model.n_head % model.n_kv_head != 0:
        raise ValueError("n_head must be a multiple of n_kv_head for GQA")

    return Config(name=raw["name"], model=model, data=data, train=train)
```

- [ ] **Step 4: Write the config file**

Create `configs/quipu-114m.toml`:

```toml
name = "quipu-114m"

[model]
vocab_size = 50257     # tiktoken gpt2
d_model    = 768
n_layer    = 12
n_head     = 12
n_kv_head  = 4         # GQA; the recipe that scales to 350M
ffn_hidden = 2048      # SwiGLU, three matrices
context    = 1024
rope_base  = 10000.0
norm_eps   = 1e-6

[data]
dataset      = "HuggingFaceFW/fineweb-edu"
subset       = "sample-10BT"
shard_dir    = "data/shards"
shard_tokens = 100_000_000   # ~200 MB per shard as uint16
val_tokens   = 10_000_000

[train]
total_tokens  = 2_500_000_000
batch_tokens  = 524_288       # 512 sequences x 1024 tokens
micro_batch   = 8             # tuned in Task 14; grad_accum absorbs any change
lr            = 6e-4
lr_min        = 6e-5
warmup_steps  = 200
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0
seed          = 1337
ckpt_dir      = "checkpoints"
ckpt_every    = 100
eval_every    = 100
eval_batches  = 20
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 6: Mutation-test the divisibility guard**

Temporarily change `!= 0` to `== 0` in `quipu/config.py`. Run
`uv run pytest tests/test_config.py -v` and confirm
`test_rejects_a_batch_that_does_not_divide_evenly` FAILS. Restore the line.

A guard that cannot fail is not a guard.

- [ ] **Step 7: Commit**

```bash
git add quipu/__init__.py quipu/config.py configs/quipu-114m.toml tests/test_config.py
git commit -m "Load a run from one TOML file and derive its step counts once

Step count and gradient accumulation are computed in the config rather than at
call sites, so nothing can disagree about how many tokens a run actually sees."
```

---

## Task 3: Tokenizer wrapper

**Files:**
- Create: `quipu/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tokenizer.py`:

```python
from quipu.tokenizer import Tokenizer


def test_vocab_size_matches_the_config():
    assert Tokenizer().vocab_size == 50257


def test_round_trips_text():
    tok = Tokenizer()
    text = "The quipu encoded records in knotted cords."
    assert tok.decode(tok.encode(text)) == text


def test_eot_is_the_documented_id():
    # 50256 is <|endoftext|> in the GPT-2 vocabulary. data.py separates documents
    # with it, so a change here silently changes the shard format.
    assert Tokenizer().eot == 50256


def test_every_token_id_fits_in_uint16():
    # Shards are uint16. A vocabulary above 65535 would wrap silently.
    assert Tokenizer().vocab_size <= 65536
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.tokenizer'`

- [ ] **Step 3: Write the tokenizer**

Create `quipu/tokenizer.py`:

```python
"""GPT-2 BPE, wrapped.

The wrapper exists for one reason: sub-project 2 may train a tokenizer on a
code-heavy mixture, and every call site should keep working when it does.
"""
from __future__ import annotations

import tiktoken


class Tokenizer:
    def __init__(self, encoding: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
```

`encode_ordinary` skips special-token parsing, so a document containing the literal text
`<|endoftext|>` is encoded as ordinary characters rather than raising.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quipu/tokenizer.py tests/test_tokenizer.py
git commit -m "Wrap the GPT-2 tokenizer so sub-project 2 can swap it"
```

---

## Task 4: Shard format

Spec test 1. A shard bug is invisible in the loss curve, which is exactly why it gets its own test.

**Files:**
- Create: `quipu/data.py`
- Test: `tests/test_data_shards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_shards.py`:

```python
import numpy as np
import pytest

from quipu.data import read_shard, tokenize_documents, write_shard
from quipu.tokenizer import Tokenizer


def test_shard_round_trips_exactly(tmp_path):
    tokens = np.array([1, 2, 3, 50256, 4, 5], dtype=np.uint16)
    path = tmp_path / "shard_000.bin"
    write_shard(path, tokens)
    assert np.array_equal(read_shard(path), tokens)


def test_shard_reads_back_as_uint16(tmp_path):
    path = tmp_path / "shard_000.bin"
    write_shard(path, np.array([7, 8, 9], dtype=np.uint16))
    assert read_shard(path).dtype == np.uint16


def test_shard_length_is_exact(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "shard_000.bin"
    write_shard(path, tokens)
    assert len(read_shard(path)) == 1000


def test_write_rejects_the_wrong_dtype(tmp_path):
    # int32 tokens would write twice the bytes and every later read would be garbage.
    with pytest.raises(ValueError, match="uint16"):
        write_shard(tmp_path / "x.bin", np.array([1, 2, 3], dtype=np.int32))


def test_documents_are_separated_by_eot():
    tok = Tokenizer()
    out = tokenize_documents(["hello", "world"], tok)
    # Each document is followed by exactly one EOT.
    assert out.tolist().count(tok.eot) == 2
    assert out[-1] == tok.eot


def test_tokenized_output_is_uint16():
    out = tokenize_documents(["hello"], Tokenizer())
    assert out.dtype == np.uint16


def test_empty_documents_are_skipped():
    tok = Tokenizer()
    # A blank document would contribute a bare EOT and teach the model that EOT
    # follows EOT, which is not a pattern in the corpus.
    out = tokenize_documents(["hello", "", "   ", "world"], tok)
    assert out.tolist().count(tok.eot) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_data_shards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.data'`

- [ ] **Step 3: Write the data module**

Create `quipu/data.py`:

```python
"""Shard format: a flat little-endian uint16 array of token ids, nothing else.

No header, no index, no compression. The file length divided by two is the token
count, which makes a shard trivially memory-mappable and impossible to misparse.
The vocabulary is 50257, so uint16 is exact; tokenizer.py asserts that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from quipu.tokenizer import Tokenizer


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        raise ValueError(f"shards are uint16, got {tokens.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tofile writes raw little-endian on every platform we target.
    tokens.astype("<u2", copy=False).tofile(path)


def read_shard(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2")


def tokenize_documents(docs: Iterable[str], tok: Tokenizer) -> np.ndarray:
    """Concatenate documents, each terminated by one EOT.

    Blank documents are dropped: a bare EOT would teach the model that EOT follows
    EOT, which never happens in the corpus.
    """
    out: list[int] = []
    for doc in docs:
        if not doc or not doc.strip():
            continue
        out.extend(tok.encode(doc))
        out.append(tok.eot)
    return np.array(out, dtype=np.uint16)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_data_shards.py -v`
Expected: 7 passed.

- [ ] **Step 5: Mutation-test the dtype guard**

Delete the `if tokens.dtype != np.uint16: raise` block. Run the tests and confirm
`test_write_rejects_the_wrong_dtype` FAILS. Restore it.

- [ ] **Step 6: Commit**

```bash
git add quipu/data.py tests/test_data_shards.py
git commit -m "Define the shard format: a flat uint16 array and nothing else

No header means the file length is the token count and a shard cannot be
misparsed. Blank documents are dropped so EOT never follows EOT."
```

---

## Task 5: Model primitives

RMSNorm, RoPE and SwiGLU, each tested alone before anything composes them.

**Files:**
- Create: `quipu/model.py`
- Test: `tests/test_model_primitives.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_primitives.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_model_primitives.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.model'`

- [ ] **Step 3: Write the primitives**

Create `quipu/model.py`:

```python
"""The transformer. No training code, no I/O, no retrieval.

Retrieval and memory arrive in sub-project 4. This file stays deliberately boring
so that the ablation later has something honest to be ablated against.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quipu.config import ModelConfig


class RMSNorm(nn.Module):
    """Scale by the root-mean-square. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # The reduction runs in fp32: in bf16 the sum of squares over 768 elements
        # loses enough precision to shift the norm visibly.
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each shaped (1, 1, seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, H, T, D) by position. Splits the head in halves, GPT-NeoX style."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[..., : x1.shape[-1]].to(x.dtype)
    sin = sin[..., : x1.shape[-1]].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """SiLU-gated feed-forward. Three matrices, no biases."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_model_primitives.py -v`
Expected: 8 passed.

- [ ] **Step 5: Mutation-test the RoPE rotation**

In `apply_rope`, change `x1 * cos - x2 * sin` to `x1 * cos + x2 * sin`. Run the tests and confirm
`test_rope_preserves_vector_norm` FAILS (the sign makes it a shear, not a rotation). Restore it.

- [ ] **Step 6: Commit**

```bash
git add quipu/model.py tests/test_model_primitives.py
git commit -m "Add RMSNorm, RoPE and SwiGLU, each tested alone

RMSNorm reduces in fp32 because bf16 loses too much over 768 elements. The RoPE
tests assert it is a rotation and that it is position-dependent, which a no-op
implementation would pass neither of."
```

---

## Task 6: Attention, block and the model

**Files:**
- Modify: `quipu/model.py` (append)
- Test: `tests/test_model_shapes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_shapes.py`:

```python
import torch

from quipu.config import ModelConfig, load_config
from quipu.model import Quipu


def tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=32, rope_base=10000.0, norm_eps=1e-6,
    )


def test_forward_returns_logits_over_the_vocabulary():
    cfg = tiny()
    model = Quipu(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (2, 8)))
    assert out.shape == (2, 8, cfg.vocab_size)


def test_embeddings_are_tied():
    # Tied weights save 38.6M parameters at the real scale. If the tie silently
    # breaks, the parameter count test below is the only thing that notices.
    model = Quipu(tiny())
    assert model.lm_head.weight is model.embed.weight


def test_parameter_count_is_exactly_the_documented_figure():
    cfg = load_config("configs/quipu-114m.toml").model
    total = sum(p.numel() for p in Quipu(cfg).parameters())
    assert total == 114_114_048


def test_causal_mask_does_not_leak():
    """The single most important test in this sub-project.

    A mask that lets position t see position t+1 produces a beautiful loss curve
    and a model that cannot generate. Changing the LAST token must leave every
    earlier position bit-identical.
    """
    torch.manual_seed(0)
    cfg = tiny()
    model = Quipu(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        before = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (int(ids2[0, -1]) + 1) % cfg.vocab_size
    with torch.no_grad():
        after = model(ids2)

    assert torch.equal(before[0, :-1], after[0, :-1])
    # And the changed position really did change, or the test proves nothing.
    assert not torch.equal(before[0, -1], after[0, -1])


def test_accepts_a_sequence_shorter_than_the_context():
    cfg = tiny()
    assert Quipu(cfg)(torch.randint(0, cfg.vocab_size, (1, 5))).shape == (1, 5, cfg.vocab_size)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_model_shapes.py -v`
Expected: FAIL with `ImportError: cannot import name 'Quipu'`

- [ ] **Step 3: Append attention, block and model to `quipu/model.py`**

```python
class Attention(nn.Module):
    """Grouped-query attention with RoPE and a causal mask."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:, :, :T], sin[:, :, :T])
        k = apply_rope(k, cos[:, :, :T], sin[:, :, :T])

        # is_causal=True is the mask. enable_gqa lets 12 query heads share 4 KV heads
        # without materialising the repeat.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class Quipu(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tied: saves 38.6M parameters

        cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_base)
        # Buffers, not parameters: they are derived constants and must not be trained
        # or saved into the optimiser state.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        # Scale the residual projections by depth, as GPT-2 does, so the residual
        # stream does not grow with n_layer.
        for name, p in self.named_parameters():
            if name.endswith("o.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm(x))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_model_shapes.py -v`
Expected: 5 passed. If `test_parameter_count_is_exactly_the_documented_figure` fails, print the
actual number and reconcile it against §4 of the spec before changing either.

- [ ] **Step 5: Mutation-test the causal mask**

Change `is_causal=True` to `is_causal=False` in `Attention.forward`. Run
`uv run pytest tests/test_model_shapes.py -v` and confirm `test_causal_mask_does_not_leak` FAILS.
Restore it.

This is the mutation that matters most in the whole plan. Do not skip it.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add quipu/model.py tests/test_model_shapes.py
git commit -m "Add GQA attention, the block and the model, at exactly 114,114,048 parameters

The causal-mask test is the important one: a leaking mask trains to a good loss
curve and produces a model that cannot generate, and nothing else would catch it."
```

---

## Task 7: Resumable token stream

Checkpoint state, not a convenience. A loader that restarts from zero silently re-trains on the
same tokens and the loss curve will not show it.

**Files:**
- Create: `quipu/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_loader.py`:

```python
import numpy as np
import torch

from quipu.data import write_shard
from quipu.loader import TokenStream


def make_shards(tmp_path, n_shards=3, per_shard=1000):
    for i in range(n_shards):
        start = i * per_shard
        write_shard(
            tmp_path / f"shard_{i:03d}.bin",
            np.arange(start, start + per_shard, dtype=np.uint16),
        )
    return tmp_path


def test_yields_inputs_and_targets_offset_by_one(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    x, y = stream.next_batch()
    assert x.shape == (2, 8) and y.shape == (2, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_position_advances(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    assert stream.position == 0
    stream.next_batch()
    # Each batch consumes micro_batch x context tokens, plus the one-token lookahead.
    assert stream.position == 2 * 8


def test_state_round_trip_resumes_the_same_batch(tmp_path):
    a = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    for _ in range(5):
        a.next_batch()
    state = a.state_dict()
    expected_x, expected_y = a.next_batch()

    b = TokenStream(tmp_path, micro_batch=2, context=8)
    b.load_state_dict(state)
    got_x, got_y = b.next_batch()

    assert torch.equal(expected_x, got_x)
    assert torch.equal(expected_y, got_y)


def test_wraps_around_at_the_end_of_the_data(tmp_path):
    stream = TokenStream(make_shards(tmp_path, n_shards=1, per_shard=64),
                         micro_batch=2, context=8)
    for _ in range(20):          # far past the 64 tokens available
        x, _ = stream.next_batch()
        assert x.shape == (2, 8)


def test_crosses_a_shard_boundary_without_a_gap(tmp_path):
    # Shards hold 0..999, 1000..1999, 2000..2999. Reading across the join must be
    # contiguous, or the model sees a discontinuity it will happily learn.
    stream = TokenStream(make_shards(tmp_path), micro_batch=1, context=4)
    stream.load_state_dict({"position": 998})
    x, _ = stream.next_batch()
    assert x[0].tolist() == [998, 999, 1000, 1001]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.loader'`

- [ ] **Step 3: Write the loader**

Create `quipu/loader.py`:

```python
"""A position in the token stream, which is checkpoint state.

Shards are concatenated into one logical array and read sequentially. The position
is a token offset into that array, so resuming is exact and crossing a shard
boundary is not a special case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from quipu.data import read_shard


class TokenStream:
    def __init__(self, shard_dir: str | Path, micro_batch: int, context: int) -> None:
        self.shard_dir = Path(shard_dir)
        self.micro_batch = micro_batch
        self.context = context

        paths = sorted(self.shard_dir.glob("shard_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no shards in {self.shard_dir}")
        # Concatenating keeps the boundary logic in one place. At 2.5B tokens this
        # is 5 GB, which fits in 31.7 GB of RAM; if it ever does not, this becomes
        # a memmap and nothing else changes.
        self.tokens = np.concatenate([read_shard(p) for p in paths])
        self.position = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.micro_batch * self.context + 1   # +1 for the shifted target
        if self.position + need > len(self.tokens):
            self.position = 0
        chunk = self.tokens[self.position : self.position + need].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.micro_batch, self.context)
        y = torch.from_numpy(chunk[1:]).view(self.micro_batch, self.context)
        self.position += self.micro_batch * self.context
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_loader.py -v`
Expected: 5 passed.

- [ ] **Step 5: Mutation-test the resume**

In `load_state_dict`, change the body to `self.position = 0`. Run the tests and confirm
`test_state_round_trip_resumes_the_same_batch` FAILS. Restore it.

- [ ] **Step 6: Commit**

```bash
git add quipu/loader.py tests/test_loader.py
git commit -m "Stream tokens from shards with a position that survives a restart

The position is checkpoint state. A loader that silently restarts from zero
re-trains on the same tokens and the loss curve does not show it."
```

---

## Task 8: Run logging

Ported from `ai-chatbot-softrobotics/training/`. Nothing in RESULTS.md is ever typed by hand.

**Files:**
- Create: `quipu/runlog.py`
- Test: `tests/test_runlog.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runlog.py`:

```python
import json

from quipu.runlog import RunLog


def test_writes_a_record_with_config_and_metrics(tmp_path):
    log = RunLog(tmp_path, run_id="test-run", config={"name": "quipu-114m"})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=524288)
    log.log_step(step=2, train_loss=4.5, lr=2e-4, tokens=1048576)
    log.finish(status="completed")

    record = json.loads((tmp_path / "test-run.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "test-run"
    assert record["config"]["name"] == "quipu-114m"
    assert record["status"] == "completed"
    assert len(record["steps"]) == 2
    assert record["steps"][-1]["train_loss"] == 4.5


def test_survives_being_killed_mid_run(tmp_path):
    # A 20-hour run that loses its log because the laptop slept is not acceptable.
    log = RunLog(tmp_path, run_id="killed", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    record = json.loads((tmp_path / "killed.json").read_text(encoding="utf-8"))
    assert record["status"] == "running"
    assert len(record["steps"]) == 1


def test_records_validation_loss_separately(tmp_path):
    log = RunLog(tmp_path, run_id="val", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    log.log_eval(step=1, val_loss=4.9)
    record = json.loads((tmp_path / "val.json").read_text(encoding="utf-8"))
    assert record["evals"] == [{"step": 1, "val_loss": 4.9}]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_runlog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.runlog'`

- [ ] **Step 3: Write the run log**

Create `quipu/runlog.py`:

```python
"""One JSON file per run, flushed on every step.

Flushing every step rather than at the end is deliberate: a run that is killed at
hour 19 must still have its history. The cost is one small write per step, which is
nothing beside a training step.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, out_dir: str | Path, run_id: str, config: dict[str, Any]) -> None:
        self.path = Path(out_dir) / f"{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "config": config,
            "steps": [],
            "evals": [],
        }
        self._flush()

    def log_step(self, step: int, train_loss: float, lr: float, tokens: int, **extra: Any) -> None:
        self.record["steps"].append(
            {"step": step, "train_loss": train_loss, "lr": lr, "tokens": tokens, **extra}
        )
        self._flush()

    def log_eval(self, step: int, val_loss: float) -> None:
        self.record["evals"].append({"step": step, "val_loss": val_loss})
        self._flush()

    def finish(self, status: str) -> None:
        self.record["status"] = status
        self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self.record, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_runlog.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add quipu/runlog.py tests/test_runlog.py
git commit -m "Log each run to JSON, flushed every step

Flushing per step means a run killed at hour 19 still has its history."
```

---

## Task 9: Evaluation

**Files:**
- Create: `quipu/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval.py`:

```python
import numpy as np
import torch

from quipu.config import ModelConfig
from quipu.data import write_shard
from quipu.eval import estimate_loss, generate
from quipu.loader import TokenStream
from quipu.model import Quipu


def tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=32, rope_base=10000.0, norm_eps=1e-6,
    )


def test_estimate_loss_is_near_ln_vocab_at_initialisation(tmp_path):
    # An untrained model is uniform over the vocabulary, so cross-entropy should be
    # about ln(128) = 4.85. A number far from this means the head is mis-initialised.
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    loss = estimate_loss(Quipu(tiny()), stream, batches=5, device="cpu")
    assert 4.0 < loss < 5.6


def test_estimate_loss_leaves_the_model_in_training_mode(tmp_path):
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    model = Quipu(tiny())
    model.train()
    estimate_loss(model, stream, batches=2, device="cpu")
    assert model.training, "estimate_loss must restore training mode"


def test_estimate_loss_does_not_move_the_stream_position(tmp_path):
    # Evaluation must not consume training tokens.
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    stream.load_state_dict({"position": 100})
    estimate_loss(Quipu(tiny()), stream, batches=3, device="cpu")
    assert stream.position == 100


def test_generate_returns_the_requested_number_of_new_tokens():
    model = Quipu(tiny()).eval()
    prompt = torch.randint(0, 128, (1, 4))
    out = generate(model, prompt, max_new_tokens=6, device="cpu")
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


def test_generate_never_exceeds_the_context():
    cfg = tiny()
    model = Quipu(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.context))
    # Asking for more tokens than the context must crop, not crash.
    out = generate(model, prompt, max_new_tokens=5, device="cpu")
    assert out.shape == (1, cfg.context + 5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.eval'`

- [ ] **Step 3: Write the eval module**

Create `quipu/eval.py`:

```python
"""Held-out loss and generation probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from quipu.loader import TokenStream
from quipu.model import Quipu


@torch.no_grad()
def estimate_loss(model: Quipu, stream: TokenStream, batches: int, device: str) -> float:
    """Mean cross-entropy over `batches` batches, leaving the stream where it was.

    Evaluation must not consume training tokens, so the position is saved and
    restored rather than shared.
    """
    was_training = model.training
    saved = stream.state_dict()
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = stream.next_batch()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
    stream.load_state_dict(saved)
    if was_training:
        model.train()
    return total / batches


@torch.no_grad()
def generate(
    model: Quipu,
    idx: torch.Tensor,
    max_new_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = 50,
) -> torch.Tensor:
    """Sample continuations. Used only for eyeballing coherence, never for a metric."""
    model.eval()
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        # The model has no KV cache and RoPE is built for `context` positions, so the
        # window is cropped rather than grown.
        window = idx[:, -model.cfg.context :]
        logits = model(window)[:, -1, :] / temperature
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 5 passed.

- [ ] **Step 5: Mutation-test the position restore**

Delete `stream.load_state_dict(saved)` from `estimate_loss`. Run the tests and confirm
`test_estimate_loss_does_not_move_the_stream_position` FAILS. Restore it.

- [ ] **Step 6: Commit**

```bash
git add quipu/eval.py tests/test_eval.py
git commit -m "Add held-out loss and generation probes

estimate_loss saves and restores the stream position: evaluation must not
consume training tokens, and at 20 batches every 100 steps it silently would."
```

---

## Task 10: Training loop and checkpointing

**Files:**
- Create: `quipu/train.py`
- Test: `tests/test_checkpoint_resume.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint_resume.py`:

```python
import numpy as np
import torch

from quipu.config import ModelConfig, TrainConfig
from quipu.data import write_shard
from quipu.train import Trainer


def tiny_model() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=16, rope_base=10000.0, norm_eps=1e-6,
    )


def tiny_train(tmp_path) -> TrainConfig:
    return TrainConfig(
        total_tokens=16 * 2 * 20, batch_tokens=16 * 2, micro_batch=2, context=16,
        lr=1e-3, lr_min=1e-4, warmup_steps=2, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, seed=7, ckpt_dir=str(tmp_path / "ckpt"), ckpt_every=5,
        eval_every=1000, eval_batches=1,
    )


def make_data(tmp_path):
    write_shard(tmp_path / "shard_000.bin",
                np.random.RandomState(0).randint(0, 128, 8192).astype(np.uint16))
    return tmp_path


def build(tmp_path, data_dir):
    return Trainer(
        model_cfg=tiny_model(), train_cfg=tiny_train(tmp_path),
        shard_dir=data_dir, device="cpu", run_dir=tmp_path / "runs", run_id="t",
    )


def test_loss_decreases_on_repeated_data(tmp_path):
    data = make_data(tmp_path)
    trainer = build(tmp_path, data)
    first = trainer.train_step()
    for _ in range(30):
        last = trainer.train_step()
    assert last < first


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    """Save at step N, reload, and the next step must match what an uninterrupted
    run would have produced. Anything less and a 20-hour run silently diverges."""
    data = make_data(tmp_path)

    a = build(tmp_path, data)
    for _ in range(6):
        a.train_step()
    a.save_checkpoint()
    expected = a.train_step()

    b = build(tmp_path, data)
    b.load_checkpoint()
    got = b.train_step()

    assert abs(expected - got) < 1e-5, f"resume diverged: {expected} vs {got}"


def test_checkpoint_restores_every_piece_of_state(tmp_path):
    data = make_data(tmp_path)
    a = build(tmp_path, data)
    for _ in range(6):
        a.train_step()
    a.save_checkpoint()

    b = build(tmp_path, data)
    b.load_checkpoint()

    assert b.step == a.step
    assert b.stream.position == a.stream.position
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)


def test_learning_rate_warms_up_then_decays(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    cfg = trainer.train_cfg
    assert trainer.lr_at(0) < cfg.lr                 # warming up
    assert abs(trainer.lr_at(cfg.warmup_steps) - cfg.lr) < 1e-9   # peak
    assert abs(trainer.lr_at(cfg.steps) - cfg.lr_min) < 1e-9      # floor
    assert trainer.lr_at(cfg.steps // 2) < cfg.lr                 # decaying
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_checkpoint_resume.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.train'`

- [ ] **Step 3: Write the trainer**

Create `quipu/train.py`:

```python
"""The training loop, gradient accumulation, and a checkpoint that resumes exactly."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from quipu.config import Config, ModelConfig, TrainConfig, load_config
from quipu.eval import estimate_loss
from quipu.loader import TokenStream
from quipu.model import Quipu
from quipu.runlog import RunLog


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        shard_dir: str | Path,
        device: str,
        run_dir: str | Path,
        run_id: str,
        val_dir: str | Path | None = None,
    ) -> None:
        torch.manual_seed(train_cfg.seed)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.step = 0

        self.model = Quipu(model_cfg).to(device)
        self.stream = TokenStream(shard_dir, train_cfg.micro_batch, model_cfg.context)
        self.val_stream = (
            TokenStream(val_dir, train_cfg.micro_batch, model_cfg.context) if val_dir else None
        )

        # Weight decay on matrices only. Norms and embeddings are excluded because
        # decaying them shrinks the residual scale rather than regularising anything.
        decay = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=train_cfg.lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )
        self.log = RunLog(run_dir, run_id, {"model": vars(model_cfg), "train": vars(train_cfg)})

    def lr_at(self, step: int) -> float:
        cfg = self.train_cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if step >= cfg.steps:
            return cfg.lr_min
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine

    def train_step(self) -> float:
        cfg = self.train_cfg
        lr = self.lr_at(self.step)
        for group in self.opt.param_groups:
            group["lr"] = lr

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        use_amp = self.device.startswith("cuda")
        for _ in range(cfg.grad_accum):
            x, y = self.stream.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            # Divide before backward so the accumulated gradient is the mean over the
            # whole batch, not the sum over micro-batches.
            (loss / cfg.grad_accum).backward()
            total += loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step += 1
        self.log.log_step(
            step=self.step, train_loss=total, lr=lr,
            tokens=self.step * cfg.batch_tokens,
        )
        return total

    def save_checkpoint(self) -> Path:
        path = Path(self.train_cfg.ckpt_dir) / f"step_{self.step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "stream": self.stream.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )
        latest = path.parent / "latest.pt"
        torch.save({"path": str(path)}, latest)
        return path

    def load_checkpoint(self, path: str | Path | None = None) -> None:
        if path is None:
            pointer = Path(self.train_cfg.ckpt_dir) / "latest.pt"
            path = torch.load(pointer, weights_only=False)["path"]
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.step = state["step"]
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.stream.load_state_dict(state["stream"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def run(self) -> None:
        cfg = self.train_cfg
        try:
            while self.step < cfg.steps:
                loss = self.train_step()
                if self.step % cfg.eval_every == 0 and self.val_stream is not None:
                    val = estimate_loss(self.model, self.val_stream, cfg.eval_batches, self.device)
                    self.log.log_eval(self.step, val)
                    print(f"step {self.step:>6}  train {loss:.4f}  val {val:.4f}")
                if self.step % cfg.ckpt_every == 0:
                    self.save_checkpoint()
            self.save_checkpoint()
            self.log.finish("completed")
        except KeyboardInterrupt:
            self.save_checkpoint()
            self.log.finish("interrupted")
            raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--run-id", default="quipu-114m-001")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model_cfg=cfg.model, train_cfg=cfg.train,
        shard_dir=Path(cfg.data.shard_dir) / "train",
        val_dir=Path(cfg.data.shard_dir) / "val",
        device=device, run_dir="results/runs", run_id=args.run_id,
    )
    if args.resume:
        trainer.load_checkpoint()
        print(f"resumed at step {trainer.step}, stream position {trainer.stream.position}")
    trainer.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_checkpoint_resume.py -v`
Expected: 4 passed.

- [ ] **Step 5: Mutation-test the resume**

In `load_checkpoint`, delete the `self.stream.load_state_dict(state["stream"])` line. Run the
tests and confirm `test_resume_reproduces_the_uninterrupted_run` FAILS. Restore it.

Then delete `self.opt.load_state_dict(state["optimizer"])` and confirm the same test FAILS again
(Adam's moments matter). Restore it.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add quipu/train.py tests/test_checkpoint_resume.py
git commit -m "Train with gradient accumulation and a checkpoint that resumes exactly

The resume test asserts the step after a reload matches the uninterrupted run.
Dropping either the optimiser moments or the stream position breaks it, which is
what makes it worth having on a 20-hour run."
```

---

## Task 11: Build the shards

The long-running data step. Run it before the throughput measurement so Task 12 has real data.

**Files:**
- Create: `scripts/build_shards.py`

- [ ] **Step 1: Write the shard builder**

Create `scripts/build_shards.py`:

```python
"""Download FineWeb-Edu, tokenize, and write uint16 shards.

Streams rather than downloading the whole set: sample-10BT is far larger than the
2.5B tokens this run needs, and there is no reason to store the remainder.

Run: uv run python scripts/build_shards.py --config configs/quipu-114m.toml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from quipu.config import load_config
from quipu.data import write_shard
from quipu.tokenizer import Tokenizer


def build(out_dir: Path, target_tokens: int, shard_tokens: int, stream, tok: Tokenizer) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    written = 0
    shard = 0
    progress = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc=out_dir.name)

    for row in stream:
        text = row.get("text") or ""
        if not text.strip():
            continue
        buffer.extend(tok.encode(text))
        buffer.append(tok.eot)

        while len(buffer) >= shard_tokens and written < target_tokens:
            chunk = np.array(buffer[:shard_tokens], dtype=np.uint16)
            write_shard(out_dir / f"shard_{shard:03d}.bin", chunk)
            buffer = buffer[shard_tokens:]
            written += len(chunk)
            shard += 1
            progress.update(len(chunk))
        if written >= target_tokens:
            break

    # Flush the tail, so the requested token count is met rather than approached.
    if written < target_tokens and buffer:
        chunk = np.array(buffer[: target_tokens - written], dtype=np.uint16)
        write_shard(out_dir / f"shard_{shard:03d}.bin", chunk)
        written += len(chunk)
        progress.update(len(chunk))

    progress.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tok = Tokenizer()
    root = Path(cfg.data.shard_dir)

    stream = load_dataset(
        cfg.data.dataset, name=cfg.data.subset, split="train", streaming=True
    )

    # Validation first, from the head of the stream, then skipped for training so the
    # two never overlap. An overlapping split makes the val loss meaningless.
    val_written = build(root / "val", cfg.data.val_tokens,
                        min(cfg.data.val_tokens, cfg.data.shard_tokens), stream, tok)
    print(f"val:   {val_written:,} tokens")

    train_written = build(root / "train", cfg.train.total_tokens,
                          cfg.data.shard_tokens, stream, tok)
    print(f"train: {train_written:,} tokens")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Do a small dry run first**

Never start a multi-hour download without proving the path works. Run:

```bash
uv run python -c "
from datasets import load_dataset
s = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
row = next(iter(s))
print(sorted(row.keys()))
print(row['text'][:200])
"
```

Expected: prints the column names (including `text`) and 200 characters of a document.

- [ ] **Step 3: Build the shards**

Run: `uv run python scripts/build_shards.py --config configs/quipu-114m.toml`

Expected: `val: 10,000,000 tokens` then `train: 2,500,000,000 tokens`. This takes a while — the
tokenizer is the bottleneck, not the network. About 5 GB lands in `data/shards/`.

- [ ] **Step 4: Verify what was written**

Run:
```bash
uv run python -c "
from pathlib import Path
from quipu.data import read_shard
for split in ['train', 'val']:
    paths = sorted(Path('data/shards', split).glob('shard_*.bin'))
    total = sum(len(read_shard(p)) for p in paths)
    print(f'{split}: {len(paths)} shards, {total:,} tokens')
"
```
Expected: train ~2.5B, val 10M, and no shard of length zero.

- [ ] **Step 5: Commit the builder (not the shards)**

`data/shards/` is already in `.gitignore`.

```bash
git add scripts/build_shards.py
git commit -m "Build FineWeb-Edu shards by streaming, validation split taken first

The validation tokens come off the head of the stream and training continues from
where that stopped, so the two cannot overlap."
```

---

## Task 12: Measure real throughput and decide the budget

The spec says the 19–24 hour estimate is unverified. This is where it gets verified, with the
real model on the real data — Task 1 measured a bare matmul, which always flatters.

**Files:**
- Create: `scripts/measure_throughput.py`

- [ ] **Step 1: Write the measurement**

Create `scripts/measure_throughput.py`:

```python
"""Time real training steps and project the full run.

Task 1 measured a dense matmul. This measures the actual loop, including the
optimiser, the accumulation and the data, which is the number that matters.

Run: uv run python scripts/measure_throughput.py --micro-batch 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from quipu.config import load_config
from quipu.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides={"train": {"micro_batch": args.micro_batch}})
    trainer = Trainer(
        model_cfg=cfg.model, train_cfg=cfg.train,
        shard_dir=Path(cfg.data.shard_dir) / "train",
        device="cuda", run_dir="results/throughput", run_id=f"mb{args.micro_batch}",
    )
    print(f"micro_batch={cfg.train.micro_batch}  grad_accum={cfg.train.grad_accum}")

    for _ in range(2):            # warm up; the first steps include allocator growth
        trainer.train_step()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.steps):
        trainer.train_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    per_step = elapsed / args.steps
    tps = cfg.train.batch_tokens / per_step
    hours = cfg.train.steps * per_step / 3600
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n{per_step:.2f} s/step   {tps:,.0f} tokens/s")
    print(f"peak VRAM: {peak:.2f} GB of 8.15")
    print(f"projected full run ({cfg.train.steps:,} steps): {hours:.1f} hours")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Find the largest micro-batch that fits**

Run each and record s/step and peak VRAM:

```bash
uv run python scripts/measure_throughput.py --micro-batch 4
uv run python scripts/measure_throughput.py --micro-batch 8
uv run python scripts/measure_throughput.py --micro-batch 16
```

Expected: throughput improves with micro-batch until VRAM runs out, then raises
`torch.OutOfMemoryError`. Pick the largest that fits under about 7.5 GB, leaving headroom for
fragmentation over 20 hours.

- [ ] **Step 3: Write the chosen micro-batch into the config**

Edit `configs/quipu-114m.toml` and set `micro_batch` to the value chosen. `grad_accum` is derived,
so the tokens-per-step figure does not change.

- [ ] **Step 4: The decision gate**

If the projected run exceeds **36 hours**, reduce `total_tokens` so it lands under 30, and record
the new budget in the commit message. Do not start a run you are not prepared to finish.

- [ ] **Step 5: Commit**

```bash
git add scripts/measure_throughput.py configs/quipu-114m.toml
git commit -m "Measure the real training loop and fix the micro-batch

<N> s/step at micro_batch=<M>, <T> tokens/s, peak <V> GB of 8.15, projecting
<H> hours for <K> steps. Task 1's matmul figure was <X> TFLOPS, which as expected
flatters the full loop."
```

---

## Task 13: Results table

Spec success criterion 4: RESULTS.md is generated, never typed.

**Files:**
- Create: `quipu/results_table.py`
- Test: `tests/test_results_table.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_results_table.py`:

```python
import json

from quipu.results_table import build_table


def write_run(tmp_path, run_id, final_train, final_val, status="completed"):
    record = {
        "run_id": run_id,
        "status": status,
        "config": {"model": {"d_model": 768, "n_layer": 12}, "train": {"total_tokens": 2_500_000_000}},
        "steps": [{"step": 1, "train_loss": 10.0, "lr": 1e-4, "tokens": 1},
                  {"step": 2, "train_loss": final_train, "lr": 1e-4, "tokens": 2}],
        "evals": [{"step": 2, "val_loss": final_val}],
    }
    (tmp_path / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")


def test_table_has_one_row_per_run(tmp_path):
    write_run(tmp_path, "run-a", 3.9, 4.0)
    write_run(tmp_path, "run-b", 3.5, 3.6)
    table = build_table(tmp_path)
    assert "run-a" in table and "run-b" in table
    assert table.count("\n|") >= 3          # header, separator, two rows


def test_table_reports_the_final_losses(tmp_path):
    write_run(tmp_path, "run-a", 3.9123, 4.0456)
    table = build_table(tmp_path)
    assert "3.9123" in table and "4.0456" in table


def test_unfinished_runs_are_marked_not_hidden(tmp_path):
    # A run that died at hour 19 is still evidence and must not vanish from the table.
    write_run(tmp_path, "killed", 5.0, 5.1, status="interrupted")
    assert "interrupted" in build_table(tmp_path)


def test_a_run_with_no_evals_does_not_crash_the_table(tmp_path):
    record = {"run_id": "noeval", "status": "running", "config": {},
              "steps": [{"step": 1, "train_loss": 9.0, "lr": 1e-4, "tokens": 1}], "evals": []}
    (tmp_path / "noeval.json").write_text(json.dumps(record), encoding="utf-8")
    assert "noeval" in build_table(tmp_path)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_results_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.results_table'`

- [ ] **Step 3: Write the generator**

Create `quipu/results_table.py`:

```python
"""Generate RESULTS.md from run records. Nothing here is ever typed by hand.

Run: uv run python -m quipu.results_table results/runs > RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build_table(run_dir: str | Path) -> str:
    rows = []
    for path in sorted(Path(run_dir).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        steps = record.get("steps") or []
        evals = record.get("evals") or []
        rows.append(
            "| {run} | {status} | {n} | {train} | {val} |".format(
                run=record.get("run_id", path.stem),
                status=record.get("status", "?"),
                n=f"{steps[-1]['step']:,}" if steps else "0",
                train=f"{steps[-1]['train_loss']:.4f}" if steps else "-",
                val=f"{evals[-1]['val_loss']:.4f}" if evals else "-",
            )
        )
    header = (
        "| Run | Status | Steps | Final train loss | Final val loss |\n"
        "|---|---|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows) + "\n"


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/runs"
    print("# Quipu results\n")
    print("Generated by `quipu/results_table.py`. Do not edit by hand.\n")
    print(build_table(run_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_results_table.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quipu/results_table.py tests/test_results_table.py
git commit -m "Generate RESULTS.md from run records

Interrupted runs get a row rather than disappearing: a run that died at hour 19
is still evidence."
```

---

## Task 14: The run

**Files:**
- Modify: `RESULTS.md` (generated)
- Create: `README.md`

- [ ] **Step 1: Run the whole suite one last time**

Run: `uv run pytest -q`
Expected: everything passes. Do not start a 20-hour run on a red suite.

- [ ] **Step 2: Start training**

Run: `uv run python -m quipu.train --config configs/quipu-114m.toml --run-id quipu-114m-001`

Expected: a line every 100 steps with train and val loss. Loss starts near **ln(50257) ≈ 10.82**
and should be under 7 within a few hundred steps. If it is flat after 500 steps, stop: something
is wrong with the data or the learning rate, and 20 hours will not fix it.

- [ ] **Step 3: Prove the resume works on the real run**

After at least 200 steps, press Ctrl+C. Then run:

`uv run python -m quipu.train --config configs/quipu-114m.toml --run-id quipu-114m-001 --resume`

Expected: prints the resumed step and stream position, and the loss continues from where it was
rather than jumping. **This is spec success criterion 1 and it is proved once, deliberately, on
the real run — not assumed from the unit test.**

- [ ] **Step 4: Let it finish**

~4,768 steps. Check in occasionally; the run log is on disk after every step, so nothing is lost
if the machine sleeps.

- [ ] **Step 5: Generate the results table**

Run: `uv run python -m quipu.results_table results/runs > RESULTS.md`

- [ ] **Step 6: Eyeball the generation probes**

Run:
```bash
uv run python -c "
import torch
from quipu.config import load_config
from quipu.eval import generate
from quipu.model import Quipu
from quipu.tokenizer import Tokenizer

cfg = load_config('configs/quipu-114m.toml')
tok = Tokenizer()
model = Quipu(cfg.model).cuda()
state = torch.load('checkpoints/latest.pt', weights_only=False)
model.load_state_dict(torch.load(state['path'], weights_only=False)['model'])
for prompt in ['The capital of France is', 'Water boils at', 'The first step is to']:
    ids = torch.tensor([tok.encode(prompt)])
    out = generate(model, ids, max_new_tokens=40, device='cuda')
    print(repr(tok.decode(out[0].tolist())), '\n')
"
```

Expected: locally coherent English. **That is the whole bar** — not factuality, not instruction
following. A 114M model trained on 2.5B tokens will say false things confidently, and the spec
says so.

- [ ] **Step 7: Write the README with the measured numbers**

Create `README.md` recording: the final train and val loss from RESULTS.md, the measured
tokens/second and wall-clock from Task 12, peak VRAM, and an honest statement of what the model
cannot do. Quote no figure that is not in `results/`.

- [ ] **Step 8: Commit**

```bash
git add README.md RESULTS.md results/runs
git commit -m "Train quipu-114m: <N> steps, final val loss <V>, <H> hours on one RTX 5060

The pipeline is validated: resume was exercised on the real run and the loss
continued without a discontinuity. The model itself is a validation artifact."
```

---

## Self-review

**Spec coverage.** Every section maps to a task: §3 decision 1 (FineWeb only) → Task 11; decision 2
(GPT-2 BPE) → Task 3; decision 3 (Windows) → environment facts; decisions 4–5 (tied, GQA) → Task 6
and its tests. §4 architecture → Tasks 5–6. §5 data → Tasks 4, 11. §6 training → Task 10; the
throughput caveat → Task 12. §7 components → all files present, plus `loader.py` which the plan
declares as an addition. §8 tests 1–4 → Tasks 4, 6, 6, 10 respectively. §9 criteria 1–5 → Task 14
steps 3, 2, 2, 5, 6. §10 risks → Task 1 (Blackwell), Task 12 (thermals, OOM), config (`torch.compile`
is never enabled).

**Gap found and closed.** The spec's §10 lists `torch.compile` as a risk handled by "off by
default". No task enables it and no code references it, so the risk is handled by omission. That
is correct, and stated here so a reader does not go looking for it.

**Type consistency.** `ModelConfig`/`TrainConfig`/`DataConfig` field names are used identically in
Tasks 2, 6, 9, 10 and 12. `TokenStream.state_dict()`/`load_state_dict()` match between Tasks 7, 9
and 10. `RunLog.log_step`/`log_eval`/`finish` match between Tasks 8 and 10. `estimate_loss` and
`generate` signatures match between Tasks 9 and 14.

**One known soft spot.** `test_resume_reproduces_the_uninterrupted_run` uses a 1e-5 tolerance
rather than exact equality, because cuBLAS reductions are not bit-deterministic across process
boundaries. On CPU, where that test runs, it would likely pass at exact equality; the tolerance is
there so the same test can later be pointed at CUDA without being rewritten.
