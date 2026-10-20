# Low-Memory Training on Kaggle 2x T4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the existing data-parallel trainer continue pretraining quipu-moe-1B-A149M on Kaggle's 2x T4 (16 GB, sm_75, no bf16) by adding fp16 + loss scaling, bf16 Muon momentum, block activation checkpointing and a chunked loss, all off by default.

**Architecture:** Four independent config switches in `TrainConfig` (`precision = "fp16"`, `optimizer_state_dtype`, `activation_checkpoint`, `loss_chunk_tokens`), each implemented in its own small unit (`quipu/lossscale.py`, `quipu/optim/muon.py`, the models' forward, `quipu/lossfn.py`) and wired into `quipu/train.py`'s `Trainer.train_step`. A device guard refuses settings a GPU can't run. A Kaggle config layer and a probe script close it out.

**Tech Stack:** Python 3.12, PyTorch 2.11 (`torch.utils.checkpoint`, `torch.autocast`), pytest. Spec: `docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md`.

**Repo rules for every commit:** stage files by explicit path; commit message with NO trailers (no Co-Authored-By); never stage `quipu/model_card.py` (owner's uncommitted work); never push. Run tests with `python -m pytest` from the repo root (`C:\Users\aneek.chattopadhyay\Desktop\Other Projects\quipu`, venv `.venv`).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `quipu/config.py` | modify | new TrainConfig fields + validation |
| `quipu/fp8.py` | modify | `apply_precision` passes `"fp16"` through |
| `quipu/lossscale.py` | create | `LossScaler`: dynamic fp16 loss scale |
| `quipu/optim/muon.py` | modify | `state_dtype` group option, `cast_state_()`, NS work dtype by GPU |
| `quipu/optim/groups.py` | modify | pass `state_dtype` into Muon groups; `apply_config_lrs` re-sets it |
| `quipu/lossfn.py` | create | `chunked_cross_entropy` |
| `quipu/model.py`, `quipu/model_moe.py` | modify | `return_hidden=` and `checkpoint_blocks` |
| `quipu/eval.py` | modify | `amp_dtype=` on `estimate_loss`, `mean_loss` |
| `quipu/train.py` | modify | wire everything; device guard; checkpoint the scaler |
| `configs/quipu-moe-cpt-t4.toml` | create | Kaggle 2x T4 layer over `quipu-moe-cpt.toml` |
| `scripts/lowmem_probe.py` | create | GPU parity and memory probe (manual runs) |
| `docs/superpowers/specs/2026-10-20-kaggle-t4-runbook.md` | create | how to run a Kaggle session |
| `tests/test_lowmem.py` | create | all new tests |

---

### Task 1: Config fields and validation

**Files:**
- Modify: `quipu/config.py` (constants near line 120, `TrainConfig` after `decay_shape`, `TRAIN_ZERO_INTS`, `_check_train_extras`, cross-check after `_check_train_extras(train)` near line 659)
- Modify: `quipu/fp8.py:105-116`
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lowmem.py`:

```python
"""Low-memory training on T4 (spec docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md)."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


def smoke_cfg(tmp_path, model=None, **train):
    over = {"data": {"shard_dir": str(tmp_path / "shards")},
            "train": {"ckpt_dir": str(tmp_path / "ckpt"), **train}}
    if model:
        over["model"] = model
    return load_config(SMOKE, over)


# ---- config ----------------------------------------------------------------------

def test_new_train_fields_default_to_todays_behaviour(tmp_path):
    t = smoke_cfg(tmp_path).train
    assert t.precision == "bf16"
    assert t.optimizer_state_dtype == "fp32"
    assert t.activation_checkpoint == "none"
    assert t.loss_chunk_tokens == 0


def test_t4_settings_load(tmp_path):
    t = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16",
                  optimizer_state_dtype="bf16", activation_checkpoint="block",
                  loss_chunk_tokens=300).train
    assert (t.precision, t.optimizer_state_dtype, t.activation_checkpoint,
            t.loss_chunk_tokens) == ("fp16", "bf16", "block", 300)


@pytest.mark.parametrize("field,value", [
    ("precision", "fp32"), ("optimizer_state_dtype", "fp16"),
    ("activation_checkpoint", "layer"), ("loss_chunk_tokens", -1),
])
def test_bad_values_are_refused(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        smoke_cfg(tmp_path, model={"attnres_blocks": 0}, **{field: value})


def test_block_checkpointing_refuses_attnres(tmp_path):
    # The smoke config has attnres_blocks = 2; AttnRes has its own checkpoint option.
    with pytest.raises(ValueError, match="activation_checkpoint"):
        smoke_cfg(tmp_path, activation_checkpoint="block")


def test_apply_precision_leaves_the_model_alone_for_fp16():
    from quipu.fp8 import apply_precision
    m = torch.nn.Linear(4, 4)
    assert apply_precision(m, "fp16", "cpu") is m
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_lowmem.py -v`
Expected: FAIL (`TrainConfig` has no field `optimizer_state_dtype`; `precision 'fp16'` refused).

- [ ] **Step 3: Implement**

In `quipu/config.py`, change the constants block:

```python
PRECISIONS = ("bf16", "fp8", "fp16")
OPTIMIZER_STATE_DTYPES = ("fp32", "bf16")
ACTIVATION_CHECKPOINTS = ("none", "block")
```

and `TRAIN_ZERO_INTS`:

```python
TRAIN_ZERO_INTS = frozenset({"seed", "decay_start_step", "decay_steps", "loss_chunk_tokens"})
```

In `TrainConfig`, update the `precision` comment to list `"fp16"` and append after `decay_shape`:

```python
    # Low memory (spec 2026-10-20-t4-low-memory-training-design.md), for GPUs such as
    # Kaggle's T4. precision "fp16" (above) is fp16 autocast with a dynamic loss
    # scale (quipu.lossscale), for GPUs without bf16. optimizer_state_dtype "bf16"
    # stores Muon's momentum in bf16 (math stays fp32). activation_checkpoint "block"
    # recomputes each block's activations in backward. loss_chunk_tokens > 0 computes
    # lm_head + cross-entropy that many tokens at a time (quipu.lossfn), so the full
    # logits never exist at once; 0 = off.
    optimizer_state_dtype: str = "fp32"  # "fp32" | "bf16"
    activation_checkpoint: str = "none"  # "none" | "block"
    loss_chunk_tokens: int = 0
```

In `_check_train_extras`, after the `precision` check:

```python
    _check_choice("optimizer_state_dtype", train.optimizer_state_dtype, OPTIMIZER_STATE_DTYPES)
    _check_choice("activation_checkpoint", train.activation_checkpoint, ACTIVATION_CHECKPOINTS)
```

(`loss_chunk_tokens < 0` is already refused by `_validate_int_fields`; check the message contains the field name, and if it does not, add `if train.loss_chunk_tokens < 0: raise ValueError(f"loss_chunk_tokens must be >= 0, got {train.loss_chunk_tokens!r}")`.)

After the existing `if train.precision == "fp8" and model.kind != "moe":` block in `load_config`:

```python
    if train.activation_checkpoint == "block" and model.attnres_blocks:
        raise ValueError("activation_checkpoint 'block' does not cover Block AttnRes "
                         "(attnres_blocks > 0); use model.attnres_checkpoint there")
```

In `quipu/fp8.py` `apply_precision`, change `if precision == "bf16":` to:

```python
    if precision in ("bf16", "fp16"):
        return model
```

and its docstring's first line to `"""The model for train.precision: "bf16" and "fp16" return it untouched, ...`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py tests/test_config.py tests/test_fp8.py -v`
Expected: all PASS. If a test in `test_config.py` pins `PRECISIONS == ("bf16", "fp8")` or the error text listing choices, update that assertion to include `"fp16"`.

- [ ] **Step 5: Commit**

```bash
git add quipu/config.py quipu/fp8.py tests/test_lowmem.py
git commit -m "config: fp16 precision, optimizer_state_dtype, activation_checkpoint, loss_chunk_tokens"
```

---

### Task 2: LossScaler

**Files:**
- Create: `quipu/lossscale.py`
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_lowmem.py`)

```python
# ---- loss scaler -------------------------------------------------------------------

from quipu.lossscale import LossScaler


def test_disabled_scaler_is_identity():
    s = LossScaler(enabled=False)
    assert s.scale == 1.0
    p = torch.nn.Parameter(torch.ones(3)); p.grad = torch.full((3,), 4.0)
    s.unscale_([p])
    assert torch.equal(p.grad, torch.full((3,), 4.0))


def test_unscale_divides_grads():
    s = LossScaler(enabled=True, init_scale=8.0)
    p = torch.nn.Parameter(torch.ones(3)); p.grad = torch.full((3,), 4.0)
    q = torch.nn.Parameter(torch.ones(2))       # no grad: skipped
    s.unscale_([p, q])
    assert torch.equal(p.grad, torch.full((3,), 0.5)) and q.grad is None


def test_overflow_halves_and_resets_the_clean_count():
    s = LossScaler(enabled=True, init_scale=1024.0, growth_interval=3)
    s.good_step(); s.good_step()
    s.overflow()
    assert s.scale == 512.0 and s.clean_steps == 0


def test_growth_after_interval():
    s = LossScaler(enabled=True, init_scale=2.0, growth_interval=3)
    for _ in range(3):
        s.good_step()
    assert s.scale == 4.0 and s.clean_steps == 0


def test_scale_never_goes_below_the_floor():
    s = LossScaler(enabled=True, init_scale=2.0, floor=1.0)
    assert not s.at_floor
    s.overflow()
    assert s.scale == 1.0 and s.at_floor
    s.overflow()
    assert s.scale == 1.0


def test_state_round_trips():
    s = LossScaler(enabled=True, init_scale=64.0, growth_interval=5)
    s.overflow(); s.good_step()
    t = LossScaler(enabled=True)
    t.load_state_dict(s.state_dict())
    assert (t.scale, t.clean_steps) == (32.0, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k "scaler or unscale or overflow or growth or floor or round_trips" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.lossscale'`.

- [ ] **Step 3: Implement** `quipu/lossscale.py`

```python
"""Dynamic loss scale for fp16 training (train.precision "fp16").

fp16 underflows below ~6e-8, where many gradients live, so each micro-batch's loss is
multiplied by `scale` before backward and the gradients divided by it afterwards
(unscale_, after the all-reduce, before clipping). A step whose gradients are not
finite is an overflow: the trainer skips it and calls overflow(), which halves the
scale; after growth_interval clean steps in a row good_step() doubles it. The scale
never drops below `floor`; an overflow at the floor is a real non-finite step and
counts toward the trainer's poison streak (at_floor).

Not torch.amp.GradScaler: the trainer reduces gradients at one known point, has its
own non-finite skip, and steps two optimizers (Muon + AdamW).
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

INIT_SCALE = 2.0 ** 16
GROWTH_INTERVAL = 2000
FLOOR = 1.0


class LossScaler:
    def __init__(self, enabled: bool, init_scale: float = INIT_SCALE,
                 growth_interval: int = GROWTH_INTERVAL, floor: float = FLOOR) -> None:
        self.enabled = enabled
        self.scale = float(init_scale) if enabled else 1.0
        self.growth_interval = growth_interval
        self.floor = floor
        self.clean_steps = 0

    @property
    def at_floor(self) -> bool:
        return self.scale <= self.floor

    @torch.no_grad()
    def unscale_(self, params: Iterable[torch.nn.Parameter]) -> None:
        if not self.enabled or self.scale == 1.0:
            return
        inv = 1.0 / self.scale
        for p in params:
            if p.grad is not None:
                p.grad.mul_(inv)

    def overflow(self) -> None:
        if self.enabled:
            self.scale = max(self.floor, self.scale / 2)
            self.clean_steps = 0

    def good_step(self) -> None:
        if not self.enabled:
            return
        self.clean_steps += 1
        if self.clean_steps >= self.growth_interval:
            self.scale *= 2
            self.clean_steps = 0

    def state_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "clean_steps": self.clean_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.enabled:
            self.scale = float(state["scale"])
            self.clean_steps = int(state["clean_steps"])
```

- [ ] **Step 4: Run tests** — same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quipu/lossscale.py tests/test_lowmem.py
git commit -m "lossscale: dynamic loss scale for fp16 training"
```

---

### Task 3: Muon momentum dtype and the Newton-Schulz work dtype

**Files:**
- Modify: `quipu/optim/muon.py` (`newton_schulz`, `Muon.__init__`, `Muon.step`, new `Muon.cast_state_`)
- Modify: `quipu/optim/groups.py` (`muon_common`, `apply_config_lrs`)
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- Muon ----------------------------------------------------------------------------

from quipu.optim import Muon, newton_schulz
from quipu.optim.muon import cuda_work_dtype


def test_cuda_work_dtype_by_capability():
    assert cuda_work_dtype((7, 5)) is torch.float16      # T4
    assert cuda_work_dtype((8, 0)) is torch.bfloat16     # A100
    assert cuda_work_dtype((12, 0)) is torch.bfloat16    # RTX 50xx


def test_newton_schulz_in_fp16_matches_fp32():
    g = torch.randn(48, 80, generator=torch.Generator().manual_seed(3))
    ref = newton_schulz(g, work_dtype=torch.float32)
    out = newton_schulz(g, work_dtype=torch.float16)
    assert out.dtype == g.dtype
    assert (out - ref).norm() / ref.norm() < 0.02


def _muon_run(state_dtype, steps=20):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(32, 48))
    opt = Muon([{"params": [w], "state_dtype": state_dtype}], lr=0.02)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        w.grad = torch.randn(32, 48, generator=gen)
        opt.step()
    return w.detach().clone(), opt


def test_bf16_momentum_tracks_fp32():
    w32, _ = _muon_run("fp32")
    w16, opt = _muon_run("bf16")
    buf = next(iter(opt.state.values()))["momentum_buffer"]
    assert buf.dtype is torch.bfloat16
    assert (w16 - w32).norm() / w32.norm() < 0.01


def test_cast_state_converts_loaded_momentum():
    _, opt = _muon_run("fp32", steps=2)
    for g in opt.param_groups:
        g["state_dtype"] = "bf16"
    opt.cast_state_()
    buf = next(iter(opt.state.values()))["momentum_buffer"]
    assert buf.dtype is torch.bfloat16


def test_build_optimizers_passes_state_dtype(tmp_path):
    from quipu.model_factory import build_model
    from quipu.optim import build_optimizers
    cfg = smoke_cfg(tmp_path, optimizer_state_dtype="bf16")
    opts = build_optimizers(build_model(cfg.model), cfg.train)
    muon = [o for o in opts if isinstance(o, Muon)][0]
    assert {g["state_dtype"] for g in muon.param_groups} == {"bf16"}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k "work_dtype or newton or momentum or cast_state or state_dtype" -v`
Expected: FAIL (`cannot import name 'cuda_work_dtype'`).

- [ ] **Step 3: Implement**

In `quipu/optim/muon.py`, above `newton_schulz`:

```python
STATE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def cuda_work_dtype(capability: tuple[int, int]) -> torch.dtype:
    """Newton-Schulz's matmul dtype on a CUDA device of this compute capability: bf16
    from sm_80 (A100 on), fp16 below (T4, sm_75, has no bf16 tensor cores). fp16 is
    safe here: X is divided by its Frobenius norm first, so entries stay in [-1, 1]."""
    return torch.bfloat16 if capability >= (8, 0) else torch.float16
```

Change `newton_schulz`'s signature and its `work = ...` line:

```python
def newton_schulz(g: Tensor, steps: int = 5, work_dtype: torch.dtype | None = None) -> Tensor:
```

```python
    if work_dtype is not None:
        work = work_dtype
    elif g.is_cuda:
        work = cuda_work_dtype(torch.cuda.get_device_capability(g.device))
    else:
        work = torch.bfloat16 if g.dtype == torch.bfloat16 else torch.float32
```

Update the module docstring line "On CUDA the iteration runs in bf16 whatever the input dtype" to "On CUDA the iteration runs in bf16 (fp16 below sm_80, see cuda_work_dtype)". Update the `newton_schulz` docstring's last sentence the same way.

In `Muon.__init__`, add a parameter `state_dtype: str = "fp32"` and put `state_dtype=state_dtype` into `defaults`. In `_check_group`, add:

```python
        if group["state_dtype"] not in STATE_DTYPES:
            raise ValueError(f"state_dtype must be one of {list(STATE_DTYPES)}, "
                             f"got {group['state_dtype']!r}")
```

Replace the per-parameter body of `Muon.step` (from `state = self.state[p]` to `p.add_(...)`) with:

```python
                state = self.state[p]
                dt = STATE_DTYPES[group["state_dtype"]]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=dt)
                buf = state["momentum_buffer"]
                if buf.dtype != dt:
                    buf = state["momentum_buffer"] = buf.to(dt)
                if dt == g.dtype:
                    buf.lerp_(g, 1 - mom)
                    m = buf
                else:
                    # Stored low, updated in the gradient's dtype (fp32).
                    m = buf.to(g.dtype).lerp_(g, 1 - mom)
                    buf.copy_(m)
                direction = g.lerp(m, mom) if group["nesterov"] else m
                update = orthogonalize(direction, group["ns_steps"], group["head_dim"],
                                       group["in_out"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
```

Add a method to `Muon`:

```python
    @torch.no_grad()
    def cast_state_(self) -> None:
        """Every momentum buffer into its group's state_dtype. load_state_dict casts
        state to the parameter's dtype (fp32), so the trainer calls this right after
        loading a checkpoint: the fp32 copy is dropped before any gradient exists."""
        for group in self.param_groups:
            dt = STATE_DTYPES[group["state_dtype"]]
            for p in group["params"]:
                st = self.state.get(p)
                if st and "momentum_buffer" in st and st["momentum_buffer"].dtype != dt:
                    st["momentum_buffer"] = st["momentum_buffer"].to(dt)
```

In `quipu/optim/groups.py`, add `"state_dtype": cfg.optimizer_state_dtype` to `muon_common`, and in `apply_config_lrs`'s `if is_muon:` branch add `group["state_dtype"] = cfg.optimizer_state_dtype`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py tests/test_muon.py -v`
Expected: all PASS. If a `test_muon.py` test compares a group dict exactly, add `"state_dtype": "fp32"` to its expected dict.

- [ ] **Step 5: Commit**

```bash
git add quipu/optim/muon.py quipu/optim/groups.py tests/test_lowmem.py
git commit -m "muon: optional bf16 momentum; fp16 Newton-Schulz below sm_80"
```

---

### Task 4: Chunked loss and `return_hidden`

**Files:**
- Create: `quipu/lossfn.py`
- Modify: `quipu/model.py:149-157` (`Quipu.forward`), `quipu/model_moe.py:112-124` (`QuipuMoE.forward`)
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- chunked loss --------------------------------------------------------------------

import torch.nn.functional as F
from quipu.lossfn import chunked_cross_entropy


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_chunked_ce_equals_full_ce_and_grads(chunk):
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(2, 30, 16, generator=gen, requires_grad=True)
    w = torch.randn(50, 16, generator=gen, requires_grad=True)
    y = torch.randint(50, (2, 30), generator=gen)
    y[0, :5] = -100                                   # ignored targets (SFT)
    full = F.cross_entropy((h @ w.T).view(-1, 50), y.view(-1), ignore_index=-100,
                           reduction="sum")
    gh, gw = torch.autograd.grad(full, (h, w))
    out = chunked_cross_entropy(h, w, y, chunk, ignore_index=-100)
    ch, cw = torch.autograd.grad(out, (h, w))
    torch.testing.assert_close(out, full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ch, gh, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cw, gw, rtol=1e-5, atol=1e-6)


def test_models_return_hidden_matching_logits(tmp_path):
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    torch.manual_seed(0)
    m = build_model(cfg.model).eval()
    x = torch.randint(cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        logits = m(x)
        h = m(x, return_hidden=True)
    torch.testing.assert_close(m.lm_head(h), logits)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k "chunked or return_hidden" -v`
Expected: FAIL (`No module named 'quipu.lossfn'`).

- [ ] **Step 3: Implement**

Create `quipu/lossfn.py`:

```python
"""Cross-entropy over the vocabulary a chunk of tokens at a time (train.loss_chunk_tokens).

The full logits of one micro-batch are tokens x vocab (2,048 x 49,152 for quipu-moe,
~400 MB in fp32 per sequence, plus their gradient). Here each chunk's lm_head matmul
and cross-entropy run under torch.utils.checkpoint, so only one chunk's logits exist at
a time, in forward and in backward. Returns the SUM over non-ignored targets; the
caller divides (by the token count for pretraining, by the step's target count N for
the SFT), so the result equals F.cross_entropy(..., reduction="sum") up to float order.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_sum(h: torch.Tensor, weight: torch.Tensor, y: torch.Tensor,
               ignore_index: int) -> torch.Tensor:
    logits = F.linear(h, weight).float()
    return F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="sum")


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                          chunk: int, ignore_index: int = -100) -> torch.Tensor:
    """Summed cross-entropy of F.linear(hidden, weight) against targets. hidden
    [..., d], targets [...] (same leading shape), weight [vocab, d]."""
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk!r}")
    h = hidden.reshape(-1, hidden.size(-1))
    y = targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, h.size(0), chunk):
        total = total + checkpoint(_chunk_sum, h[s:s + chunk], weight, y[s:s + chunk],
                                   ignore_index, use_reentrant=False)
    return total
```

In `quipu/model.py`, `Quipu.forward`:

```python
    def forward(self, idx: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
```

and its last line:

```python
        h = self.norm(x)
        return h if return_hidden else self.lm_head(h)
```

Same two changes in `quipu/model_moe.py` `QuipuMoE.forward`. Add to each forward's docstring (or as a comment above the return): `# return_hidden: the final normed hidden states, for quipu.lossfn's chunked loss.`

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py tests/test_model_moe.py tests/test_model_shapes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quipu/lossfn.py quipu/model.py quipu/model_moe.py tests/test_lowmem.py
git commit -m "lossfn: chunked cross-entropy; models can return final hidden states"
```

---

### Task 5: Block activation checkpointing

**Files:**
- Modify: `quipu/model.py` (`Quipu.__init__`, `Quipu.forward`), `quipu/model_moe.py` (`QuipuMoE.__init__`, `QuipuMoE.forward`)
- Test: `tests/test_lowmem.py`

The MoE layer writes `self._last_scores` during forward (moe.py `MoELayer.forward`) and the model writes `self.last_stats[l]` from the block's return value. Under non-reentrant checkpointing the recompute runs `block(x, cos, sin)` again inside backward: it overwrites `_last_scores` with the same values (same input, same balancer bias: the bias only moves after the optimizer step) and its returned stats are discarded, because the `self.last_stats[l] = ...` assignment happens only in the original forward. The test below pins this.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- activation checkpointing --------------------------------------------------------

def _grads_after_backward(model, x):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).view(-1, model.cfg.vocab_size), x.view(-1))
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                           if p.grad is not None}


def test_moe_block_checkpointing_changes_nothing(tmp_path):
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    torch.manual_seed(0)
    plain = build_model(cfg.model).train()
    ckpt = build_model(cfg.model).train()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.checkpoint_blocks = True
    x = torch.randint(cfg.model.vocab_size, (2, 32), generator=torch.Generator().manual_seed(1))
    l1, g1 = _grads_after_backward(plain, x)
    l2, g2 = _grads_after_backward(ckpt, x)
    torch.testing.assert_close(l2, l1)
    assert g1.keys() == g2.keys()
    for n in g1:
        torch.testing.assert_close(g2[n], g1[n], rtol=1e-5, atol=1e-7, msg=n)
    for b1, b2 in zip(plain.blocks, ckpt.blocks):
        torch.testing.assert_close(b2.moe._last_scores, b1.moe._last_scores)
    for s1, s2 in zip(plain.last_stats, ckpt.last_stats):
        assert torch.equal(s1.counts, s2.counts) and torch.equal(s1.dropped, s2.dropped)


def test_dense_block_checkpointing_changes_nothing():
    from quipu.config import load_config
    from quipu.model import Quipu
    mc = dataclasses.replace(load_config(ROOT / "configs" / "quipu-114m.toml").model,
                             vocab_size=256, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                             ffn_hidden=128, context=32)
    torch.manual_seed(0)
    plain, ckpt = Quipu(mc).train(), Quipu(mc).train()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.checkpoint_blocks = True
    x = torch.randint(256, (2, 32), generator=torch.Generator().manual_seed(1))
    l1, g1 = _grads_after_backward(plain, x)
    l2, g2 = _grads_after_backward(ckpt, x)
    torch.testing.assert_close(l2, l1)
    for n in g1:
        torch.testing.assert_close(g2[n], g1[n], rtol=1e-5, atol=1e-7, msg=n)


def test_checkpointing_is_off_in_eval_and_no_grad(tmp_path, monkeypatch):
    import quipu.model_moe as mm
    from quipu.model_factory import build_model
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    m = build_model(cfg.model)
    m.checkpoint_blocks = True
    calls = []
    monkeypatch.setattr(mm, "checkpoint", lambda *a, **k: calls.append(1) or a[0](*a[1:]))
    x = torch.randint(cfg.model.vocab_size, (1, 8))
    with torch.no_grad():
        m.train()(x)
    m.eval()(x)
    assert calls == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k checkpointing -v`
Expected: FAIL (the models have no `checkpoint_blocks` effect / `mm.checkpoint` missing).

- [ ] **Step 3: Implement**

In `quipu/model_moe.py`, add `from torch.utils.checkpoint import checkpoint` to the imports. In `QuipuMoE.__init__`, after `self.last_stats = ...`:

```python
        # train.activation_checkpoint "block" (set by the trainer): each block's
        # activations are recomputed in backward (training with grad only). The
        # recompute re-runs the router on the same input, so it leaves _last_scores as
        # the first pass did, and its stats are discarded (last_stats is assigned
        # here, from the first pass only).
        self.checkpoint_blocks = False
```

Replace the plain loop in `QuipuMoE.forward`:

```python
        if self.attnres is None:
            ckpt = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
            for l, block in enumerate(self.blocks):
                if ckpt:
                    x, self.last_stats[l] = checkpoint(block, x, cos, sin, use_reentrant=False)
                else:
                    x, self.last_stats[l] = block(x, cos, sin)
```

In `quipu/model.py`, add the same import; in `Quipu.__init__` add `self.checkpoint_blocks = False` (comment: `# train.activation_checkpoint "block": recompute each block in backward.`); replace its loop:

```python
        ckpt = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            x = checkpoint(block, x, cos, sin, use_reentrant=False) if ckpt else block(x, cos, sin)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py tests/test_model_moe.py tests/test_moe.py tests/test_model_shapes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quipu/model.py quipu/model_moe.py tests/test_lowmem.py
git commit -m "model: optional per-block activation checkpointing"
```

---

### Task 6: Eval follows the training autocast dtype

**Files:**
- Modify: `quipu/eval.py` (`estimate_loss`, `mean_loss`)
- Modify: `quipu/train.py` (`evaluate_by_source` signature and its autocast)
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing test** (append)

```python
# ---- eval dtype ----------------------------------------------------------------------

def test_eval_functions_take_amp_dtype():
    import inspect
    from quipu.eval import estimate_loss, mean_loss
    from quipu.train import evaluate_by_source
    for fn in (estimate_loss, mean_loss, evaluate_by_source):
        p = inspect.signature(fn).parameters["amp_dtype"]
        assert p.default is torch.bfloat16
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k amp_dtype -v`
Expected: FAIL with `KeyError: 'amp_dtype'`.

- [ ] **Step 3: Implement**

`estimate_loss`: signature becomes

```python
def estimate_loss(
    model: nn.Module, stream: TokenStream, batches: int, device: str, amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> float:
```

and `dtype=torch.bfloat16` in its autocast becomes `dtype=amp_dtype`. Add to its docstring: `amp_dtype is the training autocast's (fp16 under train.precision "fp16").` Same for `mean_loss` (add `amp_dtype: torch.dtype = torch.bfloat16,` after `amp`). In `quipu/train.py`, `evaluate_by_source(model, root, context, device, amp_dtype: torch.dtype = torch.bfloat16)` and its autocast `dtype=amp_dtype`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py tests/test_eval.py tests/test_eval_mix.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quipu/eval.py quipu/train.py tests/test_lowmem.py
git commit -m "eval: autocast dtype follows the training precision"
```

---

### Task 7: Wire it into the trainer

**Files:**
- Modify: `quipu/train.py`: `Trainer.__init__` (after `self.model = apply_precision(...)`, near line 579), `_trial_compiled_step` (line ~798), `train_step` (lines ~893-990), `_write_checkpoint` (~1119), `load_checkpoint` (~1220), `_load_optimizers` (~1286), eval calls (`estimate_loss` ~1403, `mean_loss` ~756, `evaluate_by_source` ~1497), module docstring
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- trainer ---------------------------------------------------------------------------

from quipu.data import write_shard
from quipu.train import Trainer, UsageError, check_device_support


def make_data(tmp_path, tokens=6000, seed=0):
    d = tmp_path / "data"
    write_shard(d / "shard_000.bin",
                np.random.RandomState(seed).randint(0, 512, tokens).astype(np.uint16))
    return d


def build(tmp_path, cfg, data, run_id="t", resume=False):
    return Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data, device="cpu",
                   run_dir=tmp_path / "runs", run_id=run_id, resume=resume)


def _losses(tmp_path, name, steps=3, **train):
    cfg = smoke_cfg(tmp_path / name, model={"attnres_blocks": 0}, **train)
    t = build(tmp_path / name, cfg, make_data(tmp_path / name))
    return [t.train_step() for _ in range(steps)], t


def test_memory_options_do_not_change_training_on_cpu(tmp_path):
    base, tb = _losses(tmp_path, "base")
    low, tl = _losses(tmp_path, "low", precision="fp16", optimizer_state_dtype="fp32",
                      activation_checkpoint="block", loss_chunk_tokens=300)
    np.testing.assert_allclose(low, base, rtol=1e-5)
    for (n, p), (_, q) in zip(tb.model.named_parameters(), tl.model.named_parameters()):
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-6, msg=n)


def test_bf16_state_trains(tmp_path):
    losses, t = _losses(tmp_path, "bf16state", steps=5, optimizer_state_dtype="bf16")
    assert all(math.isfinite(x) for x in losses)
    muon = t.optimizers[0]
    assert all(s["momentum_buffer"].dtype is torch.bfloat16 for s in muon.state.values())


def test_fp16_overflow_halves_the_scale_and_skips_without_counting(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    t = build(tmp_path, cfg, make_data(tmp_path))
    scale0 = t.scaler.scale
    p = next(t.model.parameters())
    h = p.register_hook(lambda g: torch.full_like(g, float("inf")))
    t.train_step()
    h.remove()
    assert t.step == 0 and t.scaler.scale == scale0 / 2
    assert t.nonfinite_streak == 0 and t.overflow_steps == 1
    t.train_step()
    assert t.step == 1


def test_loss_scale_is_checkpointed(tmp_path):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16")
    data = make_data(tmp_path)
    t = build(tmp_path, cfg, data)
    t.scaler.overflow()
    t.train_step()
    t.save_checkpoint()
    r = build(tmp_path, cfg, data, run_id="t", resume=True)
    r.resume_from_latest()
    assert r.scaler.state_dict() == t.scaler.state_dict()


def test_fp32_momentum_checkpoint_resumes_into_bf16_state(tmp_path):
    data = make_data(tmp_path)
    cfg32 = smoke_cfg(tmp_path, model={"attnres_blocks": 0})
    t = build(tmp_path, cfg32, data)
    t.train_step()
    path = t.save_checkpoint()
    cfg16 = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, optimizer_state_dtype="bf16")
    r = build(tmp_path, cfg16, data, run_id="u")
    r.init_weights_from(path, optimizers=True)
    muon = r.optimizers[0]
    assert all(s["momentum_buffer"].dtype is torch.bfloat16 for s in muon.state.values())


@pytest.mark.parametrize("cap,dispatch,precision,ok", [
    ((7, 5), "loop", "fp16", True),
    ((7, 5), "grouped", "fp16", False),
    ((7, 5), "loop", "bf16", False),
    ((9, 0), "grouped", "bf16", True),
    ((8, 0), "grouped", "bf16", False),
    (None, "grouped", "bf16", True),          # CPU: nothing to check
])
def test_device_guard(tmp_path, cap, dispatch, precision, ok):
    cfg = smoke_cfg(tmp_path, model={"attnres_blocks": 0, "moe_dispatch": dispatch},
                    precision=precision)
    if ok:
        check_device_support(cfg.model, cfg.train, cap)
    else:
        with pytest.raises(UsageError):
            check_device_support(cfg.model, cfg.train, cap)
```

Before writing Step 3, check the names these tests assume against `quipu/train.py`: `save_checkpoint()` returns the path (line ~1091, it does); the `--init-from` method is the one at line ~1236 (open it and use its real name and signature in the test; it is `init_weights_from(path, optimizers=True)`); `resume_from_latest()` exists. Fix the test to the real names before running.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k "memory_options or bf16_state or overflow or loss_scale or resumes_into or device_guard" -v`
Expected: FAIL (`cannot import name 'check_device_support'`).

- [ ] **Step 3: Implement**

3a. Imports at the top of `quipu/train.py`:

```python
from quipu.lossfn import chunked_cross_entropy
from quipu.lossscale import LossScaler
from quipu.optim import Muon
```

(`Muon` may already be imported; don't duplicate.)

3b. Module-level function, next to `resolve_checkpoint`:

```python
def check_device_support(model_cfg: ModelConfig, train_cfg: TrainConfig,
                         capability: tuple[int, int] | None) -> None:
    """Refuse at startup what this GPU cannot run, rather than failing mid-run.
    capability: torch.cuda.get_device_capability() of the training device, None on
    CPU (nothing to check). bf16 tensor cores start at sm_80; torch._grouped_mm (the
    "grouped" expert dispatch) at sm_90."""
    if capability is None:
        return
    sm = f"sm_{capability[0]}{capability[1]}"
    if model_cfg.kind == "moe" and model_cfg.moe_dispatch == "grouped" and capability < (9, 0):
        raise UsageError(f"model.moe_dispatch 'grouped' needs sm_90 or newer, this GPU is "
                         f"{sm}; use 'loop' or 'padded'")
    if train_cfg.precision == "bf16" and capability < (8, 0):
        raise UsageError(f"train.precision 'bf16' needs sm_80 or newer, this GPU is {sm}; "
                         "use 'fp16'")
```

3c. In `Trainer.__init__`, right after `self.model = apply_precision(...)` and before `dist.broadcast_params`:

```python
        check_device_support(
            model_cfg, train_cfg,
            torch.cuda.get_device_capability(device) if device.startswith("cuda") else None)
        # Low memory (spec 2026-10-20): autocast dtype, the fp16 loss scale, block
        # activation checkpointing.
        self.amp_dtype = torch.float16 if train_cfg.precision == "fp16" else torch.bfloat16
        self.scaler = LossScaler(enabled=train_cfg.precision == "fp16")
        self.overflow_steps = 0     # fp16 loss-scale overflows skipped (not poison)
        self.model.checkpoint_blocks = train_cfg.activation_checkpoint == "block"
```

3d. Every `torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)` inside `Trainer` (in `_trial_compiled_step` and `train_step`) becomes `dtype=self.amp_dtype`. Pass `amp_dtype=self.amp_dtype` to `estimate_loss(...)` (~1403), `mean_loss(...)` in `_eval_mix` (~756), and `evaluate_by_source(...)` (~1497, `amp_dtype=trainer.amp_dtype`).

3e. In `train_step`, replace the loss computation inside `with torch.autocast(...)` and the two backward branches. The new micro-batch body (keep the surrounding code):

```python
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
                    chunk = cfg.loss_chunk_tokens
                    if chunk:
                        h = self.forward_model(x, return_hidden=True)
                        nll = chunked_cross_entropy(h, self.model.lm_head.weight, y, chunk,
                                                    ignore_index=IGNORE_INDEX)
                        loss = nll / (n_targets if sft else y.numel())
                    else:
                        logits = self.forward_model(x)
                        if sft:
                            # Targets outside assistant turns are IGNORE_INDEX; every
                            # block has one at least (MaskedBlockStream), so N > 0.
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                   y.reshape(-1), ignore_index=IGNORE_INDEX,
                                                   reduction="sum") / n_targets
                        else:
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                   y.reshape(-1))
                scale = self.scaler.scale
                if sft:
                    (loss * scale).backward()
                    total_t += loss.detach().double()
                else:
                    # Divide before backward so the accumulated gradient is the mean
                    # over the whole batch, not the sum over micro-batches.
                    (loss * (scale / accum)).backward()
                    total_t += loss.detach().double() / accum
```

After the `if world > 1:` all-reduce block and before `total = total_t.item()`:

```python
            # fp16: the gradients carry the loss scale until here (after the
            # all-reduce, before clipping); a no-op otherwise.
            self.scaler.unscale_(self.model.parameters())
```

Replace the non-finite branch's opening condition and add the overflow path:

```python
            total = total_t.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            grads_ok = bool(torch.isfinite(grad_norm))
            if (math.isfinite(total) and not grads_ok and self.scaler.enabled
                    and not self.scaler.at_floor):
                # fp16 overflow: expected now and then. Halve the scale and skip the
                # step; not a poisoned batch, so the streak is untouched.
                self._zero_grad()
                if self.is_moe:
                    self.model.clear_balance_scores()
                self.scaler.overflow()
                self.overflow_steps += 1
                if dist.is_main():
                    print(f"fp16 overflow at step {self.step}: loss scale -> "
                          f"{self.scaler.scale:g}, step skipped", flush=True)
                return total
            if not (math.isfinite(total) and grads_ok):
```

(the rest of the existing non-finite branch stays as it is). After `for opt in self.optimizers: opt.step()`:

```python
            self.scaler.good_step()
```

3f. `_write_checkpoint`: add to the saved dict:

```python
                "loss_scale": self.scaler.state_dict() if self.scaler.enabled else None,
```

`load_checkpoint`, after `self.skipped_steps = ...`:

```python
        if state.get("loss_scale") is not None:     # absent before fp16 existed
            self.scaler.load_state_dict(state["loss_scale"])
```

3g. `_load_optimizers`, after `apply_config_lrs(self.optimizers, self.train_cfg)`:

```python
        # load_state_dict cast Muon's momentum to fp32 (the parameters' dtype);
        # back to the configured dtype now, before any gradient takes memory.
        for opt in self.optimizers:
            if isinstance(opt, Muon):
                opt.cast_state_()
```

3h. Module docstring of `quipu/train.py`: add a paragraph after the fp8 one:

```
train.precision "fp16" (spec 2026-10-20, Kaggle T4) runs the forward under fp16
autocast with a dynamic loss scale (quipu.lossscale): an overflowing step is skipped
and halves the scale without counting toward the non-finite streak, unless the scale
is at its floor. optimizer_state_dtype, activation_checkpoint and loss_chunk_tokens
trade compute for memory (quipu.optim.muon, the models' checkpoint_blocks,
quipu.lossfn). check_device_support refuses bf16 below sm_80 and the grouped
dispatch below sm_90 at startup.
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py -v`
Expected: PASS. Then the trainer suites:
Run: `python -m pytest tests/test_train_moe.py tests/test_checkpoint_resume.py tests/test_chat_sft.py tests/test_dist.py tests/test_train_exit_codes.py -v`
Expected: PASS, except the known pre-existing failure `test_chat_sft::test_the_sft_config_inherits_the_full_model_and_scales_the_lr` (situ_glu vs swiglu pin), which fails before this change too. Check that with `git stash; python -m pytest tests/test_chat_sft.py -k inherits; git stash pop` if in doubt.

- [ ] **Step 5: Commit**

```bash
git add quipu/train.py tests/test_lowmem.py
git commit -m "train: fp16 loss scaling, chunked loss, block checkpointing, bf16 momentum, device guard"
```

---

### Task 8: Kaggle config layer

**Files:**
- Create: `configs/quipu-moe-cpt-t4.toml`
- Test: `tests/test_lowmem.py`

- [ ] **Step 1: Write the failing test** (append)

```python
# ---- Kaggle config -------------------------------------------------------------------

def test_t4_config_is_the_cpt_model_with_low_memory_training():
    cpt = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    t4 = load_config(ROOT / "configs" / "quipu-moe-cpt-t4.toml")
    assert dataclasses.replace(t4.model, moe_dispatch=cpt.model.moe_dispatch) == cpt.model
    assert t4.model.moe_dispatch == "loop"
    t = t4.train
    assert (t.precision, t.optimizer_state_dtype, t.activation_checkpoint) == \
        ("fp16", "bf16", "block")
    assert t.loss_chunk_tokens == 2048 and t.compile is False
    assert t.batch_tokens == cpt.train.batch_tokens
    assert t.grad_accum % 2 == 0                      # splits over 2 GPUs
    assert t.ckpt_keep == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_lowmem.py -k t4_config -v`
Expected: FAIL (file not found).

- [ ] **Step 3: Create** `configs/quipu-moe-cpt-t4.toml`

```toml
name = "quipu-moe-cpt-t4"

# quipu-moe-cpt on Kaggle's 2x T4 (16 GB, sm_75: no bf16, no grouped matmul).
# Spec: docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md.
# Runbook: docs/superpowers/specs/2026-10-20-kaggle-t4-runbook.md.
# Same model and schedule as the B200 run; only how a step is computed changes.
inherit = "quipu-moe-cpt.toml"

[model]
moe_dispatch = "loop"

[train]
precision             = "fp16"
optimizer_state_dtype = "bf16"
activation_checkpoint = "block"
loss_chunk_tokens     = 2048
compile               = false
micro_batch           = 2         # starting point; the first session measures memory
ckpt_dir              = "/kaggle/working/ckpt"
ckpt_keep             = 1         # /kaggle/working holds 20 GB
# Steps take ~25-60 s on 2x T4 (estimate); about every 30-45 minutes. Retune from
# the first session's step time.
ckpt_every            = 60
eval_every            = 60
# The spend guard doubles as the session clock: 1 "dollar" per hour stops the run
# cleanly with a checkpoint after 11.25 h, inside Kaggle's 12-hour limit.
usd_per_hour          = 1.0
budget_usd            = 11.25
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lowmem.py -k t4_config -v`
Expected: PASS. If `tests/test_modal_train.py` iterates over every `configs/*cpt*.toml`, run it too: `python -m pytest tests/test_modal_train.py -v`.

- [ ] **Step 5: Commit**

```bash
git add configs/quipu-moe-cpt-t4.toml tests/test_lowmem.py
git commit -m "configs: quipu-moe-cpt-t4 for Kaggle 2x T4"
```

---

### Task 9: GPU probe script (parity and memory)

**Files:**
- Create: `scripts/lowmem_probe.py`

Manual GPU tool, no unit test (it needs a GPU and real shards). It must import and print `--help` on CPU.

- [ ] **Step 1: Create** `scripts/lowmem_probe.py`

```python
"""Probe the low-memory training options on a GPU (spec 2026-10-20, Testing 2-5).

  parity: train a reduced quipu-moe (the 1B's width and expert layout, --layers deep)
          twice from the same seed and data, once bf16 with no memory options and once
          with the T4 settings (fp16, bf16 momentum, block checkpointing, chunked
          loss), and compare. Needs a bf16-capable GPU (the laptop's RTX 5060).
  memory: one training step at the given settings; peak memory, step time, tokens/s.

  python scripts/lowmem_probe.py parity --shards data/shards-moe/train --steps 200
  python scripts/lowmem_probe.py memory --shards data/shards-moe/train --layers 4 --micro-batch 2 --t4
  torchrun --standalone --nproc_per_node=2 scripts/lowmem_probe.py memory --shards ... --layers 16 --t4

Writes a JSON summary to --out (default results/lowmem_probe.json) and prints it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu import dist  # noqa: E402
from quipu.config import load_config  # noqa: E402
from quipu.train import Trainer  # noqa: E402

T4 = {"precision": "fp16", "optimizer_state_dtype": "bf16",
      "activation_checkpoint": "block", "loss_chunk_tokens": 2048}
REFERENCE = {"precision": "bf16", "optimizer_state_dtype": "fp32",
             "activation_checkpoint": "none", "loss_chunk_tokens": 0}


def make_trainer(args, train_over: dict, run_dir: Path, run_id: str) -> Trainer:
    cfg = load_config(ROOT / "configs" / "quipu-moe-cpt.toml")
    model = dataclasses.replace(cfg.model, n_layer=args.layers, moe_dispatch="loop")
    train = dataclasses.replace(
        cfg.train, micro_batch=args.micro_batch,
        batch_tokens=args.micro_batch * model.context * args.accum * dist.world_size(),
        compile=False, budget_usd=0.0, usd_per_hour=0.0, eval_every=10**9,
        ckpt_every=10**9, ckpt_dir=str(run_dir / "ckpt"), **train_over)
    return Trainer(model_cfg=model, train_cfg=train, shard_dir=args.shards,
                   device=f"cuda:{dist.local_rank()}" if dist.world_size() > 1 else "cuda",
                   run_dir=run_dir, run_id=run_id)


def run(args, over: dict, run_dir: Path, run_id: str, steps: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t = make_trainer(args, over, run_dir, run_id)
    losses, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        losses.append(t.train_step())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    tokens = t.train_cfg.batch_tokens
    step_s = sorted(times)[len(times) // 2]
    out = {"settings": over, "layers": args.layers, "micro_batch": args.micro_batch,
           "losses": losses, "nonfinite": sum(not math.isfinite(x) for x in losses),
           "overflow_steps": getattr(t, "overflow_steps", 0), "steps_taken": t.step,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "median_step_s": step_s, "tokens_per_s": tokens / step_s}
    del t
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["parity", "memory"])
    p.add_argument("--shards", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4, help="micro-batches per step per rank")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--t4", action="store_true", help="memory mode: use the T4 settings")
    p.add_argument("--out", default=str(ROOT / "results" / "lowmem_probe.json"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 2
    dist.init_from_env(cuda=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.mode == "parity":
            ref = run(args, REFERENCE, tmp / "ref", "ref", args.steps)
            low = run(args, T4, tmp / "low", "low", args.steps)
            tail = lambda r: sum(r["losses"][-10:]) / 10  # noqa: E731
            summary = {"reference": ref, "t4": low,
                       "final_loss_rel_diff": abs(tail(low) - tail(ref)) / tail(ref),
                       "pass": abs(tail(low) - tail(ref)) / tail(ref) < 0.01
                       and low["nonfinite"] == low["overflow_steps"]}
        else:
            summary = run(args, T4 if args.t4 else REFERENCE, tmp / "mem", "mem",
                          max(3, min(args.steps, 5)))
    if dist.is_main():
        text = json.dumps(summary, indent=2)
        print(text)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    dist.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Before running: check `quipu.dist.init_from_env`'s real signature (`dist.py:64`, it takes `cuda=`) and that `Trainer` accepts a missing `val_dir` (it does, default None). The expert-load "spread unchanged" check from the spec is read from the two runs' logs in `tmp/*/runs/*.json` ("moe" entries); if the trainer only logs those at eval steps, set `eval_every` to `args.steps` instead of 10**9 and add the last "moe" entry of each run to the summary.

- [ ] **Step 2: Check it imports**

Run: `python scripts/lowmem_probe.py --help`
Expected: usage text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/lowmem_probe.py
git commit -m "scripts: lowmem_probe for fp16 parity and memory on a GPU"
```

---

### Task 10: Kaggle runbook

**Files:**
- Create: `docs/superpowers/specs/2026-10-20-kaggle-t4-runbook.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Kaggle 2x T4 runbook (quipu-moe continued pretraining)

Config: configs/quipu-moe-cpt-t4.toml. Spec: 2026-10-20-t4-low-memory-training-design.md.

## Once (owner)

1. Private Kaggle dataset `quipu-data-v3-part1`: about 1B tokens of data v3 chunk 0
   shards (uint16 .bin, ~2 GB) from the Modal volume quipu-pool, plus the val shard.
   Download with `modal volume get` (Git Bash, MSYS_NO_PATHCONV=1), upload in the
   Kaggle UI.
2. Private Kaggle dataset `quipu-moe-step8129`: checkpoints/.../step_008129.pt (8.1 GB).
3. Notebook settings: Accelerator "GPU T4 x2", Internet on (pip install from GitHub),
   Persistence: files.

## Every session

```bash
pip install -q git+https://github.com/Aneek1/quipu.git@main   # or clone and pip install -e .
nvidia-smi                                                  # two T4s
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-cpt \
  --override data.shard_dir=<the trainer view of /kaggle/input/quipu-data-v3-part1> \
  FIRST SESSION ONLY:  --init-from /kaggle/input/quipu-moe-step8129/step_008129.pt --init-optimizers
  LATER SESSIONS:      --resume   (with the previous session's /kaggle/working/ckpt and runs/
                                   attached as input and copied back to /kaggle/working)
```

Run it as "Save Version → Save & Run All (commit)" so it continues in the background.
The config's budget (11.25 "dollars" at 1 per hour) stops it with a checkpoint before
Kaggle's 12-hour limit.

## First session: checks before a long run (about 1 hour)

- `python scripts/lowmem_probe.py memory --shards ... --layers 16 --micro-batch 2 --t4`
  under torchrun with 2 processes: peak GiB per rank must be under 14.5.
- First logged train loss within 1% of the bf16 run's logged loss at step 8129
  (remote-runs/quipu-moe/results logs); 50 steps with no non-finite loss (fp16
  overflows that halve the scale are fine).
- Note step time and tokens/s; set micro_batch (largest that fits) and ckpt_every
  (~30-45 min) and commit the change to configs/quipu-moe-cpt-t4.toml.

If the loss check fails or non-finite steps repeat: stop, T4s go back to eval and
generation only (spec, "Testing").
```

(Before saving, open `quipu/train.py`'s argparse block (~line 1560-1590) and `quipu/shard_view.py` and replace the `--override data.shard_dir=<...>` line with the exact way the CPT launcher (`scripts/remote/modal_train.py`) builds the trainer's shard view, so the runbook command is copy-pasteable.)

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-10-20-kaggle-t4-runbook.md
git commit -m "docs: Kaggle 2x T4 runbook"
```

---

### Task 11: Laptop GPU checks (manual, with the owner)

Warn the owner first: these use the laptop GPU for ~20-40 minutes.

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: everything passes except the known `test_chat_sft::test_the_sft_config_inherits_the_full_model_and_scales_the_lr`.

- [ ] **Step 2: Parity on the RTX 5060**

Run: `python scripts/lowmem_probe.py parity --shards data/shards-moe/train --layers 4 --micro-batch 1 --steps 200`
(use whatever English train shard directory exists locally; `ls data`.)
Expected: `"pass": true` (final-loss difference under 1%, no non-finite steps beyond overflows).

- [ ] **Step 3: Memory at T4 settings**

Run: `python scripts/lowmem_probe.py memory --shards data/shards-moe/train --layers 4 --micro-batch 2 --t4`
and the same without `--t4`. Record both peaks. Projection for 16 layers: static
(weights + grads + optimizer, ~10.3 GB at bf16 momentum, scales with parameter count:
compute it from the 4-layer run's parameter count) + per-layer activation x 16.
Expected: projected peak per T4 under 14.5 GiB.

- [ ] **Step 4: Write results into the spec**

Append a "Results (laptop)" section to `docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md` with the parity numbers, both memory peaks and the projection; commit:

```bash
git add docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md results/lowmem_probe.json
git commit -m "spec: laptop parity and memory results for T4 training"
```

(If `results/` is untracked by design, commit only the spec.)
```
