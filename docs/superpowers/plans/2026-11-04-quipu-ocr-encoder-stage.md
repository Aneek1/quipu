# Quipu-OCR Encoder Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the encoder stage of Quipu-OCR v1 (spec `docs/superpowers/specs/2026-11-04-quipu-ocr-line-design.md`): a from-scratch 22M ViT that reads one line of printed or handwritten English through a CTC head, with its text and image data, training script and evaluation against Tesseract and TrOCR-small.

**Architecture:** Text lines are cut from Quipu's own token shards and rendered on the fly into 64-px-high images with open-licensed Google Fonts (printed and handwriting styles) and scan-like noise. A ViT turns each image into a 4 × C grid of patch features; the CTC head averages the 4 rows into C column vectors and predicts characters. Evaluation runs on fixed, hashed held-out sets (held-out text and held-out fonts) plus a nonsense set.

**Tech Stack:** Python 3.13, PyTorch 2.11 (`F.ctc_loss`, `scaled_dot_product_attention`), Pillow 12 for rendering, numpy, pytest. Optional at eval time: the `tesseract` CLI and `transformers` TrOCR-small.

**Scope:** This plan is spec sections 3.1–3.3, 4, 5.1 and 6 (with the CTC model as the system under test). The joint stage (spec 3.4, 5.2: connector, two new tokens, training Quipu) is a second plan, written once this stage's CER numbers exist, because its settings depend on how good the encoder turns out.

**One deviation from the spec's file table:** the renderer is a library module, `quipu/ocr_render.py`, because the data loader in `quipu/ocr_data.py` imports it; `scripts/ocr/render_lines.py` is a thin CLI over it that writes samples for eyeballing. Everything else matches the table.

**Repo rules for the executor:**
- Python is `.venv/Scripts/python` (Windows). In Git Bash set `MSYS_NO_PATHCONV=1` before commands with `/`-leading arguments.
- Stage files by explicit path. Never stage `quipu/model_card.py` (the user's uncommitted work) or `git add -A`.
- Commit messages carry no AI attribution lines (no Co-Authored-By).
- Never push without the user's fresh yes. Any network download (Task 5, Task 9's TrOCR weights) is announced to the user with what and how big before it runs.
- Never use text written by Claude or any other model as training text, and never SP Manufacturing documents.

---

## File Structure

| File | Responsibility |
|---|---|
| `quipu/ocr_ctc.py` | character set, label encoding, CTC length check, `CTCHead`, `LineCTC` model, `ctc_loss`, greedy decode |
| `quipu/vision.py` | `ViTConfig`, `LineViT` encoder (patches, factorised positions, masked attention) |
| `quipu/ocr_render.py` | font manifest loading, licence check, held-out font split, `render_line` (printed and hand), augmentation |
| `quipu/ocr_data.py` | `preprocess` (resize, pad, to tensor), `collate`, `LineStream` (on-the-fly training stream) |
| `quipu/ocr_metrics.py` | Levenshtein distance, CER, WER, exact match |
| `scripts/ocr/build_text.py` | cut lines from token shards, random lines, held-out split, write `train.txt` / `held.txt` |
| `scripts/ocr/fetch_fonts.py` | download pinned Google Fonts families with their licence files, write `fonts.json` |
| `scripts/ocr/render_lines.py` | CLI: render sample lines to PNGs for a visual check |
| `scripts/ocr/train_ctc.py` | encoder-stage training: config, loop, eval, checkpoints, resume, early stop |
| `scripts/ocr/eval_ocr.py` | `make` fixed eval sets; `run` CTC model and baselines, write the results table |
| `configs/ocr-ctc.toml` | full encoder-stage config |
| `configs/ocr-ctc-smoke.toml` | tiny CPU config for the smoke test |
| `configs/ocr-fonts.toml` | the pinned font family list, by style |
| `docs/ocr/handwriting-licences.md` | licence review of real handwriting datasets |
| `tests/test_ocr_ctc.py`, `tests/test_vision.py`, `tests/test_ocr_render.py`, `tests/test_ocr_data.py`, `tests/test_ocr_metrics.py`, `tests/test_ocr_build_text.py`, `tests/test_ocr_train_ctc.py`, `tests/test_ocr_eval.py` | unit and smoke tests |
| `tests/ocr_fixtures.py` | a test font directory built from matplotlib's bundled DejaVu Sans |

Data lives under `data/ocr/` (git-ignored like the rest of `data/`): `data/ocr/text/`, `data/ocr/fonts/`, `data/ocr/eval/`. Runs go to `runs/ocr-ctc/`.

---

### Task 1: Character set, CTC length check and greedy decode

**Files:**
- Create: `quipu/ocr_ctc.py`
- Test: `tests/test_ocr_ctc.py`

- [ ] **Step 1: Write the failing tests**

```python
"""quipu/ocr_ctc.py: character set, labels, CTC length check, greedy decode."""
import pytest
import torch

from quipu import ocr_ctc as oc


def test_charset_is_printable_ascii_plus_extras():
    assert oc.CHARSET[:95] == "".join(chr(c) for c in range(32, 127))
    assert oc.CHARSET[95:] == "°£€×–‘’"
    assert oc.N_CLASSES == len(oc.CHARSET) + 1
    assert oc.BLANK == 0


def test_encode_text_is_one_based_and_rejects_unknown():
    assert oc.encode_text(" !") == [1, 2]
    assert oc.encode_text("a") == [oc.CHARSET.index("a") + 1]
    with pytest.raises(ValueError, match="not in the OCR character set"):
        oc.encode_text("naïve")


def test_in_charset():
    assert oc.in_charset("Hello, world! £5")
    assert not oc.in_charset("tab\there")


def test_ctc_steps_needed_counts_repeats():
    assert oc.ctc_steps_needed("abc") == 3
    assert oc.ctc_steps_needed("hello") == 6      # one "ll"
    assert oc.ctc_steps_needed("aaa") == 5
    assert oc.ctc_steps_needed("") == 0


def _one_hot_path(path, n_classes):
    lp = torch.full((len(path), 1, n_classes), -30.0)
    for t, k in enumerate(path):
        lp[t, 0, k] = 0.0
    return lp


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    a, b = oc.encode_text("a")[0], oc.encode_text("b")[0]
    lp = _one_hot_path([a, a, oc.BLANK, a, b, b, oc.BLANK], oc.N_CLASSES)
    assert oc.ctc_greedy_decode(lp, torch.tensor([7])) == ["aab"]


def test_greedy_decode_respects_lengths():
    a, b = oc.encode_text("a")[0], oc.encode_text("b")[0]
    lp = _one_hot_path([a, oc.BLANK, b, b], oc.N_CLASSES)
    assert oc.ctc_greedy_decode(lp, torch.tensor([2])) == ["a"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_ctc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.ocr_ctc'`

- [ ] **Step 3: Write the implementation**

```python
"""Character set, CTC head and greedy decoding for the OCR encoder stage.

Spec docs/superpowers/specs/2026-11-04-quipu-ocr-line-design.md, section 3.3. Class 0
is the CTC blank; character i of CHARSET is class i + 1. The set is fixed here so the
trainer, the data filter and the evaluation can never disagree about it."""
from __future__ import annotations

import torch

EXTRA = "°£€×–‘’"
CHARSET = "".join(chr(c) for c in range(32, 127)) + EXTRA
BLANK = 0
N_CLASSES = len(CHARSET) + 1
_INDEX = {ch: i + 1 for i, ch in enumerate(CHARSET)}


def in_charset(text: str) -> bool:
    return all(ch in _INDEX for ch in text)


def encode_text(text: str) -> list[int]:
    try:
        return [_INDEX[ch] for ch in text]
    except KeyError as e:
        raise ValueError(f"character {e.args[0]!r} is not in the OCR character set") from None


def ctc_steps_needed(text: str) -> int:
    """The fewest CTC steps that can emit `text`: one per character plus a blank between
    each pair of equal neighbours."""
    return len(text) + sum(1 for a, b in zip(text, text[1:]) if a == b)


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    """log_probs [T, B, K], lengths [B] (valid steps per item) -> one string per item."""
    best = log_probs.argmax(-1).transpose(0, 1).tolist()
    out = []
    for seq, n in zip(best, lengths.tolist()):
        chars, prev = [], BLANK
        for k in seq[:n]:
            if k != prev and k != BLANK:
                chars.append(CHARSET[k - 1])
            prev = k
        out.append("".join(chars))
    return out
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_ctc.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add quipu/ocr_ctc.py tests/test_ocr_ctc.py
git commit -m "ocr: character set, CTC length check, greedy decode"
```

---

### Task 2: The line ViT encoder

**Files:**
- Create: `quipu/vision.py`
- Test: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_vision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.vision'`

- [ ] **Step 3: Write the implementation**

```python
"""The OCR line encoder: a small ViT over 16x16 patches of a 64-px-high line image.

Spec section 3.2. Positions are a learned row table (4) plus a learned column table
(64), added, so width can vary per batch. Padding columns are masked out as attention
keys, so the features of real columns do not depend on how much padding follows."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ViTConfig:
    height: int = 64
    max_width: int = 1024
    patch: int = 16
    d_model: int = 384
    n_layer: int = 12
    n_head: int = 6
    mlp: int = 1536

    @property
    def rows(self) -> int:
        return self.height // self.patch

    @property
    def max_cols(self) -> int:
        return self.max_width // self.patch


class _Block(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.mlp), nn.GELU(),
                                 nn.Linear(cfg.mlp, cfg.d_model))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = (self.qkv(self.ln1(x)).view(B, N, 3, self.n_head, D // self.n_head)
                   .permute(2, 0, 3, 1, 4))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.out(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.ln2(x))


class LineViT(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(1, cfg.d_model, cfg.patch, stride=cfg.patch)
        self.row_pos = nn.Parameter(torch.randn(cfg.rows, cfg.d_model) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(cfg.max_cols, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, images: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
        """images [B, 1, 64, W] (ink 1, background 0), col_mask [B, W/16] True on real
        columns -> features [B, 4, W/16, d_model]."""
        cfg = self.cfg
        B, _, H, W = images.shape
        if H != cfg.height:
            raise ValueError(f"image height {H}, expected {cfg.height}")
        if W % cfg.patch:
            raise ValueError(f"image width {W} is not a multiple of {cfg.patch}")
        if W > cfg.max_width:
            raise ValueError(f"image width {W} is over max_width {cfg.max_width}")
        C = W // cfg.patch
        if tuple(col_mask.shape) != (B, C):
            raise ValueError(f"col_mask shape {tuple(col_mask.shape)}, expected {(B, C)}")
        x = self.patch(images).permute(0, 2, 3, 1)                     # B, R, C, D
        R = x.shape[1]
        x = x + self.row_pos[:R, None, :] + self.col_pos[None, :C, :]
        x = x.reshape(B, R * C, -1)
        keys = col_mask[:, None, :].expand(B, R, C).reshape(B, 1, 1, R * C)
        for blk in self.blocks:
            x = blk(x, keys)
        return self.norm(x).reshape(B, R, C, -1)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_vision.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add quipu/vision.py tests/test_vision.py
git commit -m "ocr: line ViT encoder with factorised positions and masked padding"
```

---

### Task 3: CTC head, model and loss

**Files:**
- Modify: `quipu/ocr_ctc.py` (append)
- Test: `tests/test_ocr_ctc.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ocr_ctc.py`)

```python
from quipu.vision import ViTConfig

TINY = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)


def test_line_ctc_shapes():
    m = oc.LineCTC(TINY)
    images = torch.rand(2, 1, 64, 128)
    mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    lp, lens = m(images, mask)
    assert lp.shape == (8, 2, oc.N_CLASSES)
    assert lp.dtype == torch.float32
    assert lens.tolist() == [8, 5]
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(8, 2))


def test_ctc_loss_goes_down_when_overfitting_two_lines():
    torch.manual_seed(0)
    m = oc.LineCTC(TINY)
    images = torch.rand(2, 1, 64, 128)
    mask = torch.ones(2, 8, dtype=torch.bool)
    texts = ["ab", "cab"]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        lp, lens = m(images, mask)
        loss = oc.ctc_loss(lp, lens, texts)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first
    lp, lens = m(images, mask)
    assert oc.ctc_greedy_decode(lp, lens) == texts
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_ctc.py -v`
Expected: FAIL with `AttributeError: module 'quipu.ocr_ctc' has no attribute 'LineCTC'`

- [ ] **Step 3: Write the implementation** (append to `quipu/ocr_ctc.py`; add the imports at the top of the file)

```python
import torch.nn.functional as F
from torch import nn

from quipu.vision import LineViT, ViTConfig


class CTCHead(nn.Module):
    """Mean over the 4 patch rows of each column, then a linear layer to the classes."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, N_CLASSES)

    def forward(self, feats: torch.Tensor, col_mask: torch.Tensor):
        """feats [B, R, C, D], col_mask [B, C] -> (log_probs [C, B, K] float32, lengths [B])."""
        logits = self.proj(feats.mean(dim=1)).float()
        return logits.log_softmax(-1).transpose(0, 1), col_mask.sum(1)


class LineCTC(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.encoder = LineViT(cfg)
        self.head = CTCHead(cfg.d_model)

    def forward(self, images: torch.Tensor, col_mask: torch.Tensor):
        return self.head(self.encoder(images, col_mask), col_mask)


def ctc_loss(log_probs: torch.Tensor, lengths: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """Mean CTC loss over the batch (PyTorch divides each item by its target length)."""
    targets = [encode_text(t) for t in texts]
    flat = torch.tensor([k for t in targets for k in t], dtype=torch.long,
                        device=log_probs.device)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return F.ctc_loss(log_probs, flat, lengths.cpu(), tgt_lens, blank=BLANK,
                      reduction="mean", zero_infinity=True)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_ctc.py -v`
Expected: 8 passed (the overfit test takes a few seconds on CPU)

- [ ] **Step 5: Commit**

```bash
git add quipu/ocr_ctc.py tests/test_ocr_ctc.py
git commit -m "ocr: CTC head, LineCTC model and loss"
```

---

### Task 4: CER, WER and exact match

**Files:**
- Create: `quipu/ocr_metrics.py`
- Test: `tests/test_ocr_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
"""quipu/ocr_metrics.py."""
from quipu.ocr_metrics import cer, edit_distance, exact_match, wer


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_cer_is_total_edits_over_total_reference_chars():
    assert cer(["abcd", "xy"], ["abcd", "xz"]) == 1 / 6
    assert cer(["ab"], [""]) == 1.0


def test_wer_counts_words():
    assert wer(["the cat sat"], ["the bat sat"]) == 1 / 3


def test_exact_match():
    assert exact_match(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3


def test_empty_reference_set_raises():
    import pytest
    with pytest.raises(ValueError):
        cer([], [])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.ocr_metrics'`

- [ ] **Step 3: Write the implementation**

```python
"""OCR metrics: character and word error rate (Levenshtein), exact-line match.

Rates are corpus-level: total edits over total reference length, not a mean of
per-line rates, so long lines weigh more."""
from __future__ import annotations

from typing import Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def _rate(refs: list[Sequence], hyps: list[Sequence]) -> float:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    total = sum(len(r) for r in refs)
    if total == 0:
        raise ValueError("no reference characters to score against")
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / total


def cer(refs: list[str], hyps: list[str]) -> float:
    return _rate(refs, hyps)


def wer(refs: list[str], hyps: list[str]) -> float:
    return _rate([r.split() for r in refs], [h.split() for h in hyps])


def exact_match(refs: list[str], hyps: list[str]) -> float:
    if not refs:
        raise ValueError("no references")
    return sum(r == h for r, h in zip(refs, hyps)) / len(refs)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add quipu/ocr_metrics.py tests/test_ocr_metrics.py
git commit -m "ocr: CER, WER and exact-match metrics"
```

---

### Task 5: Font list, fetcher and manifest

**Files:**
- Create: `configs/ocr-fonts.toml`
- Create: `scripts/ocr/fetch_fonts.py`

No unit test for the network part; the licence check it relies on is tested in Task 6. Before running Step 3, tell the user: "downloading about 45 font families (roughly 60–120 MB) from github.com/google/fonts into data/ocr/fonts", and wait for a yes.

- [ ] **Step 1: Write the font list**

```toml
# Font families for OCR rendering (spec section 4.2, 4.3), from github.com/google/fonts.
# Each entry is the family's directory in that repo; its licence file (OFL.txt or
# LICENSE.txt) is downloaded with it, and the renderer refuses a font without one.
# Families missing at the pinned commit are skipped with a warning; fetch_fonts.py
# fails if fewer than min_per_style of a style are fetched.

min_per_style = 8

printed = [
  "ofl/lora", "ofl/merriweather", "ofl/opensans", "ofl/lato", "ofl/inter",
  "ofl/sourcecodepro", "ofl/ibmplexmono", "ofl/ibmplexsans", "ofl/ibmplexserif",
  "ofl/ptserif", "ofl/ptsans", "ofl/librebaskerville", "ofl/firasans", "ofl/firamono",
  "ofl/nunito", "ofl/oswald", "ofl/ebgaramond", "ofl/crimsontext", "ofl/spacemono",
  "ofl/courierprime", "ofl/notosans", "ofl/notoserif", "ofl/worksans", "ofl/rubik",
  "ofl/karla", "ofl/abrilfatface", "ofl/bebasneue", "apache/robotoslab",
]

hand = [
  "ofl/caveat", "ofl/indieflower", "ofl/patrickhand", "ofl/shadowsintolight",
  "ofl/kalam", "ofl/gloriahallelujah", "ofl/architectsdaughter", "ofl/handlee",
  "ofl/nanumpenscript", "ofl/reeniebeanie", "apache/homemadeapple", "ofl/dancingscript",
  "apache/rocksalt", "apache/justanotherhand", "ofl/coveredbyyourgrace",
  "ofl/sueellenfrancisco", "ofl/zeyada", "ofl/marckscript", "ofl/kristi",
]
```

- [ ] **Step 2: Write the fetcher**

```python
"""Download the OCR font families listed in configs/ocr-fonts.toml from
github.com/google/fonts, each with its licence file, and write data/ocr/fonts/fonts.json.

    .venv/Scripts/python scripts/ocr/fetch_fonts.py [--out data/ocr/fonts] [--ref <commit>]

--ref defaults to the current commit of google/fonts main, resolved once and recorded
in fonts.json, so a later run with --ref <that sha> fetches the same files. Uses the
GitHub contents API without a token (60 requests/hour is enough for ~50 families)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.github.com/repos/google/fonts"
RAW = "https://raw.githubusercontent.com/google/fonts"
LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt", "UFL.txt")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quipu-ocr-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def resolve_ref(ref: str | None) -> str:
    if ref:
        return ref
    return json.loads(_get(f"{API}/commits/main"))["sha"]


def fetch_family(family_dir: str, style: str, ref: str, out: Path) -> list[dict]:
    try:
        listing = json.loads(_get(f"{API}/contents/{family_dir}?ref={ref}"))
    except urllib.error.HTTPError as e:
        print(f"skip {family_dir}: HTTP {e.code}", file=sys.stderr)
        return []
    names = [e["name"] for e in listing if e["type"] == "file"]
    licence = next((n for n in LICENCE_NAMES if n in names), None)
    fonts = [n for n in names if n.lower().endswith(".ttf")]
    if licence is None or not fonts:
        print(f"skip {family_dir}: licence={licence} fonts={len(fonts)}", file=sys.stderr)
        return []
    # Upright faces only: italics are a different shape and the augmenter adds slant.
    fonts = [n for n in fonts if "italic" not in n.lower()] or fonts
    dest = out / family_dir
    dest.mkdir(parents=True, exist_ok=True)
    for name in [licence, *fonts]:
        (dest / name).write_bytes(_get(f"{RAW}/{ref}/{family_dir}/{name}"))
    return [{"family": family_dir.split("/")[1], "style": style,
             "path": f"{family_dir}/{n}", "licence": f"{family_dir}/{licence}",
             "sha256": hashlib.sha256((dest / n).read_bytes()).hexdigest()} for n in fonts]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "fonts"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "ocr-fonts.toml"))
    ap.add_argument("--ref", default=None)
    args = ap.parse_args(argv)
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    ref = resolve_ref(args.ref)
    entries: list[dict] = []
    for style in ("printed", "hand"):
        got = 0
        for family_dir in cfg[style]:
            fam = fetch_family(family_dir, style, ref, out)
            got += bool(fam)
            entries.extend(fam)
        print(f"{style}: {got} families")
        if got < cfg["min_per_style"]:
            print(f"only {got} {style} families, need {cfg['min_per_style']}", file=sys.stderr)
            return 1
    (out / "fonts.json").write_text(json.dumps({"repo": "google/fonts", "ref": ref,
                                                "fonts": entries}, indent=1), encoding="utf-8")
    print(f"{len(entries)} font files at {ref}; manifest {out / 'fonts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it (after the user's yes)**

Run: `.venv/Scripts/python scripts/ocr/fetch_fonts.py`
Expected: `printed: N families` and `hand: M families` with N, M ≥ 8, then `... font files at <sha>; manifest ...`. Skipped families are printed to stderr; that is fine.

- [ ] **Step 4: Commit** (the script and list only; `data/` is ignored)

```bash
git add configs/ocr-fonts.toml scripts/ocr/fetch_fonts.py
git commit -m "ocr: pinned Google Fonts list and fetcher with licence files"
```

---

### Task 6: Renderer (printed and hand) with licence check

**Files:**
- Create: `quipu/ocr_render.py`
- Create: `tests/ocr_fixtures.py`
- Test: `tests/test_ocr_render.py`

- [ ] **Step 1: Write the test fixture**

```python
"""A font directory for OCR tests, built from matplotlib's bundled DejaVu Sans so no
download is needed. The licence file is a stand-in; only its presence is checked."""
import json
import shutil
from pathlib import Path

import matplotlib

DEJAVU = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"


def make_font_dir(root: Path, families=(("printed", "alpha"), ("printed", "beta"),
                                        ("hand", "gamma"), ("hand", "delta"))) -> Path:
    entries = []
    for style, fam in families:
        d = root / "ofl" / fam
        d.mkdir(parents=True)
        shutil.copy(DEJAVU, d / f"{fam}.ttf")
        (d / "OFL.txt").write_text("test licence stand-in\n", encoding="utf-8")
        entries.append({"family": fam, "style": style, "path": f"ofl/{fam}/{fam}.ttf",
                        "licence": f"ofl/{fam}/OFL.txt", "sha256": "x"})
    (root / "fonts.json").write_text(json.dumps({"ref": "test", "fonts": entries}),
                                     encoding="utf-8")
    return root
```

- [ ] **Step 2: Write the failing tests**

```python
"""quipu/ocr_render.py: licence refusal, font split, rendering."""
import random
import shutil

import numpy as np
import pytest

from quipu import ocr_render as r
from tests.ocr_fixtures import DEJAVU, make_font_dir


def test_font_without_licence_is_refused(tmp_path):
    shutil.copy(DEJAVU, tmp_path / "x.ttf")
    with pytest.raises(r.FontLicenceError, match="no licence file"):
        r.find_licence(tmp_path / "x.ttf")
    font = r.Font("x", "printed", tmp_path / "x.ttf")
    with pytest.raises(r.FontLicenceError):
        r.render_line("hello", font, random.Random(0))


def test_load_fonts_reads_manifest(tmp_path):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    assert sorted((f.style, f.family) for f in fonts) == [
        ("hand", "delta"), ("hand", "gamma"), ("printed", "alpha"), ("printed", "beta")]


def test_split_fonts_keeps_every_style_on_both_sides(tmp_path):
    train, held = r.split_fonts(r.load_fonts(make_font_dir(tmp_path)))
    for style in ("printed", "hand"):
        assert any(f.style == style for f in train)
        assert any(f.style == style for f in held)
    assert not {f.family for f in train} & {f.family for f in held}


@pytest.mark.parametrize("style", ["printed", "hand"])
def test_render_line_is_grayscale_with_ink(tmp_path, style):
    font = next(f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == style)
    img = r.render_line("Hello, World 42", font, random.Random(1))
    assert img.mode == "L"
    arr = np.asarray(img)
    assert arr.min() < 128 < arr.max()          # dark ink on light paper
    assert img.width > img.height


def test_render_line_is_deterministic_per_seed(tmp_path):
    font = r.load_fonts(make_font_dir(tmp_path))[0]
    a = r.render_line("same seed", font, random.Random(7)).tobytes()
    b = r.render_line("same seed", font, random.Random(7)).tobytes()
    c = r.render_line("same seed", font, random.Random(8)).tobytes()
    assert a == b
    assert a != c
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.ocr_render'`

- [ ] **Step 4: Write the implementation**

```python
"""Render one line of text as a scan-like grayscale image (spec sections 4.2, 4.3).

Fonts come from the manifest that scripts/ocr/fetch_fonts.py writes; a font without a
licence file beside it is refused here, at render time, so no code path can use one.
"printed" draws characters with even spacing; "hand" adds per-character baseline,
size and spacing jitter and a per-line slant. Both then go through the same scan
augmentation. Everything random comes from the `rng` passed in, so a seed fixes the
image exactly."""
from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt", "UFL.txt")
STYLES = ("printed", "hand")


class FontLicenceError(Exception):
    pass


@dataclass(frozen=True)
class Font:
    family: str
    style: str
    path: Path


def find_licence(font_path: str | Path) -> Path:
    d = Path(font_path).parent
    for name in LICENCE_NAMES:
        if (d / name).is_file():
            return d / name
    raise FontLicenceError(f"{font_path}: no licence file ({', '.join(LICENCE_NAMES)}) "
                           "beside it; not used")


def load_fonts(font_dir: str | Path) -> list[Font]:
    font_dir = Path(font_dir)
    manifest = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))
    fonts = []
    for e in manifest["fonts"]:
        if e["style"] not in STYLES:
            raise ValueError(f"unknown style {e['style']!r} for {e['path']}")
        path = font_dir / e["path"]
        find_licence(path)
        fonts.append(Font(e["family"], e["style"], path))
    return fonts


def split_fonts(fonts: list[Font], held_every: int = 5) -> tuple[list[Font], list[Font]]:
    """Held-out fonts by family: sha256(family) % held_every == 0. Each style keeps at
    least one family on each side (the first family by name moves if needed)."""
    def held(fam: str) -> bool:
        return int(hashlib.sha256(fam.encode()).hexdigest(), 16) % held_every == 0

    held_fams = {f.family for f in fonts if held(f.family)}
    for style in STYLES:
        fams = sorted({f.family for f in fonts if f.style == style})
        if len(fams) < 2:
            raise ValueError(f"need at least 2 {style} families to split, have {len(fams)}")
        if not any(f in held_fams for f in fams):
            held_fams.add(fams[0])
        if all(f in held_fams for f in fams):
            held_fams.discard(fams[0])
    return ([f for f in fonts if f.family not in held_fams],
            [f for f in fonts if f.family in held_fams])


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    key = (str(path), size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return _FONT_CACHE[key]


def _draw(text: str, font: Font, size: int, bg: int, ink: int, rng: random.Random,
          hand: bool) -> Image.Image:
    spacing = rng.uniform(-0.5, 2.5)
    sizes = [max(8, round(size * rng.uniform(0.9, 1.1))) if hand else size for _ in text]
    advances = [_font(font.path, s).getlength(ch) for ch, s in zip(text, sizes)]
    width = int(sum(advances) + spacing * len(text) + 2 * size) + 16
    height = int(size * 2.2)
    img = Image.new("L", (width, height), bg)
    d = ImageDraw.Draw(img)
    x, base = float(size // 2), size * 0.5
    for ch, s, adv in zip(text, sizes, advances):
        dy = rng.gauss(0, size * 0.04) if hand else 0.0
        d.text((x, base + dy + (size - s) * 0.8), ch, font=_font(font.path, s), fill=ink)
        x += adv + spacing + (rng.gauss(0, size * 0.05) if hand else 0.0)
    if hand:
        img = _shear(img, rng.uniform(-0.3, 0.3), bg)
    return _crop_to_ink(img, bg, ink, rng)


def _shear(img: Image.Image, s: float, bg: int) -> Image.Image:
    w, h = img.size
    new_w = w + int(abs(s) * h)
    c = w / 2 - new_w / 2 - s * h / 2
    return img.transform((new_w, h), Image.Transform.AFFINE, (1, s, c, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=bg)


def _crop_to_ink(img: Image.Image, bg: int, ink: int, rng: random.Random) -> Image.Image:
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr < (bg + ink) / 2)
    if len(xs) == 0:
        return img
    mx, my = rng.randint(2, 16), rng.randint(3, 12)
    box = (max(0, xs.min() - mx), max(0, ys.min() - my),
           min(img.width, xs.max() + 1 + mx), min(img.height, ys.max() + 1 + my))
    return img.crop(box)


def _augment(img: Image.Image, bg: int, rng: random.Random) -> Image.Image:
    nrng = np.random.default_rng(rng.getrandbits(32))
    img = img.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BILINEAR, expand=True,
                     fillcolor=bg)
    if rng.random() < 0.3:
        w, h = img.size
        j = lambda n: rng.uniform(-0.03, 0.03) * n                      # noqa: E731
        quad = (j(w), j(h), j(w), h + j(h), w + j(w), h + j(h), w + j(w), j(h))
        img = img.transform(img.size, Image.Transform.QUAD, quad,
                            resample=Image.Resampling.BILINEAR, fillcolor=bg)
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.MinFilter(3))                       # ink bleed
    elif rng.random() < 0.15:
        img = img.filter(ImageFilter.MaxFilter(3))                       # ink erosion
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.3:                                               # paper texture
        h, w = arr.shape
        low = nrng.normal(0, 1, (max(2, h // 16), max(2, w // 16))).astype(np.float32)
        tex = np.asarray(Image.fromarray(low).resize((w, h), Image.Resampling.BICUBIC))
        arr = arr + tex * rng.uniform(4, 15)
    if rng.random() < 0.3:                                               # lighting gradient
        arr = arr + np.linspace(0, rng.uniform(-30, 30), arr.shape[1])[None, :]
    if rng.random() < 0.5:
        arr = arr + nrng.normal(0, rng.uniform(2, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(30, 95))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def render_line(text: str, font: Font, rng: random.Random) -> Image.Image:
    find_licence(font.path)
    size = rng.randint(28, 48)
    bg, ink = rng.randint(190, 255), rng.randint(0, 90)
    img = _draw(text, font, size, bg, ink, rng, hand=font.style == "hand")
    return _augment(img, bg, rng)
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_render.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add quipu/ocr_render.py tests/ocr_fixtures.py tests/test_ocr_render.py
git commit -m "ocr: line renderer (printed and hand) with font licence check"
```

---

### Task 7: Preprocess, collate and the training stream

**Files:**
- Create: `quipu/ocr_data.py`
- Test: `tests/test_ocr_data.py`

- [ ] **Step 1: Write the failing tests**

```python
"""quipu/ocr_data.py."""
import itertools

import torch
from PIL import Image

from quipu import ocr_ctc as oc
from quipu import ocr_render as r
from quipu.ocr_data import LineStream, collate, preprocess
from tests.ocr_fixtures import make_font_dir


def test_preprocess_resizes_to_64_and_pads_to_16():
    t = preprocess(Image.new("L", (300, 50), 255))
    assert t.shape == (1, 64, 384)               # 300*64/50 = 384
    t = preprocess(Image.new("L", (100, 64), 255))
    assert t.shape == (1, 64, 112)               # 100 padded to 112
    assert float(t.max()) == 0.0                 # white paper is 0, ink is 1


def test_preprocess_caps_width():
    assert preprocess(Image.new("L", (5000, 64), 0)).shape == (1, 64, 1024)


def test_collate_pads_and_masks():
    a = (torch.ones(1, 64, 32), "ab")
    b = (torch.ones(1, 64, 64), "abc")
    images, mask, texts = collate([a, b])
    assert images.shape == (2, 1, 64, 64)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert float(images[0, :, :, 32:].abs().sum()) == 0.0
    assert texts == ["ab", "abc"]


def test_line_stream_yields_fitting_lines(tmp_path):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    lines = ["hello world", "x" * 80, "naïve", "Quipu reads 42 lines"]
    stream = LineStream(lines, fonts, hand_frac=0.5, max_chars=40, seed=3)
    items = list(itertools.islice(iter(stream), 6))
    for img, text in items:
        assert text in ("hello world", "Quipu reads 42 lines")   # long and non-charset dropped
        assert img.shape[1] == 64
        assert oc.ctc_steps_needed(text) <= img.shape[-1] // 16
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quipu.ocr_data'`

- [ ] **Step 3: Write the implementation**

```python
"""OCR data: image preprocessing, batching, and the on-the-fly training stream.

Spec sections 3.1 and 4.5. Images become [1, 64, W] float tensors with ink 1 and
paper 0, W a multiple of 16 and at most 1024. A line is skipped (not truncated) when
it is too long for the encoder stage, has characters outside the set, or renders
narrower than its CTC path needs."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from quipu.ocr_ctc import ctc_steps_needed, in_charset
from quipu.ocr_render import Font, render_line

HEIGHT, MAX_WIDTH, PATCH = 64, 1024, 16


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    w = min(MAX_WIDTH, max(PATCH, round(img.width * HEIGHT / img.height)))
    img = img.resize((w, HEIGHT), Image.Resampling.BILINEAR)
    t = torch.from_numpy(1.0 - np.asarray(img, dtype=np.float32) / 255.0)[None]
    pad = (-w) % PATCH
    return F.pad(t, (0, pad)) if pad else t


def collate(items: list[tuple[torch.Tensor, str]]):
    widths = [t.shape[-1] for t, _ in items]
    W = max(widths)
    images = torch.zeros(len(items), 1, HEIGHT, W)
    for i, (t, _) in enumerate(items):
        images[i, :, :, : t.shape[-1]] = t
    cols = torch.tensor(widths) // PATCH
    mask = torch.arange(W // PATCH)[None, :] < cols[:, None]
    return images, mask, [s for _, s in items]


class LineStream(IterableDataset):
    """Endless (image, text) pairs: a random line, a random style (hand with
    probability hand_frac), a random font of that style, rendered and preprocessed."""

    def __init__(self, lines: list[str], fonts: list[Font], hand_frac: float,
                 max_chars: int, seed: int) -> None:
        self.lines = [s for s in lines if 0 < len(s) <= max_chars and in_charset(s)]
        if not self.lines:
            raise ValueError("no usable lines after filtering")
        self.by_style = {st: [f for f in fonts if f.style == st] for st in ("printed", "hand")}
        if hand_frac > 0 and not self.by_style["hand"]:
            raise ValueError("hand_frac > 0 but no hand fonts")
        self.hand_frac, self.seed = hand_frac, seed

    def __iter__(self):
        info = get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            text = rng.choice(self.lines)
            style = "hand" if rng.random() < self.hand_frac else "printed"
            img = preprocess(render_line(text, rng.choice(self.by_style[style]), rng))
            if ctc_steps_needed(text) <= img.shape[-1] // PATCH:
                yield img, text
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add quipu/ocr_data.py tests/test_ocr_data.py
git commit -m "ocr: preprocess, collate and on-the-fly line stream"
```

---

### Task 8: Text lines from Quipu's shards

**Files:**
- Create: `scripts/ocr/build_text.py`
- Test: `tests/test_ocr_build_text.py`

- [ ] **Step 1: Write the failing tests**

```python
"""scripts/ocr/build_text.py: cutting lines, random lines, the held-out split."""
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_text", ROOT / "scripts" / "ocr" / "build_text.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_cut_lines_respects_lengths_and_word_boundaries():
    doc = "The quick brown fox jumps over the lazy dog. " * 5
    lines = bt.cut_lines(doc, random.Random(0), min_chars=5, max_chars=20)
    assert lines
    for s in lines:
        assert 5 <= len(s) <= 20
        assert s == s.strip()
        assert all(w in doc for w in s.split())


def test_cut_lines_drops_characters_outside_the_set():
    lines = bt.cut_lines("café crème naïve plain words here", random.Random(0), 5, 60)
    assert all("é" not in s and "ï" not in s for s in lines)


def test_random_line_is_in_charset_and_bounded():
    from quipu.ocr_ctc import in_charset
    rng = random.Random(1)
    for _ in range(200):
        s = bt.random_line(rng, max_chars=40)
        assert 1 <= len(s) <= 40 and in_charset(s)


def test_is_held_is_stable_and_about_one_in_twenty():
    held = [bt.is_held(i) for i in range(20000)]
    assert held == [bt.is_held(i) for i in range(20000)]
    assert 0.03 < sum(held) / len(held) < 0.07


def test_build_writes_disjoint_files(tmp_path):
    docs = [f"document number {i} has some ordinary words in it for lines" for i in range(400)]
    bt.build(docs, tmp_path, random_frac=0.15, seed=0)
    train = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    held = (tmp_path / "held.txt").read_text(encoding="utf-8").splitlines()
    assert train and held
    assert not set(train) & set(held)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_build_text.py -v`
Expected: FAIL with `FileNotFoundError` for `scripts/ocr/build_text.py`

- [ ] **Step 3: Write the implementation**

```python
"""Cut OCR text lines from Quipu's token shards (spec section 4.1).

    .venv/Scripts/python scripts/ocr/build_text.py --shards data/shards-moe/train \
        --out data/ocr/text --lines 10000000

Documents are the EOT-separated runs of a shard, decoded with Quipu's tokenizer. Each
is cut into lines of 5-60 characters at word boundaries, keeping only characters in
the OCR set. About 15% of lines are random strings (codes, digits, punctuation) so a
model cannot read by guessing language. Every 20th document (by index hash) goes to
held.txt, the rest to train.txt; a line that lands in both is removed from train."""
from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_ctc import CHARSET, in_charset  # noqa: E402

MIN_CHARS, MAX_CHARS = 5, 60


def cut_lines(doc: str, rng: random.Random, min_chars: int = MIN_CHARS,
              max_chars: int = MAX_CHARS) -> list[str]:
    words = [w for w in doc.split() if in_charset(w)]
    lines, cur = [], []
    target = rng.randint(min_chars, max_chars)
    for w in words:
        if len(w) > max_chars:
            continue
        cand = " ".join(cur + [w])
        if len(cand) > target and cur:
            line = " ".join(cur)
            if len(line) >= min_chars:
                lines.append(line)
            cur, target = [w], rng.randint(min_chars, max_chars)
        else:
            cur.append(w)
    line = " ".join(cur)
    if min_chars <= len(line) <= max_chars:
        lines.append(line)
    return lines


_KINDS = (string.digits, string.ascii_uppercase + string.digits, CHARSET,
          string.ascii_letters + string.digits + "-_/.:#")


def random_line(rng: random.Random, max_chars: int = MAX_CHARS) -> str:
    alphabet = rng.choice(_KINDS)
    n = rng.randint(1, max_chars)
    s = "".join(rng.choice(alphabet) for _ in range(n)).strip()
    return s or rng.choice(string.digits)


def is_held(doc_index: int) -> bool:
    return int(hashlib.sha256(str(doc_index).encode()).hexdigest(), 16) % 20 == 0


def build(docs: Iterable[str], out: Path, random_frac: float, seed: int,
          max_lines: int | None = None) -> tuple[int, int]:
    rng = random.Random(seed)
    train, held = [], []
    for i, doc in enumerate(docs):
        dest = held if is_held(i) else train
        for line in cut_lines(doc, rng):
            dest.append(line)
            if rng.random() < random_frac / (1 - random_frac):
                dest.append(random_line(rng))
        if max_lines and len(train) + len(held) >= max_lines:
            break
    held_set = set(held)
    train = [s for s in train if s not in held_set]
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "held.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    return len(train), len(held)


def shard_documents(shard_dir: Path, tokenizer: str) -> Iterator[str]:
    from quipu.data import read_shard
    from quipu.tokenizer import make_tokenizer

    tok = make_tokenizer(tokenizer)
    for shard in sorted(shard_dir.glob("shard_*.bin")):
        ids = read_shard(shard)
        cuts = [-1, *(int(j) for j in (ids == tok.eot).nonzero()[0]), len(ids)]
        for a, b in zip(cuts, cuts[1:]):
            if b - a > 1:
                yield tok.decode(ids[a + 1:b].tolist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", default=str(ROOT / "data" / "shards-moe" / "train"))
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "text"))
    ap.add_argument("--lines", type=int, default=10_000_000)
    ap.add_argument("--random-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    n_train, n_held = build(shard_documents(Path(args.shards), args.tokenizer), Path(args.out),
                            args.random_frac, args.seed, args.lines)
    print(f"train {n_train:,} lines, held {n_held:,} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_build_text.py -v`
Expected: 5 passed

- [ ] **Step 5: Build the real text (local, CPU, a few minutes)**

Run: `.venv/Scripts/python scripts/ocr/build_text.py`
Expected: `train ~9,500,000 lines, held ~500,000 lines -> .../data/ocr/text`. The shards are fineweb-edu web text (the pretraining data), not model output.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/build_text.py tests/test_ocr_build_text.py
git commit -m "ocr: cut text lines from Quipu shards with a held-out split"
```

---

### Task 9: Sample CLI and fixed evaluation sets

**Files:**
- Create: `scripts/ocr/render_lines.py`
- Create: `scripts/ocr/eval_ocr.py` (the `make` command; `run` comes in Task 11)
- Test: `tests/test_ocr_eval.py`

- [ ] **Step 1: Write the sample CLI**

```python
"""Render sample lines to PNGs for a visual check of fonts and augmentation.

    .venv/Scripts/python scripts/ocr/render_lines.py --n 40 --out runs/ocr-samples"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_render import load_fonts, render_line  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fonts", default=str(ROOT / "data" / "ocr" / "fonts"))
    ap.add_argument("--text", default=str(ROOT / "data" / "ocr" / "text" / "held.txt"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "ocr-samples"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    fonts = load_fonts(args.fonts)
    lines = Path(args.text).read_text(encoding="utf-8").splitlines()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        font = fonts[i % len(fonts)]
        text = rng.choice(lines)
        render_line(text, font, rng).save(out / f"{i:03d}_{font.style}_{font.family}.png")
    print(f"{args.n} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing tests for `make`**

```python
"""scripts/ocr/eval_ocr.py: fixed eval sets."""
import importlib.util
from pathlib import Path

from tests.ocr_fixtures import make_font_dir

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_ocr", ROOT / "scripts" / "ocr" / "eval_ocr.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _make(tmp_path, seed=0):
    fonts = make_font_dir(tmp_path / "fonts")
    held = tmp_path / "held.txt"
    held.write_text("\n".join(f"held line number {i}" for i in range(50)) + "\n", encoding="utf-8")
    out = tmp_path / f"eval{seed}"
    ev.make_sets(held, fonts, out, n=12, seed=seed)
    return out


def test_make_sets_writes_three_sets_with_labels_and_sums(tmp_path):
    out = _make(tmp_path)
    for name in ("printed", "hand", "nonsense"):
        rows = ev.read_set(out / name)
        assert len(rows) == 12
        for png, text in rows:
            assert png.is_file() and text
    assert (out / "SHA256SUMS").is_file()
    ev.verify_sums(out)                       # does not raise


def test_make_sets_is_deterministic(tmp_path):
    a, b = _make(tmp_path, 0), _make(tmp_path / "again", 0)
    assert (a / "SHA256SUMS").read_text() == (b / "SHA256SUMS").read_text()


def test_verify_sums_catches_a_changed_file(tmp_path):
    import pytest
    out = _make(tmp_path)
    png, _ = ev.read_set(out / "printed")[0]
    png.write_bytes(png.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        ev.verify_sums(out)


def test_nonsense_words():
    import random
    w = ev.nonsense_line(random.Random(0))
    assert w and all(part.isalpha() for part in w.split())
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_eval.py -v`
Expected: FAIL with `FileNotFoundError` for `scripts/ocr/eval_ocr.py`

- [ ] **Step 4: Write `eval_ocr.py` with `make`**

```python
"""Quipu-OCR evaluation (spec section 6).

    .venv/Scripts/python scripts/ocr/eval_ocr.py make [--n 1000]
    .venv/Scripts/python scripts/ocr/eval_ocr.py run --ckpt runs/ocr-ctc/best.pt \
        [--baselines tesseract,trocr]

make: fixed sets from held-out text and held-out fonts (quipu.ocr_render.split_fonts):
printed/, hand/ and nonsense/ (random letters and made-up words, any held-out font).
Each set is PNGs plus labels.tsv; SHA256SUMS covers every file and `run` refuses a set
whose files changed."""
from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_ctc import in_charset  # noqa: E402
from quipu.ocr_render import load_fonts, render_line, split_fonts  # noqa: E402

SETS = ("printed", "hand", "nonsense")
MAX_CHARS = 40          # the encoder stage's limit, so the CTC model is scored fairly


def nonsense_line(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(4, 24)))
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiou"
    words = ["".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.randint(1, 4)))
             for _ in range(rng.randint(1, 5))]
    return " ".join(words)


def make_sets(held_text: Path, font_dir: Path, out: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    _, held_fonts = split_fonts(load_fonts(font_dir))
    lines = [s for s in held_text.read_text(encoding="utf-8").splitlines()
             if 0 < len(s) <= MAX_CHARS and in_charset(s)]
    for name in SETS:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        fonts = held_fonts if name == "nonsense" else [f for f in held_fonts if f.style == name]
        rows = []
        for i in range(n):
            text = nonsense_line(rng) if name == "nonsense" else rng.choice(lines)
            png = f"{i:05d}.png"
            render_line(text, rng.choice(fonts), rng).save(d / png)
            rows.append(f"{png}\t{text}")
        (d / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def verify_sums(out: Path) -> None:
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if hashlib.sha256((out / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{rel} changed since the eval set was made")


def read_set(d: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in (d / "labels.tsv").read_text(encoding="utf-8").splitlines():
        png, text = line.split("\t", 1)
        rows.append((d / png, text))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--text", default=str(ROOT / "data" / "ocr" / "text" / "held.txt"))
    mk.add_argument("--fonts", default=str(ROOT / "data" / "ocr" / "fonts"))
    mk.add_argument("--out", default=str(ROOT / "data" / "ocr" / "eval"))
    mk.add_argument("--n", type=int, default=1000)
    mk.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make_sets(Path(args.text), Path(args.fonts), Path(args.out), args.n, args.seed)
        print(f"eval sets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_eval.py -v`
Expected: 4 passed

- [ ] **Step 6: Make the real sets and look at samples**

Run: `.venv/Scripts/python scripts/ocr/render_lines.py` then `.venv/Scripts/python scripts/ocr/eval_ocr.py make`
Expected: 40 PNGs in `runs/ocr-samples/`; open a handful (printed and hand) and check the text is legible to a person, nothing is clipped, and the noise looks like a scan. Fix the renderer before going on if not. Then `eval sets -> .../data/ocr/eval`.

- [ ] **Step 7: Commit**

```bash
git add scripts/ocr/render_lines.py scripts/ocr/eval_ocr.py tests/test_ocr_eval.py
git commit -m "ocr: sample renderer CLI and fixed, hashed eval sets"
```

---

### Task 10: Encoder-stage training script

**Files:**
- Create: `configs/ocr-ctc.toml`, `configs/ocr-ctc-smoke.toml`
- Create: `scripts/ocr/train_ctc.py`
- Test: `tests/test_ocr_train_ctc.py`

- [ ] **Step 1: Write the configs**

`configs/ocr-ctc.toml`:

```toml
# Quipu-OCR encoder stage (spec section 5.1). Kaggle T4: precision "fp16".
[model]
d_model = 384
n_layer = 12
n_head  = 6
mlp     = 1536

[data]
text      = "data/ocr/text/train.txt"
fonts     = "data/ocr/fonts"
eval_dir  = "data/ocr/eval"
hand_frac = 0.3
max_chars = 40
workers   = 4

[train]
steps        = 60000
batch        = 128
lr           = 1e-3
warmup       = 1000
weight_decay = 0.05
eval_every   = 2000
eval_n       = 500        # lines per eval set during training (the full sets at the end)
patience     = 2          # evaluations without a better printed CER before stopping
precision    = "bf16"     # "bf16" | "fp16" | "fp32"
seed         = 0
ckpt_dir     = "runs/ocr-ctc"
```

`configs/ocr-ctc-smoke.toml`:

```toml
[model]
d_model = 32
n_layer = 2
n_head  = 4
mlp     = 64

[data]
text      = "PLACEHOLDER_SET_BY_TEST"
fonts     = "PLACEHOLDER_SET_BY_TEST"
eval_dir  = "PLACEHOLDER_SET_BY_TEST"
hand_frac = 0.3
max_chars = 40
workers   = 0

[train]
steps        = 6
batch        = 4
lr           = 1e-3
warmup       = 2
weight_decay = 0.05
eval_every   = 3
eval_n       = 4
patience     = 2
precision    = "fp32"
seed         = 0
ckpt_dir     = "PLACEHOLDER_SET_BY_TEST"
```

The smoke config's paths are overridden with `--set data.text=...` by the test; `PLACEHOLDER_SET_BY_TEST` is a literal value that makes a run fail fast if someone forgets.

- [ ] **Step 2: Write the failing smoke test**

```python
"""scripts/ocr/train_ctc.py end to end on CPU with the tiny config."""
import importlib.util
import json
from pathlib import Path

import torch

from tests.ocr_fixtures import make_font_dir

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_ctc", ROOT / "scripts" / "ocr" / "train_ctc.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)
_spec2 = importlib.util.spec_from_file_location("eval_ocr", ROOT / "scripts" / "ocr" / "eval_ocr.py")
ev = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ev)


def _setup(tmp_path):
    fonts = make_font_dir(tmp_path / "fonts")
    text = tmp_path / "train.txt"
    text.write_text("\n".join(f"line {i} of text" for i in range(100)) + "\n", encoding="utf-8")
    ev.make_sets(text, fonts, tmp_path / "eval", n=4, seed=0)
    return ["--config", str(ROOT / "configs" / "ocr-ctc-smoke.toml"),
            "--set", f"data.text={text}", "--set", f"data.fonts={fonts}",
            "--set", f"data.eval_dir={tmp_path / 'eval'}",
            "--set", f"train.ckpt_dir={tmp_path / 'run'}"]


def test_smoke_trains_checkpoints_and_resumes(tmp_path):
    args = _setup(tmp_path)
    assert tc.main(args) == 0
    run = tmp_path / "run"
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert ck["step"] == 6
    assert (run / "best.pt").is_file()
    log = [json.loads(s) for s in (run / "log.jsonl").read_text().splitlines()]
    assert any("cer_printed" in e for e in log)
    # resume with more steps: starts at 6, ends at 9
    assert tc.main(args + ["--set", "train.steps=9"]) == 0
    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 9


def test_placeholder_paths_fail_fast(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="PLACEHOLDER"):
        tc.main(["--config", str(ROOT / "configs" / "ocr-ctc-smoke.toml")])
```

- [ ] **Step 3: Run the test to see it fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_train_ctc.py -v`
Expected: FAIL with `FileNotFoundError` for `scripts/ocr/train_ctc.py`

- [ ] **Step 4: Write the training script**

```python
"""Train the Quipu-OCR encoder with a CTC head (spec section 5.1).

    .venv/Scripts/python scripts/ocr/train_ctc.py --config configs/ocr-ctc.toml \
        [--set train.precision=fp16] [--set train.steps=40000]

Single GPU (the model is ~22M parameters; a second T4 is not worth DDP here). Logs one
JSON line per eval to <ckpt_dir>/log.jsonl. Checkpoints: latest.pt every eval (resumed
automatically), best.pt whenever the printed-set CER improves. Stops early when that
CER has not improved for `patience` evals in a row. The data stream is seeded by
seed + step on resume, so a resumed run does not replay the same images."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from quipu.ocr_ctc import LineCTC, ctc_greedy_decode, ctc_loss  # noqa: E402
from quipu.ocr_data import LineStream, collate, preprocess  # noqa: E402
from quipu.ocr_metrics import cer  # noqa: E402
from quipu.ocr_render import load_fonts, split_fonts  # noqa: E402
from quipu.vision import ViTConfig  # noqa: E402
from PIL import Image  # noqa: E402

EVAL_SETS = ("printed", "hand", "nonsense")


def load_config(path: str, sets: list[str]) -> dict:
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for item in sets:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        old = cfg[section][name]
        cfg[section][name] = type(old)(value) if not isinstance(old, bool) else value == "true"
    for section in cfg.values():
        for k, v in section.items():
            if isinstance(v, str) and "PLACEHOLDER" in v:
                raise SystemExit(f"config value {k}={v!r}: PLACEHOLDER not replaced")
    return cfg


def lr_at(step: int, cfg: dict) -> float:
    t = cfg["train"]
    if step < t["warmup"]:
        return t["lr"] * (step + 1) / t["warmup"]
    p = (step - t["warmup"]) / max(1, t["steps"] - t["warmup"])
    return t["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, eval_dir: Path, n: int, device, autocast) -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "ocr"))
    from eval_ocr import read_set
    model.eval()
    out = {}
    for name in EVAL_SETS:
        rows = read_set(eval_dir / name)[:n]
        refs, hyps = [], []
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            images, mask, texts = collate([(preprocess(Image.open(p)), t) for p, t in chunk])
            with autocast():
                lp, lens = model(images.to(device), mask.to(device))
            hyps += ctc_greedy_decode(lp.float().cpu(), lens.cpu())
            refs += texts
        out[f"cer_{name}"] = cer(refs, hyps)
    model.train()
    return out


def save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    t, d = cfg["train"], cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[t["precision"]]
    autocast = (lambda: torch.autocast(device, dtype=dtype)) if dtype else (
        lambda: torch.autocast(device, enabled=False))
    scaler = torch.amp.GradScaler(device, enabled=t["precision"] == "fp16")

    ckpt_dir = Path(t["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vit = ViTConfig(**cfg["model"])
    model = LineCTC(vit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    step, best, bad = 0, float("inf"), 0
    if (ckpt_dir / "latest.pt").is_file():
        ck = torch.load(ckpt_dir / "latest.pt", weights_only=False, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, bad = ck["step"], ck["best_cer"], ck["bad_evals"]
        print(f"resumed at step {step} (best printed CER {best:.4f})")

    train_fonts, _ = split_fonts(load_fonts(d["fonts"]))
    lines = Path(d["text"]).read_text(encoding="utf-8").splitlines()
    stream = LineStream(lines, train_fonts, d["hand_frac"], d["max_chars"], t["seed"] + step)
    loader = iter(DataLoader(stream, batch_size=t["batch"], collate_fn=collate,
                             num_workers=d["workers"], persistent_workers=d["workers"] > 0))
    log = open(ckpt_dir / "log.jsonl", "a", encoding="utf-8")
    t0, seen = time.time(), 0
    model.train()
    while step < t["steps"]:
        images, mask, texts = next(loader)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        with autocast():
            lp, lens = model(images.to(device), mask.to(device))
        loss = ctc_loss(lp.float(), lens, texts)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += len(texts)
        if step % t["eval_every"] == 0 or step == t["steps"]:
            metrics = evaluate(model, Path(d["eval_dir"]), t["eval_n"], device, autocast)
            entry = {"step": step, "loss": round(loss.item(), 4),
                     "lines_per_s": round(seen / (time.time() - t0), 1), **metrics}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if metrics["cer_printed"] < best:
                best, bad = metrics["cer_printed"], 0
                save(ckpt_dir / "best.pt", {"model": model.state_dict(), "vit": vars(vit),
                                            "step": step, "cer_printed": best})
            else:
                bad += 1
            save(ckpt_dir / "latest.pt", {"model": model.state_dict(), "opt": opt.state_dict(),
                                          "scaler": scaler.state_dict(), "step": step,
                                          "best_cer": best, "bad_evals": bad, "vit": vars(vit)})
            if bad >= t["patience"]:
                print(f"stopping: printed CER has not improved for {bad} evals")
                break
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`vars(vit)` works on a frozen dataclass (it has a `__dict__`); `ViTConfig(**ck["vit"])` rebuilds it.

- [ ] **Step 5: Run the smoke test**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_train_ctc.py -v`
Expected: 2 passed

- [ ] **Step 6: Run the whole OCR test set**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_ctc.py tests/test_vision.py tests/test_ocr_render.py tests/test_ocr_data.py tests/test_ocr_metrics.py tests/test_ocr_build_text.py tests/test_ocr_eval.py tests/test_ocr_train_ctc.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add configs/ocr-ctc.toml configs/ocr-ctc-smoke.toml scripts/ocr/train_ctc.py tests/test_ocr_train_ctc.py
git commit -m "ocr: encoder-stage CTC training with eval, checkpoints, resume, early stop"
```

---

### Task 11: Evaluation run and baselines

**Files:**
- Modify: `scripts/ocr/eval_ocr.py` (add `run`)
- Test: `tests/test_ocr_eval.py` (append)

- [ ] **Step 1: Write the failing test** (append)

```python
def test_run_scores_a_ctc_checkpoint_and_writes_a_table(tmp_path):
    import torch
    from quipu.ocr_ctc import LineCTC
    from quipu.vision import ViTConfig
    out = _make(tmp_path)
    vit = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)
    ck = tmp_path / "best.pt"
    torch.save({"model": LineCTC(vit).state_dict(), "vit": vars(vit), "step": 1}, ck)
    table = tmp_path / "results.md"
    ev.main(["run", "--ckpt", str(ck), "--sets", str(out), "--out", str(table),
             "--baselines", ""])
    text = table.read_text(encoding="utf-8")
    assert "| quipu-ocr-ctc" in text
    for name in ("printed", "hand", "nonsense"):
        assert name in text
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_eval.py::test_run_scores_a_ctc_checkpoint_and_writes_a_table -v`
Expected: FAIL (argparse: invalid choice `run`)

- [ ] **Step 3: Add `run` to `eval_ocr.py`**

Add these functions above `main`:

```python
def predict_ctc(ckpt: Path, paths: list[Path]) -> list[str]:
    import torch
    from PIL import Image
    from quipu.ocr_ctc import LineCTC, ctc_greedy_decode
    from quipu.ocr_data import collate, preprocess
    from quipu.vision import ViTConfig

    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LineCTC(ViTConfig(**ck["vit"])).to(device).eval()
    model.load_state_dict(ck["model"])
    hyps = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            images, mask, _ = collate([(preprocess(Image.open(p)), "") for p in paths[i:i + 64]])
            lp, lens = model(images.to(device), mask.to(device))
            hyps += ctc_greedy_decode(lp.cpu(), lens.cpu())
    return hyps


def predict_tesseract(paths: list[Path]) -> list[str] | None:
    import shutil
    import subprocess
    exe = shutil.which("tesseract")
    if exe is None:
        print("tesseract not on PATH: skipped", file=sys.stderr)
        return None
    return [subprocess.run([exe, str(p), "stdout", "--psm", "7"], capture_output=True,
                           text=True, encoding="utf-8").stdout.strip() for p in paths]


def predict_trocr(paths: list[Path], style: str) -> list[str]:
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    name = "microsoft/trocr-small-handwritten" if style == "hand" else "microsoft/trocr-small-printed"
    proc = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).eval()
    hyps = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
            ids = model.generate(proc(images=imgs, return_tensors="pt").pixel_values,
                                 max_new_tokens=64)
            hyps += [s.strip() for s in proc.batch_decode(ids, skip_special_tokens=True)]
    return hyps


def run(ckpt: Path, sets: Path, out: Path, baselines: list[str]) -> str:
    from quipu.ocr_metrics import cer, exact_match, wer
    verify_sums(sets)
    rows = []
    for name in SETS:
        data = read_set(sets / name)
        paths, refs = [p for p, _ in data], [t for _, t in data]
        systems = {"quipu-ocr-ctc": predict_ctc(ckpt, paths)}
        if "tesseract" in baselines:
            hyp = predict_tesseract(paths)
            if hyp is not None:
                systems["tesseract"] = hyp
        if "trocr" in baselines:
            systems["trocr-small"] = predict_trocr(paths, "hand" if name == "hand" else "printed")
        for system, hyps in systems.items():
            rows.append(f"| {system} | {name} | {len(refs)} | {cer(refs, hyps):.4f} | "
                        f"{wer(refs, hyps):.4f} | {exact_match(refs, hyps):.3f} |")
    table = ("| system | set | lines | CER | WER | exact |\n|---|---|---|---|---|---|\n"
             + "\n".join(rows) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Checkpoint: `{ckpt}`. Sets: `{sets}` (SHA256SUMS verified). "
                   "Greedy decoding, no language-model rescoring.\n\n" + table, encoding="utf-8")
    return table
```

In `main`, add the subparser after `mk`:

```python
    rn = sub.add_parser("run")
    rn.add_argument("--ckpt", required=True)
    rn.add_argument("--sets", default=str(ROOT / "data" / "ocr" / "eval"))
    rn.add_argument("--out", default=str(ROOT / "results" / "ocr-encoder-stage.md"))
    rn.add_argument("--baselines", default="tesseract,trocr")
```

and after the `make` branch:

```python
    if args.cmd == "run":
        print(run(Path(args.ckpt), Path(args.sets), Path(args.out),
                  [b for b in args.baselines.split(",") if b]))
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_eval.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/eval_ocr.py tests/test_ocr_eval.py
git commit -m "ocr: eval run with Tesseract and TrOCR-small baselines"
```

---

### Task 12: Handwriting dataset licence review

**Files:**
- Create: `docs/ocr/handwriting-licences.md`

This is reading, not code. For each candidate (IAM, GNHK, CVL, RIMES, Bentham, Imgur5K, and any line-level set found while looking), open the dataset's own licence or terms page and record:

- [ ] **Step 1: Fill in one row per dataset**

```markdown
# Real handwriting datasets: licence review (spec section 4.4)

Rule: a set is used for TRAINING only if its terms allow releasing a model trained on
it with public weights. Otherwise EVALUATION only, if its terms allow that, or not used.

| Set | Line-level? | Licence / terms (link) | Registration | Train? | Eval? | Notes |
|---|---|---|---|---|---|---|
| IAM | yes | <terms page URL, quoted key sentence under 15 words> | yes | ... | ... | ... |
```

Each "Train?/Eval?" cell is yes or no with the sentence of the terms that decides it. No guessing: if the terms are unclear, the answer is no and the note says why.

- [ ] **Step 2: Commit**

```bash
git add docs/ocr/handwriting-licences.md
git commit -m "docs: handwriting dataset licence review for Quipu-OCR"
```

If a set passes for evaluation, add a `real_hand` set to `eval_ocr.py make` in a follow-up commit (its own labels.tsv from the dataset's line images, test split only, listed in SHA256SUMS). If one passes for training, a follow-up adds it to `LineStream` as a third source.

---

### Task 13: Real training run on Kaggle and results

**Files:**
- Create: `results/ocr-encoder-stage.md` (written by `eval_ocr.py run`)

Runs on Kaggle's free 2× T4 using the existing runbook `docs/superpowers/specs/2026-10-20-kaggle-t4-runbook.md` (branch clone, no pip install). Steps that leave the laptop need the user:

- [ ] **Step 1: Ask the user** to (a) approve pushing the branch so Kaggle can clone it, and (b) upload `data/ocr/text/` (about 400 MB), `data/ocr/fonts/` and `data/ocr/eval/` as a private Kaggle dataset, or approve the executor doing it. Nothing is pushed or uploaded before that yes.

- [ ] **Step 2: On Kaggle**, with the dataset mounted at `/kaggle/input/quipu-ocr`:

```bash
python scripts/ocr/train_ctc.py --config configs/ocr-ctc.toml \
  --set train.precision=fp16 \
  --set data.text=/kaggle/input/quipu-ocr/text/train.txt \
  --set data.fonts=/kaggle/input/quipu-ocr/fonts \
  --set data.eval_dir=/kaggle/input/quipu-ocr/eval \
  --set train.ckpt_dir=/kaggle/working/ocr-ctc
```

Expected: one JSON line per 2,000 steps with falling `cer_printed`. Kaggle sessions cap at 12 hours; if it stops, rerun the same command in a new session with the previous `/kaggle/working/ocr-ctc` restored (it resumes from `latest.pt`).

- [ ] **Step 3: Score and write the results** (on Kaggle or locally with `best.pt` downloaded)

```bash
python scripts/ocr/eval_ocr.py run --ckpt runs/ocr-ctc/best.pt --baselines tesseract,trocr
```

TrOCR-small downloads about 250 MB per model the first time: tell the user before running locally. Expected: a table with `quipu-ocr-ctc`, `tesseract` (if installed) and `trocr-small` rows for printed, hand and nonsense, written to `results/ocr-encoder-stage.md`.

- [ ] **Step 4: Commit the results**

```bash
git add results/ocr-encoder-stage.md
git commit -m "results: Quipu-OCR encoder stage CER/WER vs Tesseract and TrOCR-small"
```

These numbers decide the joint-stage plan: if printed CER is far above TrOCR-small's, fix the encoder (more steps, data, or size) before attaching it to Quipu.
