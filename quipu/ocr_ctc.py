"""Character set, CTC head and greedy decoding for the OCR encoder stage.

Spec docs/superpowers/specs/2026-11-04-quipu-ocr-line-design.md, section 3.3. Class 0
is the CTC blank; character i of CHARSET is class i + 1. The set is fixed here so the
trainer, the data filter and the evaluation can never disagree about it."""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from quipu.vision import LineViT, ViTConfig

EXTRA = "°£€×–‘’"
CHARSET = "".join(chr(c) for c in range(32, 127)) + EXTRA
BLANK = 0
N_CLASSES = len(CHARSET) + 1
STEPS_PER_COL = 2          # CTC steps emitted per 16-px patch column
_INDEX = {ch: i + 1 for i, ch in enumerate(CHARSET)}
CHARSET_SHA256 = hashlib.sha256(CHARSET.encode("utf-8")).hexdigest()


def in_charset(text: str) -> bool:
    return all(ch in _INDEX for ch in text)


def encode_text(text: str) -> list[int]:
    try:
        return [_INDEX[ch] for ch in text]
    except KeyError as e:
        raise ValueError(f"character {e.args[0]!r} is not in the OCR character set") from None


def ckpt_compat() -> dict:
    """Fields stored in every checkpoint so a model is never read with another charset
    or another number of CTC steps per column."""
    return {"charset_sha256": CHARSET_SHA256, "steps_per_col": STEPS_PER_COL}


def check_ckpt_compat(ck: dict, where: object) -> None:
    if ck.get("charset_sha256") != CHARSET_SHA256:
        raise ValueError(f"{where}: checkpoint charset sha256 {str(ck.get('charset_sha256'))[:12]} "
                         f"does not match this code's ({CHARSET_SHA256[:12]})")
    if ck.get("steps_per_col") != STEPS_PER_COL:
        raise ValueError(f"{where}: checkpoint has {ck.get('steps_per_col')} CTC steps per "
                         f"column, this code {STEPS_PER_COL}")


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


class CTCHead(nn.Module):
    """Mean over the 4 patch rows of each column, then a linear layer to
    STEPS_PER_COL x classes: column i gives CTC steps 2i and 2i+1, so a step covers 8 px.

    col_mask must be a prefix mask (real columns first, padding after): the returned
    lengths are col_mask.sum(1) * STEPS_PER_COL and the loss and decoder read the first
    `length` steps."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, N_CLASSES * STEPS_PER_COL)

    def forward(self, feats: torch.Tensor, col_mask: torch.Tensor):
        """feats [B, R, C, D], col_mask [B, C] -> (log_probs [C*S, B, K] float32, lengths [B])."""
        logits = self.proj(feats.mean(dim=1)).float()
        B, C, _ = logits.shape
        logits = logits.reshape(B, C * STEPS_PER_COL, N_CLASSES)
        return logits.log_softmax(-1).transpose(0, 1), col_mask.sum(1) * STEPS_PER_COL


class LineCTC(nn.Module):
    """ViT encoder plus CTC head. col_mask must be a bool prefix mask (real columns
    first, padding after)."""

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.encoder = LineViT(cfg)
        self.head = CTCHead(cfg.d_model)

    def forward(self, images: torch.Tensor, col_mask: torch.Tensor):
        return self.head(self.encoder(images, col_mask), col_mask)


def ctc_loss(log_probs: torch.Tensor, lengths: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """Mean CTC loss over the batch (PyTorch divides each item by its target length)."""
    for i, (t, n) in enumerate(zip(texts, lengths.tolist())):
        need = ctc_steps_needed(t)
        if need > n:
            raise ValueError(f"item {i}: text needs {need} CTC steps but has only {n}")
    targets = [encode_text(t) for t in texts]
    flat = torch.tensor([k for t in targets for k in t], dtype=torch.long,
                        device=log_probs.device)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return F.ctc_loss(log_probs, flat, lengths.cpu(), tgt_lens, blank=BLANK,
                      reduction="mean", zero_infinity=True)
