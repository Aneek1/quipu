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
