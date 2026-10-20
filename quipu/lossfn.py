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
