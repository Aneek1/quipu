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
