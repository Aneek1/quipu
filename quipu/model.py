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
    """Rotate (B, H, T, D) by position. Splits the head in halves, GPT-NeoX style.
    Computed in at least fp32 and cast back, so bf16 q/k are rotated with full-precision angles."""
    assert cos.shape[-1] == x.shape[-1] // 2, "RoPE cache width must be head_dim // 2"
    compute = torch.promote_types(x.dtype, torch.float32)
    x1, x2 = x.to(compute).chunk(2, dim=-1)
    cos, sin = cos.to(compute), sin.to(compute)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    """SiLU-gated feed-forward. Three matrices, no biases."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


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

        # is_causal=True is the mask. enable_gqa=True would materialise the repeat
        # for free in principle, but on this build (Windows torch 2.11 cu128, sm_120)
        # there is no flash-attention kernel and the memory-efficient kernel doesn't
        # accept enable_gqa, so SDPA silently falls back to the MATH kernel: far more
        # memory and much slower. Expanding K/V ourselves with repeat_interleave (so
        # query head h reads KV head h // rep, matching GQA grouping) lets SDPA pick
        # the memory-efficient kernel instead.
        rep = self.n_head // self.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        assert idx.shape[1] <= self.cfg.context, (
            f"sequence length {idx.shape[1]} exceeds context {self.cfg.context}"
        )
        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm(x))
