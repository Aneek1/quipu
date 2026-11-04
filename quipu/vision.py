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
