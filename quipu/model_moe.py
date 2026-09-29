"""quipu-moe: the quipu transformer with a routed MoE feed-forward (spec section 3).

    embed -> n_layer x [RMSNorm -> Attention -> RMSNorm -> MoELayer] -> RMSNorm -> tied lm_head

Attention, RMSNorm and RoPE are quipu.model's own; the feed-forward is
quipu.moe.MoELayer. Depth mixing is a config switch (model.attnres_blocks):

- 0: a plain pre-norm residual, exactly quipu.model.Block with the SwiGLU swapped for
  the MoE layer:  x = x + attn(norm1(x));  x = x + moe(norm2(x)).
- N > 0: Block Attention Residuals (quipu.attnres) with N blocks of n_layer / N
  layers. An AttnRes "layer" is one attention sublayer AND the MoE sublayer after
  it: AttnRes forms the input h_l of the pair, the pair runs as an ordinary pre-norm
  block on it (the MoE sublayer reads h_l + a through the usual residual inside the
  pair), and the pair's output is its residual update
        f_l(h_l) = a + m,   a = attn(norm1(h_l)),   m = moe(norm2(h_l + a)).
  Block sums, and the depth softmax, are over these f_l. This is the report's
  scheme with the layer taken as a whole transformer layer, which keeps the number
  of AttnRes steps equal to n_layer and the blocks equal to groups of layers.

After every forward, last_stats holds one quipu.moe.MoEStats per layer; after its
optimizer step the trainer calls update_balance() to move every layer's Quantile
Balancing bias.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from quipu.attnres import BlockAttnRes
from quipu.config import ModelConfig
from quipu.model import Attention, Quipu, RMSNorm, build_rope_cache
from quipu.moe import MoELayer, MoEStats


class MoEBlock(nn.Module):
    """One layer: pre-norm attention then pre-norm MoE."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        """Plain pre-norm residual: returns (x + a + m, stats)."""
        x = x + self.attn(self.norm1(x), cos, sin)
        m, stats = self.moe(self.norm2(x))
        return x + m, stats

    def update(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        """The residual update f(h) = a + m for Block AttnRes, in h's dtype."""
        a = self.attn(self.norm1(h), cos, sin).to(h.dtype)
        m, stats = self.moe(self.norm2(h + a))
        return a + m.to(h.dtype), stats


class QuipuMoE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.kind != "moe":
            raise ValueError(f"QuipuMoE needs kind 'moe', got {cfg.kind!r}")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg.n_layer))
        self.attnres = (
            BlockAttnRes(cfg.d_model, cfg.n_layer, cfg.attnres_blocks, cfg.norm_eps)
            if cfg.attnres_blocks else None
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tied, as in quipu.model.Quipu

        cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # quipu.model's init: normal(0, 0.02) for Linear/Embedding, residual
        # projections scaled by depth. MoELayer has already initialised its router and
        # expert bank (plain Parameters, untouched by _init); its shared experts are
        # Linear, so they are re-drawn here and their down projections re-scaled.
        # AttnRes pseudo-queries stay 0.
        self.apply(Quipu._init)
        for name, p in self.named_parameters():
            if name.endswith("o.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

        self.last_stats: list[MoEStats] = []

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.shape[1] <= self.cfg.context, (
            f"sequence length {idx.shape[1]} exceeds context {self.cfg.context}"
        )
        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        stats: list[MoEStats] = []
        if self.attnres is None:
            for block in self.blocks:
                x, s = block(x, cos, sin)
                stats.append(s)
        else:
            def step(l: int, h: torch.Tensor) -> torch.Tensor:
                f, s = self.blocks[l].update(h, cos, sin)
                stats.append(s)
                return f
            x = self.attnres(x, step)
        self.last_stats = stats
        return self.lm_head(self.norm(x))

    def update_balance(self) -> None:
        """One Quantile Balancing step in every layer, from its latest forward."""
        for block in self.blocks:
            block.moe.update_balance()
