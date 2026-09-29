"""quipu-moe: the quipu transformer with a routed MoE feed-forward (spec section 3).

    embed -> n_layer x [RMSNorm -> Attention -> RMSNorm -> MoELayer] -> RMSNorm -> tied lm_head

Attention, RMSNorm and RoPE are quipu.model's own; the feed-forward is
quipu.moe.MoELayer. Depth mixing is a config switch (model.attnres_blocks):

- 0: a plain pre-norm residual, exactly quipu.model.Block with the SwiGLU swapped for
  the MoE layer:  x = x + attn(norm1(x));  x = x + moe(norm2(x)).
- N > 0: Block Attention Residuals (quipu.attnres). As in the reference, every
  SUB-LAYER is its own AttnRes step with its own pseudo-query, so there are 2 *
  n_layer steps (plus the head's query):
        step 2l:     h = mix();  a = attn(norm1(h));  partial += a
        step 2l + 1: h = mix();  m = moe(norm2(h));   partial += m
  Each block holds n_layer / N whole layers (2 * n_layer / N sub-layers), so
  n_layer must divide by N. With zero pseudo-queries (the init) every step input is
  the plain residual stream divided by the number of sources, and the norms are
  scale-invariant, so the model starts as the plain residual model.
  model.attnres_checkpoint recomputes the depth mix in backward while training.

Batch dependence: AttnRes and attention are per position / causal, but with
moe_dispatch "padded" each expert keeps only its first `capacity` assignments of
the whole flattened batch (Switch-style): which tokens are dropped depends on the
rest of the batch, later rows and later tokens drop more, and a row's logits can
change when other rows change. Inside one row earlier positions never depend on
later ones. Evaluation and generation should run with loop dispatch
(set_dispatch("loop"), or build with moe_dispatch "loop").

After every forward, last_stats holds one quipu.moe.MoEStats per layer (a fixed
slot per layer, overwritten, so a checkpoint recompute never adds entries); after
its optimizer step the trainer calls update_balance() to move every layer's Quantile
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


class QuipuMoE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.kind != "moe":
            raise ValueError(f"QuipuMoE needs kind 'moe', got {cfg.kind!r}")
        if cfg.attnres_blocks and cfg.n_layer % cfg.attnres_blocks != 0:
            raise ValueError(f"n_layer ({cfg.n_layer}) must divide evenly into "
                             f"attnres_blocks ({cfg.attnres_blocks})")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(MoEBlock(cfg) for _ in range(cfg.n_layer))
        self.attnres = (
            BlockAttnRes(cfg.d_model, 2 * cfg.n_layer, cfg.attnres_blocks, cfg.norm_eps,
                         checkpoint=cfg.attnres_checkpoint)
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

        self.last_stats: list[MoEStats | None] = [None] * cfg.n_layer

    def _sublayer(self, i: int, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """AttnRes step i: attention of layer i // 2 (even i) or its MoE (odd i).
        Returns the sub-layer's output in h's dtype; the MoE step records its stats
        in layer i // 2's slot."""
        block = self.blocks[i // 2]
        if i % 2 == 0:
            return block.attn(block.norm1(h), cos, sin).to(h.dtype)
        m, stats = block.moe(block.norm2(h))
        self.last_stats[i // 2] = stats
        return m.to(h.dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.shape[1] <= self.cfg.context, (
            f"sequence length {idx.shape[1]} exceeds context {self.cfg.context}"
        )
        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        self.last_stats = [None] * self.cfg.n_layer
        if self.attnres is None:
            for l, block in enumerate(self.blocks):
                x, self.last_stats[l] = block(x, cos, sin)
        else:
            x = self.attnres(x, lambda i, h: self._sublayer(i, h, cos, sin))
        return self.lm_head(self.norm(x))

    def update_balance(self) -> None:
        """One Quantile Balancing step in every layer, from its latest forward."""
        for block in self.blocks:
            block.moe.update_balance()

    def set_dispatch(self, mode: str) -> None:
        """Switch every layer's routed-expert dispatch ("loop" | "padded"). Evaluation
        and generation should use "loop": padded drops depend on the whole batch."""
        for block in self.blocks:
            block.moe.dispatch = mode

    def no_decay_param_names(self) -> list[str]:
        """Names of the 1-D parameters (every RMSNorm scale and every AttnRes
        pseudo-query, attnres.queries.<i>), which an optimizer should not decay.
        The Quantile Balancing bias is a buffer, not a parameter, so it is not here."""
        return [n for n, p in self.named_parameters() if p.ndim <= 1]
