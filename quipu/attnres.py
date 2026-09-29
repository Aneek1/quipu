"""Block Attention Residuals (Kimi K3 report; spec section 3, "Depth mixing").

A plain residual stream feeds sub-layer i the running sum embedding + f_1 + ... +
f_{i-1} with every term weighted 1. Block AttnRes replaces that fixed sum with a
learned, input-dependent softmax over a short list of depth "sources".

The unit of depth is a SUB-LAYER, as in the reference: in a transformer every
attention sub-layer and every feed-forward (MoE) sub-layer is its own AttnRes step,
with its own pseudo-query, and adds its own output to the current block's partial
sum. With n_steps sub-layers cut into n_blocks equal blocks:

- b_0 is the token embedding;
- inside block n the sub-layer outputs are summed, b_n^i = f_(first) + ... +
  f_(i-th sub-layer of block n), and the completed block n is represented by its
  full sum b_n;
- the first sub-layer of block n attends over [b_0, b_1, ..., b_{n-1}]; every later
  sub-layer of block n attends over [b_0, ..., b_{n-1}, b_n^{i-1}] (the partial sum
  of the block so far);
- the output head attends over every block representation [b_0, b_1, ..., b_N],
  with its own pseudo-query.

The weights for step i are alpha_j = softmax_j( w_i . RMSNorm(v_j) ) with w_i a
learnable pseudo-query of width d_model, initialised to 0 (so at init every source
counts equally), and the step input is h_i = sum_j alpha_j v_j with the values
un-normalised. The key RMSNorm has no learnable scale: a scale g would only ever
appear as w_i . (g * k), which the pseudo-query already covers.

Why sub-layers and not whole layers: with zero queries h_i is the MEAN of its
sources, which is the plain residual stream divided by the number of sources m.
Every sub-layer reads its input through a scale-invariant RMSNorm, so at init the
model computes what the plain pre-norm residual model computes (up to the norm's
eps). Pairing attention and feed-forward into one step would instead feed the
feed-forward norm(h + a), with h the stream / m but a at full scale: a would be
over-weighted by up to m.

Causality: the softmax runs over SOURCES (depth), separately at every token
position, so AttnRes itself mixes nothing across positions. Whether a whole model
is causal is a separate matter: quipu.moe's padded dispatch drops tokens depending
on the rest of the batch (see MoELayer).

Memory: the mix never stacks the sources or materialises normalised keys.
    logit_j = (v_j . w) * rsqrt(mean(v_j^2) + eps)
is computed in fp32 from a per-position dot product and a per-position inverse RMS;
the inverse RMS of each completed block is computed once, when the block completes,
and reused by every later step; the weighted sum is accumulated source by source
(addcmul). Autograd therefore keeps only the sources themselves (which the forward
holds anyway) and per-position scalars. With checkpoint=True the mix is also
recomputed in backward (torch.utils.checkpoint, non-reentrant) while training.

Mixing and the running sums are in fp32 (the dtype of the embedding under
autocast), like the plain residual stream of quipu.model.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# step(i, h) -> f_i(h): sub-layer i's residual update (its output, without h added back).
StepFn = Callable[[int, torch.Tensor], torch.Tensor]


class BlockAttnRes(nn.Module):
    """Depth mixing over block representations for n_steps sub-layers in n_blocks blocks.

    Holds n_steps + 1 pseudo-queries: queries[i] forms step i's input and
    queries[n_steps] forms the output head's input. Each is a separate 1-D parameter
    (not rows of one matrix), so a matrix optimizer never treats them as a weight
    matrix; all are named attnres.queries.<i> for optimizer grouping. queries[0]
    never receives a gradient: the first step has only one source (b_0), and a
    softmax over one element is constant 1."""

    def __init__(self, dim: int, n_steps: int, n_blocks: int, eps: float = 1e-6,
                 checkpoint: bool = False) -> None:
        super().__init__()
        if n_blocks < 1 or n_steps < 1 or n_steps % n_blocks != 0:
            raise ValueError(f"n_steps ({n_steps}) must divide evenly into n_blocks "
                             f"({n_blocks}), both positive")
        self.dim = dim
        self.n_steps = n_steps
        self.n_blocks = n_blocks
        self.steps_per_block = n_steps // n_blocks
        self.eps = eps
        self.checkpoint = bool(checkpoint)
        self.queries = nn.ParameterList(
            nn.Parameter(torch.zeros(dim)) for _ in range(n_steps + 1)
        )

    def inv_rms(self, v: torch.Tensor) -> torch.Tensor:
        """rsqrt(mean(v^2) + eps) over the last dim, in fp32; shape v.shape[:-1]."""
        with torch.autocast(device_type=v.device.type, enabled=False):
            return torch.rsqrt(v.float().pow(2).mean(-1) + self.eps)

    def mix(self, index: int, sources: list[torch.Tensor],
            inv_rms: list[torch.Tensor | None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax-weighted sum of sources with pseudo-query `index`.

        sources: m tensors (..., d) of one dtype. inv_rms: optional cached
        self.inv_rms(source), one per source; missing or None entries are computed
        here. Returns (h, alpha): h (..., d) in the sources' dtype, alpha (m, ...)
        fp32 summing to 1 over the first dim."""
        caches = list(inv_rms or [])
        caches += [None] * (len(sources) - len(caches))
        with torch.autocast(device_type=sources[0].device.type, enabled=False):
            w = self.queries[index].float()
            vs = [v.float() for v in sources]
            rs = [self.inv_rms(v) if c is None else c for v, c in zip(vs, caches)]
            logits = torch.stack([(v @ w) * r for v, r in zip(vs, rs)])   # (m, ...)
            alpha = logits.softmax(dim=0)
            h = vs[0] * alpha[0].unsqueeze(-1)
            for j in range(1, len(vs)):
                h = torch.addcmul(h, vs[j], alpha[j].unsqueeze(-1))
        return h.to(sources[0].dtype), alpha

    def _mix_h(self, index: int, n_sources: int, *tensors: torch.Tensor) -> torch.Tensor:
        """mix() over tensors = sources + the leading sources' cached inverse RMS."""
        return self.mix(index, list(tensors[:n_sources]), list(tensors[n_sources:]))[0]

    def _input(self, index: int, sources: list[torch.Tensor],
               caches: list[torch.Tensor]) -> torch.Tensor:
        """Step `index`'s input. caches covers the completed blocks (a prefix of
        sources); the partial sum's inverse RMS, if any, is computed inside."""
        if self.checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._mix_h, index, len(sources), *sources, *caches,
                              use_reentrant=False)
        return self.mix(index, sources, caches)[0]

    def forward(self, x0: torch.Tensor, step: StepFn) -> torch.Tensor:
        """Run all n_steps sub-layers with AttnRes inputs; return the output head's
        input (before the final norm). x0 is the token embedding, b_0."""
        blocks: list[torch.Tensor] = [x0]
        block_rms: list[torch.Tensor] = [self.inv_rms(x0)]
        partial: torch.Tensor | None = None
        for i in range(self.n_steps):
            sources = blocks if partial is None else blocks + [partial]
            h = self._input(i, sources, block_rms)
            out = step(i, h).to(x0.dtype)
            partial = out if partial is None else partial + out
            if (i + 1) % self.steps_per_block == 0:
                blocks.append(partial)
                block_rms.append(self.inv_rms(partial))
                partial = None
        return self._input(self.n_steps, blocks, block_rms)
