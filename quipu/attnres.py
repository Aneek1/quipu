"""Block Attention Residuals (Kimi K3 report; spec section 3, "Depth mixing").

A plain residual stream feeds layer l the running sum embedding + f_1 + ... + f_{l-1}
with every term weighted 1. Block AttnRes replaces that fixed sum with a learned,
input-dependent softmax over a short list of depth "sources":

- b_0 is the token embedding;
- the n_layer layers are cut into n_blocks equal blocks; inside block n the layer
  outputs are summed, b_n^i = f_(first) + ... + f_(i-th layer of block n), and the
  completed block n is represented by its full sum b_n;
- the first layer of block n attends over [b_0, b_1, ..., b_{n-1}]; every later
  layer of block n attends over [b_0, ..., b_{n-1}, b_n^{i-1}] (the partial sum of
  the block so far);
- the output head attends over every block representation [b_0, b_1, ..., b_N],
  with its own pseudo-query.

The weights for layer l are alpha_i = softmax_i( w_l . RMSNorm(v_i) ) with w_l a
learnable pseudo-query of width d_model, initialised to 0 (so at init every source
counts equally), and the layer input is h_l = sum_i alpha_i v_i with the values
un-normalised. The softmax runs over SOURCES (depth), separately at every token
position, so AttnRes mixes nothing across positions and cannot break causality.

The key RMSNorm has no learnable scale: a scale g would only ever appear as
w_l . (g * k), which the pseudo-query already covers.

"Layer" here is whatever the caller's step function computes; quipu.model_moe uses
one attention sublayer plus one MoE sublayer (see QuipuMoE). Mixing and the running
sums are in fp32 (the dtype of the embedding under autocast), like the plain
residual stream of quipu.model.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# step(l, h) -> f_l(h): layer l's residual update (its output, without h added back).
StepFn = Callable[[int, torch.Tensor], torch.Tensor]


class BlockAttnRes(nn.Module):
    """Depth mixing over block representations for n_layer layers in n_blocks blocks.

    Holds n_layer + 1 pseudo-queries: queries[l] forms layer l's input and
    queries[n_layer] forms the output head's input. Each is a separate 1-D parameter
    (not rows of one matrix), so a matrix optimizer never treats them as a weight
    matrix. queries[0] never receives a gradient: the first layer has only one
    source (b_0), and a softmax over one element is constant 1."""

    def __init__(self, dim: int, n_layer: int, n_blocks: int, eps: float = 1e-6) -> None:
        super().__init__()
        if n_blocks < 1 or n_layer < 1 or n_layer % n_blocks != 0:
            raise ValueError(f"n_layer ({n_layer}) must divide evenly into n_blocks "
                             f"({n_blocks}), both positive")
        self.dim = dim
        self.n_layer = n_layer
        self.n_blocks = n_blocks
        self.layers_per_block = n_layer // n_blocks
        self.eps = eps
        self.queries = nn.ParameterList(
            nn.Parameter(torch.zeros(dim)) for _ in range(n_layer + 1)
        )

    def mix(self, index: int, sources: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax-weighted sum of sources with pseudo-query `index`.

        sources: m tensors (..., d) of one dtype. Returns (h, alpha): h (..., d) in
        the sources' dtype, alpha (m, ...) fp32 summing to 1 over the first dim."""
        v = torch.stack(sources, dim=0)                                  # (m, ..., d)
        with torch.autocast(device_type=v.device.type, enabled=False):
            k = F.rms_norm(v.float(), (self.dim,), eps=self.eps)
            logits = k @ self.queries[index].float()                     # (m, ...)
            alpha = logits.softmax(dim=0)
            h = (alpha.unsqueeze(-1) * v.float()).sum(dim=0)
        return h.to(v.dtype), alpha

    def sources(self, blocks: list[torch.Tensor], partial: torch.Tensor | None) -> list[torch.Tensor]:
        """The keys/values for the next layer: every completed block representation
        (b_0 first), plus the current block's partial sum once it has one."""
        return blocks if partial is None else blocks + [partial]

    def forward(self, x0: torch.Tensor, step: StepFn) -> torch.Tensor:
        """Run all n_layer layers with AttnRes inputs; return the output head's input
        (before the final norm). x0 is the token embedding, b_0."""
        blocks: list[torch.Tensor] = [x0]
        partial: torch.Tensor | None = None
        for l in range(self.n_layer):
            h, _ = self.mix(l, self.sources(blocks, partial))
            out = step(l, h).to(x0.dtype)
            partial = out if partial is None else partial + out
            if (l + 1) % self.layers_per_block == 0:
                blocks.append(partial)
                partial = None
        h, _ = self.mix(self.n_layer, blocks)
        return h
