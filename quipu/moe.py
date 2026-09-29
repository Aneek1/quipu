"""The routed feed-forward of quipu-moe (spec sections 2-3).

Pieces, smallest first:

- situ_glu / SiTUGLU: SwiGLU with soft caps, from the Kimi K3 report,
      h = (b1 * tanh(g / b1) * sigmoid(g)) * (b2 * tanh(u / b2)),  y = W_down h
  where g = W_gate x and u = W_up x. Near zero tanh(z / b) * b ~ z, so it is SwiGLU;
  for large inputs |h| <= b1 * b2, which keeps bf16 activations bounded.
- ExpertBank: n experts stored as stacked weights ([n, d, h] gate/up, [n, h, d] down),
  run on tokens already sorted by expert, one contiguous slice per expert.
- Router: fp32 softmax scores; Top-k chosen on score + bias, weights taken from the
  unbiased scores and renormalised over the chosen k.
- QuantileBalancer: the auxiliary-loss-free bias b. It never receives gradients; it is
  moved after each step by the Quantile Balancing rule (see QuantileBalancer.update).
- MoELayer: shared (always-on) experts plus the routed bank.

Router and balancing math run in fp32 even under bf16 autocast: the Top-k decision
and the quantile cutoffs compare scores that differ in the third decimal place.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quipu.config import ModelConfig
from quipu.model import SwiGLU

INIT_STD = 0.02


def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta_gate: float, beta_up: float) -> torch.Tensor:
    """SiTU-GLU hidden activation. Bounded by beta_gate * beta_up in absolute value."""
    g = beta_gate * torch.tanh(gate / beta_gate) * torch.sigmoid(gate)
    return g * (beta_up * torch.tanh(up / beta_up))


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU hidden activation, the same formula as quipu.model.SwiGLU."""
    return F.silu(gate) * up


class SiTUGLU(SwiGLU):
    """Soft-capped SwiGLU. Same parameters and state_dict keys as quipu.model.SwiGLU
    (gate, up, down), so the two are drop-in replacements for one another."""

    def __init__(self, dim: int, hidden: int, beta_gate: float = 4.0, beta_up: float = 25.0) -> None:
        super().__init__(dim, hidden)
        if beta_gate <= 0 or beta_up <= 0:
            raise ValueError(f"SiTU-GLU betas must be positive, got {beta_gate}, {beta_up}")
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)

    def act(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return situ_glu(gate, up, self.beta_gate, self.beta_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x), self.up(x)))


def make_ffn(activation: str, dim: int, hidden: int, beta_gate: float, beta_up: float) -> SwiGLU:
    """A single full-width expert of the configured activation."""
    if activation == "swiglu":
        return SwiGLU(dim, hidden)
    if activation == "situ_glu":
        return SiTUGLU(dim, hidden, beta_gate, beta_up)
    raise ValueError(f"unknown activation {activation!r}")


class ExpertBank(nn.Module):
    """n independent gated experts held as three stacked weight tensors.

    forward takes tokens already sorted by expert and the per-expert counts; expert e
    runs on the contiguous slice [offset_e, offset_e + counts[e]). The only Python loop
    is over experts (a handful of matmuls each), never over tokens."""

    def __init__(
        self,
        n_experts: int,
        dim: int,
        hidden: int,
        activation: str = "swiglu",
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        n_layer: int = 1,
    ) -> None:
        super().__init__()
        if activation not in ("swiglu", "situ_glu"):
            raise ValueError(f"unknown activation {activation!r}")
        self.n_experts = n_experts
        self.activation = activation
        self.beta_gate = float(beta_gate)
        self.beta_up = float(beta_up)
        self.gate = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.up = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.down = nn.Parameter(torch.empty(n_experts, hidden, dim))
        self.reset_parameters(n_layer)

    def reset_parameters(self, n_layer: int) -> None:
        # quipu.model conventions: normal(0, 0.02), residual (down) projections scaled
        # by depth so the residual stream does not grow with n_layer.
        nn.init.normal_(self.gate, mean=0.0, std=INIT_STD)
        nn.init.normal_(self.up, mean=0.0, std=INIT_STD)
        nn.init.normal_(self.down, mean=0.0, std=INIT_STD / (2 * n_layer) ** 0.5)

    def act(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.activation == "situ_glu":
            return situ_glu(gate, up, self.beta_gate, self.beta_up)
        return swiglu(gate, up)

    def forward(self, x_sorted: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        # One host sync for all offsets rather than one per expert.
        sizes = counts.tolist()
        chunks = torch.split(x_sorted, sizes, dim=0)
        outs = []
        for e, chunk in enumerate(chunks):
            if chunk.shape[0] == 0:
                continue
            h = self.act(chunk @ self.gate[e], chunk @ self.up[e])
            outs.append(h @ self.down[e])
        if not outs:
            return x_sorted.new_zeros(0, self.down.shape[-1])
        return torch.cat(outs, dim=0)


def select_experts(
    scores: torch.Tensor, bias: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k on the biased scores; weights from the unbiased scores, renormalised.

    The bias only decides WHICH experts a token goes to; it never changes how much
    each chosen expert's output counts."""
    idx = (scores + bias).topk(top_k, dim=-1).indices
    picked = scores.gather(-1, idx)
    weights = picked / picked.sum(-1, keepdim=True)
    return weights, idx


class Router(nn.Module):
    """Linear d -> n_experts, softmax scores, Top-k selection. All in fp32."""

    def __init__(self, dim: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        if not n_experts >= top_k >= 1:
            raise ValueError(f"need n_experts >= top_k >= 1, got {n_experts}, {top_k}")
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, mean=0.0, std=INIT_STD)

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (T, d). Returns weights (T, k) fp32, indices (T, k), scores (T, n) fp32."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.weight.float())
            scores = logits.softmax(dim=-1)
            weights, idx = select_experts(scores, bias.detach().float(), self.top_k)
        return weights, idx, scores


class QuantileBalancer(nn.Module):
    """Auxiliary-loss-free load balancing (Quantile Balancing, Kimi K3).

    Holds the per-expert bias b as a persistent fp32 buffer: it is saved with the
    model, but it is not a parameter, so no optimizer ever sees it and it never has
    a gradient. update() is the only thing that changes it."""

    def __init__(self, n_experts: int, top_k: int, update_rate: float) -> None:
        super().__init__()
        if not n_experts >= top_k >= 1:
            raise ValueError(f"need n_experts >= top_k >= 1, got {n_experts}, {top_k}")
        if not 0.0 < update_rate <= 1.0:
            raise ValueError(f"update_rate must be in (0, 1], got {update_rate}")
        self.n_experts = n_experts
        self.top_k = top_k
        self.update_rate = float(update_rate)
        self.register_buffer("bias", torch.zeros(n_experts, dtype=torch.float32))

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        """One balancing step from a batch of unbiased router scores (T, n).

        Token i's margin to expert j is m_ij = s_ij + b_j - alpha_i, where alpha_i is
        the (k+1)-th largest biased score of token i: j is among token i's Top-k exactly
        when m_ij > 0. The target load is q = k*T/n tokens per expert. Shifting b_j by
        -c_j, where c_j is the (q+1)-th largest margin in column j, leaves exactly q
        tokens with a positive margin. Every expert's cutoff moves every token's
        alpha, so the step is taken as an EMA: b_j <- b_j - rate * c_j."""
        k, n = self.top_k, self.n_experts
        if k >= n:
            return  # every expert takes every token; nothing to balance
        s = scores.detach().float().reshape(-1, n)
        T = s.shape[0]
        if T < 2:
            return
        biased = s + self.bias
        alpha = biased.topk(k + 1, dim=-1).values[:, -1]
        margins = biased - alpha[:, None]
        q = min(max(int(round(k * T / n)), 0), T - 1)
        # (q+1)-th largest per column == value at 0-based rank q in descending order.
        cutoff = margins.topk(q + 1, dim=0).values[q]
        self.bias.sub_(self.update_rate * cutoff)


class MoELayer(nn.Module):
    """Shared experts (full width, always on) plus top_k of n_experts routed experts.

    forward(x) -> (y, counts): y has x's shape; counts (n_experts,) is how many tokens
    each routed expert processed in this call (sums to T * top_k). The batch's router
    scores are kept so the trainer can call update_balance() after its optimizer step."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.kind != "moe":
            raise ValueError(f"MoELayer needs kind 'moe', got {cfg.kind!r}")
        d = cfg.d_model
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = Router(d, cfg.n_experts, cfg.top_k)
        self.balancer = QuantileBalancer(cfg.n_experts, cfg.top_k, cfg.balance_update_rate)
        self.experts = ExpertBank(
            cfg.n_experts, d, cfg.expert_hidden, cfg.activation,
            cfg.situ_beta_gate, cfg.situ_beta_up, n_layer=cfg.n_layer,
        )
        self.shared = nn.ModuleList(
            make_ffn(cfg.activation, d, cfg.shared_hidden, cfg.situ_beta_gate, cfg.situ_beta_up)
            for _ in range(cfg.shared_experts)
        )
        for ffn in self.shared:
            nn.init.normal_(ffn.gate.weight, mean=0.0, std=INIT_STD)
            nn.init.normal_(ffn.up.weight, mean=0.0, std=INIT_STD)
            nn.init.normal_(ffn.down.weight, mean=0.0, std=INIT_STD / (2 * cfg.n_layer) ** 0.5)
        self._last_scores: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        T, k = xf.shape[0], self.top_k

        weights, idx, scores = self.router(xf, self.balancer.bias)
        self._last_scores = scores.detach()

        # (token, expert, weight) triples, grouped by expert.
        flat_expert = idx.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        token_of = torch.arange(T, device=x.device).repeat_interleave(k)[order]
        counts = torch.bincount(flat_expert, minlength=self.n_experts)

        out = self.experts(xf[token_of], counts)
        out = out.float() * weights.reshape(-1)[order, None]
        y = torch.zeros(T, shape[-1], device=x.device, dtype=torch.float32)
        y = y.index_add(0, token_of, out)
        for ffn in self.shared:
            y = y + ffn(xf).float()
        out_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return y.to(out_dtype).reshape(shape), counts

    def update_balance(self) -> None:
        """Apply one Quantile Balancing step from the latest forward's scores."""
        if self._last_scores is not None:
            self.balancer.update(self._last_scores)
            self._last_scores = None
