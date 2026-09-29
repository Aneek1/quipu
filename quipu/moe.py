"""The routed feed-forward of quipu-moe (spec sections 2-3).

Pieces, smallest first:

- situ_glu / SiTUGLU: SwiGLU with soft caps, from the Kimi K3 report,
      h = (b1 * tanh(g / b1) * sigmoid(g)) * (b2 * tanh(u / b2)),  y = W_down h
  where g = W_gate x and u = W_up x. Near zero tanh(z / b) * b ~ z, so it is SwiGLU;
  for large inputs |h| <= b1 * b2, which keeps bf16 activations bounded.
- ExpertBank: n experts stored as stacked weights ([n, d, h] gate/up, [n, h, d] down).
  Two dispatches over tokens already sorted by expert: "loop" runs each expert on its
  contiguous slice; "padded" pads every slice to a fixed capacity and runs all experts
  in one batched matmul, dropping overflow. Padded drops are batch-dependent
  (Switch-style capacity over the whole flattened batch): see MoELayer.
- Router: fp32 softmax scores; Top-k chosen on score + bias, weights taken from the
  unbiased scores and renormalised over the chosen k.
- QuantileBalancer: the auxiliary-loss-free bias b. It never receives gradients; it is
  moved after each step by the Quantile Balancing rule (see QuantileBalancer.update).
- MoELayer: shared (always-on) experts plus the routed bank.

Router and balancing math run in fp32 even under bf16 autocast: the Top-k decision
and the quantile cutoffs compare scores that differ in the third decimal place.

Determinism: the combine never scatters with atomics. Each (token, slot) pair is
one row of the sorted buffer, so going back is an inverse permutation followed by a
sum over the k slots of each token; the backward passes are the same two operations
in reverse, and two identical steps give bit-identical gradients.
"""
from __future__ import annotations

import math
from typing import NamedTuple

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


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype autocast would run a matmul in, or x's own dtype outside autocast."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


class ExpertBank(nn.Module):
    """n independent gated experts held as three stacked weight tensors.

    Both dispatches take tokens already sorted by expert, the routing weight of each
    row, and return one output row per input row (weight already applied: h is scaled
    by w before the down projection). The weights are cast to the compute dtype once
    per forward, not once per expert."""

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

    def _weights(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gate.to(dtype), self.up.to(dtype), self.down.to(dtype)

    def forward_loop(
        self, x_sorted: torch.Tensor, w_sorted: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Expert e runs on rows [offset_e, offset_e + counts[e]). The only Python loop
        is over experts, never over tokens; one host sync reads all the counts."""
        dtype = _compute_dtype(x_sorted)
        # unbind, not gate[e] per expert: each gate[e] would backward into its own
        # full-size zero [n, d, h] gradient (64 fills and adds of the whole bank);
        # unbind's backward stacks the per-expert gradients once.
        gate, up, down = (w.unbind(0) for w in self._weights(dtype))
        x_sorted = x_sorted.to(dtype)
        sizes = counts.tolist()
        outs = []
        for e, (xs, ws) in enumerate(zip(torch.split(x_sorted, sizes), torch.split(w_sorted, sizes))):
            if xs.shape[0] == 0:
                continue
            h = self.act(xs @ gate[e], xs @ up[e]) * ws[:, None].to(dtype)
            outs.append(h @ down[e])
        if not outs:
            return x_sorted.new_zeros(0, self.down.shape[-1])
        return torch.cat(outs, dim=0)

    def forward_padded(
        self,
        x_sorted: torch.Tensor,
        w_sorted: torch.Tensor,
        expert_sorted: torch.Tensor,
        counts: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All experts in one torch.bmm over [n, capacity, d]. Each expert keeps its
        first `capacity` rows (token order, as sorted); rows past that are dropped and
        get a zero output. No host sync: shapes depend only on n and capacity.
        Returns (out, dropped) with dropped a 0-d tensor.

        Whether a row is dropped depends on every row sorted before it, i.e. on the
        rest of the batch: later tokens and later batch rows drop more, and a token's
        output can change when an EARLIER row of the batch changes. Within one
        sequence an earlier position never depends on a later one. Use loop dispatch
        for evaluation and generation."""
        n, N, d = self.n_experts, x_sorted.shape[0], x_sorted.shape[-1]
        dtype = _compute_dtype(x_sorted)
        gate, up, down = self._weights(dtype)
        dev = x_sorted.device

        offsets = torch.cumsum(counts, 0) - counts                         # [n]
        slot = torch.arange(capacity, device=dev)
        valid = slot[None, :] < counts[:, None]                             # [n, C]
        # Gather each (expert, slot) from its sorted row; empty slots read a zero row
        # appended at index N. Real rows are read at most once.
        src = torch.where(valid, offsets[:, None] + slot[None, :], N)
        x_pad = torch.cat([x_sorted.to(dtype), x_sorted.new_zeros(1, d, dtype=dtype)])
        w_pad = torch.cat([w_sorted, w_sorted.new_zeros(1)])
        xb = x_pad[src]                                                     # [n, C, d]
        wb = w_pad[src].to(dtype)                                           # [n, C]

        h = self.act(torch.bmm(xb, gate), torch.bmm(xb, up)) * wb[..., None]
        ob = torch.bmm(h, down)                                             # [n, C, d]

        # Back to one row per sorted input; overflow rows read an appended zero row.
        pos = torch.arange(N, device=dev) - offsets[expert_sorted]
        kept = pos < capacity
        dst = torch.where(kept, expert_sorted * capacity + pos, n * capacity)
        o_pad = torch.cat([ob.reshape(n * capacity, d), ob.new_zeros(1, d)])
        return o_pad[dst], (~kept).sum()


def select_experts(
    scores: torch.Tensor, bias: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k on the biased scores; weights from the unbiased scores, renormalised.

    The bias only decides WHICH experts a token goes to; it never changes how much
    each chosen expert's output counts."""
    idx = (scores + bias).topk(top_k, dim=-1).indices
    picked = scores.gather(-1, idx)
    # Softmax scores can underflow to 0 for all chosen experts when the bias forces
    # far-down experts in; the clamp keeps that a 0 weight rather than a NaN.
    denom = picked.sum(-1, keepdim=True).clamp_min(torch.finfo(picked.dtype).tiny)
    weights = picked / denom
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

        Margins are two-sided, measured against the score that decides membership
        from where each (token, expert) pair currently stands:
          - expert j chosen for token i: m_ij = biased_ij - (k+1)-th biased score of
            token i (> 0; how far b_j can fall before i is lost);
          - expert j not chosen:        m_ij = biased_ij - k-th biased score of
            token i (<= 0; how far b_j must rise before i is won).
        So j holds token i exactly when m_ij > 0, and a starved expert that is every
        token's runner-up sees negative margins and is raised (a one-sided margin
        against the (k+1)-th score would be 0 for it and never move it).

        The target load is q = k*T/n tokens per expert. Shifting b_j by -c_j, where
        c_j is the (q+1)-th largest margin in column j, leaves q tokens with a
        positive margin. Every expert's shift moves the others' thresholds, so the
        step is taken as an EMA, b_j <- b_j - rate * c_j, and the bias is then
        re-centred to mean 0 (adding a constant to every b_j never changes a Top-k,
        so this only stops the biases drifting together).

        Skipped when k >= n (every expert takes every token) or k*T < n (the batch is
        too small for a per-expert target of at least one token)."""
        k, n = self.top_k, self.n_experts
        s = scores.detach().float().reshape(-1, n)
        T = s.shape[0]
        if k >= n or k * T < n:
            return
        biased = s + self.bias
        top = biased.topk(k + 1, dim=-1)
        chosen = torch.zeros_like(biased, dtype=torch.bool).scatter_(1, top.indices[:, :k], True)
        thr = torch.where(chosen, top.values[:, k:k + 1], top.values[:, k - 1:k])
        margins = biased - thr
        q = min(int(round(k * T / n)), T - 1)
        # (q+1)-th largest per column == value at 0-based rank q in descending order.
        cutoff = margins.topk(q + 1, dim=0).values[q]
        self.bias.sub_(self.update_rate * cutoff)
        self.bias.sub_(self.bias.mean())


class MoEStats(NamedTuple):
    counts: torch.Tensor   # (n_experts,) tokens routed to each expert; sums to T * top_k
    dropped: torch.Tensor  # 0-d; assignments dropped by the padded dispatch's capacity


class MoELayer(nn.Module):
    """Shared experts (full width, always on) plus top_k of n_experts routed experts.

    forward(x) -> (y, MoEStats): y has x's shape (the autocast dtype under autocast,
    else x's dtype). The batch's router scores are kept so the trainer can call
    update_balance() after its optimizer step.

    dispatch ("loop" | "padded") is a validated, mutable attribute: a model trained
    with padded dispatch can be switched to loop for evaluation. Padded dispatch
    drops overflow Switch-style, so its drops (and a token's output) depend on the
    rest of the batch; loop dispatch drops nothing and is per-token."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.kind != "moe":
            raise ValueError(f"MoELayer needs kind 'moe', got {cfg.kind!r}")
        d = cfg.d_model
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.dispatch = cfg.moe_dispatch
        self.capacity_factor = float(cfg.capacity_factor)
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
        self._step_scores: list[torch.Tensor] = []

    @property
    def dispatch(self) -> str:
        return self._dispatch

    @dispatch.setter
    def dispatch(self, mode: str) -> None:
        if mode not in ("loop", "padded"):
            raise ValueError(f"unknown moe_dispatch {mode!r}")
        self._dispatch = mode

    def capacity(self, n_tokens: int) -> int:
        """Per-expert slots for the padded dispatch: ceil(factor * T * k / n)."""
        return max(1, math.ceil(self.capacity_factor * n_tokens * self.top_k / self.n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        shape = x.shape
        d = shape[-1]
        xf = x.reshape(-1, d)
        T, k = xf.shape[0], self.top_k

        weights, idx, scores = self.router(xf, self.balancer.bias)
        self._last_scores = scores.detach()

        # Row r = t*k + slot of the flattened (token, slot) assignments; stable sort
        # groups them by expert with tokens in order inside each group.
        flat_expert = idx.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        counts = torch.bincount(flat_expert, minlength=self.n_experts)
        w_sorted = weights.reshape(-1)[order]
        # Gather through a (T, k, d) broadcast view rather than xf[order // k]: the
        # backward of this index writes each row once (no duplicate-index
        # accumulation), and the expand's backward is a plain sum over k.
        xv = xf.unsqueeze(1).expand(T, k, d)
        x_sorted = xv[order // k, order % k]

        if self.dispatch == "padded":
            out, dropped = self.experts.forward_padded(
                x_sorted, w_sorted, flat_expert[order], counts, self.capacity(T))
        else:
            out = self.experts.forward_loop(x_sorted, w_sorted, counts)
            dropped = counts.new_zeros(())

        # Inverse permutation back to (token, slot) order, then sum the k slots in fp32.
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        y = out[inv].view(T, k, d).sum(1, dtype=torch.float32)
        for ffn in self.shared:
            y = y + ffn(xf).float()
        return y.to(_compute_dtype(x)).reshape(shape), MoEStats(counts, dropped)

    def accumulate_balance_scores(self, max_rows: int | None = None) -> None:
        """Stash the latest forward's router scores for this optimizer step's balance
        update. Under gradient accumulation the trainer calls this once per
        micro-batch, so update_balance() sees the whole step, not the last micro-batch.

        max_rows caps what one micro-batch keeps (memory: T x n_experts fp32 per
        micro-batch per layer): every stride-th token, stride = ceil(T / max_rows),
        starting at an offset that rotates with the micro-batch index so successive
        micro-batches do not all sample the same positions. Deterministic, so a
        resumed run balances exactly as the uninterrupted one did."""
        s = self._last_scores
        if s is None:
            return
        if max_rows is not None and s.shape[0] > max_rows:
            stride = math.ceil(s.shape[0] / max_rows)
            # A copy: the strided view would keep the whole (T, n) tensor alive.
            s = s[len(self._step_scores) % stride :: stride].clone()
        self._step_scores.append(s)
        self._last_scores = None

    def clear_balance_scores(self) -> None:
        """Drop stashed and latest scores (a skipped or abandoned step)."""
        self._step_scores = []
        self._last_scores = None

    def update_balance(self) -> None:
        """Apply one Quantile Balancing step: from every score stashed by
        accumulate_balance_scores() since the last update when there are any, else
        from the latest forward's. Scores are consumed either way."""
        if self._step_scores:
            self.balancer.update(torch.cat(self._step_scores))
        elif self._last_scores is not None:
            self.balancer.update(self._last_scores)
        self.clear_balance_scores()
