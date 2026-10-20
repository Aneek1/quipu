"""Muon (spec section 6.1): momentum, then orthogonalise the update.

For a weight matrix W the update is not the momentum M itself but its nearest
semi-orthogonal matrix, approximately U V^T for M = U S V^T, found by a few
Newton-Schulz iterations. Every direction of the update then moves by about the
same amount, whatever the gradient's spectrum; the learning rate is a step size in
spectral norm.

Pieces:

- newton_schulz(g, steps): the quintic iteration of Jordan et al. (Muon, 2024),
      X <- a X + (b A + c A^2) X,   A = X X^T,   (a, b, c) = (3.4445, -4.7750, 2.0315)
  on X = g / ||g||_F. The coefficients trade exactness for speed: singular values
  land near 1 (roughly [0.7, 1.2]), not exactly on it, which is all Muon needs.
  Wide orientation is used (a tall matrix is transposed and back), so A is the
  smaller Gram matrix. On CUDA the iteration runs in bf16 (fp16 below sm_80, see
  cuda_work_dtype) whatever the input dtype (training grads are fp32; Keller's
  reference does G.bfloat16() too), to within a few percent of fp32; on CPU it runs in fp32 unless the input is bf16.
  The result is cast back to the input's dtype.
- orthogonalize(update, ...): newton_schulz plus the shape scale
  max(1, fan_out / fan_in) ** 0.5, so a tall matrix's rows get RMS comparable to a
  square one's. Two variants:
    head_dim: the output dimension is cut into blocks of head_dim rows (one per
        attention head) and each block is orthogonalised on its own. Heads are
        independent functions; orthogonalising Q as one matrix would couple them.
    in_out:   the tensor's last two dims are (fan_in, fan_out), ExpertBank's
        x @ W layout, instead of nn.Linear's (fan_out, fan_in). Only the shape
        scale cares: NS(M^T) = NS(M)^T.
  A 3-D tensor [n, ., .] (an ExpertBank weight) is n separate matrices.
- Muon: Nesterov momentum (EMA form), orthogonalise, decoupled weight decay
      W <- W (1 - lr wd) - lr orthogonalize(g + momentum (buf - g))
  head_dim / in_out are param-group options, not tensor attributes, so they travel
  with the optimizer's state_dict. So does state_dtype ("fp32" | "bf16"): the dtype
  the momentum buffer is stored in. The update itself is always computed in the
  gradient's dtype; bf16 storage halves the buffer's memory.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor

NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7


STATE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def cuda_work_dtype(capability: tuple[int, int]) -> torch.dtype:
    """Newton-Schulz's matmul dtype on a CUDA device of this compute capability: bf16
    from sm_80 (A100 on), fp16 below (T4, sm_75, has no bf16 tensor cores). fp16 is
    safe here: X is divided by its Frobenius norm first, so entries stay in [-1, 1]."""
    return torch.bfloat16 if capability >= (8, 0) else torch.float16


def newton_schulz(g: Tensor, steps: int = 5, work_dtype: torch.dtype | None = None) -> Tensor:
    """Approximately orthogonalise g (2-D, or a batch of matrices as 3-D [n, r, c],
    each done on its own). Returns g's shape and dtype. A zero matrix stays zero.
    Iterates in work_dtype if given; else bf16 on CUDA (fp16 below sm_80, see
    cuda_work_dtype), fp32 on CPU (bf16 if g is bf16)."""
    if g.ndim not in (2, 3):
        raise ValueError(f"newton_schulz needs a 2-D or 3-D tensor, got shape {tuple(g.shape)}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive int, got {steps!r}")
    a, b, c = NS_COEFFS
    # bf16 on CUDA (as Keller's reference, G.bfloat16()): tensor-core matmuls, and
    # the iteration only needs singular values in a loose band; fp16 on pre-Ampere
    # GPUs, which have no bf16 tensor cores. fp32 on CPU, where bf16 matmul is slow,
    # unless the input is already bf16.
    if work_dtype is not None:
        work = work_dtype
    elif g.is_cuda:
        work = cuda_work_dtype(torch.cuda.get_device_capability(g.device))
    else:
        work = torch.bfloat16 if g.dtype == torch.bfloat16 else torch.float32
    # Frobenius norm >= spectral norm, so every singular value starts <= 1, inside
    # the iteration's basin.
    if work == torch.float16:
        # fp16's range is narrow (min normal 6e-5, max 65504): a raw gradient can
        # underflow or its norm overflow. Normalise in fp32 first, then cast.
        base = g if g.dtype == torch.float32 else g.float()
        x = (base / (base.norm(dim=(-2, -1), keepdim=True) + NS_EPS)).to(work)
        tall = x.size(-2) > x.size(-1)
        if tall:
            x = x.mT
    else:
        x = g.to(work)
        tall = x.size(-2) > x.size(-1)
        if tall:
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + NS_EPS)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


def orthogonalize(
    update: Tensor, steps: int = 5, head_dim: int | None = None, in_out: bool = False
) -> Tensor:
    """The Muon direction for one parameter's update (see the module docstring).

    update: [out, in] (nn.Linear), or [n, in, out] with in_out=True (ExpertBank);
    2-D in_out and 3-D [n, out, in] are accepted too. head_dim (2-D, not in_out
    only) splits the out dimension into blocks orthogonalised separately."""
    if update.ndim not in (2, 3):
        raise ValueError(f"Muon needs 2-D or 3-D parameters, got shape {tuple(update.shape)}")
    m = update if update.ndim == 3 else update.unsqueeze(0)
    if in_out:
        m = m.mT                                   # -> [n, out, in]
    n, fan_out, fan_in = m.shape
    if head_dim is not None:
        if update.ndim != 2 or in_out:
            raise ValueError("head_dim applies to 2-D [out, in] matrices only, got shape "
                             f"{tuple(update.shape)} (in_out={in_out})")
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 1:
            raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
        if fan_out % head_dim != 0:
            raise ValueError(f"head_dim ({head_dim}) must divide the output dim ({fan_out})")
        blocks = m.reshape(n * (fan_out // head_dim), head_dim, fan_in)
        out = newton_schulz(blocks, steps) * max(1.0, head_dim / fan_in) ** 0.5
        out = out.reshape(n, fan_out, fan_in)
    else:
        out = newton_schulz(m, steps) * max(1.0, fan_out / fan_in) ** 0.5
    if in_out:
        out = out.mT
    return out.reshape(update.shape)


class Muon(torch.optim.Optimizer):
    """Muon over 2-D / 3-D parameters. Group options: lr, momentum, weight_decay,
    ns_steps, nesterov, head_dim (None = whole matrix), in_out (ExpertBank layout),
    state_dtype ("fp32" | "bf16", the momentum buffer's storage dtype).
    Any extra group keys (base_lr, for the LR schedule) are kept as given."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
        head_dim: int | None = None,
        in_out: bool = False,
        state_dtype: str = "fp32",
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov, head_dim=head_dim,
                        in_out=in_out, state_dtype=state_dtype)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        self._check_group(group)

    @staticmethod
    def _check_group(group: dict[str, Any]) -> None:
        lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
        if not (isinstance(lr, (int, float)) and lr >= 0):
            raise ValueError(f"lr must be >= 0, got {lr!r}")
        if not (isinstance(mom, (int, float)) and 0 <= mom < 1):
            raise ValueError(f"momentum must be in [0, 1), got {mom!r}")
        if not (isinstance(wd, (int, float)) and wd >= 0):
            raise ValueError(f"weight_decay must be >= 0, got {wd!r}")
        steps = group["ns_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"ns_steps must be a positive int, got {steps!r}")
        if group["state_dtype"] not in STATE_DTYPES:
            raise ValueError(f"state_dtype must be one of {list(STATE_DTYPES)}, "
                             f"got {group['state_dtype']!r}")
        for p in group["params"]:
            if p.ndim not in (2, 3):
                raise ValueError(f"Muon takes 2-D or 3-D parameters, got shape {tuple(p.shape)}; "
                                 "give vectors to AdamW")
            if group["head_dim"] is not None:
                # Validated here, at build time, rather than at the first step.
                if p.ndim != 2 or group["in_out"]:
                    raise ValueError("head_dim groups take 2-D [out, in] matrices only, got "
                                     f"shape {tuple(p.shape)}")
                if p.shape[0] % group["head_dim"] != 0:
                    raise ValueError(f"head_dim ({group['head_dim']}) must divide the output "
                                     f"dim ({p.shape[0]})")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                dt = STATE_DTYPES[group.get("state_dtype", "fp32")]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, dtype=dt)
                buf = state["momentum_buffer"]
                if buf.dtype != dt:
                    buf = state["momentum_buffer"] = buf.to(dt)
                if dt == g.dtype:
                    buf.lerp_(g, 1 - mom)
                    m = buf
                else:
                    # Stored low, updated in the gradient's dtype (fp32).
                    m = buf.to(g.dtype).lerp_(g, 1 - mom)
                    buf.copy_(m)
                direction = g.lerp(m, mom) if group["nesterov"] else m
                update = orthogonalize(direction, group["ns_steps"], group["head_dim"],
                                       group["in_out"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
        return loss

    @torch.no_grad()
    def cast_state_(self) -> None:
        """Every momentum buffer into its group's state_dtype. load_state_dict casts
        state to the parameter's dtype (fp32), so the trainer calls this right after
        loading a checkpoint: the fp32 copy is dropped before any gradient exists."""
        for group in self.param_groups:
            dt = STATE_DTYPES[group.get("state_dtype", "fp32")]
            for p in group["params"]:
                st = self.state.get(p)
                if st and "momentum_buffer" in st and st["momentum_buffer"].dtype != dt:
                    st["momentum_buffer"] = st["momentum_buffer"].to(dt)
