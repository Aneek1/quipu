"""Held-out loss and generation probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from quipu.loader import TokenStream
from quipu.model import Quipu


@torch.no_grad()
def estimate_loss(
    model: Quipu, stream: TokenStream, batches: int, device: str, amp: bool = True
) -> float:
    """Mean cross-entropy over `batches` batches, leaving the stream where it was.

    Evaluation must not consume training tokens, so the position is saved and
    restored rather than shared.

    Training runs under bf16 autocast; eval defaults to the same numerics so the
    train and val curves are comparable, and because fp32 eval is markedly slower.
    CPU stays fp32 regardless of `amp`, since autocast("cuda", ...) is a no-op there.
    """
    was_training = model.training
    saved = stream.state_dict()
    model.eval()
    total = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")):
        for _ in range(batches):
            x, y = stream.next_batch()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
    stream.load_state_dict(saved)
    if was_training:
        model.train()
    return total / batches


@torch.no_grad()
def generate(
    model: Quipu,
    idx: torch.Tensor,
    max_new_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = 50,
) -> torch.Tensor:
    """Sample continuations. Used only for eyeballing coherence, never for a metric."""
    was_training = model.training
    model.eval()
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        # The model has no KV cache and RoPE is built for `context` positions, so the
        # window is cropped rather than grown.
        window = idx[:, -model.cfg.context :]
        logits = model(window)[:, -1, :] / temperature
        if top_k is not None:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    if was_training:
        model.train()
    return idx
