"""Held-out loss, bits per byte and generation probes.

Every entry point runs a quipu-moe model with loop dispatch (loop_dispatch below):
the padded dispatch drops tokens depending on the rest of the batch, so a metric or
a sample taken under it would depend on what else happened to be in the batch.
The model's own dispatch is restored afterwards. Dense models pass straight through.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from quipu.loader import TokenStream


def _moe_layers(model: nn.Module) -> list[nn.Module]:
    """The routed layers of a model (through a torch.compile wrapper too): anything
    with a validated `dispatch` attribute, i.e. quipu.moe.MoELayer."""
    from quipu.moe import MoELayer   # local: quipu.moe imports quipu.model, not eval

    return [m for m in model.modules() if isinstance(m, MoELayer)]


@contextlib.contextmanager
def loop_dispatch(model: nn.Module) -> Iterator[None]:
    """Run every MoE layer with loop dispatch inside the block, then put each layer's
    own dispatch back (per layer: they can differ). A no-op for dense models."""
    layers = _moe_layers(model)
    saved = [layer.dispatch for layer in layers]
    for layer in layers:
        layer.dispatch = "loop"
    try:
        yield
    finally:
        for layer, mode in zip(layers, saved):
            layer.dispatch = mode


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def estimate_loss(
    model: nn.Module, stream: TokenStream, batches: int, device: str, amp: bool = True
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
    try:
        with loop_dispatch(model), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp and str(device).startswith("cuda")
        ):
            for _ in range(batches):
                x, y = stream.next_batch()
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
    finally:
        stream.load_state_dict(saved)
        if was_training:
            model.train()
    return total / batches


@torch.no_grad()
def bits_per_byte(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    tokenizer: Any,
    device: str | torch.device | None = None,
    amp: bool = True,
) -> float:
    """Bits per UTF-8 byte of text: mean NLL (nats) x tokens / (bytes x ln 2), i.e.
    the summed NLL over every target token divided by the bytes those targets
    decode to. Comparable across tokenizers, unlike loss per token.

    batches yields (x, y) token tensors of shape (B, T), y the next-token targets.
    Bytes are counted per TOKEN, from tokenizer.token_byte_lengths(): each token's
    raw byte length (special tokens 0). Raw bytes, not decoded text: a byte-level
    BPE can split one character across tokens (and across rows), and decoding a
    fragment would count replacement characters instead of the bytes it holds.
    `device` defaults to the model's; amp as in estimate_loss (bf16 on CUDA only).
    """
    nll, _, total_bytes = nll_tokens_bytes(model, batches, tokenizer.token_byte_lengths(),
                                           device, amp)
    if total_bytes == 0:
        raise ValueError("bits_per_byte: the targets hold 0 bytes")
    return nll / (total_bytes * math.log(2))


@torch.no_grad()
def nll_tokens_bytes(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    byte_lengths: list[int] | torch.Tensor,
    device: str | torch.device | None = None,
    amp: bool = True,
) -> tuple[float, int, int]:
    """(summed NLL in nats, target tokens, target bytes) over `batches`, with loop
    dispatch; the sums behind bits_per_byte, for callers that also want the mean loss
    per token (nll / tokens) or the sample size. byte_lengths as
    tokenizer.token_byte_lengths()."""
    device = torch.device(device) if device is not None else _device_of(model)
    lens = torch.as_tensor(byte_lengths, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    nll = torch.zeros((), dtype=torch.float64, device=device)
    n_bytes = torch.zeros((), dtype=torch.long, device=device)
    n_tokens = 0
    try:
        with loop_dispatch(model), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
        ):
            for x, y in batches:
                y = y.to(device)
                logits = model(x.to(device))
                nll += F.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                    reduction="sum",
                ).double()
                n_bytes += lens[y].sum()
                n_tokens += y.numel()
    finally:
        if was_training:
            model.train()
    return float(nll), n_tokens, int(n_bytes)


@torch.no_grad()
def generate(
    model: nn.Module,
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
    with loop_dispatch(model):
        for _ in range(max_new_tokens):
            # The model has no KV cache and RoPE is built for `context` positions, so
            # the window is cropped rather than grown.
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
