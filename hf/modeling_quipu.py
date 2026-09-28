"""Standalone loader for quipu-114m. Needs only torch, safetensors, tiktoken and
huggingface_hub; it does not import the training package.

    from modeling_quipu import load, generate
    model = load("quipu-lm/quipu-114m")            # or a local directory
    print(generate(model, "Photosynthesis is the process by which", max_new_tokens=60))

The architecture matches `quipu/model.py` in the training repo exactly; the export
script checks that both produce the same logits before anything is uploaded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QuipuConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    ffn_hidden: int
    context: int
    rope_base: float
    norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX half-split rotation, computed in at least fp32."""
    compute = torch.promote_types(x.dtype, torch.float32)
    x1, x2 = x.to(compute).chunk(2, dim=-1)
    cos, sin = cos.to(compute), sin.to(compute)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Grouped-query attention: 12 query heads share 4 key/value heads."""

    def __init__(self, cfg: QuipuConfig) -> None:
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos[:, :, :T], sin[:, :, :T])
        k = apply_rope(k, cos[:, :, :T], sin[:, :, :T])
        rep = self.n_head // self.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg: QuipuConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class Quipu(nn.Module):
    def __init__(self, cfg: QuipuConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied input/output embeddings
        cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.shape[1] <= self.cfg.context, "sequence longer than the 1,024-token context"
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm(x))


def _resolve(repo_or_dir: str, filename: str, revision: str | None) -> str:
    if os.path.isdir(repo_or_dir):
        return os.path.join(repo_or_dir, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_or_dir, filename, revision=revision)


def load(
    repo_or_dir: str = "quipu-lm/quipu-114m",
    weights: str = "model.safetensors",
    device: str = "cpu",
    revision: str | None = None,
) -> Quipu:
    """Load the final model, or a milestone with weights="milestones/step_001000.safetensors"."""
    from safetensors.torch import load_file

    with open(_resolve(repo_or_dir, "config.json", revision), encoding="utf-8") as f:
        raw = json.load(f)
    cfg = QuipuConfig(**{k: raw[k] for k in QuipuConfig.__dataclass_fields__})
    model = Quipu(cfg)
    state = load_file(_resolve(repo_or_dir, weights, revision))
    # lm_head.weight is not stored: it is the embedding matrix (tied).
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    missing = {k for k, _ in model.named_parameters()} - set(state) - {"lm_head.weight"}
    assert not missing, f"weights file is missing {sorted(missing)}"
    assert model.lm_head.weight is model.embed.weight
    return model.to(device).eval()


@torch.no_grad()
def generate(
    model: Quipu,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_k: int = 50,
    seed: int | None = None,
) -> str:
    """temperature=0 is greedy. The tokenizer is tiktoken's GPT-2 encoding."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.cfg.context:])[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        idx = torch.cat([idx, nxt], dim=1)
    return enc.decode(idx[0].tolist())
