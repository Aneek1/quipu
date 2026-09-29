"""Standalone loader for quipu-moe (AneekC/quipu-moe-1B-A149M and its -chat variant).
Needs only torch, safetensors, tokenizers and huggingface_hub; it does not import the
training package.

    from modeling_quipu_moe import load, load_tokenizer, generate, chat
    model = load("AneekC/quipu-moe-1B-A149M")          # or a local directory
    tok = load_tokenizer("AneekC/quipu-moe-1B-A149M")
    print(generate(model, tok, "def fibonacci(n):", max_new_tokens=60))

    # int4 weights (group-wise, dequantized to fp32 on load: smaller download, same
    # memory once loaded; see the model card for the measured quality loss)
    model = load("AneekC/quipu-moe-1B-A149M", weights="model-int4.safetensors")

    # the chat model
    model = load("AneekC/quipu-moe-1B-A149M-chat")
    print(chat(model, tok, [{"role": "user", "content": "Apa ibu kota Indonesia?"}]))

The architecture matches quipu/model_moe.py in the training repository: a decoder-only
transformer whose feed-forward is a mixture of experts (shared experts always on, plus
top-k of n routed experts chosen on router score + a balancing bias, weights
renormalised over the chosen k), optionally with Block Attention Residuals (a learned
softmax over block outputs replaces the plain residual sum) and SiTU-GLU (soft-capped
SwiGLU). Routed experts run one after another on the tokens routed to them: nothing
is dropped and a token's output never depends on the rest of the batch. The export
script checks that this file reproduces the training model's logits before anything
is uploaded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIAL_TOKENS = ("<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "=== FILE: ", "=== END FILE ===")


@dataclass(frozen=True)
class QuipuMoEConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    context: int
    rope_base: float
    norm_eps: float
    n_experts: int
    top_k: int
    expert_hidden: int
    shared_experts: int
    shared_hidden: int
    activation: str = "swiglu"
    situ_beta_gate: float = 4.0
    situ_beta_up: float = 25.0
    attnres_blocks: int = 0

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


def glu(gate: torch.Tensor, up: torch.Tensor, cfg: QuipuMoEConfig) -> torch.Tensor:
    """SwiGLU, or SiTU-GLU: (b1 tanh(g / b1) sigmoid(g)) * (b2 tanh(u / b2))."""
    if cfg.activation == "situ_glu":
        b1, b2 = cfg.situ_beta_gate, cfg.situ_beta_up
        return (b1 * torch.tanh(gate / b1) * torch.sigmoid(gate)) * (b2 * torch.tanh(up / b2))
    return F.silu(gate) * up


class Attention(nn.Module):
    """Grouped-query attention with RoPE, causal."""

    def __init__(self, cfg: QuipuMoEConfig) -> None:
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


class SharedExpert(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.shared_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.shared_hidden, bias=False)
        self.down = nn.Linear(cfg.shared_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(glu(self.gate(x), self.up(x), self.cfg))


class Router(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model))


class Balancer(nn.Module):
    """Holds the balancing bias learned in training (it only decides which experts a
    token goes to, never how much each counts)."""

    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.register_buffer("bias", torch.zeros(cfg.n_experts))


class Experts(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        n, d, h = cfg.n_experts, cfg.d_model, cfg.expert_hidden
        self.gate = nn.Parameter(torch.empty(n, d, h))
        self.up = nn.Parameter(torch.empty(n, d, h))
        self.down = nn.Parameter(torch.empty(n, h, d))


class MoE(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.balancer = Balancer(cfg)
        self.experts = Experts(cfg)
        self.shared = nn.ModuleList(SharedExpert(cfg) for _ in range(cfg.shared_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, d, k = x.shape, x.shape[-1], self.cfg.top_k
        xf = x.reshape(-1, d)
        # Routing in fp32: softmax scores, Top-k on score + bias, weights renormalised.
        scores = F.linear(xf.float(), self.router.weight.float()).softmax(-1)
        idx = (scores + self.balancer.bias.float()).topk(k, dim=-1).indices
        picked = scores.gather(-1, idx)
        w = picked / picked.sum(-1, keepdim=True).clamp_min(torch.finfo(picked.dtype).tiny)
        y = torch.zeros(xf.shape[0], d, dtype=torch.float32, device=x.device)
        dtype = xf.dtype
        for e in range(self.cfg.n_experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xs = xf[tok]
            h = glu(xs @ self.experts.gate[e].to(dtype), xs @ self.experts.up[e].to(dtype), self.cfg)
            h = h * w[tok, slot][:, None].to(dtype)
            y.index_add_(0, tok, (h @ self.experts.down[e].to(dtype)).float())
        for ffn in self.shared:
            y = y + ffn(xf).float()
        return y.to(dtype).reshape(shape)


class Block(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoE(cfg)


class AttnRes(nn.Module):
    """Block Attention Residuals: every sub-layer (attention or MoE) reads a softmax
    over depth sources (the embedding, each completed block's summed outputs, and the
    current block's partial sum), weighted by its own pseudo-query against the
    sources' RMS-normalised values."""

    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.n_steps = 2 * cfg.n_layer
        self.per_block = self.n_steps // cfg.attnres_blocks
        self.eps = cfg.norm_eps
        self.queries = nn.ParameterList(nn.Parameter(torch.zeros(cfg.d_model))
                                        for _ in range(self.n_steps + 1))

    def mix(self, i: int, sources: list[torch.Tensor]) -> torch.Tensor:
        w = self.queries[i].float()
        vs = [v.float() for v in sources]
        logits = torch.stack([(v @ w) * torch.rsqrt(v.pow(2).mean(-1) + self.eps) for v in vs])
        alpha = logits.softmax(0)
        h = vs[0] * alpha[0].unsqueeze(-1)
        for j in range(1, len(vs)):
            h = torch.addcmul(h, vs[j], alpha[j].unsqueeze(-1))
        return h.to(sources[0].dtype)

    def forward(self, x0: torch.Tensor, step) -> torch.Tensor:
        blocks, partial = [x0], None
        for i in range(self.n_steps):
            h = self.mix(i, blocks if partial is None else blocks + [partial])
            out = step(i, h).to(x0.dtype)
            partial = out if partial is None else partial + out
            if (i + 1) % self.per_block == 0:
                blocks.append(partial)
                partial = None
        return self.mix(self.n_steps, blocks)


class QuipuMoE(nn.Module):
    def __init__(self, cfg: QuipuMoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.attnres = AttnRes(cfg) if cfg.attnres_blocks else None
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied input/output embeddings
        cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _step(self, i: int, h: torch.Tensor) -> torch.Tensor:
        b = self.blocks[i // 2]
        if i % 2 == 0:
            return b.attn(b.norm1(h), self.rope_cos, self.rope_sin).to(h.dtype)
        return b.moe(b.norm2(h)).to(h.dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.shape[1] <= self.cfg.context, f"sequence longer than the {self.cfg.context}-token context"
        x = self.embed(idx)
        if self.attnres is None:
            for b in self.blocks:
                x = x + b.attn(b.norm1(x), self.rope_cos, self.rope_sin)
                x = x + b.moe(b.norm2(x))
        else:
            x = self.attnres(x, self._step)
        return self.lm_head(self.norm(x))


# ---- weights -----------------------------------------------------------------------------

def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor,
                    shape: list[int], group_size: int) -> torch.Tensor:
    """Group-wise asymmetric int4: two 4-bit codes per byte (low nibble first), groups of
    group_size consecutive values along the last dimension, w = code * scale + min."""
    lead, last = list(shape[:-1]), shape[-1]
    groups = scales.shape[-1]
    q = qweight.reshape(*lead, groups, group_size // 2)
    codes = torch.stack([q & 0x0F, q >> 4], dim=-1).reshape(*lead, groups, group_size).float()
    w = codes * scales.float()[..., None] + mins.float()[..., None]
    return w.reshape(*lead, groups * group_size)[..., :last].contiguous()


def load_state(path: str) -> dict[str, torch.Tensor]:
    """A weights file's tensors as float, dequantizing int4 entries (named in the
    file's "int4" metadata)."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        state = {k: f.get_tensor(k) for k in f.keys()}
    quant = json.loads(meta.get("int4", "{}"))
    out = {}
    for name, q in quant.items():
        out[name] = dequantize_int4(state.pop(name + ".qweight"), state.pop(name + ".scales"),
                                    state.pop(name + ".mins"), q["shape"], q["group_size"])
    out.update({k: v.float() for k, v in state.items()})
    return out


def _resolve(repo_or_dir: str, filename: str, revision: str | None) -> str:
    if os.path.isdir(repo_or_dir):
        return os.path.join(repo_or_dir, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_or_dir, filename, revision=revision)


def load(repo_or_dir: str = "AneekC/quipu-moe-1B-A149M", weights: str = "model.safetensors",
         device: str = "cpu", revision: str | None = None,
         dtype: torch.dtype = torch.float32) -> QuipuMoE:
    """The final model; a milestone with weights="milestones/step_NNNNNN.safetensors";
    int4 with weights="model-int4.safetensors"."""
    with open(_resolve(repo_or_dir, "config.json", revision), encoding="utf-8") as f:
        raw = json.load(f)
    cfg = QuipuMoEConfig(**{fl.name: raw[fl.name] for fl in fields(QuipuMoEConfig) if fl.name in raw})
    model = QuipuMoE(cfg)
    state = load_state(_resolve(repo_or_dir, weights, revision))
    # lm_head.weight is not stored: it is the embedding matrix (tied).
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, f"weights mismatch: missing {missing}, unexpected {unexpected}"
    assert model.lm_head.weight is model.embed.weight
    return model.to(device=device, dtype=dtype).eval()


# ---- tokenizer and generation ------------------------------------------------------------

class Tokenizer:
    """tokenizer.json via the `tokenizers` package. encode() treats special-token
    strings in the text as text; encode_chat() parses them."""

    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer as HFTokenizer
        self._plain = HFTokenizer.from_file(path)
        self._plain.encode_special_tokens = True
        self._special = HFTokenizer.from_file(path)

    def special_id(self, name: str) -> int:
        return self._special.token_to_id(name)

    @property
    def eot(self) -> int:
        return self.special_id("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self._plain.encode(text, add_special_tokens=False).ids

    def encode_with_special(self, text: str) -> list[int]:
        return self._special.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._plain.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(repo_or_dir: str = "AneekC/quipu-moe-1B-A149M",
                   revision: str | None = None) -> Tokenizer:
    return Tokenizer(_resolve(repo_or_dir, "tokenizer.json", revision))


def render_chat(messages: list[dict]) -> str:
    """<|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>, then an open
    <|assistant|> for the reply."""
    out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
    return out + "<|assistant|>"


@torch.no_grad()
def _continue(model: QuipuMoE, ids: list[int], max_new_tokens: int, temperature: float,
              top_k: int, seed: int | None, stop_ids: set[int]) -> list[int]:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    idx = torch.tensor([ids], device=device)
    new: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.cfg.context:])[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        t = int(nxt)
        if t in stop_ids:
            break
        new.append(t)
        idx = torch.cat([idx, nxt], dim=1)
    return new


def generate(model: QuipuMoE, tok: Tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.0, top_k: int = 50, seed: int | None = None) -> str:
    """Continue plain text (base model). temperature=0 is greedy."""
    ids = tok.encode(prompt)
    return prompt + tok.decode(_continue(model, ids, max_new_tokens, temperature, top_k, seed,
                                         {tok.eot}))


def chat(model: QuipuMoE, tok: Tokenizer, messages: list[dict], max_new_tokens: int = 200,
         temperature: float = 0.0, top_k: int = 50, seed: int | None = None) -> str:
    """The assistant's reply to a conversation (chat model), up to <|end|>."""
    ids = tok.encode_with_special(render_chat(messages))
    return tok.decode(_continue(model, ids, max_new_tokens, temperature, top_k, seed,
                                {tok.special_id("<|end|>"), tok.eot}))
