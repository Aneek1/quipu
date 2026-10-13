"""Token streaming for the standalone QuipuMoE loader, with a KV cache.

The published loader (modeling_quipu_moe.py) recomputes the whole sequence for every
new token. This reuses its modules and weights but keeps each layer's keys and values,
so after the prompt each step only runs the newest token. The maths per layer is the
loader's: same projections, RoPE at the token's absolute position, grouped-query
attention, MoE routing and shared experts. Only models without Block Attention
Residuals (attnres_blocks == 0) take the cached path; others fall back to the loader."""
import torch
import torch.nn.functional as F


def _attn(attn, x, cos, sin, pos, cache):
    B, T, _ = x.shape
    q = attn.q(x).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
    k = attn.k(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    v = attn.v(x).view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    c, s = cos[:, :, pos:pos + T], sin[:, :, pos:pos + T]
    q = attn_rope(q, c, s)
    k = attn_rope(k, c, s)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    new_cache = (k, v)
    rep = attn.n_head // attn.n_kv_head
    kr, vr = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    # Prefill (no cache, T tokens): causal. One new token over the cache: it may see all.
    y = F.scaled_dot_product_attention(q, kr, vr, is_causal=cache is None and T > 1)
    return attn.o(y.transpose(1, 2).contiguous().view(B, T, -1)), new_cache


attn_rope = None   # set by bind(): the loader's apply_rope


def bind(mq) -> None:
    global attn_rope
    attn_rope = mq.apply_rope


@torch.no_grad()
def forward_cached(model, idx, pos, caches):
    """Logits for idx (B, T) starting at absolute position pos; updates caches in place."""
    x = model.embed(idx)
    for i, b in enumerate(model.blocks):
        h, caches[i] = _attn(b.attn, b.norm1(x), model.rope_cos, model.rope_sin, pos, caches[i])
        x = x + h
        x = x + b.moe(b.norm2(x))
    return model.lm_head(model.norm(x))


def _pick(logits, new, temperature, repetition_penalty, generator=None):
    logits = logits.float().clone()
    if repetition_penalty > 1.0 and new:
        # CTRL-style penalty on tokens already in the reply: stops greedy loops.
        seen = torch.tensor(sorted(set(new)))
        vals = logits[0, seen]
        logits[0, seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return int(logits.argmax(-1))
    logits = logits / float(temperature)
    v, _ = torch.topk(logits, 50)
    logits[logits < v[:, [-1]]] = -float("inf")
    return int(torch.multinomial(F.softmax(logits, -1), 1, generator=generator))


@torch.no_grad()
def stream(model, ids, max_new_tokens, stop_ids, temperature=0.0, repetition_penalty=1.0,
           use_cache=True, generator=None):
    """Yield the reply's token ids one at a time (stops at a stop id, not yielded)."""
    ctx = model.cfg.context
    cached = use_cache and model.attnres is None and len(ids) + max_new_tokens <= ctx
    new: list[int] = []
    if cached:
        caches = [None] * len(model.blocks)
        logits = forward_cached(model, torch.tensor([ids]), 0, caches)[:, -1, :]
        pos = len(ids)
    else:
        idx = torch.tensor([ids])
    for _ in range(int(max_new_tokens)):
        if not cached:
            logits = model(idx[:, -ctx:])[:, -1, :]
        t = _pick(logits, new, temperature, repetition_penalty, generator)
        if t in stop_ids:
            return
        new.append(t)
        yield t
        if cached:
            logits = forward_cached(model, torch.tensor([[t]]), pos, caches)[:, -1, :]
            pos += 1
        else:
            idx = torch.cat([idx, torch.tensor([[t]])], dim=1)
