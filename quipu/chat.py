"""The chat format of quipu-moe's chat fine-tune (spec section 13): render, tokenize with
an assistant-only loss mask, parse back, fit to a context, and generate a reply.

A conversation is a list of {"role", "content"} messages. It renders as

    <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|user|>...

with no other separators: exactly the Hugging Face chat template the export writes
(scripts/export_hf.py CHAT_TEMPLATE) and hf/modeling_quipu_moe.render_chat, which
appends an open <|assistant|> for the reply (add_generation_prompt).

Tokens. A turn is [role token] + encode_with_special(content) + [<|end|>]. That is
what the tokenizer gives for the whole rendered string (special tokens split the
text before BPE, so nothing merges across them), so training tokens equal the
tokens the HF template and the standalone loader produce at inference. The one
consequence: the step-builder's FILE markers ("=== FILE: ", "=== END FILE ===") in
content become their special tokens, as the tokenizer reserved them for. Content
that contains a chat special string (<|user|> and the rest, CHAT_SPECIALS) is
refused (ChatFormatError): parsed at inference it would forge a turn boundary.

Loss mask (spec 13): 1 exactly on the assistant turns' content tokens and their
closing <|end|>; 0 on role tokens, system and user turns, and padding. mask[i] says
whether token i is a training TARGET (the trainer shifts it with the tokens).

Roles: an optional system message first, then user and assistant alternating,
starting with user (validate()).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

ROLES = ("system", "user", "assistant")
ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
END = "<|end|>"
EOT = "<|endoftext|>"
# Strings that must not occur inside content: each is one token under
# encode_with_special and would change the conversation's structure.
CHAT_SPECIALS = (EOT, "<|system|>", "<|user|>", "<|assistant|>", END)
# Targets the loss ignores (torch's cross_entropy default ignore_index).
IGNORE_INDEX = -100


class ChatFormatError(ValueError):
    """A conversation that does not follow the chat format, or tokens that do not
    parse as one."""


def validate(messages: Sequence[dict[str, Any]]) -> None:
    """Raise ChatFormatError unless `messages` is: optional system first, then user /
    assistant alternating from user; every content a string with no chat special
    string in it."""
    if not messages:
        raise ChatFormatError("a conversation needs at least one message")
    expect = "user"
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or set(m) - {"role", "content"} or "role" not in m \
                or "content" not in m:
            raise ChatFormatError(f"message {i} must be {{'role', 'content'}}, got {m!r}")
        role, content = m["role"], m["content"]
        if role not in ROLES:
            raise ChatFormatError(f"message {i}: unknown role {role!r}")
        if not isinstance(content, str):
            raise ChatFormatError(f"message {i}: content must be a string")
        bad = [s for s in CHAT_SPECIALS if s in content]
        if bad:
            raise ChatFormatError(f"message {i} ({role}) contains the special string(s) {bad}")
        if role == "system":
            if i != 0:
                raise ChatFormatError(f"message {i}: a system message may only come first")
            continue
        if role != expect:
            raise ChatFormatError(f"message {i}: expected a {expect} turn, got {role}")
        expect = "assistant" if role == "user" else "user"


def render(messages: Sequence[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    """The conversation as text in the chat template (validated first)."""
    validate(messages)
    out = "".join(f"{ROLE_TOKENS[m['role']]}{m['content']}{END}" for m in messages)
    return out + (ROLE_TOKENS["assistant"] if add_generation_prompt else "")


def _ids(tok: Any) -> dict[str, int]:
    return {name: tok.special_id(name) for name in CHAT_SPECIALS}


def encode_turn(tok: Any, message: dict[str, Any]) -> tuple[list[int], list[int]]:
    """One turn's tokens and loss mask (1 on an assistant's content and <|end|>)."""
    role_id = tok.special_id(ROLE_TOKENS[message["role"]])
    content = tok.encode_with_special(message["content"]) if message["content"] else []
    ids = [role_id] + content + [tok.special_id(END)]
    on = 1 if message["role"] == "assistant" else 0
    return ids, [0] + [on] * (len(content) + 1)


def encode(tok: Any, messages: Sequence[dict[str, Any]],
           add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """(token ids, loss mask) of the rendered conversation; with add_generation_prompt
    an open <|assistant|> (mask 0) follows, ready for the reply."""
    validate(messages)
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        t, k = encode_turn(tok, m)
        ids += t
        mask += k
    if add_generation_prompt:
        ids.append(tok.special_id(ROLE_TOKENS["assistant"]))
        mask.append(0)
    return ids, mask


def parse(tok: Any, ids: Sequence[int]) -> list[dict[str, str]]:
    """The conversation `ids` encodes (encode's inverse). Trailing <|endoftext|>
    padding is allowed; anything else out of place is a ChatFormatError."""
    sp = _ids(tok)
    role_of = {sp[ROLE_TOKENS[r]]: r for r in ROLES}
    messages: list[dict[str, str]] = []
    i, n = 0, len(ids)
    while i < n:
        t = int(ids[i])
        if t == sp[EOT]:
            if any(int(x) != sp[EOT] for x in ids[i:]):
                raise ChatFormatError(f"tokens after <|endoftext|> padding at {i}")
            break
        if t not in role_of:
            raise ChatFormatError(f"expected a role token at {i}, got {t}")
        j = i + 1
        while j < n and int(ids[j]) != sp[END]:
            if int(ids[j]) in role_of or int(ids[j]) == sp[EOT]:
                raise ChatFormatError(f"turn starting at {i} is not closed by <|end|>")
            j += 1
        if j == n:
            raise ChatFormatError(f"turn starting at {i} is not closed by <|end|>")
        messages.append({"role": role_of[t], "content": tok.decode(list(ids[i + 1:j]))})
        i = j + 1
    validate(messages)
    return messages


@dataclass(frozen=True)
class Fitted:
    messages: list[dict[str, Any]]
    ids: list[int]
    mask: list[int]
    truncated: bool       # later turns were cut to fit


def fit(tok: Any, messages: Sequence[dict[str, Any]], max_tokens: int) -> Fitted | None:
    """The longest prefix of the conversation that ends with an assistant turn and
    fits in max_tokens tokens, cut at a turn boundary; None when not even the first
    assistant turn fits (or there is none). Turns are never cut in the middle."""
    validate(messages)
    ids: list[int] = []
    mask: list[int] = []
    best: Fitted | None = None
    for k, m in enumerate(messages):
        t, mk = encode_turn(tok, m)
        if len(ids) + len(t) > max_tokens:
            break
        ids += t
        mask += mk
        if m["role"] == "assistant":
            best = Fitted(list(messages[:k + 1]), list(ids), list(mask), False)
    if best is not None and len(best.messages) < len(messages):
        best = Fitted(best.messages, best.ids, best.mask, True)
    return best


# ---- generation ----------------------------------------------------------------------------

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Logits outside the smallest set of tokens whose probability reaches top_p set
    to -inf (the most likely token is always kept). logits: [V] or [B, V]."""
    if top_p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    # Drop a token when the mass BEFORE it already reaches top_p.
    drop = probs.cumsum(-1) - probs >= top_p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, order, sorted_logits)


@torch.no_grad()
def generate_reply(model: Callable[[torch.Tensor], torch.Tensor], prompt_ids: Sequence[int],
                   stop_ids: Sequence[int], *, max_new_tokens: int, context: int,
                   temperature: float = 0.0, top_p: float = 1.0,
                   generator: torch.Generator | None = None,
                   on_token: Callable[[int], None] | None = None,
                   device: str | torch.device = "cpu") -> tuple[list[int], str]:
    """Sample a reply after `prompt_ids` until a stop id (not included in the
    result), max_new_tokens, whichever first. model(idx [1, T]) -> logits [1, T, V];
    the window is the last `context` tokens (no KV cache). temperature 0 is greedy;
    otherwise softmax(logits / temperature) restricted to top_p. Returns (reply ids,
    "stop" | "length"). on_token is called with each kept token as it comes."""
    stops = {int(s) for s in stop_ids}
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context:])[0, -1].float()
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = F.softmax(top_p_filter(logits / temperature, top_p), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        if nxt in stops:
            return out, "stop"
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long, device=idx.device)], 1)
    return out, "length"


def stop_ids(tok: Any) -> list[int]:
    """A reply ends at <|end|> (and, defensively, at <|endoftext|>)."""
    return [tok.special_id(END), tok.special_id(EOT)]
