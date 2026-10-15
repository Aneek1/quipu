"""Quipu chat demo: quipu-moe-1B-A149M-chat-distill on a free CPU Space.

Uses the model repo's own standalone loader (modeling_quipu_moe.py), so the demo runs
exactly the published weights, with a KV cache added in kv_generate.py (same tokens as the
loader's own generation, about 6x faster). Replies stream token by token."""
import dataclasses
import importlib.util
import os
import re
import sys
import threading

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

import kv_generate as kv

REPO = os.environ.get("QUIPU_REPO", "AneekC/quipu-moe-1B-A149M-chat-distill")
CONTEXT = 4096            # trained at 2,048; stretched 2x with NTK-aware RoPE scaling (below)
MAX_NEW = 1536            # prompt + history + reply share the context
MIN_REPLY = 256           # a long chat keeps at least this much room for the reply

def usable_cpus() -> int:
    # Spaces report the host's CPU count; the container only gets its share.
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 4))


torch.set_num_threads(usable_cpus())

path = (os.path.join(REPO, "modeling_quipu_moe.py") if os.path.isdir(REPO)       # local test
        else hf_hub_download(REPO, "modeling_quipu_moe.py"))
spec = importlib.util.spec_from_file_location("modeling_quipu_moe", path)
mq = importlib.util.module_from_spec(spec)
sys.modules["modeling_quipu_moe"] = mq
spec.loader.exec_module(mq)
kv.bind(mq)

model = mq.load(REPO).eval()
tok = mq.load_tokenizer(REPO)


def stretch_context(model, length: int) -> None:
    """Serve a longer context than the model was trained on by raising the RoPE base
    (NTK-aware scaling). Measured on held-out books and code (scripts/context_stretch.py):
    at 2x the loss on the first 2,048 positions rises under 1% and the positions after
    2,048 are still predicted well, so the extra text is used. No retraining."""
    cfg = model.cfg
    s = length / cfg.context
    if s <= 1:
        return
    base = cfg.rope_base * s ** (cfg.head_dim / (cfg.head_dim - 2))
    model.rope_cos, model.rope_sin = mq.build_rope_cache(length, cfg.head_dim, base)
    model.cfg = dataclasses.replace(cfg, context=length, rope_base=base)


stretch_context(model, CONTEXT)
END, EOT = tok.special_id("<|end|>"), tok.eot
lock = threading.Lock()    # one generation at a time on a 2-vCPU box


def encode(messages: list[dict]) -> list[int]:
    """Same encoding as mq.chat: role token + text + <|end|>, then an open <|assistant|>."""
    ids: list[int] = []
    for m in messages:
        ids.append(tok.special_id(f"<|{m['role']}|>"))
        ids += tok.encode(m["content"]) if m["content"] else []
        ids.append(END)
    ids.append(tok.special_id("<|assistant|>"))
    return ids


TAG = re.compile(r"</?[A-Za-z!][^>]*>")


def show(reply: str) -> str:
    """Markdown for the chat window that shows code as code. Text outside ``` fences that
    contains HTML tags is put in an ```html fence: rendered as-is the tags would vanish,
    and escaping them breaks inside Markdown's indented code blocks."""
    parts = reply.split("```")
    for i in range(0, len(parts), 2):          # even pieces are outside fences
        if TAG.search(parts[i]):
            parts[i] = "\n```html\n" + parts[i].strip("\n") + "\n```\n"
    return "```".join(parts)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def respond(message, history, temperature, repetition_penalty, max_new_tokens):
    msgs = []
    for h in history or []:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            msgs.append({"role": h["role"], "content": text_of(h.get("content"))})
    msgs.append({"role": "user", "content": text_of(message)})
    ids = encode(msgs)
    ctx = model.cfg.context
    want = int(max_new_tokens)
    turns = len(msgs)
    while len(ids) + want > ctx and len(msgs) > 1:    # drop the oldest turns first
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
        ids = encode(msgs)
    notes = []
    if len(msgs) < turns:
        notes.append("older messages in this chat were left out to fit the window")
    if len(ids) > ctx - MIN_REPLY:                      # a very long single message
        notes.append(f"your message was too long, so only its last {ctx - MIN_REPLY:,} tokens were used")
        ids = ids[-(ctx - MIN_REPLY):]
    max_new = min(want, ctx - len(ids))                 # stay inside the window (KV-cache path)
    head = (f"*(The model reads {ctx:,} tokens at most: " + "; ".join(notes) + ".)*\n\n") if notes else ""

    with lock:
        new: list[int] = []
        for t in kv.stream(model, ids, max_new, {END, EOT},
                           temperature=float(temperature), repetition_penalty=float(repetition_penalty)):
            new.append(t)
            yield head + show(tok.decode(new))
        if not new:
            yield head + "(no reply)"


DESCRIPTION = """**quipu-moe-1B-A149M-chat-distill**: a 1B-parameter mixture of experts (149M active per token),
trained from scratch for under $20, then fine-tuned on human-written chats plus coding answers that passed
their unit tests. It solves 7.4% of MBPP. **It is small and often wrong**: don't trust its facts or its code.
It does best on short Python functions and simple questions in English; other languages and longer tasks
such as whole web pages are still beyond it. Runs on a small CPU machine.
[Model card](https://huggingface.co/AneekC/quipu-moe-1B-A149M-chat-distill) ·
[Project page](https://quipulm.dev) · [Code](https://github.com/Aneek1/quipu)"""

EXAMPLES = [       # chosen by trying them on this model (temperature 0.3, two seeds each)
    ["Write a Python function that checks whether a string is a palindrome."],
    ["Write a Python function that removes duplicates from a list but keeps the order."],
    ["Write a Python function that counts the vowels in a string."],
    ["What is the capital of France?"],
]

kwargs = dict(
    fn=respond,
    title="Quipu chat",
    description=DESCRIPTION,
    examples=EXAMPLES,
    additional_inputs=[
        gr.Slider(0.0, 1.2, value=0.3, step=0.1, label="Temperature (0 = greedy)"),
        gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="Repetition penalty (1 = off)"),
        gr.Slider(64, MAX_NEW, value=768, step=64, label=f"Max new tokens (the window is {CONTEXT:,} incl. your message)"),
    ],
    cache_examples=False,
)
try:
    demo = gr.ChatInterface(type="messages", **kwargs)
except TypeError:          # newer Gradio: messages format is the only one
    demo = gr.ChatInterface(**kwargs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
