"""Quipu chat demo: quipu-moe-1B-A149M-chat-distill on a free CPU Space.

Uses the model repo's own standalone loader (modeling_quipu_moe.py), so the demo runs
exactly the published weights, with a KV cache added in kv_generate.py (same tokens as the
loader's own generation, about 6x faster). Replies stream token by token."""
import importlib.util
import os
import sys
import threading

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

import kv_generate as kv

REPO = os.environ.get("QUIPU_REPO", "AneekC/quipu-moe-1B-A149M-chat-distill")
MAX_NEW = 256
MAX_PROMPT = 1024          # keep the most recent turns (context is 2,048 tokens)

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
    while len(ids) > MAX_PROMPT and len(msgs) > 1:    # drop the oldest turns first
        msgs = msgs[2:] if len(msgs) > 2 else msgs[1:]
        ids = encode(msgs)
    ids = ids[-MAX_PROMPT:]

    with lock:
        new: list[int] = []
        for t in kv.stream(model, ids, int(max_new_tokens), {END, EOT},
                           temperature=float(temperature), repetition_penalty=float(repetition_penalty)):
            new.append(t)
            yield tok.decode(new)
        if not new:
            yield "(no reply)"


DESCRIPTION = """**quipu-moe-1B-A149M-chat-distill**: a 1B-parameter mixture of experts (149M active per token),
trained from scratch for under $20, then fine-tuned on human-written chats plus coding answers that passed
their unit tests. It solves 7.4% of MBPP. **It is small and often wrong**: don't trust its facts or its code.
Runs on a small CPU machine.
[Model card](https://huggingface.co/AneekC/quipu-moe-1B-A149M-chat-distill) ·
[Project page](https://quipu-lm.vercel.app) · [Code](https://github.com/Aneek1/quipu)"""

EXAMPLES = [
    ["Write a Python function that checks whether a string is a palindrome."],
    ["Write a Python function that returns the n-th Fibonacci number."],
    ["What is photosynthesis? Explain it in two sentences."],
    ["Apa itu fotosintesis?"],
]

kwargs = dict(
    fn=respond,
    title="Quipu chat",
    description=DESCRIPTION,
    examples=EXAMPLES,
    additional_inputs=[
        gr.Slider(0.0, 1.2, value=0.3, step=0.1, label="Temperature (0 = greedy)"),
        gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="Repetition penalty (1 = off)"),
        gr.Slider(16, MAX_NEW, value=160, step=16, label="Max new tokens"),
    ],
    cache_examples=False,
)
try:
    demo = gr.ChatInterface(type="messages", **kwargs)
except TypeError:          # newer Gradio: messages format is the only one
    demo = gr.ChatInterface(**kwargs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
