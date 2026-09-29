"""Chat with a quipu-moe chat model in the terminal (spec 13).

    python -m uv run python scripts/chat.py --config configs/quipu-moe-sft.toml \
        [--checkpoint latest|PATH] [--temperature 0.7] [--top-p 0.9] [--system TEXT]
    python -m uv run python scripts/chat.py --hf hf_export_moe_chat [...]

The model is either a training checkpoint (--config gives the model and tokenizer;
--checkpoint "latest" is the one latest.pt in train.ckpt_dir points at, or a path
to any checkpoint or milestone) or an exported Hugging Face folder (--hf DIR, loaded
with the folder's own standalone modeling_quipu_moe.py, no training code). Routed
experts always run with loop dispatch (nothing dropped; a reply never depends on
padding), which the standalone loader does by construction.

Each turn: the conversation so far in the chat template (quipu.chat), an open
<|assistant|>, then tokens sampled until <|end|> (or <|endoftext|>, or
--max-new-tokens). --temperature 0 is greedy; otherwise sampling from softmax
(logits / T) restricted to --top-p. The reply is printed as it comes. When the
conversation no longer fits the context with room for a reply, the oldest turns
are dropped (the system message is kept). Commands: /reset (new conversation),
/system TEXT (set the system message and reset), /quit (or end of input).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quipu import chat  # noqa: E402

HF_MODULE = "modeling_quipu_moe.py"


class StreamPrinter:
    """Prints a reply token by token: the decoded text so far minus what is already
    printed, held back while it ends in a partial UTF-8 character (U+FFFD)."""

    def __init__(self, tok: Any, out: TextIO) -> None:
        self.tok, self.out = tok, out
        self.ids: list[int] = []
        self.shown = 0

    def __call__(self, token: int) -> None:
        self.ids.append(token)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return
        self.out.write(text[self.shown:])
        self.out.flush()
        self.shown = len(text)

    def finish(self) -> str:
        text = self.tok.decode(self.ids)
        self.out.write(text[self.shown:] + "\n")
        self.out.flush()
        return text


def prompt_ids(tok: Any, messages: list[dict[str, str]], budget: int) -> list[int]:
    """The conversation ready for the reply, oldest user/assistant turns dropped
    until it fits `budget` tokens (the system message stays). The newest user turn
    alone is left-cropped if even it is too long."""
    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    turns = messages[len(system):]
    while True:
        ids, _ = chat.encode(tok, system + turns, add_generation_prompt=True)
        if len(ids) <= budget or len(turns) <= 1:
            return ids[-budget:]
        turns = turns[2:]


def run_chat(model: Callable[[torch.Tensor], torch.Tensor], tok: Any, *, context: int,
             lines: Iterable[str], out: TextIO, temperature: float = 0.7, top_p: float = 0.9,
             max_new_tokens: int = 512, system: str | None = None, seed: int | None = None,
             device: str = "cpu", wrap: Callable[[], Any] = contextlib.nullcontext
             ) -> list[dict[str, str]]:
    """The chat loop over `lines` (user inputs); returns the final conversation.
    `wrap` is entered around every generation (loop dispatch for a quipu model)."""
    def fresh() -> list[dict[str, str]]:
        return [{"role": "system", "content": system}] if system else []

    messages = fresh()
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    stops = chat.stop_ids(tok)
    budget = max(1, context - max_new_tokens)
    for line in lines:
        text = line.rstrip("\r\n")
        if not text.strip():
            continue
        if text.strip() in ("/quit", "/exit"):
            break
        if text.strip() == "/reset":
            messages = fresh()
            out.write("(new conversation)\n")
            continue
        if text.startswith("/system"):
            system = text[len("/system"):].strip() or None
            messages = fresh()
            out.write("(system message set; new conversation)\n")
            continue
        try:
            candidate = messages + [{"role": "user", "content": text}]
            ids = prompt_ids(tok, candidate, budget)
        except chat.ChatFormatError as exc:
            out.write(f"(not sent: {exc})\n")
            continue
        out.write("quipu> ")
        printer = StreamPrinter(tok, out)
        with wrap():
            reply_ids, why = chat.generate_reply(
                model, ids, stops, max_new_tokens=max_new_tokens, context=context,
                temperature=temperature, top_p=top_p,
                generator=gen if temperature > 0 else None, on_token=printer, device=device)
        reply = printer.finish()
        if why == "length":
            out.write(f"(stopped at --max-new-tokens {max_new_tokens})\n")
        # A reply that contains a chat special string cannot go back into the
        # template; it is kept as text without them.
        for s in chat.CHAT_SPECIALS:
            reply = reply.replace(s, "")
        messages = candidate + [{"role": "assistant", "content": reply}]
    return messages


def _input_lines(prompt: str = "you> ") -> Iterable[str]:
    while True:
        try:
            yield input(prompt)
        except EOFError:
            return


def load_hf(folder: str | Path, device: str) -> tuple[Any, Any, int]:
    """(model, tokenizer, context) from an exported folder via its standalone loader."""
    folder = Path(folder)
    spec = importlib.util.spec_from_file_location("modeling_quipu_moe", folder / HF_MODULE)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no {HF_MODULE} in {folder}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod        # its dataclasses look the module up there
    spec.loader.exec_module(mod)
    model = mod.load(str(folder), device=device)
    return model, mod.load_tokenizer(str(folder)), model.cfg.context


def load_checkpoint(config: str, checkpoint: str, device: str) -> tuple[Any, Any, int]:
    from quipu import evalsets
    from quipu.config import load_config
    from quipu.tokenizer import make_tokenizer

    cfg = load_config(config)
    path = (evalsets.latest_checkpoint(cfg.train.ckpt_dir) if checkpoint == "latest"
            else Path(checkpoint))
    model = evalsets.load_model(cfg.model, path, device)
    return model, make_tokenizer(cfg.data.tokenizer), cfg.model.context


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chat with a quipu-moe chat model.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--config", default="configs/quipu-moe-sft.toml")
    src.add_argument("--hf", default=None, help="an exported Hugging Face folder")
    ap.add_argument("--checkpoint", default="latest", help="latest | a checkpoint path")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--system", default=None, help="a system message")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(argv)
    if not 0 < args.top_p <= 1 or args.temperature < 0 or args.max_new_tokens < 1:
        print("error: need 0 < --top-p <= 1, --temperature >= 0, --max-new-tokens >= 1",
              file=sys.stderr)
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    if args.hf:
        model, tok, context = load_hf(args.hf, device)
        wrap: Callable[[], Any] = contextlib.nullcontext
    else:
        from quipu.eval import loop_dispatch
        model, tok, context = load_checkpoint(args.config, args.checkpoint, device)
        wrap = lambda: loop_dispatch(model)  # noqa: E731
    if not hasattr(tok, "encode_with_special"):
        print("error: chat needs the BPE tokenizer with chat tokens", file=sys.stderr)
        return 2
    if args.max_new_tokens >= context:
        print(f"error: --max-new-tokens must be below the context ({context})", file=sys.stderr)
        return 2
    print("quipu-moe chat. /reset, /system TEXT, /quit. A small model: often wrong.",
          flush=True)
    run_chat(model, tok, context=context, lines=_input_lines(), out=sys.stdout,
             temperature=args.temperature, top_p=args.top_p,
             max_new_tokens=args.max_new_tokens, system=args.system, seed=args.seed,
             device=device, wrap=wrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
