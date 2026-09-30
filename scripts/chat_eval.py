"""The chat model's samples for its model card (spec 13, plan M12).

    python -m uv run python scripts/chat_eval.py --config configs/quipu-moe-sft.toml \
        [--checkpoint latest|PATH] [--out results/sft/chat_samples.json] \
        [--val-by-source results/sft/val_by_source.json] [--device auto]

Spec 13: two fixed prompts per language, answered at temperature 0 and shown verbatim
in the card, good and bad alike. Each prompt (CHAT_PROMPTS: English and the ten
spec-11 languages, Chinese in both scripts) is one user turn rendered with the chat
template (quipu.chat: <|user|>...<|end|><|assistant|>), then the reply is decoded
greedily until <|end|> (or <|endoftext|>, or --max-new-tokens), routed experts on
loop dispatch. The model is the CHAT model (the SFT checkpoint: --checkpoint
"latest" is the one latest.pt in the SFT config's train.ckpt_dir points at), never
the base model's pretraining samples (results/moe/milestones/samples.json).

The output JSON also carries the held-out chat loss per source that the SFT wrote
at its end (quipu.train, --val-by-source-out, default results/sft/val_by_source.json)
when that file exists, so the chat card (quipu/model_card.py) reads both from one
place. Run it on the box after the SFT (a GPU makes the ~22 greedy replies take a
minute or two; there is no KV cache), or on the laptop from the copied-back
checkpoint (memory-mapped; slow on the CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quipu import chat  # noqa: E402

# Two fixed user prompts per language (spec 13): one plain question, one short
# explanation or task. Nothing needs looking up beyond general knowledge.
CHAT_PROMPTS: dict[str, tuple[str, str]] = {
    "eng_Latn": ("What is photosynthesis? Explain it in two sentences.",
                 "Write a Python function that checks whether a number is prime."),
    "ind_Latn": ("Apa ibu kota Indonesia?",
                 "Jelaskan dengan singkat apa itu fotosintesis."),
    "zsm_Latn": ("Apakah makanan tradisional Malaysia yang terkenal?",
                 "Terangkan secara ringkas bagaimana hujan terbentuk."),
    "zho_Hans": ("中国的首都是哪里？", "请简单解释什么是光合作用。"),
    "zho_Hant": ("台灣最有名的小吃是什麼？", "請簡單說明地震是怎麼發生的。"),
    "jpn_Jpan": ("日本で一番高い山は何ですか？", "光合成とは何か、簡単に説明してください。"),
    "kor_Hang": ("한국의 수도는 어디입니까?", "광합성이 무엇인지 간단히 설명해 주세요."),
    "tam_Taml": ("தமிழ்நாட்டின் தலைநகரம் எது?",
                 "ஒளிச்சேர்க்கை என்றால் என்ன என்று சுருக்கமாக விளக்குங்கள்."),
    "hin_Deva": ("भारत की राजधानी क्या है?", "प्रकाश संश्लेषण क्या है, संक्षेप में समझाइए।"),
    "hin_Latn": ("Bharat ki rajdhani kya hai?",
                 "Photosynthesis kya hota hai, thode mein samjhao."),
    "urd_Latn": ("Pakistan ka darul hukumat kaun sa shehar hai?",
                 "Barish kaise hoti hai, mukhtasar mein batayein."),
}
MAX_NEW_TOKENS = 200


@torch.no_grad()
def chat_samples(model: Any, tok: Any, context: int, device: str,
                 prompts: dict[str, tuple[str, ...]] = CHAT_PROMPTS,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, list[dict[str, Any]]]:
    """{language: [{"prompt", "reply", "finish": "stop" | "length", "reply_tokens"}]},
    every reply greedy (temperature 0) after one rendered user turn."""
    from quipu.eval import loop_dispatch

    stops = chat.stop_ids(tok)
    out: dict[str, list[dict[str, Any]]] = {}
    was_training = model.training
    model.eval()
    try:
        for lang, pair in prompts.items():
            rows = []
            for prompt in pair:
                ids, _ = chat.encode(tok, [{"role": "user", "content": prompt}],
                                     add_generation_prompt=True)
                with loop_dispatch(model):
                    reply, why = chat.generate_reply(
                        model, ids, stops, max_new_tokens=max_new_tokens, context=context,
                        temperature=0.0, device=device)
                rows.append({"prompt": prompt, "reply": tok.decode(reply), "finish": why,
                             "reply_tokens": len(reply)})
            out[lang] = rows
    finally:
        model.train(was_training)
    return out


def run(cfg: Any, checkpoint: Path, device: str, out: Path, val_by_source: Path | None,
        prompts: dict[str, tuple[str, ...]] = CHAT_PROMPTS,
        max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    """Load the chat checkpoint, sample, write `out` (JSON). Returns what was written."""
    from quipu import evalsets
    from quipu.fsio import write_text_atomic
    from quipu.tokenizer import make_tokenizer

    tok = make_tokenizer(cfg.data.tokenizer)
    if not hasattr(tok, "encode_with_special"):
        raise ValueError("chat samples need the BPE tokenizer with chat tokens")
    model = evalsets.load_model(cfg.model, checkpoint, device)
    samples = chat_samples(model, tok, cfg.model.context, device, prompts, max_new_tokens)
    held_out = None
    if val_by_source is not None and Path(val_by_source).is_file():
        held_out = json.loads(Path(val_by_source).read_text(encoding="utf-8"))
    result = {
        "model": "chat", "checkpoint": str(checkpoint),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decoding": {"temperature": 0.0, "max_new_tokens": max_new_tokens,
                     "template": "quipu.chat: <|user|>PROMPT<|end|><|assistant|>, "
                                 "stops at <|end|>"},
        "samples": samples,
        "val_by_source": held_out,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/quipu-moe-sft.toml")
    ap.add_argument("--checkpoint", default="latest", help="latest | a checkpoint path")
    ap.add_argument("--out", default="results/sft/chat_samples.json")
    ap.add_argument("--val-by-source", default="results/sft/val_by_source.json")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(argv)

    from quipu import evalsets
    from quipu.config import load_config

    cfg = load_config(args.config)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    path = (evalsets.latest_checkpoint(cfg.train.ckpt_dir) if args.checkpoint == "latest"
            else Path(args.checkpoint))
    if args.max_new_tokens < 1 or args.max_new_tokens >= cfg.model.context:
        print(f"error: --max-new-tokens must be 1..{cfg.model.context - 1}", file=sys.stderr)
        return 2
    result = run(cfg, path, device, Path(args.out), Path(args.val_by_source),
                 max_new_tokens=args.max_new_tokens)
    n = sum(len(v) for v in result["samples"].values())
    print(f"{n} chat samples from {path} -> {args.out}"
          + ("" if result["val_by_source"] else
             f" (no held-out loss: {args.val_by_source} is missing)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
