"""Distillation CUDA arm, step 1: the teacher answers every prompt in the prompt set.

Reads data/distill-prompts/prompts.jsonl (scripts/build_distill_prompts.py) and
writes one teacher reply per prompt to data/distill-cuda/generations.jsonl:
id, teacher, revision, response, finish ("eos" | "length"), new_tokens.

- Teacher: Qwen/Qwen2.5-Coder-1.5B-Instruct (Apache-2.0) at a pinned revision,
  bf16, on one GPU, in the teacher's own chat template (one user turn: the prompt).
- Greedy decoding (deterministic), up to --max-new-tokens new tokens.
- Prompts are batched by length; a CUDA out-of-memory halves the batch and retries.
- Resumable: ids already in the output are skipped, and each finished batch is
  appended and flushed, so an interrupted run continues where it stopped.

Run:  uv run python scripts/distill_generate.py [--limit N] [--batch 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEACHER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEACHER_REV = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def load_prompts(path: Path, done: set[str], limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]
    return [r for r in rows if r["id"] not in done]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            out.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            continue  # a torn last line from an interrupted run
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", default="data/distill-prompts/prompts.jsonl")
    ap.add_argument("--out", default="data/distill-cuda/generations.jsonl")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--revision", default=TEACHER_REV)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = load_prompts(ROOT / args.prompts, done_ids(out), args.limit)
    print(f"{len(todo)} prompts to answer", flush=True)
    if not todo:
        return 0

    tok = AutoTokenizer.from_pretrained(args.teacher, revision=args.revision)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.teacher, revision=args.revision,
                                                 dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda").eval()

    texts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
             for r in todo}
    todo.sort(key=lambda r: len(texts[r["id"]]))
    batch, i, t0, n_done, n_tok = args.batch, 0, time.time(), 0, 0
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        while i < len(todo):
            chunk = todo[i:i + batch]
            enc = tok([texts[r["id"]] for r in chunk], return_tensors="pt",
                      padding=True).to("cuda")
            try:
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, temperature=None, top_p=None,
                                         top_k=None, pad_token_id=tok.pad_token_id)
            except torch.cuda.OutOfMemoryError:
                del enc
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                print(f"out of memory: batch -> {batch}", flush=True)
                continue
            new = gen[:, enc["input_ids"].shape[1]:]
            for r, ids in zip(chunk, new):
                ids = ids.tolist()
                eos = [j for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)]
                n = eos[0] if eos else len(ids)
                f.write(json.dumps({"id": r["id"], "teacher": args.teacher,
                                    "revision": args.revision,
                                    "response": tok.decode(ids[:n], skip_special_tokens=True),
                                    "finish": "eos" if eos else "length",
                                    "new_tokens": n}, ensure_ascii=False) + "\n")
                n_tok += n
            f.flush()
            i += len(chunk)
            n_done += len(chunk)
            el = time.time() - t0
            print(f"{n_done}/{len(todo)} answered, {n_tok / el:.0f} tok/s, "
                  f"eta {(len(todo) - n_done) * el / n_done / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
