"""How far does quipu-moe read past its 2,048-token training window without training?

The model uses RoPE, so the context is only a table size; the question is quality.
Each long document is scored at 4k and 8k tokens with four ways of building the RoPE
table, and the loss is reported per position bucket. If longer context is being used,
the loss after 2k should stay at or below the loss before it (more text to lean on);
if the stretch breaks the model, it shoots up.

- none: the same table, extended to more positions (extrapolation)
- pi:   position interpolation, positions divided by the stretch factor
- ntk:  NTK-aware, a larger RoPE base (base * s ** (d / (d - 2)))
- yarn: YaRN, interpolation only on the low-frequency dimensions, plus its
        attention temperature (0.1 ln s + 1, applied through cos/sin)

Text: books from the PG-19 test split (never in Quipu's training data); code: long
Python files from this machine's standard library. Never modifies a checkpoint.

Run: .venv/Scripts/python.exe scripts/context_stretch.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quipu import evalsets                     # noqa: E402
from quipu.config import load_config           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "remote-runs/quipu-moe/checkpoints/quipu-moe/milestones/step_008129.pt"
PG19 = "https://huggingface.co/datasets/emozilla/pg19-test/resolve/main/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
METHODS = ["none", "pi", "ntk", "yarn"]


def rope_table(method: str, length: int, head_dim: int, base: float, train_ctx: int):
    """(cos, sin), each (1, 1, length, head_dim // 2), for a stretch to `length`."""
    s = max(1.0, length / train_ctx)
    dims = torch.arange(0, head_dim, 2).float() / head_dim
    inv = 1.0 / (base ** dims)
    pos = torch.arange(length).float()
    scale = 1.0
    if method == "pi":
        pos = pos / s
    elif method == "ntk":
        inv = 1.0 / ((base * s ** (head_dim / (head_dim - 2))) ** dims)
    elif method == "yarn":
        wavelength = 2 * math.pi / inv
        r = train_ctx / wavelength                       # turns a dimension makes in the training window
        ramp = ((r - 1.0) / (32.0 - 1.0)).clamp(0, 1)    # 0: interpolate fully, 1: leave as trained
        inv = inv / s * (1 - ramp) + inv * ramp
        scale = 0.1 * math.log(s) + 1.0
    elif method != "none":
        raise ValueError(method)
    freqs = torch.outer(pos, inv)
    return (freqs.cos() * scale)[None, None], (freqs.sin() * scale)[None, None]


def pg19_books(cache: Path, n: int) -> list[str]:
    path = cache / "pg19-test.parquet"
    if not path.exists():
        import urllib.request
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PG19, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    import pyarrow.parquet as pq
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return [t[t.find("\n\n", 2000) + 2:] for t in texts[:n]]     # skip the Gutenberg header


def stdlib_code(n: int, min_chars: int) -> list[str]:
    lib = Path(sys.base_prefix) / "Lib"
    files = sorted((p for p in lib.glob("*.py") if p.stat().st_size >= min_chars),
                   key=lambda p: p.name)
    return [p.read_text(encoding="utf-8", errors="replace") for p in files[:n]]


@torch.no_grad()
def position_losses(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-position next-token loss (fp32) for one sequence of length L+1."""
    x, y = ids[None, :-1], ids[None, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    out = []
    for i in range(0, logits.shape[1], 2048):                  # keeps the fp32 copy small
        out.append(F.cross_entropy(logits[0, i:i + 2048].float(), y[0, i:i + 2048], reduction="none"))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8, help="documents per kind")
    ap.add_argument("--lengths", default="4096,8192")
    ap.add_argument("--weights", default=str(CKPT), help="a milestone .pt or an exported model.safetensors")
    ap.add_argument("--out", default=str(ROOT / "results/context_stretch"))
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    longest = max(lengths)

    cfg = load_config(ROOT / "configs/quipu-moe.toml")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / cfg.data.tokenizer))
    kinds = {"books": pg19_books(ROOT / "data/context_stretch", args.docs * 2),
             "code": stdlib_code(args.docs * 3, 40_000)}
    docs: dict[str, list[torch.Tensor]] = {}
    for kind, texts in kinds.items():
        seqs = [torch.tensor(tok.encode(t).ids, dtype=torch.long) for t in texts]
        docs[kind] = [s[:longest + 1] for s in seqs if len(s) > longest][:args.docs]
        print(f"{kind}: {len(docs[kind])} documents of {longest + 1:,}+ tokens", flush=True)

    if args.weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(args.weights)
        state.setdefault("lm_head.weight", state["embed.weight"])
    else:
        state = evalsets.checkpoint_state(args.weights)
    model_cfg = dataclasses.replace(cfg.model, context=longest)
    model = evalsets.load_model(model_cfg, state=state).to(torch.bfloat16).cuda()
    train_ctx = cfg.model.context

    results = []
    runs = [(2048, "none")] + [(L, m) for L in lengths for m in METHODS]
    for L, method in runs:
        cos, sin = rope_table(method, L, cfg.model.head_dim, cfg.model.rope_base, train_ctx)
        model.rope_cos, model.rope_sin = cos.cuda(), sin.cuda()
        t0 = time.time()
        for kind, seqs in docs.items():
            loss = torch.stack([position_losses(model, s[:L + 1].cuda()) for s in seqs]).mean(0).cpu()
            row = {"length": L, "method": method, "kind": kind}
            for lo, hi in BUCKETS:
                if hi <= L:
                    row[f"{lo}-{hi}"] = round(loss[lo:hi].mean().item(), 4)
            results.append(row)
            print(json.dumps(row), f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    cols = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    lines = ["| kind | length | method | " + " | ".join(cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for r in results:
        lines.append(f"| {r['kind']} | {r['length']} | {r['method']} | "
                     + " | ".join(f"{r[c]:.3f}" if c in r else "" for c in cols) + " |")
    (out / "summary.md").write_text(
        "Loss (nats per token) by position, quipu-moe step 8129, trained at 2,048 tokens.\n\n"
        + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
