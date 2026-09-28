"""Needle-in-a-haystack test for approach A (retrieval memory), weekend-run spec §5.2.

A 5-digit access code is planted in a long haystack (FineWeb val text, or held-out
code val) and the model is asked to complete the sentence that states it. Quipu sees
1,024 tokens at a time, so without memory it can only answer when the needle sits in
the last window; with memory, retrieval has to bring the needle's chunk into view.

Every trial is scored twice so a failure can be attributed:
- **hit**: was the needle in the window the model saw? (retrieval's job)
- **pass**: does the greedy 8-token continuation contain the exact value? (copying)
"copy | hit" is pass rate among hits, which is the model's share of the blame.

Conditions: `off` (the haystack's tail plus the prompt), and memory on with `bm25`,
`dense` (multilingual-e5-small) or `fused` (reciprocal-rank fusion of both).

How trials stay cheap: a (haystack, size) pair is chunked and indexed once. The
needle then joins the one chunk it lands in (that chunk grows by the needle's
length) instead of shifting every later chunk boundary, so a trial patches exactly
one chunk in each index and restores it afterwards. Offsets stay consistent: the
patched chunk list concatenates back into exactly the planted document.

A size-N haystack is at most N tokens including the needle: the base slice is
N - NEEDLE_RESERVE tokens long. That keeps the 1k case genuinely inside the window.

Never modifies a checkpoint. Writes results/needle/<haystack>_<size>.json and
results/needle/summary.md, atomically.

Run: uv run python scripts/needle_eval.py            (full default run)
     uv run python scripts/needle_eval.py --quick    (smoke run, a minute or two)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from quipu.config import Config, load_config
from quipu.data import read_shard
from quipu.fsio import replace_with_retry
from quipu.memory.bm25 import BM25Index
from quipu.memory.chunker import CHUNK_TOKENS, Chunk, chunk_spans
from quipu.memory.dense import DenseIndex, E5Embedder, Embedder
from quipu.memory.fusion import reciprocal_rank_fusion
from quipu.memory.window import pack_window
from quipu.model import Quipu
from quipu.tokenizer import Tokenizer
from quipu.train import LATEST

HAYSTACK_DIRS = {"text": "val", "code": "code_val"}
CONDITIONS = ("off", "bm25", "dense", "fused")
DEFAULT_SIZES = (1_000, 32_000, 128_000, 1_000_000)
DEFAULT_DEPTHS = (0, 25, 50, 75, 100)
DEFAULT_TRIALS = 20
DEFAULT_SEED = 1234
MAX_NEW_TOKENS = 8
# Tokens kept free in every haystack for the needle; checked per needle, and the
# longest needle (newline + sentence + newline) is ~15 tokens.
NEEDLE_RESERVE = 24
# How deep each retriever's ranking goes. Only three or four chunks fit a window,
# but RRF needs the lists to overlap past the top few to mean anything.
RETRIEVE_K = 32

ADJECTIVES = (
    "amber", "brave", "silent", "crimson", "hidden", "northern", "golden", "frozen",
    "ancient", "hollow", "quiet", "iron", "silver", "broken", "distant", "emerald",
    "velvet", "scarlet", "copper", "misty",
)


# --------------------------------------------------------------------------
# needles
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Needle:
    kind: str
    adjective: str
    value: str
    text: str
    prompt: str


def trial_rng(seed: int, kind: str, size: int, depth: int, trial: int) -> random.Random:
    """One independent, reproducible stream per trial. A string seed is hashed with
    SHA-512 by random.Random, so it's stable across runs and Python processes (unlike
    hash()), and adding a size or depth never changes another trial's needle."""
    return random.Random(f"{seed}:{kind}:{size}:{depth}:{trial}")


def make_needle(kind: str, rng: random.Random) -> Needle:
    adjective = rng.choice(ADJECTIVES)
    value = str(rng.randint(10000, 99999))
    if kind == "text":
        prompt = f"The access code for the {adjective} vault is"
        text = f"{prompt} {value}."
    elif kind == "code":
        prompt = f"{adjective.upper()}_VAULT_CODE ="
        text = f"{prompt} {value}"
    else:
        raise ValueError(f"unknown haystack kind {kind!r}; expected one of {sorted(HAYSTACK_DIRS)}")
    return Needle(kind=kind, adjective=adjective, value=value, text=text, prompt=prompt)


def needle_token_ids(needle: Needle, tok: Tokenizer) -> list[int]:
    """Newline, needle, newline, each encoded on its own so the splice lands on token
    boundaries and never merges with the haystack's neighbouring tokens."""
    nl = tok.encode("\n")
    return nl + tok.encode(needle.text) + nl


def plant(base: np.ndarray, needle_ids: Sequence[int], depth: int) -> tuple[np.ndarray, int]:
    """(planted document, insertion position). depth 0 = before the first token,
    100 = after the last."""
    if not 0 <= depth <= 100:
        raise ValueError(f"depth must be in [0, 100], got {depth}")
    pos = round(depth / 100 * len(base))
    ids = np.asarray(needle_ids, dtype=base.dtype)
    return np.concatenate([base[:pos], ids, base[pos:]]), pos


def needle_chunk_index(pos: int, spans: Sequence[Chunk]) -> int:
    """The chunk the needle joins: the one containing base position `pos`, or the
    last chunk when the needle goes after the final token."""
    for c in spans:
        if c.start <= pos < c.end:
            return c.index
    if pos == spans[-1].end:
        return spans[-1].index
    raise ValueError(f"position {pos} is outside the haystack [0, {spans[-1].end}]")


def planted_chunk(base: np.ndarray, span: Chunk, pos: int, needle_ids: Sequence[int]) -> list[int]:
    return base[span.start:pos].tolist() + list(needle_ids) + base[pos:span.end].tolist()


# --------------------------------------------------------------------------
# sizes, haystacks, model
# --------------------------------------------------------------------------


def parse_size(s: str) -> int:
    s = s.strip()
    mult = 1
    if s[-1:] in ("k", "K"):
        mult, s = 1_000, s[:-1]
    elif s[-1:] in ("m", "M"):
        mult, s = 1_000_000, s[:-1]
    value = int(s) * mult
    if value < 1:
        raise ValueError(f"size must be positive, got {value}")
    return value


def format_size(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def load_haystack(shard_dir: Path, kind: str) -> np.ndarray:
    """Every shard of that split, concatenated in name order. Sizes are contiguous
    prefixes of this, so a larger haystack contains every smaller one."""
    d = Path(shard_dir) / HAYSTACK_DIRS[kind]
    shards = sorted(d.glob("shard_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no shards for the {kind} haystack in {d}")
    return np.concatenate([read_shard(p) for p in shards])


def resolve_checkpoint(cfg: Config, ckpt: str) -> Path:
    if ckpt != "latest":
        return Path(ckpt)
    ckpt_dir = Path(cfg.train.ckpt_dir)
    pointer = torch.load(ckpt_dir / LATEST, map_location="cpu", weights_only=True)
    return ckpt_dir / pointer["file"]


def load_model(cfg: Config, path: Path, device: str) -> Quipu:
    """Read-only. A training checkpoint holds {"model": ..., "optimizer": ...}; a
    milestone is the bare (bf16) state_dict. Either loads strictly, so a mismatched
    file fails loudly rather than evaluating half-random weights."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    model = Quipu(cfg.model)
    model.load_state_dict(state, strict=True)
    assert model.lm_head.weight is model.embed.weight, "lm_head/embed tie was broken"
    return model.to(device).eval()


@torch.no_grad()
def greedy_continue(model: Quipu, tokens: Sequence[int], n: int, device: str) -> list[int]:
    """Argmax decoding of n tokens. Callers guarantee len(tokens) + n <= context, so
    the window is never cropped (cropping would silently drop the needle)."""
    assert len(tokens) + n <= model.cfg.context
    idx = torch.tensor([list(tokens)], dtype=torch.long, device=device)
    out: list[int] = []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        for _ in range(n):
            next_id = model(idx)[:, -1, :].argmax(dim=-1, keepdim=True)
            out.append(int(next_id))
            idx = torch.cat([idx, next_id], dim=1)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def _rates(records: list[dict]) -> dict:
    n = len(records)
    hits = [r for r in records if r["hit"]]
    passes = sum(1 for r in records if r["pass"])
    return {
        "trials": n,
        "hit_rate": len(hits) / n if n else None,
        "copy_given_hit": (sum(1 for r in hits if r["pass"]) / len(hits)) if hits else None,
        "accuracy": passes / n if n else None,
        # A random 5-digit guess is right 1 time in 90,000; anything here is worth a look.
        "passes_without_hit": sum(1 for r in records if r["pass"] and not r["hit"]),
    }


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Per condition: overall rates, rates by depth, median retrieval latency."""
    out: dict[str, dict] = {}
    for cond in dict.fromkeys(r["condition"] for r in records):
        recs = [r for r in records if r["condition"] == cond]
        agg = _rates(recs)
        agg["by_depth"] = {
            str(d): _rates([r for r in recs if r["depth"] == d])
            for d in sorted({r["depth"] for r in recs})
        }
        lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
        agg["latency_ms_median"] = statistics.median(lat) if lat else None
        out[cond] = agg
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    replace_with_retry(tmp, path)


def _cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def run_one(
    *, model: Quipu, tok: Tokenizer, kind: str, haystack: np.ndarray, size: int,
    depths: Sequence[int], trials: int, conditions: Sequence[str], device: str,
    seed: int, embedder: Embedder | None, chunk_tokens: int, log: Callable[[str], None],
) -> dict:
    budget = model.cfg.context - MAX_NEW_TOKENS
    base = haystack[: size - NEEDLE_RESERVE]
    spans = chunk_spans(len(base), chunk_tokens)
    base_chunks = [base[c.start:c.end].tolist() for c in spans]
    sep = tok.encode("\n")

    build_s: dict[str, float | None] = {"bm25": None, "dense": None}
    bm25 = dense = None
    base_texts: list[str] = []
    if {"bm25", "fused"} & set(conditions):
        t0 = time.perf_counter()
        bm25 = BM25Index(base_chunks)
        build_s["bm25"] = time.perf_counter() - t0
    if {"dense", "fused"} & set(conditions):
        t0 = time.perf_counter()
        base_texts = [tok.decode(c) for c in base_chunks]
        dense = DenseIndex(base_texts, embedder)
        build_s["dense"] = time.perf_counter() - t0
    log(f"  {kind} {format_size(size)}: {len(spans)} chunks, index build "
        f"bm25={build_s['bm25'] if build_s['bm25'] is None else round(build_s['bm25'], 2)}s "
        f"dense={build_s['dense'] if build_s['dense'] is None else round(build_s['dense'], 2)}s")

    records: list[dict] = []
    peak: dict[str, float | None] = {c: None for c in conditions}
    for depth in depths:
        for trial in range(trials):
            needle = make_needle(kind, trial_rng(seed, kind, size, depth, trial))
            ids = needle_token_ids(needle, tok)
            if len(ids) > NEEDLE_RESERVE:
                raise ValueError(f"needle is {len(ids)} tokens, over NEEDLE_RESERVE={NEEDLE_RESERVE}")
            prompt = tok.encode(needle.prompt)
            planted, pos = plant(base, ids, depth)
            ci = needle_chunk_index(pos, spans)
            chunks = list(base_chunks)
            chunks[ci] = planted_chunk(base, spans[ci], pos, ids)
            if bm25 is not None:
                bm25.patch(ci, chunks[ci])
            if dense is not None:
                dense.patch(ci, tok.decode(chunks[ci]))

            try:
                for cond in conditions:
                    if _cuda(device):
                        torch.cuda.reset_peak_memory_stats()
                    ranking: list[int] = []
                    latency = None
                    if cond == "off":
                        # Only the last `budget` tokens can matter, so never turn
                        # the whole 1M-token document into a Python list.
                        offset = max(len(planted) - budget, 0)
                        full = planted[offset:].tolist() + sep + prompt
                        cut = max(len(full) - budget, 0)
                        window = full[cut:]
                        hit = pos >= offset + cut
                        window_chunks: list[int] = []
                    else:
                        t0 = time.perf_counter()
                        if cond == "bm25":
                            ranking = bm25.search(prompt, RETRIEVE_K)
                        elif cond == "dense":
                            ranking = dense.search(needle.prompt, RETRIEVE_K)
                        elif cond == "fused":
                            ranking = reciprocal_rank_fusion([
                                bm25.search(prompt, RETRIEVE_K),
                                dense.search(needle.prompt, RETRIEVE_K),
                            ])
                        else:
                            raise ValueError(f"unknown condition {cond!r}")
                        latency = (time.perf_counter() - t0) * 1000.0
                        packed = pack_window(ranking, chunks, prompt, budget, separator=sep)
                        window = list(packed.tokens)
                        window_chunks = list(packed.chunk_ids)
                        hit = ci in packed.chunk_ids
                    cont_ids = greedy_continue(model, window, MAX_NEW_TOKENS, device)
                    cont = tok.decode(cont_ids)
                    if _cuda(device):
                        mb = torch.cuda.max_memory_allocated() / 2**20
                        peak[cond] = mb if peak[cond] is None else max(peak[cond], mb)
                    records.append({
                        "haystack": kind, "size": size, "depth": depth, "trial": trial,
                        "condition": cond, "adjective": needle.adjective, "value": needle.value,
                        "needle_pos": pos, "needle_chunk": ci,
                        "needle_rank": ranking.index(ci) if ci in ranking else None,
                        "window_chunks": window_chunks, "window_tokens": len(window),
                        "hit": bool(hit), "pass": needle.value in cont,
                        "continuation": cont, "latency_ms": latency,
                    })
            finally:
                # Restore the base chunk so the next trial starts from the clean index.
                if bm25 is not None:
                    bm25.patch(ci, base_chunks[ci])
                if dense is not None:
                    dense.patch(ci, base_texts[ci])

    aggregates = aggregate(records)
    for cond in conditions:
        aggregates[cond]["peak_vram_mb"] = peak[cond]
        aggregates[cond]["index_build_s"] = {
            "off": None, "bm25": build_s["bm25"], "dense": build_s["dense"],
            "fused": (build_s["bm25"] or 0.0) + (build_s["dense"] or 0.0)
            if build_s["bm25"] is not None and build_s["dense"] is not None else None,
        }[cond]
    return {
        "haystack": kind, "size": size, "size_label": format_size(size),
        "base_tokens": len(base), "chunks": len(spans), "chunk_tokens": chunk_tokens,
        "window_budget": budget, "depths": list(depths), "trials_per_depth": trials,
        "seed": seed, "index_build_s": build_s, "aggregates": aggregates, "trials": records,
    }


def run(
    *, model: Quipu, tok: Tokenizer, haystacks: dict[str, np.ndarray], sizes: Sequence[int],
    depths: Sequence[int], trials: int, conditions: Sequence[str], device: str, out_dir: Path,
    seed: int = DEFAULT_SEED, embedder_factory: Callable[[], Embedder] | None = None,
    chunk_tokens: int = CHUNK_TOKENS, log: Callable[[str], None] = print,
) -> list[dict]:
    """Evaluate every (haystack, size); write one JSON per pair as it finishes (so a
    crash late in the run keeps the finished pairs) and the summary at the end."""
    unknown = set(conditions) - set(CONDITIONS)
    if unknown or not conditions:
        raise ValueError(f"conditions must be a non-empty subset of {CONDITIONS}, got {conditions!r}")
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    for d in depths:
        if not 0 <= d <= 100:
            raise ValueError(f"depths must be in [0, 100], got {d}")
    for kind, hay in haystacks.items():
        for size in sizes:
            if size <= NEEDLE_RESERVE:
                raise ValueError(f"size {size} must exceed NEEDLE_RESERVE ({NEEDLE_RESERVE})")
            if len(hay) < size:
                raise ValueError(f"the {kind} haystack has {len(hay)} tokens, fewer than size {size}")
    embedder = None
    if {"dense", "fused"} & set(conditions):
        if embedder_factory is None:
            raise ValueError("dense/fused conditions need an embedder_factory")
        embedder = embedder_factory()

    out_dir = Path(out_dir)
    results = []
    for kind, hay in haystacks.items():
        for size in sizes:
            t0 = time.perf_counter()
            res = run_one(
                model=model, tok=tok, kind=kind, haystack=hay, size=size, depths=depths,
                trials=trials, conditions=conditions, device=device, seed=seed,
                embedder=embedder, chunk_tokens=chunk_tokens, log=log,
            )
            res["seconds"] = time.perf_counter() - t0
            _write_text_atomic(out_dir / f"{kind}_{format_size(size)}.json",
                               json.dumps(res, indent=1))
            line = ", ".join(
                f"{c}: hit {a['hit_rate']:.2f} acc {a['accuracy']:.2f}"
                for c, a in res["aggregates"].items()
            )
            log(f"  {kind} {format_size(size)} done in {res['seconds']:.1f}s -- {line}")
            results.append(res)
    write_summary(out_dir)
    return results


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _num(x: float | None, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def write_summary(out_dir: Path) -> None:
    """One table over every result file in out_dir, so runs of different haystacks
    or sizes into the same directory add up rather than overwrite each other."""
    out_dir = Path(out_dir)
    results = []
    for p in out_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "aggregates" in data and "haystack" in data:
            results.append(data)
    results.sort(key=lambda d: (d["haystack"], d["size"]))
    lines = [
        "# Needle-in-a-haystack: approach A (retrieval memory)",
        "",
        "A 5-digit value is planted in a haystack and the model completes the sentence "
        "that states it (greedy, 8 tokens). **hit** = the needle's chunk was in the "
        "window the model saw; **copy|hit** = pass rate among hits; **accuracy** = "
        "overall pass rate. Latency is the median retrieval time per query; VRAM is "
        "peak allocated during that condition (Quipu and, when loaded, the e5 embedder "
        "are both resident). Index build is per (haystack, size).",
        "",
        "| haystack | size | condition | trials | hit | copy\\|hit | accuracy "
        "| accuracy by depth % | latency ms | peak VRAM MB | index build s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in results:
        for cond, a in d["aggregates"].items():
            by_depth = " ".join(f"{k}:{_pct(v['accuracy'])}" for k, v in a["by_depth"].items())
            lines.append(
                f"| {d['haystack']} | {d['size_label']} | {cond} | {a['trials']} "
                f"| {_pct(a['hit_rate'])} | {_pct(a['copy_given_hit'])} | {_pct(a['accuracy'])} "
                f"| {by_depth} | {_num(a['latency_ms_median'], '.1f')} "
                f"| {_num(a.get('peak_vram_mb'), '.0f')} | {_num(a.get('index_build_s'), '.2f')} |"
            )
    if results:
        r = results[0]
        lines += [
            "",
            f"Chunks of {r['chunk_tokens']} tokens; window budget {r['window_budget']} tokens "
            f"(context minus the {MAX_NEW_TOKENS} generated); seed {r['seed']}; "
            f"{r['trials_per_depth']} trials per depth.",
        ]
    _write_text_atomic(out_dir / "summary.md", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--ckpt", default="latest", help="checkpoint path, or 'latest'")
    parser.add_argument("--sizes", default=None, help="comma list, e.g. 1k,32k,128k,1M")
    parser.add_argument("--depths", default=None, help="comma list of percents, e.g. 0,50,100")
    parser.add_argument("--trials", type=int, default=None, help="trials per (haystack, size, depth)")
    parser.add_argument("--haystacks", default="text,code")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default=None, help="default results/needle (results/needle/quick with --quick)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: sizes 1k,32k; depths 0,50,100; 2 trials")
    args = parser.parse_args(argv)

    sizes = [parse_size(s) for s in args.sizes.split(",")] if args.sizes else (
        [1_000, 32_000] if args.quick else list(DEFAULT_SIZES))
    depths = _int_list(args.depths) if args.depths else ([0, 50, 100] if args.quick else list(DEFAULT_DEPTHS))
    trials = args.trials if args.trials is not None else (2 if args.quick else DEFAULT_TRIALS)
    out = Path(args.out) if args.out else Path("results/needle/quick" if args.quick else "results/needle")
    kinds = [k.strip() for k in args.haystacks.split(",") if k.strip()]
    for k in kinds:
        if k not in HAYSTACK_DIRS:
            parser.error(f"unknown haystack {k!r}; choose from {sorted(HAYSTACK_DIRS)}")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    cfg = load_config(args.config)
    device = _pick_device(args.device)
    ckpt = resolve_checkpoint(cfg, args.ckpt)
    print(f"needle_eval: {ckpt} on {device}; sizes {[format_size(s) for s in sizes]}, "
          f"depths {depths}, {trials} trials, conditions {conditions}", flush=True)
    t0 = time.perf_counter()
    model = load_model(cfg, ckpt, device)
    tok = Tokenizer()
    haystacks = {k: load_haystack(Path(cfg.data.shard_dir), k) for k in kinds}
    run(
        model=model, tok=tok, haystacks=haystacks, sizes=sizes, depths=depths, trials=trials,
        conditions=conditions, device=device, out_dir=out, seed=args.seed,
        embedder_factory=lambda: E5Embedder(device=device), chunk_tokens=args.chunk_tokens,
        log=lambda m: print(m, flush=True),
    )
    print(f"needle_eval: done in {time.perf_counter() - t0:.0f}s; wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
