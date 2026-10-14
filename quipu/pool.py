"""The data v3 pool (spec docs/superpowers/specs/2026-10-14-data-v3-continued-pretraining-design.md,
"Components > 1. Pool builder"): many sources, each built on its own, mixed into rolling
chunks of ~10B tokens.

Layout under the pool root (a local directory, or the Modal Volume "quipu-pool"):

    sources/<name>/state.json        the source's stream position, totals, stats
    sources/<name>/part_NNNNN.bin    its documents, tokenized (uint16, each ends in EOT)
    sources/<name>/prov_NNNNN.jsonl.gz  one provenance line per document of that part
    sources/<name>/hashes.u64        content hashes of every kept document (exact dedup)
    mixed/shard_NNNNN.bin            the training shards (100M tokens), indices continue
    pool.json                        what the chunks have used of every source
    manifest.json                    the training-facing manifest (revisions, attribution)

Two steps, so that parallel builders never write the same files:

1. build_source (one process / container per source) reads the source's stream from
   where its last build stopped and appends unique tokens until it has `target` more.
   Only sources/<name>/ is written. A source whose stream ends is marked exhausted.
2. merge_chunk (one process) decides each source's tokens for the chunk (shares, with
   the epoch rule below), interleaves their documents deterministically and appends
   the chunk's shards to mixed/.

The epoch rule: a source that is exhausted is repeated, its documents read again from
the start, until it has given max_epochs x its unique tokens in all (over every chunk);
after that its share is redistributed over the others in proportion to theirs
(chunk_allocation). A live source is never cut short: when it has not built enough for
its share, merge_chunk raises NeedMore with the tokens each such source still needs,
and the runner builds those and merges again.
"""
from __future__ import annotations

import copy
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

HEX40 = re.compile(r"[0-9a-f]{40}")
HUB_KINDS = ("hf_parquet", "hf_jsonl_gz")


# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class SourceSpec:
    """One source of the pool. opts: every other key of its config table (repo,
    revision, files, text_field, ...), after its defaults table is applied."""
    name: str
    kind: str
    slice: str
    weight: float
    share: float  # fraction of every chunk
    opts: dict[str, Any] = field(default_factory=dict, compare=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.opts.get(key, default)


@dataclass(frozen=True)
class PoolConfig:
    path: str
    tokenizer: str
    tokenizer_sha256: str | None
    vocab_size: int | None
    shard_tokens: int
    chunk_tokens: int
    max_epochs: int
    max_doc_tokens: int
    humaneval_revision: str
    mbpp_revision: str
    lid_model: str | None
    lid_revision: str | None
    stackexchange_cutoff: str
    slices: dict[str, float]
    sources: dict[str, SourceSpec]

    def shares(self) -> dict[str, float]:
        return {n: s.share for n, s in self.sources.items()}


def _date_key(s: str) -> str:
    """'2024-04' or '2024-04-11' as a comparable 'YYYY-MM-DD' (a month is its first day)."""
    s = str(s)
    if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        raise ValueError(f"not a date (YYYY-MM or YYYY-MM-DD): {s!r}")
    return s if len(s) == 10 else s + "-01"


def check_stackexchange_date(name: str, opts: dict[str, Any], cutoff: str) -> None:
    """Refuse a Stack Exchange source whose dump (or file) is dated after the cutoff:
    dumps since June 2024 forbid LLM training. Both dates must be given."""
    for key in ("dump_date", "last_modified"):
        if key not in opts:
            raise ValueError(f"source {name}: a Stack Exchange source needs {key} "
                             f"(the dump must be dated, <= {cutoff})")
        if _date_key(opts[key]) > _date_key(cutoff):
            raise ValueError(f"source {name}: {key} {opts[key]} is after the Stack Exchange "
                             f"cutoff {cutoff}; dumps since June 2024 may not be used to "
                             "train LLMs")


def load_pool_config(path: str | Path) -> PoolConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    p = raw.get("pool", {})
    slices = {k: float(v) for k, v in raw.get("slices", {}).items()}
    if not slices or abs(sum(slices.values()) - 100.0) > 1e-6:
        raise ValueError(f"{path}: [slices] must sum to 100 (got {sum(slices.values())})")
    defaults = raw.get("defaults", {})
    cutoff = str(p.get("stackexchange_cutoff", "2024-04-30"))
    tables: dict[str, dict[str, Any]] = {}
    for name, t in raw.get("sources", {}).items():
        t = copy.deepcopy(t)
        base = t.pop("defaults", None)
        if base is not None:
            if base not in defaults:
                raise ValueError(f"source {name}: no [defaults.{base}]")
            t = {**copy.deepcopy(defaults[base]), **t}
        for key in ("kind", "slice", "weight"):
            if key not in t:
                raise ValueError(f"source {name}: missing {key}")
        if t["slice"] not in slices:
            raise ValueError(f"source {name}: unknown slice {t['slice']!r}")
        if float(t["weight"]) <= 0:
            raise ValueError(f"source {name}: weight must be positive")
        if t["kind"] in HUB_KINDS or t["kind"] == "http_jsonl":
            if not HEX40.fullmatch(str(t.get("revision", ""))):
                raise ValueError(f"source {name}: revision must be pinned to a 40-hex commit; "
                                 f"got {t.get('revision')!r}")
        if t.get("render") == "stackexchange_redpajama" or t["kind"] == "se_posts_xml":
            check_stackexchange_date(name, t, cutoff)
        tables[name] = t
    for s in slices:
        if not any(t["slice"] == s for t in tables.values()):
            raise ValueError(f"slice {s} has no source")
    totals: dict[str, float] = {}
    for t in tables.values():
        totals[t["slice"]] = totals.get(t["slice"], 0.0) + float(t["weight"])
    sources = {}
    for name, t in tables.items():
        share = slices[t["slice"]] / 100.0 * float(t["weight"]) / totals[t["slice"]]
        opts = {k: v for k, v in t.items() if k not in ("kind", "slice", "weight")}
        sources[name] = SourceSpec(name, t["kind"], t["slice"], float(t["weight"]), share, opts)
    for key in ("humaneval_revision", "mbpp_revision"):
        if not HEX40.fullmatch(str(p.get(key, ""))):
            raise ValueError(f"{path}: [pool] {key} must be a 40-hex commit")
    return PoolConfig(
        path=str(path), tokenizer=p.get("tokenizer", "artifacts/tokenizer/tokenizer.json"),
        tokenizer_sha256=p.get("tokenizer_sha256"), vocab_size=p.get("vocab_size"),
        shard_tokens=int(p.get("shard_tokens", 100_000_000)),
        chunk_tokens=int(p.get("chunk_tokens", 10_000_000_000)),
        max_epochs=int(p.get("max_epochs", 4)),
        max_doc_tokens=int(p.get("max_doc_tokens", 16_000)),
        humaneval_revision=p["humaneval_revision"], mbpp_revision=p["mbpp_revision"],
        lid_model=p.get("lid_model"), lid_revision=p.get("lid_revision"),
        stackexchange_cutoff=cutoff, slices=slices, sources=sources)


# ------------------------------------------------------------------ allocation

class SourceTotals(NamedTuple):
    """What the mixer knows of a source: unique tokens built, tokens the chunks have
    used (repeats included) and whether its stream has ended."""
    unique: int
    consumed: int
    exhausted: bool


def epoch_budget(unique: int, consumed: int, max_epochs: int) -> int:
    """Tokens an exhausted source can still give: max_epochs passes over its unique
    tokens in all, minus what the chunks have used."""
    return max(0, max_epochs * unique - consumed)


def _integer(alloc: dict[str, float], total: int) -> dict[str, int]:
    """Floor every allocation, then hand the remaining tokens out by largest remainder
    (ties: config order), so the integers sum to round(sum(alloc))."""
    out = {k: int(math.floor(v + 1e-9)) for k, v in alloc.items()}
    left = total - sum(out.values())
    order = sorted(alloc, key=lambda k: -(alloc[k] - out[k]))
    for k in order[:max(0, left)]:
        out[k] += 1
    return out


def chunk_allocation(total: int, shares: dict[str, float], bounds: dict[str, int]
                     ) -> tuple[dict[str, int], int]:
    """Tokens per source for a chunk of `total`: proportional to `shares`, a source in
    `bounds` (an exhausted source: its epoch budget) never above its bound, the rest of
    its share water-filled over the others in proportion to theirs. Returns
    (allocation, shortfall); shortfall > 0 only when every source is bounded."""
    from quipu.shard_mix import allocate

    weights = {k: v for k, v in shares.items() if v > 0}
    alloc, short = allocate(float(total), weights, available={k: float(v) for k, v in
                                                               bounds.items() if k in weights})
    placed = int(round(total - short))
    ints = _integer(alloc, placed)
    for k, b in bounds.items():  # rounding never lifts a bounded source over its bound
        if k in ints and ints[k] > b:
            ints[k] = b
    short = total - sum(ints.values())
    return {k: ints.get(k, 0) for k in shares}, short


def bounds_for(states: dict[str, SourceTotals], max_epochs: int) -> dict[str, int]:
    return {k: epoch_budget(s.unique, s.consumed, max_epochs)
            for k, s in states.items() if s.exhausted}


def build_targets(total: int, shares: dict[str, float], states: dict[str, SourceTotals], *,
                  max_epochs: int) -> dict[str, int]:
    """New unique tokens each live source must build for a chunk of `total`: its
    allocation minus what it has built and the chunks have not used yet. Exhausted
    sources build nothing; their repeats come from what they already have."""
    want, _ = chunk_allocation(total, shares, bounds_for(states, max_epochs))
    out = {}
    for k, w in want.items():
        s = states.get(k, SourceTotals(0, 0, False))
        if s.exhausted:
            continue
        need = w - (s.unique - s.consumed)
        if need > 0:
            out[k] = need
    return out
