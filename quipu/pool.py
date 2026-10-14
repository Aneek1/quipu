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


# ------------------------------------------------------------------ per-source state

STATE = "state.json"
HASHES = "hashes.u64"
STATE_VERSION = 1
HTML_CAP_WARMUP_TOKENS = 1_000_000  # as data v2 (scripts/build_shards.py)


class ResumeError(RuntimeError):
    """A saved source (or pool) does not match what this build would make of it."""


class NeedMore(RuntimeError):
    """merge_chunk: live sources have not built enough for the chunk; .targets says
    how many more tokens each must build."""

    def __init__(self, targets: dict[str, int]) -> None:
        self.targets = dict(targets)
        super().__init__("sources must build more before the chunk can be mixed: "
                         + ", ".join(f"{k} +{v:,}" for k, v in sorted(targets.items())))


class Row(NamedTuple):
    """One row of a source stream. text None: the adapter dropped it (meta["_drop"] is
    the stat to count, e.g. "dropped_license"). meta: provenance fields (kept per
    document) and private keys ("_attrib": counted in the attribution summary);
    "language" feeds the HTML cap, the spec's path_field the code path rules. cursor:
    the stream position AFTER this row (JSON), where a resumed build starts."""
    text: str | None
    meta: dict[str, Any]
    cursor: dict[str, Any]


def _nest(flat: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted(flat):
        if ":" in k:
            name, sub = k.split(":", 1)
            out.setdefault(name, {})[sub] = flat[k]
        else:
            out[k] = flat[k]
    return out


def _flat(nested: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in nested.items():
        if isinstance(v, dict):
            for sub, n in v.items():
                out[f"{k}:{sub}"] = n
        else:
            out[k] = v
    return out


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, obj: Any) -> None:
    import json
    import os

    from quipu.fsio import replace_with_retry

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    replace_with_retry(tmp, path)


@dataclass
class SourceState:
    """sources/<name>/state.json. parts: [{"file", "prov", "tokens", "documents"}]."""
    name: str
    identity: dict[str, Any] = field(default_factory=dict)
    cursor: dict[str, Any] | None = None
    exhausted: bool = False
    parts: list[dict[str, Any]] = field(default_factory=list)
    unique_tokens: int = 0
    documents: int = 0
    hashes: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    attribution: dict[str, int] = field(default_factory=dict)
    builds: list[dict[str, Any]] = field(default_factory=list)
    version: int = STATE_VERSION

    @classmethod
    def load(cls, directory: str | Path) -> "SourceState":
        import json
        d = json.loads((Path(directory) / STATE).read_text(encoding="utf-8"))
        return cls(**d)

    @classmethod
    def load_or_new(cls, directory: str | Path, name: str) -> "SourceState":
        p = Path(directory) / STATE
        return cls.load(directory) if p.exists() else cls(name=name)

    def save(self, directory: str | Path) -> None:
        from dataclasses import asdict
        write_json_atomic(Path(directory) / STATE, asdict(self))

    def totals(self, consumed: int) -> SourceTotals:
        return SourceTotals(self.unique_tokens, consumed, self.exhausted)


def recover(directory: Path, st: SourceState) -> int:
    """Put a source directory back to its saved state: delete parts / provenance files
    the state does not list (a build killed between writing a part and saving the
    state) and temp files, and cut hashes.u64 back to the saved count. Returns the
    number of files removed."""
    keep = {p["file"] for p in st.parts} | {p["prov"] for p in st.parts}
    removed = 0
    for pat in ("part_*.bin", "prov_*.jsonl.gz", "*.tmp"):
        for f in directory.glob(pat):
            if f.name not in keep:
                f.unlink()
                removed += 1
    h = directory / HASHES
    size = st.hashes * 8
    if h.exists() and h.stat().st_size != size:
        if h.stat().st_size < size:
            raise ResumeError(f"{h}: {h.stat().st_size // 8} hashes on disk but the state "
                              f"says {st.hashes}")
        with open(h, "r+b") as f:
            f.truncate(size)
    elif not h.exists() and size:
        raise ResumeError(f"{h} is missing (the state says {st.hashes} hashes)")
    return removed


# ------------------------------------------------------------------ document processing

@dataclass(frozen=True)
class BuildSetup:
    """What processing a document needs, picklable for worker processes: a zero-argument
    tokenizer factory, the decontaminator (find(text) -> (benchmark, task, rule) | None),
    the optional language-ID classifier (predict) and the code length limit."""
    tokenizer: Any
    decontam: Any = None
    lid: Any = None
    lid_threshold: float = 0.0
    max_doc_tokens: int = 16_000


OK = "ok"  # a document's outcome: (status, ids, detail); status "ok" or the stat to count


def process_doc(text: str, lid_source: str | None, code: bool, setup: BuildSetup,
                tok: Any) -> tuple[str, Any, str | None]:
    """Decontaminate, language-ID (when lid_source) and tokenize one document."""
    import numpy as np

    from quipu.data import encode_document

    if setup.decontam is not None:
        hit = setup.decontam.find(text)
        if hit is not None:
            return "dropped_contamination", None, hit[0]
    if lid_source is not None and setup.lid is not None:
        from quipu.shard_mix import lid_decision
        label, prob, probs = setup.lid.predict(text)
        _, keep = lid_decision(lid_source, label, prob, probs, setup.lid_threshold)
        if not keep:
            return "dropped_lid", None, label
    ids = encode_document(text, tok)
    if ids is None:
        return "dropped_blank", None, None
    if code and len(ids) > setup.max_doc_tokens:
        return "dropped_too_long", None, None
    return OK, np.asarray(ids, dtype=np.uint16), None


_WORKER: dict[str, Any] = {}


def _init_worker(setup: BuildSetup) -> None:
    import os

    from quipu.shard_mix import WORKER_ENV
    for k, v in WORKER_ENV.items():
        os.environ[k] = v
    _WORKER["setup"] = setup
    _WORKER["tok"] = setup.tokenizer()


def _work(job: tuple[list[str], str | None, bool]) -> list[tuple[str, Any, str | None]]:
    texts, lid_source, code = job
    setup, tok = _WORKER["setup"], _WORKER["tok"]
    return [process_doc(t, lid_source, code, setup, tok) for t in texts]


def content_hash(text: str) -> int:
    """Exact dedup identity (data v2's): blake2b-8 of the UTF-8 text."""
    from quipu.shard_mix import content_hash as h
    return h(text)


_MINIFIED_RE = re.compile(r"\.min\.(js|css)$|\.map$", re.IGNORECASE)
_VENDORED_DIRS = frozenset(["vendor", "node_modules", "dist", "build", "bower_components",
                            "third_party", "external"])
_VENDORED_NAME_PARTS = ("bundle", ".pack.")


def code_path_reason(path: str | None) -> str | None:
    """data v2's path rules (scripts/build_shards.py CODE_PATH_SKIP_RULES, the same
    patterns; tests/test_pool_sources.py checks they agree): "minified", "vendored" or
    None."""
    if not path:
        return None
    parts = [p for p in str(path).replace("\\", "/").lower().split("/") if p]
    if not parts:
        return None
    name = parts[-1]
    if _MINIFIED_RE.search(name):
        return "minified"
    if _VENDORED_DIRS.intersection(parts[:-1]) or any(k in name for k in _VENDORED_NAME_PARTS):
        return "vendored"
    return None


# ------------------------------------------------------------------ the builder

def _part_names(index: int) -> tuple[str, str]:
    return f"part_{index:05d}.bin", f"prov_{index:05d}.jsonl.gz"


def build_source(out_dir: str | Path, spec: SourceSpec, target: int, *, setup: BuildSetup,
                 open_rows: Any, shard_tokens: int, workers: int = 0, batch_docs: int = 256,
                 limit_docs: int | None = None, identity: dict[str, Any] | None = None,
                 on_checkpoint: Any = None, log: Any = None) -> dict[str, Any]:
    """Append at least `target` new unique tokens (stopping after the document that
    reaches it) of `spec`'s stream to out_dir, continuing from the saved cursor.

    open_rows(cursor) yields Rows from that position (None: the start). Each row is,
    in order: dropped by the adapter (text None); blank; a code path rule (minified /
    vendored; spec opts code + path_field); an exact duplicate (blake2b-8 of the text,
    against every document this source ever kept); contaminated (setup.decontam, every
    source); another language (setup.lid, when spec opts lid names the language); too
    long (code only); over the HTML cap (spec opts html_cap); or kept. Processing runs in
    `workers` processes (0: in this process); decisions are made in stream order, so the
    output does not depend on the number of workers.

    Kept documents are buffered and written as a part (cut at the first document end
    at or after shard_tokens), with one provenance line per document. The state is saved
    after every part, so a killed build resumes from its last part (the rows after it are
    read again). The stream ending marks the source exhausted. `identity` (revision, file
    list hash) must match the saved one. on_checkpoint() is called after every save
    (a Modal Volume commit). limit_docs stops after that many kept documents (smoke
    runs; not exhaustion). Returns the build's summary.
    """
    import gzip
    import json
    import time
    from collections import Counter, deque

    import numpy as np

    from quipu.data import write_shard
    from quipu.fsio import replace_with_retry

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = SourceState.load_or_new(out, spec.name)
    identity = dict(identity or {})
    if st.identity and identity:
        diff = sorted(k for k in set(st.identity) | set(identity)
                      if st.identity.get(k) != identity.get(k))
        if diff:
            raise ResumeError(f"source {spec.name}: the saved build differs in "
                              f"{', '.join(diff)} ({ {k: st.identity.get(k) for k in diff} } "
                              f"vs { {k: identity.get(k) for k in diff} }); build it in a new "
                              "directory")
    if identity:
        st.identity = identity
    recover(out, st)
    started, t0, cpu0 = _now(), time.time(), time.process_time()
    summary: dict[str, Any] = {"source": spec.name, "target": target, "tokens": 0,
                               "documents": 0, "rows": 0, "exhausted": st.exhausted}
    if st.exhausted or target <= 0:
        return summary

    stats: Counter = Counter(_flat(st.stats))
    attribution: Counter = Counter(st.attribution)
    seen: set[int] = set()
    hpath = out / HASHES
    if hpath.exists():
        seen.update(np.fromfile(hpath, dtype="<u8").tolist())
    lid_source = spec.get("lid")
    code = bool(spec.get("code"))
    path_field = spec.get("path_field")
    html_cap = spec.get("html_cap")
    html_warmup = int(spec.get("html_warmup", HTML_CAP_WARMUP_TOKENS))
    cursor = st.cursor

    buf: list[Any] = []
    prov: list[str] = []
    new_hashes: list[int] = []
    buf_tokens = 0
    built = docs_this = rows_this = 0
    process_seconds = 0.0

    def save() -> None:
        nonlocal buf, prov, new_hashes, buf_tokens
        if buf:
            name, pname = _part_names(len(st.parts))
            write_shard(out / name, np.concatenate(buf))
            tmp = out / (pname + ".tmp")
            tmp.write_bytes(gzip.compress("".join(prov).encode("utf-8"), compresslevel=6))
            replace_with_retry(tmp, out / pname)
            st.parts.append({"file": name, "prov": pname, "tokens": buf_tokens,
                             "documents": len(prov)})
        if new_hashes:
            with open(hpath, "ab") as f:
                np.asarray(new_hashes, dtype="<u8").tofile(f)
            st.hashes += len(new_hashes)
        buf, prov, new_hashes, buf_tokens = [], [], [], 0
        st.cursor = cursor
        st.stats = _nest(dict(stats))
        st.attribution = dict(attribution)
        st.save(out)
        if on_checkpoint is not None:
            on_checkpoint()

    executor = None
    if workers > 0:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(workers,
                                       mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_init_worker, initargs=(setup,))
    tok = setup.tokenizer() if executor is None else None

    def prepare(rows: list[Row]) -> tuple[list[Row], list[Any], Any]:
        """The cheap filters, here; the rest is sent to the workers (or done inline)."""
        nonlocal process_seconds
        pre: list[Any] = []
        texts: list[str] = []
        for r in rows:
            if r.text is None:
                pre.append((r.meta.get("_drop") or "dropped_by_source", None))
                continue
            if not r.text.strip():
                pre.append(("dropped_blank", None))
                continue
            if code and path_field:
                reason = code_path_reason(r.meta.get(path_field))
                if reason is not None:
                    pre.append((f"skipped_{reason}", None))
                    continue
            h = content_hash(r.text)
            if h in seen:
                pre.append(("dropped_duplicate", None))
                continue
            pre.append((None, h))
            texts.append(r.text)
        if executor is None:
            c0 = time.process_time()
            res = [process_doc(t, lid_source, code, setup, tok) for t in texts]
            process_seconds += time.process_time() - c0
            return rows, pre, res
        return rows, pre, executor.submit(_work, (texts, lid_source, code))

    def batches() -> Any:
        batch: list[Row] = []
        for r in open_rows(cursor):
            batch.append(r)
            if len(batch) >= batch_docs:
                yield batch
                batch = []
        if batch:
            yield batch

    done = ended = False
    pending: deque = deque()
    source_it = batches()
    depth = max(2, 2 * workers)
    try:
        while not done:
            while len(pending) < depth:
                try:
                    pending.append(prepare(next(source_it)))
                except StopIteration:
                    break
            if not pending:
                ended = True
                break
            rows, pre, res = pending.popleft()
            results = res if isinstance(res, list) else res.result()
            k = 0
            for r, (status, h) in zip(rows, pre):
                rows_this += 1
                stats["rows"] += 1
                cursor = r.cursor
                if status is not None:
                    stats[status] += 1
                    continue
                status, ids, detail = results[k]
                k += 1
                if status != OK:
                    stats[status] += 1
                    if detail is not None:
                        stats[f"{status}_as:{detail}"] += 1
                    continue
                if h in seen:  # a duplicate of a document in the batches in flight
                    stats["dropped_duplicate"] += 1
                    continue
                n = len(ids)
                if html_cap is not None and r.meta.get("language") == "HTML":
                    total = st.unique_tokens
                    if (total >= html_warmup
                            and stats["html_tokens"] + n > html_cap * (total + n)):
                        stats["skipped_html_cap"] += 1
                        continue
                    stats["html_tokens"] += n
                seen.add(h)
                new_hashes.append(h)
                buf.append(ids)
                meta = {k2: v for k2, v in r.meta.items() if not k2.startswith("_")}
                prov.append(json.dumps(meta, ensure_ascii=False, default=str) + "\n")
                if r.meta.get("_attrib") is not None:
                    attribution[str(r.meta["_attrib"])] += 1
                buf_tokens += n
                built += n
                docs_this += 1
                st.unique_tokens += n
                st.documents += 1
                stats["kept"] += 1
                if buf_tokens >= shard_tokens:
                    save()
                if built >= target or (limit_docs is not None and docs_this >= limit_docs):
                    done = True
                    break
            if log is not None:
                log(spec.name, built, target, rows_this)
        if ended:
            st.exhausted = True
        save()
    finally:
        for _, _, res in pending:
            if not isinstance(res, list):
                res.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        source_it.close()

    wall = time.time() - t0
    entry = {"started": started, "finished": _now(), "target": target, "tokens": built,
             "documents": docs_this, "rows": rows_this, "wall_seconds": round(wall, 3),
             "cpu_seconds_main": round(time.process_time() - cpu0, 3),
             "process_seconds": round(process_seconds, 3) if executor is None else None,
             "workers": workers, "exhausted": st.exhausted, "cursor": st.cursor}
    st.builds.append(entry)
    st.save(out)
    if on_checkpoint is not None:
        on_checkpoint()
    summary.update(tokens=built, documents=docs_this, rows=rows_this, exhausted=st.exhausted,
                   wall_seconds=entry["wall_seconds"], process_seconds=entry["process_seconds"],
                   cpu_seconds_main=entry["cpu_seconds_main"])
    return summary


# ------------------------------------------------------------------ the pool and the mixer

SOURCES = "sources"
MIXED = "mixed"
POOL_FILE = "pool.json"
MANIFEST = "manifest.json"
POOL_VERSION = 1


def source_dir(root: str | Path, name: str) -> Path:
    return Path(root) / SOURCES / name


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.bin"


@dataclass
class PoolState:
    """pool.json: tokens each source has given to the chunks (repeats included), the
    next mixed shard index, and every chunk's record."""
    consumed: dict[str, int] = field(default_factory=dict)
    next_shard: int = 0
    chunks: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    version: int = POOL_VERSION

    @classmethod
    def load(cls, root: str | Path) -> "PoolState":
        import json
        p = Path(root) / POOL_FILE
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def save(self, root: str | Path) -> None:
        from dataclasses import asdict
        write_json_atomic(Path(root) / POOL_FILE, asdict(self))


def load_states(root: str | Path, names: Any) -> dict[str, SourceState | None]:
    out: dict[str, SourceState | None] = {}
    for n in names:
        d = source_dir(root, n)
        out[n] = SourceState.load(d) if (d / STATE).exists() else None
    return out


def pool_totals(root: str | Path, cfg: PoolConfig, ps: PoolState | None = None
                ) -> dict[str, SourceTotals]:
    ps = ps or PoolState.load(root)
    out = {}
    for n, st in load_states(root, cfg.sources).items():
        c = int(ps.consumed.get(n, 0))
        out[n] = st.totals(c) if st is not None else SourceTotals(0, c, False)
    return out


def missing_tokens(total: int, cfg: PoolConfig, totals: dict[str, SourceTotals]
                   ) -> tuple[dict[str, int], dict[str, int], int]:
    """(allocation, what each live source still lacks for it, shortfall)."""
    want, short = chunk_allocation(total, cfg.shares(), bounds_for(totals, cfg.max_epochs))
    need = {}
    for n, w in want.items():
        t = totals[n]
        if not t.exhausted and t.unique - t.consumed < w:
            need[n] = w - (t.unique - t.consumed)
    return want, need, short


def fit_chunk_tokens(root: str | Path, cfg: PoolConfig) -> int:
    """The largest chunk the sources' built tokens can fill without building more
    (smoke runs: a few hundred documents per source set the chunk size)."""
    totals = pool_totals(root, cfg)
    lo, hi = 0, max(1, sum(t.unique for t in totals.values()) * cfg.max_epochs)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if missing_tokens(mid, cfg, totals)[1]:
            hi = mid - 1
        else:
            lo = mid
    return lo


class _Docs:
    """A source's documents from token position `start` (a document boundary), read
    from its parts (memory-mapped); with wrap, the stream starts again at the first
    document when it ends (the repeat epochs of an exhausted source)."""

    def __init__(self, directory: Path, st: SourceState, start: int, eot: int,
                 wrap: bool) -> None:
        self.dir, self.parts, self.eot, self.wrap = directory, st.parts, eot, wrap
        self.unique = st.unique_tokens
        if self.unique <= 0:
            raise ValueError(f"{st.name}: no tokens built")
        self.pos = start % self.unique if wrap else start
        if not wrap and start > self.unique:
            raise ValueError(f"{st.name}: position {start} beyond its {self.unique} tokens")
        self._load(self.pos)

    def _load(self, pos: int) -> None:
        import numpy as np
        acc = 0
        for i, p in enumerate(self.parts):
            if pos < acc + p["tokens"]:
                self.part, self.base = i, acc
                self.arr = np.memmap(self.dir / p["file"], dtype="<u2", mode="r")
                self.ends = np.flatnonzero(self.arr == self.eot)
                off = pos - acc
                if off and self.arr[off - 1] != self.eot:
                    raise ValueError(f"{self.dir.name}: position {pos} is not a document end")
                self.k = int(np.searchsorted(self.ends, off))
                return
            acc += p["tokens"]
        self.part = None  # at the end

    def __iter__(self) -> "_Docs":
        return self

    def __next__(self) -> Any:
        while True:
            if self.part is None:
                if not self.wrap or self.pos == 0:
                    raise StopIteration
                self.pos = 0
                self._load(0)
                continue
            if self.k < len(self.ends):
                lo = 0 if self.k == 0 else int(self.ends[self.k - 1]) + 1
                hi = int(self.ends[self.k]) + 1
                self.k += 1
                self.pos = self.base + hi
                return self.arr[lo:hi]
            nxt = self.base + self.parts[self.part]["tokens"]
            self.pos = nxt
            self._load(nxt)
            if self.part is None and self.wrap:
                self.pos = 0
                self._load(0)


def _recover_mixed(mixed: Path, next_shard: int) -> int:
    """Delete shards a killed merge wrote beyond pool.json's next_shard, and temp files."""
    removed = 0
    if not mixed.exists():
        return 0
    for f in list(mixed.glob("shard_*.bin")) + list(mixed.glob("*.tmp")):
        m = re.fullmatch(r"shard_(\d+)\.bin", f.name)
        if m is None or int(m.group(1)) >= next_shard:
            f.unlink()
            removed += 1
    return removed


def merge_chunk(root: str | Path, cfg: PoolConfig, chunk_tokens: int | None = None, *,
                eot: int, provenance: dict[str, Any] | None = None,
                log: Any = None) -> dict[str, Any]:
    """Mix the next chunk of `chunk_tokens` (default cfg.chunk_tokens) from what the
    sources have built and append it to mixed/ as shards of cfg.shard_tokens, indices
    continuing from the last chunk.

    Allocation: chunk_allocation over the shares, exhausted sources bounded by their
    epoch budget. A live source that has not built its allocation raises NeedMore
    (nothing is written). Before each document the source furthest below its
    allocated share of the tokens written so far supplies it (ties: config order);
    a source stops at its allocation (its last document may pass it), and the chunk
    ends at the first document end at or after its total, so no document is ever cut.
    The last shard of a chunk holds the remainder. pool.json is saved only after the
    shards are written; shards past it (a killed merge) are deleted first, so a rerun
    writes the same bytes.
    """
    import numpy as np

    from quipu.data import write_shard

    root = Path(root)
    T = int(chunk_tokens if chunk_tokens is not None else cfg.chunk_tokens)
    ps = PoolState.load(root)
    mixed = root / MIXED
    _recover_mixed(mixed, ps.next_shard)
    states = load_states(root, cfg.sources)
    totals = pool_totals(root, cfg, ps)
    want, need, short = missing_tokens(T, cfg, totals)
    if need:
        raise NeedMore(need)
    total = T - short
    order = [n for n in cfg.sources if want[n] > 0]
    if not order:
        raise RuntimeError("no source has tokens for this chunk")
    readers = {n: _Docs(source_dir(root, n), states[n], ps.consumed.get(n, 0), eot,
                        wrap=states[n].exhausted) for n in order}
    shares = {n: want[n] / total for n in order}
    taken = dict.fromkeys(order, 0)
    docs = dict.fromkeys(order, 0)
    active = list(order)
    mixed.mkdir(parents=True, exist_ok=True)
    first = ps.next_shard
    index = first
    buf = np.empty(cfg.shard_tokens, dtype=np.uint16)
    fill = written = 0
    shards: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal fill, index
        write_shard(mixed / shard_name(index), buf[:fill].copy())
        shards.append({"file": shard_name(index), "tokens": fill})
        index += 1
        fill = 0

    while written < total:
        if not active:
            raise RuntimeError(f"chunk ran out of documents after {written:,} of {total:,} "
                               "tokens")
        best = max(active, key=lambda n: (shares[n] * written - taken[n], -order.index(n)))
        try:
            doc = next(readers[best])
        except StopIteration:
            raise RuntimeError(f"{best} ran out of documents after {taken[best]:,} tokens "
                               f"(allocated {want[best]:,})") from None
        pos = 0
        while pos < len(doc):
            take = min(cfg.shard_tokens - fill, len(doc) - pos)
            buf[fill:fill + take] = doc[pos:pos + take]
            fill += take
            pos += take
            if fill == cfg.shard_tokens:
                flush()
        taken[best] += len(doc)
        docs[best] += 1
        written += len(doc)
        if taken[best] >= want[best]:
            active.remove(best)
        if log is not None and docs[best] % 100_000 == 0:
            log(written, total)
    if fill:
        flush()

    by_source = {}
    for n in cfg.sources:
        st = states[n]
        c0 = ps.consumed.get(n, 0)
        got = taken.get(n, 0)
        u = st.unique_tokens if st is not None else 0
        by_source[n] = {"target_share": cfg.sources[n].share, "allocated": want[n],
                        "tokens": got, "documents": docs.get(n, 0),
                        "share": round(got / written, 6) if written else 0.0,
                        "epochs_before": round(c0 / u, 4) if u else 0.0,
                        "epochs_after": round((c0 + got) / u, 4) if u else 0.0,
                        "exhausted": bool(st and st.exhausted)}
        ps.consumed[n] = c0 + got
    rec = {"index": len(ps.chunks), "target_tokens": T, "tokens": written,
           "shortfall": short, "shards": [first, index - 1], "finished": _now(),
           "by_source": by_source}
    ps.chunks.append(rec)
    ps.next_shard = index
    if provenance:
        ps.provenance = provenance
    ps.save(root)
    write_manifest(root, cfg, ps)
    return rec


def write_manifest(root: str | Path, cfg: PoolConfig, ps: PoolState | None = None,
                   top: int = 50) -> dict[str, Any]:
    """manifest.json: the mixed shards, every chunk, and per source its revision,
    unique tokens and documents, repeats, filter counts and attribution summary."""
    import hashlib
    import json

    root = Path(root)
    ps = ps or PoolState.load(root)
    states = load_states(root, cfg.sources)
    sources = {}
    for n, s in cfg.sources.items():
        st = states[n]
        u = st.unique_tokens if st else 0
        c = ps.consumed.get(n, 0)
        attr = sorted((st.attribution if st else {}).items(), key=lambda kv: (-kv[1], kv[0]))
        sources[n] = {
            "kind": s.kind, "slice": s.slice, "share": s.share,
            **{k: s.opts[k] for k in ("repo", "url", "via", "revision", "files", "licence",
                                      "dump_date", "last_modified", "path", "split", "lid")
               if k in s.opts},
            "identity": st.identity if st else {},
            "unique_tokens": u, "documents": st.documents if st else 0,
            "exhausted": bool(st and st.exhausted), "consumed": c,
            "epochs_used": round(c / u, 4) if u else 0.0,
            "cursor": st.cursor if st else None, "stats": st.stats if st else {},
            "attribution": {"distinct": len(attr), "top": dict(attr[:top])},
            "builds": len(st.builds) if st else 0,
        }
    shards = []
    for rec in ps.chunks:
        lo, hi = rec["shards"]
        for i in range(lo, hi + 1):
            p = root / MIXED / shard_name(i)
            shards.append({"file": shard_name(i), "chunk": rec["index"],
                           "tokens": p.stat().st_size // 2 if p.exists() else None})
    cfg_bytes = Path(cfg.path).read_bytes() if Path(cfg.path).exists() else b""
    m = {"format": "data v3 pool (quipu/pool.py)", "version": POOL_VERSION,
         "config": cfg.path, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
         "dtype": "uint16 little-endian", "shard_dir": MIXED,
         "shard_tokens": cfg.shard_tokens, "max_epochs": cfg.max_epochs,
         "total_tokens": sum(r["tokens"] for r in ps.chunks), "shards": shards,
         "chunks": ps.chunks, "provenance": ps.provenance, "sources": sources,
         "written": _now()}
    write_json_atomic(root / MANIFEST, m)
    return m


def run_chunk(root: str | Path, cfg: PoolConfig, build: Any, *, chunk_tokens: int | None = None,
              eot: int, max_rounds: int = 5, provenance: dict[str, Any] | None = None,
              log: Any = print) -> dict[str, Any]:
    """Build what the next chunk needs (build(targets): {source: new tokens}; the caller
    runs the sources, in parallel or not), then merge it. A source that runs dry while
    building gives its share to the others, which may then need more: another round."""
    T = int(chunk_tokens if chunk_tokens is not None else cfg.chunk_tokens)
    targets = build_targets(T, cfg.shares(), pool_totals(root, cfg), max_epochs=cfg.max_epochs)
    for rnd in range(max_rounds):
        if targets:
            if log:
                log(f"round {rnd + 1}: building {len(targets)} sources, "
                    f"{sum(targets.values()):,} tokens")
            build(targets)
        try:
            return merge_chunk(root, cfg, T, eot=eot, provenance=provenance)
        except NeedMore as exc:
            targets = exc.targets
    raise RuntimeError(f"chunk not mixed after {max_rounds} build rounds; still missing "
                       f"{targets}")
