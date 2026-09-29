"""The quipu-moe data mix (spec sections 5 and 11): weighted sampling, parallel
tokenisation and language-ID filtering, for scripts/build_shards.py.

Everything here is deterministic: a rebuild with the same sources writes the same
bytes, whatever the number of worker processes. The pieces:

- allocate: water-fills a token total over weighted buckets, bounded by what each
  bucket has and by hard caps (HTML <= html_cap of code). It is used both while
  collecting (to set quotas) and at the end (to fix each bucket's final tokens), so
  the cap is re-applied after every redistribution, never just once up front.

- Collector: admits documents of one stream (the mixed code stream, or one text
  language) by bucket until each bucket's quota is met. Quotas carry `slack` extra
  (default 10%) so that, if a bucket ends short, the others hold enough to cover it.
  A bucket is exhausted ONLY on a genuine stall: its gain stayed under stall_gain
  of its original quota for stall_windows consecutive windows (about one parquet
  file each), i.e. it has all but vanished from the stream. At the default 1e-4 a
  stalled bucket would need over 10,000 windows to fill. There is deliberately no
  time limit: a common language that needs hundreds of windows (Python at a 5B-token
  code budget) is never starved; the hard file cap on the source is the only other
  stop. An exhausted bucket's quota is frozen at what it holds and the remainder is
  re-allocated over the others.

- Runner: tokenises (and LID-labels) documents in `workers` processes (spawn, so it
  behaves the same on Windows and Linux). Batches are submitted in stream order and
  their results consumed in the same order (an ordered map with a bounded window),
  and every admission decision is made in the main process in stream order, so the
  output does not depend on scheduling. As an optimisation, a batch's documents of
  buckets that were already full when the batch was submitted are not tokenised; if
  a later redistribution makes such a document wanted after all, the main process
  tokenises it itself. Either way the decision and the bytes are the same.

- BucketStore: each collected bucket is appended to a flat uint16 file with its
  document ends, so the final train split can interleave buckets by their final
  shares (weighted_interleave) without holding anything in memory.

Language ID (spec 11): lid_decision keeps a document when the classifier's top label
is its source language (cmn_Hani: zho_Hans or zho_Hant), drops it when a different
label wins with probability >= threshold, and keeps it when the classifier is unsure.
cmn_Hani is split into zho_Hans / zho_Hant by the classifier's label (or, when unsure,
by whichever of the two it rates higher).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import multiprocessing
import os
import unicodedata
from array import array
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import numpy as np

from quipu.data import encode_document

CODE = "code"
TEXT = "text"
ENGLISH = "eng_Latn"
ZH_SOURCE = "cmn_Hani"
ZH_BUCKETS = ("zho_Hans", "zho_Hant")

# Per-document outcomes.
OK = "ok"
DEFERRED = "deferred"      # its bucket was full at submission: not tokenised (yet)
BLANK = "blank"
TOO_LONG = "too_long"
LEAK = "leak"
LID_DROP = "lid_drop"

DEFAULT_SLACK = 0.10
DEFAULT_WINDOW = 40_000        # offers per window: about one github-code-clean file
DEFAULT_STALL_WINDOWS = 10
DEFAULT_STALL_GAIN = 1e-4      # of the bucket's original quota per window
DEFAULT_LID_THRESHOLD = 0.5
LID_MAX_CHARS = 2000


def content_hash(code: str) -> int:
    """Exact-content identity for dedup: blake2b-8 of the raw UTF-8 source."""
    digest = hashlib.blake2b(code.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


# ------------------------------------------------------------------ allocation

def allocate(total: float, weights: dict[str, float], available: dict[str, float] | None = None,
             caps: dict[str, float] | None = None) -> tuple[dict[str, float], float]:
    """Split `total` over `weights` (proportionally), water-filling around bounds.

    A bucket's bound is min(available[b], caps[b] * total) (missing = unbounded).
    Each round gives every unbounded bucket its weight's share of what is left; any
    bucket over its bound is fixed AT its bound and leaves the round, and the rest
    is shared again among the remaining buckets. So a short bucket's remainder goes
    to the others in proportion to their weights, and a capped bucket (HTML) never
    receives any of it beyond its cap: the cap is re-applied after every
    redistribution. Returns (allocation, shortfall), shortfall > 0 when every
    bucket hit its bound before `total` was placed.
    """
    available = available or {}
    caps = caps or {}
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights or any(w <= 0 for w in weights.values()):
        raise ValueError("weights must be positive")

    def bound(b: str) -> float:
        cap = caps[b] * total if b in caps else math.inf
        return min(available.get(b, math.inf), cap)

    alloc: dict[str, float] = {}
    active = list(weights)
    left = float(total)
    while active and left > 0:
        w = sum(weights[b] for b in active)
        trial = {b: left * weights[b] / w for b in active}
        over = [b for b in active if trial[b] > bound(b)]
        if not over:
            alloc.update(trial)
            left = 0.0
            break
        for b in over:
            alloc[b] = bound(b)
            left -= alloc[b]
        active = [b for b in active if b not in over]
    return {b: alloc.get(b, 0.0) for b in weights}, max(left, 0.0)


def share_violations(achieved: dict[str, float], weights: dict[str, float], *,
                     min_weight: float = 0.05, max_off: float = 0.02,
                     what: str = "code") -> list[str]:
    """Buckets of weight >= min_weight whose achieved share is off by more than max_off."""
    out = []
    for b, w in weights.items():
        if w < min_weight:
            continue
        got = achieved.get(b, 0.0)
        if abs(got - w) > max_off + 1e-12:
            out.append(f"{b}: {got:.2%} of {what}, target {w:.2%} "
                       f"(off by {abs(got - w) * 100:.2f} pp > {max_off * 100:g} pp)")
    return out


class ShareError(RuntimeError):
    """The achieved mix is too far from the configured weights (see share_violations)."""


# ------------------------------------------------------------------ collection

class BucketStore:
    """Per-bucket token files: <dir>/<i>.bin (uint16, documents back to back) plus the
    end offset of every document. Append while collecting; read back with docs()."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in list(self.dir.glob("*.bin")):
            p.unlink()
        self._files: dict[str, Any] = {}
        self._names: dict[str, str] = {}
        self._ends: dict[str, array] = {}
        self.tokens: Counter[str] = Counter()

    def append(self, bucket: str, ids: np.ndarray) -> None:
        f = self._files.get(bucket)
        if f is None:
            name = f"{len(self._names):03d}.bin"  # bucket names may not be file-safe
            self._names[bucket] = name
            f = self._files[bucket] = open(self.dir / name, "wb")
            self._ends[bucket] = array("q")
        f.write(ids.astype("<u2", copy=False).tobytes())
        self.tokens[bucket] += len(ids)
        self._ends[bucket].append(self.tokens[bucket])

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {b: None for b in self._files}

    def documents(self, bucket: str) -> int:
        return len(self._ends.get(bucket, ()))

    def docs(self, bucket: str) -> Iterator[np.ndarray]:
        """The bucket's documents in collection order (memory-mapped)."""
        if self.tokens[bucket] == 0:
            return iter(())
        if self._files.get(bucket) is not None:
            raise RuntimeError("close() the store before reading it")
        data = np.memmap(self.dir / self._names[bucket], dtype="<u2", mode="r")
        ends = self._ends[bucket]

        def gen() -> Iterator[np.ndarray]:
            start = 0
            for end in ends:
                yield data[start:end]
                start = end
        return gen()


class Collector:
    """Admit documents by bucket until each bucket's quota is met (see module docstring).

    total: the stream's token target. quota (base) comes from allocate(total, weights,
    caps), with exhausted buckets bounded at what they hold; a bucket is admitted while
    under base * (1 + slack). done: every bucket not exhausted holds at least its base.
    """

    def __init__(self, weights: dict[str, float], total: float, *,
                 caps: dict[str, float] | None = None, slack: float = DEFAULT_SLACK,
                 window: int = DEFAULT_WINDOW, stall_windows: int = DEFAULT_STALL_WINDOWS,
                 stall_gain: float = DEFAULT_STALL_GAIN,
                 store: BucketStore | None = None) -> None:
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"weights sum to {sum(weights.values())}, not 1")
        if total <= 0 or slack < 0 or window <= 0 or stall_windows <= 0 or stall_gain < 0:
            raise ValueError("total, window and stall_windows must be positive; "
                             "slack and stall_gain non-negative")
        self.weights = dict(weights)
        self.total = float(total)
        self.caps = dict(caps or {})
        self.slack = slack
        self.window = window
        self.stall_windows = stall_windows
        self.stall_gain = stall_gain
        self.store = store
        self.initial = {b: w * self.total for b, w in weights.items()}
        self.taken: dict[str, int] = dict.fromkeys(weights, 0)
        self.docs: Counter[str] = Counter()
        self.exhausted: dict[str, dict[str, Any]] = {}
        self.offers = 0
        self.windows = 0
        self.redistributions = 0
        self._stalled: Counter[str] = Counter()
        self._window_start = dict(self.taken)
        self._requota()

    def _requota(self) -> None:
        held = {b: float(self.taken[b]) for b in self.exhausted}
        self.base, _ = allocate(self.total, self.weights, held, self.caps)
        self.limit = {b: self.base[b] if b in self.exhausted else self.base[b] * (1 + self.slack)
                      for b in self.weights}

    def wants(self, bucket: str) -> bool:
        return (bucket in self.weights and bucket not in self.exhausted
                and self.taken[bucket] < self.limit[bucket])

    def full(self) -> frozenset[str]:
        """Buckets that take nothing more right now."""
        return frozenset(b for b in self.weights if not self.wants(b))

    @property
    def done(self) -> bool:
        return all(self.taken[b] >= self.base[b] for b in self.weights if b not in self.exhausted)

    def admit(self, bucket: str, ids: np.ndarray) -> None:
        self.taken[bucket] += len(ids)
        self.docs[bucket] += 1
        if self.store is not None:
            self.store.append(bucket, ids)

    def tick(self) -> None:
        """Count one offered document (after the cheap filters); ends windows."""
        self.offers += 1
        if self.offers % self.window == 0:
            self._end_window()

    def _end_window(self) -> None:
        self.windows += 1
        newly = []
        for b in self.weights:
            if b in self.exhausted or self.taken[b] >= self.base[b]:
                self._stalled[b] = 0
                continue
            gain = self.taken[b] - self._window_start[b]
            self._stalled[b] = self._stalled[b] + 1 if gain < self.stall_gain * self.initial[b] else 0
            if self._stalled[b] >= self.stall_windows:
                newly.append(b)
        self._window_start = dict(self.taken)
        for b in newly:
            self.exhausted[b] = {"window": self.windows, "offers": self.offers,
                                 "tokens": self.taken[b],
                                 "of_quota": round(self.taken[b] / self.base[b], 6)
                                 if self.base[b] else 0.0}
        if newly:
            self.redistributions += 1
            self._requota()

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total, "slack": self.slack,
            "offers": self.offers, "windows": self.windows,
            "redistributions": self.redistributions,
            "exhausted": dict(self.exhausted),
            "rule": (f"window {self.window} offers; a bucket is exhausted only after "
                     f"{self.stall_windows} consecutive windows each gaining under "
                     f"{self.stall_gain:g} of its original quota (a genuine stall); "
                     "otherwise only the source's file cap ends collection"),
            "by_bucket": {b: {"weight": self.weights[b], "initial_quota": self.initial[b],
                              "quota": self.base[b], "tokens": self.taken[b],
                              "documents": self.docs[b]} for b in self.weights},
        }


# ------------------------------------------------------------------ documents

class Offer(NamedTuple):
    """A document that passed the cheap filters. source: the code language or the
    text source language; meta: kept in the main process only (never sent). A dict
    meta may carry "counts" (the source's filter counts since the previous offer,
    added to the stats when this offer is decided) and "last_file"."""
    kind: str
    source: str
    text: str
    meta: Any = None


class Result(NamedTuple):
    bucket: str
    status: str
    ids: np.ndarray | None
    label: str | None = None
    prob: float | None = None


def normalise_for_lid(text: str, max_chars: int = LID_MAX_CHARS) -> str:
    """How the specialist's training lines were cleaned: NFKC, whitespace collapsed
    to single spaces (fastText needs one line). Only the first max_chars are used."""
    return " ".join(unicodedata.normalize("NFKC", text[:max_chars]).split())


def lid_decision(source: str, label: str, prob: float, probs: dict[str, float],
                 threshold: float) -> tuple[str, bool]:
    """(bucket, keep) for a text document of `source` that the classifier labelled."""
    accepted = ZH_BUCKETS if source == ZH_SOURCE else (source,)
    if label in accepted:
        return (label if source == ZH_SOURCE else source), True
    if prob >= threshold:
        return source, False
    if source == ZH_SOURCE:  # unsure: the likelier of the two scripts
        return max(ZH_BUCKETS, key=lambda b: (probs.get(b, 0.0), b == ZH_BUCKETS[0])), True
    return source, True


def lid_labels_needed(sources: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for s in sources:
        out.update(ZH_BUCKETS if s == ZH_SOURCE else (s,))
    return out


class FastTextLid:
    """The fastText specialist of AneekC/lid-specialists-9plus1 (labels such as
    "__label__eng_Latn"). The model is loaded lazily in each process: fastText
    models do not pickle, so only the path travels to the workers."""

    def __init__(self, path: str | Path, max_chars: int = LID_MAX_CHARS) -> None:
        self.path = str(path)
        self.max_chars = max_chars
        self._model = None

    def __getstate__(self) -> dict:
        return {"path": self.path, "max_chars": self.max_chars}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["path"], state["max_chars"])

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                import fasttext
            except ImportError as exc:
                raise RuntimeError(
                    "--lid-filter needs the fastText bindings (package fasttext-numpy2, "
                    "installed by `uv sync` on Linux only; it has no Windows wheel). "
                    "Build the shards on the Linux box.") from exc
            self._model = fasttext.load_model(self.path)
        return self._model

    def labels(self) -> list[str]:
        return [lab.removeprefix("__label__") for lab in self.model.get_labels()]

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        labels, probs = self.model.predict(normalise_for_lid(text, self.max_chars), k=-1)
        pairs = {lab.removeprefix("__label__"): float(p) for lab, p in zip(labels, probs)}
        top = labels[0].removeprefix("__label__")
        return top, pairs[top], pairs


@dataclasses.dataclass(frozen=True)
class DocSetup:
    """What a worker needs, picklable: a zero-argument tokenizer factory (a module-level
    function or class, or functools.partial of one), the code length limit, and the
    optional leakage guard (find_file) and language-ID classifier (predict)."""
    tokenizer: Callable[[], Any]
    max_doc_tokens: int
    guard: Any = None
    lid: Any = None
    lid_threshold: float = DEFAULT_LID_THRESHOLD


def process_doc(kind: str, source: str, text: str, skip: frozenset[str], setup: DocSetup,
                tok: Any) -> Result:
    """Classify and tokenise one document (in a worker or, as fallback, in main)."""
    if kind == CODE:
        if source in skip:
            return Result(source, DEFERRED, None)
        if setup.guard is not None and setup.guard.find_file(text) is not None:
            return Result(source, LEAK, None)
        ids = encode_document(text, tok)
        if ids is None:
            return Result(source, BLANK, None)
        if len(ids) > setup.max_doc_tokens:
            return Result(source, TOO_LONG, None)
        return Result(source, OK, np.asarray(ids, dtype=np.uint16))
    label = prob = None
    bucket = source
    if setup.lid is not None:
        label, prob, probs = setup.lid.predict(text)
        bucket, keep = lid_decision(source, label, prob, probs, setup.lid_threshold)
        if not keep:
            return Result(source, LID_DROP, None, label, prob)
    if bucket in skip:
        return Result(bucket, DEFERRED, None, label, prob)
    ids = encode_document(text, tok)
    if ids is None:
        return Result(bucket, BLANK, None, label, prob)
    return Result(bucket, OK, np.asarray(ids, dtype=np.uint16), label, prob)


_WORKER: dict[str, Any] = {}


def _init_worker(setup: DocSetup) -> None:
    # One process per core already: keep the Rust tokenizer single-threaded.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    _WORKER["setup"] = setup
    _WORKER["tok"] = setup.tokenizer()


def _run_job(job: tuple[int, list[tuple[str, str, str]], frozenset[str]]
             ) -> tuple[int, list[Result]]:
    seq, docs, skip = job
    setup, tok = _WORKER["setup"], _WORKER["tok"]
    return seq, [process_doc(k, s, t, skip, setup, tok) for k, s, t in docs]


def ordered_map(fn: Callable[[Any], Any], jobs: Iterable[Any], pool: Any,
                ahead: int) -> Iterator[Any]:
    """fn over jobs, results in SUBMISSION order: an ordered imap with at most `ahead`
    jobs in flight (Pool.imap would read the whole job stream ahead, unbounded).
    A job is pulled only after the result `ahead` places before it was handed out,
    so what the consumer has done by then is fixed, not a matter of timing."""
    if pool is None:
        for job in jobs:
            yield fn(job)
        return
    pending: deque = deque()
    for job in jobs:
        pending.append(pool.apply_async(fn, (job,)))
        if len(pending) > ahead:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Runner:
    """Documents -> collector decisions, tokenising in `workers` processes (1: inline)."""

    def __init__(self, setup: DocSetup, workers: int = 1, *, batch_docs: int = 256,
                 ahead: int | None = None) -> None:
        if workers < 1 or batch_docs < 1:
            raise ValueError("workers and batch_docs must be at least 1")
        self.setup = setup
        self.workers = workers
        self.batch_docs = batch_docs
        self.ahead = ahead if ahead is not None else 4 * workers
        self.tok = setup.tokenizer()
        self.pool = None
        if workers > 1:
            ctx = multiprocessing.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_init_worker, initargs=(setup,))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _local(self, job: tuple[int, list[tuple[str, str, str]], frozenset[str]]
               ) -> tuple[int, list[Result]]:
        seq, docs, skip = job
        return seq, [process_doc(k, s, t, skip, self.setup, self.tok) for k, s, t in docs]

    def collect(self, offers: Iterable[Offer], collector: Collector, stats: Counter, *,
                on_admit: Callable[[Offer, Result], None] | None = None) -> None:
        """Offer documents in order until collector.done or offers run out.

        Decision order per document (fixed, so the counts are deterministic too):
        text: dropped by LID; then over quota; code: over quota; then (both) leak,
        too long, admitted. stats counts: offers, admitted, skipped_quota, lid_dropped,
        dropped_leakage, skipped_too_long, skipped_blank.
        """
        if collector.done:
            return
        pending: dict[int, tuple[list[Offer], frozenset[str]]] = {}

        def jobs() -> Iterator[tuple[int, list[tuple[str, str, str]], frozenset[str]]]:
            for seq, batch in enumerate(batched(offers, self.batch_docs)):
                skip = collector.full()
                pending[seq] = (batch, skip)
                # Code of a full language is not even sent; text goes out for its LID label.
                yield seq, [(o.kind, o.source, o.text) for o in batch
                            if not (o.kind == CODE and o.source in skip)], skip

        fn = _run_job if self.pool is not None else self._local
        for seq, results in ordered_map(fn, jobs(), self.pool, self.ahead):
            batch, skip = pending.pop(seq)
            it = iter(results)
            for offer in batch:
                if offer.kind == CODE and offer.source in skip:
                    res = Result(offer.source, DEFERRED, None)
                else:
                    res = next(it)
                self._decide(offer, res, collector, stats, on_admit)
                if collector.done:
                    return

    def _decide(self, offer: Offer, res: Result, collector: Collector, stats: Counter,
                on_admit: Callable[[Offer, Result], None] | None) -> None:
        stats["offers"] += 1
        if isinstance(offer.meta, dict):  # the source's filter counts leading up to it
            stats.update(offer.meta.get("counts") or {})
            if "last_file" in offer.meta:
                stats["last_file"] = offer.meta["last_file"]
        try:
            if res.status == LID_DROP:
                stats["lid_dropped"] += 1
                stats[f"lid_dropped_as:{res.label}"] += 1
                return
            if not collector.wants(res.bucket):
                stats["skipped_quota"] += 1
                return
            if res.status == DEFERRED:  # wanted after all: tokenise it here
                res = process_doc(offer.kind, offer.source, offer.text, frozenset(),
                                  self.setup, self.tok)
            if res.status == LEAK:
                stats["dropped_leakage"] += 1
                return
            if res.status == TOO_LONG:
                stats["skipped_too_long"] += 1
                return
            if res.status == BLANK:
                stats["skipped_blank"] += 1
                return
            collector.admit(res.bucket, res.ids)
            stats["admitted"] += 1
            if on_admit is not None:
                on_admit(offer, res)
        finally:
            collector.tick()


# ------------------------------------------------------------------ final mix

def weighted_interleave(streams: dict[str, Iterator[np.ndarray]], shares: dict[str, float]
                        ) -> Iterator[tuple[np.ndarray, str]]:
    """Merge bucket streams so each holds its share of the tokens taken so far.

    Before each document, the bucket furthest below its share (share * total - taken)
    supplies it; ties go to the earlier bucket in `streams`. Deterministic. A bucket
    running dry is an error, never a silent change of mix.
    """
    order = [b for b in streams if shares.get(b, 0.0) > 0]
    taken = dict.fromkeys(order, 0)
    total = 0
    while True:
        best, best_deficit = None, -math.inf
        for b in order:
            d = shares[b] * total - taken[b]
            if d > best_deficit:
                best, best_deficit = b, d
        if best is None:
            raise RuntimeError("train: no bucket has a positive share")
        try:
            arr = next(streams[best])
        except StopIteration:
            raise RuntimeError(f"train: {best} ran out after {taken[best]:,} tokens") from None
        taken[best] += len(arr)
        total += len(arr)
        yield arr, best
