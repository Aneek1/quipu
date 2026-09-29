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

- Runner: tokenises (and LID-labels) documents in `workers` processes (a
  concurrent.futures ProcessPoolExecutor, spawn, so it behaves the same on Windows
  and Linux). Batches are submitted in stream order and their results consumed in
  the same order (a deque of futures, bounded), and every admission decision is made
  in the main process in stream order, so the output does not depend on scheduling.
  A worker that dies (killed for memory, a crash in native code) or fails to start
  raises WorkerDied at once; it never hangs the build. As an optimisation, a batch's
  documents of buckets that were already full when the batch was submitted are not
  tokenised; if a later redistribution makes such a document wanted after all, the
  main process tokenises it itself. Either way the decision and the bytes are the same.
  A source may put MARK offers in its stream (the end of each of its files); they are
  handled in decision order too (on_mark), which is where a build checkpoints.

- BucketStore: each collected bucket is appended to a flat uint16 file
  (<bucket>.bin, named from the sanitised bucket name) with an append-only file of
  document ends (<bucket>.ends, uint64), so the final train split can interleave
  buckets by their final shares (weighted_interleave) without holding anything in
  memory, and a build killed part way can truncate the files back to its last
  checkpoint (snapshot / restore) and carry on.

Language ID (spec 11): lid_decision keeps a document when the classifier's top label
is its source language (cmn_Hani: zho_Hans or zho_Hant) and drops it when a different
label wins with probability >= threshold. The default threshold is 0.0: any mismatch
is dropped, as the spec says. A threshold t > 0 keeps a mismatched document whose top
label has probability < t ("kept while unsure", counted). cmn_Hani is split into
zho_Hans / zho_Hant by the classifier's label (or, when unsure, by whichever of the
two it rates higher).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import multiprocessing
import os
import unicodedata
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple
from urllib.parse import quote

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

MARK = "mark"  # Offer.kind of a source's end-of-file marker (no document)

DEFAULT_SLACK = 0.10
DEFAULT_WINDOW = 40_000        # offers per window: about one github-code-clean file
DEFAULT_STALL_WINDOWS = 10
DEFAULT_STALL_GAIN = 1e-4      # of the bucket's original quota per window
DEFAULT_LID_THRESHOLD = 0.0    # any mismatch is dropped (spec 11)
LID_MAX_CHARS = 2000
DEFAULT_CHARS_PER_TOKEN = 3.79  # code, the quipu-moe tokenizer (preflight estimates)

# Set in the environment the worker processes are spawned with: one process per core
# already, so the tokenizer and any BLAS must not start thread pools of their own.
WORKER_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "1"}


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


class WorkerDied(RuntimeError):
    """A tokenising worker process died or could not start."""


class ResumeError(RuntimeError):
    """A build's saved state cannot be resumed (see scripts/build_shards.py)."""


# ------------------------------------------------------------------ collection

def bucket_stem(bucket: str) -> str:
    """A file-safe, reversible name for a bucket ("C++" -> "C%2B%2B")."""
    return quote(bucket, safe="").replace(".", "%2E").replace("~", "%7E")


class BucketStore:
    """Per-bucket token files: <dir>/<stem>.bin (uint16, documents back to back) and
    <dir>/<stem>.ends (uint64, the running token count after each document), both
    append-only. Append while collecting; read back with docs() after close().

    fresh=True deletes whatever an earlier build left in the directory. fresh=False
    keeps it for restore(snapshot), which a resumed build must call before anything
    else: it truncates every file back to the snapshot and deletes files of buckets
    the snapshot does not have (written after it)."""

    def __init__(self, directory: Path, *, fresh: bool = True) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        if fresh:
            for p in list(self.dir.glob("*.bin")) + list(self.dir.glob("*.ends")):
                p.unlink()
        self._ready = fresh
        self._files: dict[str, tuple[Any, Any]] = {}
        self.tokens: Counter[str] = Counter()
        self._docs: Counter[str] = Counter()

    def _paths(self, bucket: str) -> tuple[Path, Path]:
        stem = bucket_stem(bucket)
        return self.dir / f"{stem}.bin", self.dir / f"{stem}.ends"

    def append(self, bucket: str, ids: np.ndarray) -> None:
        if not self._ready:
            raise RuntimeError("a BucketStore opened with fresh=False needs restore() first")
        files = self._files.get(bucket)
        if files is None:
            data, ends = self._paths(bucket)
            files = self._files[bucket] = (open(data, "ab"), open(ends, "ab"))
        files[0].write(ids.astype("<u2", copy=False).tobytes())
        self.tokens[bucket] += len(ids)
        self._docs[bucket] += 1
        files[1].write(np.uint64(self.tokens[bucket]).tobytes())

    def snapshot(self, *, sync: bool = True) -> dict[str, list[int]]:
        """{bucket: [tokens, documents]} after flushing (and fsyncing) every file."""
        for f in (f for pair in self._files.values() for f in pair):
            f.flush()
            if sync:
                os.fsync(f.fileno())
        return {b: [self.tokens[b], self._docs[b]] for b in sorted(self._docs)
                if self._docs[b]}

    @staticmethod
    def verify(directory: Path, snapshot: dict[str, list[int]]) -> None:
        """Raise ResumeError when a bucket file in `directory` is shorter than
        `snapshot` says (restore would fail on it); changes nothing on disk."""
        for bucket, (tokens, documents) in snapshot.items():
            stem = bucket_stem(bucket)
            data, ends = Path(directory) / f"{stem}.bin", Path(directory) / f"{stem}.ends"
            have = (data.stat().st_size if data.exists() else -1,
                    ends.stat().st_size if ends.exists() else -1)
            if have[0] < 2 * tokens or have[1] < 8 * documents:
                raise ResumeError(f"{directory}: bucket {bucket!r} has {have} bytes on disk, "
                                  f"less than its checkpoint ({2 * tokens}, {8 * documents})")

    def restore(self, snapshot: dict[str, list[int]]) -> None:
        """Back to `snapshot` (see the class docstring). Raises ResumeError when a file
        is shorter than the snapshot says (lost or damaged)."""
        self.close()
        keep = {bucket_stem(b) for b in snapshot}
        for p in list(self.dir.glob("*.bin")) + list(self.dir.glob("*.ends")):
            if p.stem not in keep:
                p.unlink()
        self.tokens, self._docs = Counter(), Counter()
        for bucket, (tokens, documents) in snapshot.items():
            data, ends = self._paths(bucket)
            have = (data.stat().st_size if data.exists() else -1,
                    ends.stat().st_size if ends.exists() else -1)
            if have[0] < 2 * tokens or have[1] < 8 * documents:
                raise ResumeError(f"{self.dir}: bucket {bucket!r} has {have} bytes on disk, "
                                  f"less than its checkpoint ({2 * tokens}, {8 * documents}); "
                                  "the work directory is damaged: rerun with --fresh")
            os.truncate(data, 2 * tokens)
            os.truncate(ends, 8 * documents)
            if documents:
                with open(ends, "rb") as f:
                    f.seek(8 * (documents - 1))
                    last = int(np.frombuffer(f.read(8), dtype="<u8")[0])
                if last != tokens:
                    raise ResumeError(f"{ends}: last end {last} != checkpoint {tokens} tokens; "
                                      "rerun with --fresh")
            self.tokens[bucket], self._docs[bucket] = tokens, documents
        self._ready = True

    def close(self) -> None:
        for pair in self._files.values():
            for f in pair:
                f.close()
        self._files = {}

    def documents(self, bucket: str) -> int:
        return self._docs.get(bucket, 0)

    def docs(self, bucket: str) -> Iterator[np.ndarray]:
        """The bucket's documents in collection order (memory-mapped)."""
        if self.tokens[bucket] == 0:
            return iter(())
        if bucket in self._files:
            raise RuntimeError("close() the store before reading it")
        data_path, ends_path = self._paths(bucket)
        data = np.memmap(data_path, dtype="<u2", mode="r", shape=(self.tokens[bucket],))
        ends = np.memmap(ends_path, dtype="<u8", mode="r", shape=(self._docs[bucket],))

        def gen() -> Iterator[np.ndarray]:
            start = 0
            for end in ends:
                end = int(end)
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
        # Documents of each bucket that reached a decision (past the cheap filters and
        # LID), wanted or not: how common the bucket is in the stream (projections).
        self.offered: Counter[str] = Counter()
        self._stalled: Counter[str] = Counter()
        self._window_start = dict(self.taken)
        self._requota()

    def snapshot(self) -> dict[str, Any]:
        """Everything needed to rebuild this collector exactly (JSON-safe)."""
        return {"params": {"weights": self.weights, "total": self.total, "caps": self.caps,
                           "slack": self.slack, "window": self.window,
                           "stall_windows": self.stall_windows, "stall_gain": self.stall_gain},
                "state": {"taken": self.taken, "docs": dict(self.docs),
                          "exhausted": self.exhausted, "offers": self.offers,
                          "windows": self.windows, "redistributions": self.redistributions,
                          "offered": dict(self.offered), "stalled": dict(self._stalled),
                          "window_start": self._window_start}}

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any], store: BucketStore | None = None
                      ) -> "Collector":
        col = cls(store=store, **snap["params"])
        st = snap["state"]
        if set(st["taken"]) != set(col.weights):
            raise ResumeError(f"collector state has buckets {sorted(st['taken'])}, "
                              f"weights have {sorted(col.weights)}")
        col.taken = {b: int(st["taken"][b]) for b in col.weights}
        col.docs = Counter(st["docs"])
        col.exhausted = {b: dict(v) for b, v in st["exhausted"].items()}
        col.offers, col.windows = int(st["offers"]), int(st["windows"])
        col.redistributions = int(st["redistributions"])
        col.offered = Counter(st["offered"])
        col._stalled = Counter(st["stalled"])
        col._window_start = {b: int(st["window_start"][b]) for b in col.weights}
        col._requota()
        return col

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
                              "documents": self.docs[b],
                              "offered_documents": self.offered[b]} for b in self.weights},
        }


def projected_available(col: Collector, files_read: int, files_left: int) -> dict[str, float]:
    """What each bucket of a collector over a file-by-file source should hold after
    `files_left` more files: what it holds plus its rate so far. The rate is counted
    from the documents OFFERED (wanted or not, so a bucket that has been full part of
    the time is not under-rated) times its mean admitted document size. An exhausted
    bucket holds what it has."""
    out = {}
    for b in col.weights:
        if b in col.exhausted or col.docs[b] == 0 or files_read <= 0:
            out[b] = float(col.taken[b])
            continue
        per_file = col.offered[b] / files_read * (col.taken[b] / col.docs[b])
        out[b] = col.taken[b] + per_file * max(files_left, 0)
    return out


def projection_violations(col: Collector, files_read: int, files_total: int, *,
                          min_weight: float = 0.05, max_off: float = 0.02,
                          what: str = "code") -> tuple[list[str], dict[str, float]]:
    """share_violations of the mix the collector is heading for at the file cap."""
    avail = projected_available(col, files_read, files_total - files_read)
    alloc, _ = allocate(col.total, col.weights, avail, col.caps)
    shares = {b: v / col.total for b, v in alloc.items()}
    return (share_violations(shares, col.weights, min_weight=min_weight, max_off=max_off,
                             what=what), avail)


def preflight_report(per_file: list[dict[str, float]], weights: dict[str, float],
                     total_tokens: float, *, caps: dict[str, float] | None = None,
                     chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                     file_bytes: list[int] | None = None, min_weight: float = 0.05,
                     max_off: float = 0.02) -> dict[str, Any]:
    """Whether the code files, in order, can fill the code mix.

    per_file: for each file the build would read, the bytes of source per language
    that pass the cheap filters (language, licence, path, size). Bytes become tokens
    at chars_per_token (code is nearly all ASCII). A language needs the files up to the
    one where its running total reaches its quota (allocate over the weights and caps);
    the build reads until every language is full, so it reads the most any language
    needs, or every file if one never fills. ok: the mix the files can give is within
    max_off of every weight >= min_weight.
    """
    caps = caps or {}
    quota, _ = allocate(total_tokens, weights, None, caps)
    running = dict.fromkeys(weights, 0.0)
    needed: dict[str, int | None] = dict.fromkeys(weights)
    for i, counts in enumerate(per_file):
        for b in weights:
            running[b] += counts.get(b, 0.0) / chars_per_token
            if needed[b] is None and running[b] >= quota[b]:
                needed[b] = i + 1
    final, short = allocate(total_tokens, weights, running, caps)
    shares = {b: v / total_tokens for b, v in final.items()}
    violations = share_violations(shares, weights, min_weight=min_weight, max_off=max_off)
    unfilled = [b for b in weights if needed[b] is None]
    files_read = (len(per_file) if unfilled or not per_file
                  else max(n for n in needed.values() if n is not None))
    limiting = (unfilled if unfilled else
                [b for b in weights if needed[b] == files_read])
    return {
        "files": len(per_file), "total_tokens": total_tokens,
        "chars_per_token": chars_per_token,
        "by_language": {b: {"weight": weights[b], "quota_tokens": quota[b],
                            "available_tokens": running[b], "files_to_fill": needed[b],
                            "projected_share": shares[b]} for b in weights},
        "files_read": files_read,
        "limiting_languages": limiting,
        "download_bytes": sum(file_bytes[:files_read]) if file_bytes else None,
        "shortfall_tokens": short,
        "violations": violations,
        "ok": not violations,
    }


def format_preflight(report: dict[str, Any]) -> str:
    """The preflight report as a table and a verdict (see preflight_report)."""
    lines = [f"{'language':<12}{'weight':>8}{'quota':>12}{'available':>12}"
             f"{'files to fill':>15}{'share':>9}"]
    for b, r in report["by_language"].items():
        need = r["files_to_fill"]
        off = abs(r["projected_share"] - r["weight"]) * 100
        flag = "" if r["weight"] < 0.05 or off <= 2 else "  <-- off target"
        lines.append(f"{b:<12}{r['weight']:>8.2%}{r['quota_tokens']:>12.3e}"
                     f"{r['available_tokens']:>12.3e}"
                     f"{(str(need) if need is not None else 'never'):>15}"
                     f"{r['projected_share']:>9.2%}{flag}")
    gb = report["download_bytes"]
    lines.append(f"the build would read {report['files_read']} of {report['files']} code files"
                 + (f" (~{gb / 1e9:,.1f} GB to download)" if gb else "")
                 + f"; limited by {', '.join(report['limiting_languages']) or '-'}")
    if report["ok"]:
        lines.append("verdict: OK - every language of weight >= 5% within 2 pp of its weight")
    else:
        lines.append("verdict: FAIL - the files cannot give the configured code mix:")
        lines += [f"  {v}" for v in report["violations"]]
    return "\n".join(lines)


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
    unsure: bool = False  # LID labelled it another language but below the threshold


def normalise_for_lid(text: str, max_chars: int = LID_MAX_CHARS) -> str:
    """How the specialist's training lines were cleaned: NFKC, whitespace collapsed
    to single spaces (fastText needs one line). Only the first max_chars are used."""
    return " ".join(unicodedata.normalize("NFKC", text[:max_chars]).split())


def lid_accepts(source: str, label: str) -> bool:
    """True when `label` is the source language (cmn_Hani: either Chinese script)."""
    return label in (ZH_BUCKETS if source == ZH_SOURCE else (source,))


def lid_decision(source: str, label: str, prob: float, probs: dict[str, float],
                 threshold: float) -> tuple[str, bool]:
    """(bucket, keep) for a text document of `source` that the classifier labelled.
    Kept when the label is the source language; otherwise dropped when the label's
    probability >= threshold (always, at the default 0.0), else kept (unsure)."""
    if lid_accepts(source, label):
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
    unsure = False
    if setup.lid is not None:
        label, prob, probs = setup.lid.predict(text)
        bucket, keep = lid_decision(source, label, prob, probs, setup.lid_threshold)
        if not keep:
            return Result(source, LID_DROP, None, label, prob)
        unsure = not lid_accepts(source, label)
    if bucket in skip:
        return Result(bucket, DEFERRED, None, label, prob, unsure)
    ids = encode_document(text, tok)
    if ids is None:
        return Result(bucket, BLANK, None, label, prob, unsure)
    return Result(bucket, OK, np.asarray(ids, dtype=np.uint16), label, prob, unsure)


_WORKER: dict[str, Any] = {}


def _init_worker(setup: DocSetup) -> None:
    # One process per core already: keep the Rust tokenizer single-threaded (the
    # spawn environment has WORKER_ENV too; this covers a caller that bypassed it).
    for key, value in WORKER_ENV.items():
        os.environ[key] = value
    _WORKER["setup"] = setup
    _WORKER["tok"] = setup.tokenizer()
    if setup.lid is not None and hasattr(setup.lid, "model"):
        setup.lid.model  # load it now: a broken install fails at start, not mid-build


def _ping() -> bool:
    return True


def _run_job(job: tuple[int, list[tuple[str, str, str]], frozenset[str]]
             ) -> tuple[int, list[Result]]:
    seq, docs, skip = job
    setup, tok = _WORKER["setup"], _WORKER["tok"]
    return seq, [process_doc(k, s, t, skip, setup, tok) for k, s, t in docs]


WORKER_DIED = ("a tokenising worker process died or failed to start (killed for memory? "
               "a crash in the tokenizer or fastText? an exception in the worker "
               "initializer?), so the build stops here rather than hang. Everything up "
               "to the last checkpoint is kept in <shard_dir>/_work: rerun the same "
               "command to resume.")


def _result(fut: Future) -> Any:
    try:
        return fut.result()
    except BrokenProcessPool as exc:
        raise WorkerDied(f"{WORKER_DIED} ({exc})") from exc


def ordered_map(fn: Callable[[Any], Any], jobs: Iterable[Any], pool: Any,
                ahead: int) -> Iterator[Any]:
    """fn over jobs, results in SUBMISSION order: a deque of futures with at most
    `ahead` jobs in flight (Executor.map would read the whole job stream ahead).
    A job is pulled only after the result `ahead` places before it was handed out,
    so what the consumer has done by then is fixed, not a matter of timing. A dead
    worker raises WorkerDied (BrokenProcessPool underneath) instead of hanging."""
    if pool is None:
        for job in jobs:
            yield fn(job)
        return
    pending: deque[Future] = deque()
    try:
        for job in jobs:
            pending.append(pool.submit(fn, job))
            if len(pending) > ahead:
                yield _result(pending.popleft())
        while pending:
            yield _result(pending.popleft())
    finally:
        for fut in pending:
            fut.cancel()


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
        self.pool: ProcessPoolExecutor | None = None
        self._saved_env: dict[str, str | None] = {}
        if workers > 1:
            # Workers are spawned on demand while the pool lives, so the environment
            # they inherit is set for the pool's lifetime and restored by close().
            for key, value in WORKER_ENV.items():
                self._saved_env[key] = os.environ.get(key)
                os.environ[key] = value
            ctx = multiprocessing.get_context("spawn")
            self.pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                                            initargs=(setup,))
            try:  # a worker that cannot start fails here, in seconds
                _result(self.pool.submit(_ping))
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            procs = list((getattr(pool, "_processes", None) or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for p in procs:  # do not wait for a batch nobody will read
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=10)
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_env = {}

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _local(self, job: tuple[int, list[tuple[str, str, str]], frozenset[str]]
               ) -> tuple[int, list[Result]]:
        seq, docs, skip = job
        return seq, [process_doc(k, s, t, skip, self.setup, self.tok) for k, s, t in docs]

    def collect(self, offers: Iterable[Offer], collector: Collector, stats: Counter, *,
                on_admit: Callable[[Offer, Result], None] | None = None,
                on_mark: Callable[[dict], None] | None = None) -> None:
        """Offer documents in order until collector.done or offers run out.

        Decision order per document (fixed, so the counts are deterministic too):
        text: dropped by LID; then over quota; code: over quota; then (both) leak,
        too long, admitted. stats counts: offers, admitted, skipped_quota, lid_dropped,
        lid_kept_unsure, dropped_leakage, skipped_too_long, skipped_blank. A MARK
        offer adds its meta["counts"] to stats and calls on_mark(meta), in the same
        order. The offers iterator is closed when collection ends (a generator's
        read-ahead threads and downloads stop with it).
        """
        offers_it = iter(offers)
        try:
            if collector.done:
                return
            self._collect(offers_it, collector, stats, on_admit, on_mark)
        finally:
            close = getattr(offers_it, "close", None)
            if close is not None:
                close()

    def _collect(self, offers: Iterator[Offer], collector: Collector, stats: Counter,
                 on_admit: Callable[[Offer, Result], None] | None,
                 on_mark: Callable[[dict], None] | None) -> None:
        pending: dict[int, tuple[list[Offer], frozenset[str]]] = {}

        def sent(o: Offer, skip: frozenset[str]) -> bool:
            # Code of a full language is not even sent; text goes out for its LID label.
            return o.kind != MARK and not (o.kind == CODE and o.source in skip)

        def jobs() -> Iterator[tuple[int, list[tuple[str, str, str]], frozenset[str]]]:
            for seq, batch in enumerate(batched(offers, self.batch_docs)):
                skip = collector.full()
                pending[seq] = (batch, skip)
                yield seq, [(o.kind, o.source, o.text) for o in batch if sent(o, skip)], skip

        fn = _run_job if self.pool is not None else self._local
        results_it = ordered_map(fn, jobs(), self.pool, self.ahead)
        try:
            for seq, results in results_it:
                batch, skip = pending.pop(seq)
                it = iter(results)
                for offer in batch:
                    if offer.kind == MARK:
                        stats.update(offer.meta.get("counts") or {})
                        if on_mark is not None:
                            on_mark(offer.meta)
                        continue
                    if offer.kind == CODE and offer.source in skip:
                        res = Result(offer.source, DEFERRED, None)
                    else:
                        res = next(it)
                    self._decide(offer, res, collector, stats, on_admit)
                    if collector.done:
                        return
        finally:
            results_it.close()

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
            if res.unsure:
                stats["lid_kept_unsure"] += 1
            collector.offered[res.bucket] += 1
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
