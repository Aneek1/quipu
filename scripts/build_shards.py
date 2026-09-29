"""Download FineWeb-Edu and full-stack code, tokenize, and write uint16 shards.

Streams rather than downloading whole datasets: both are far larger than the tokens
this run needs, and there is no reason to store the remainder.

Splits (weekend-run spec section 3):

- val       FineWeb-Edu only, taken first off the text stream (the trainer's loss).
- train     ONE merged stream of FineWeb-Edu and code, interleaved deterministically:
            before each document, the source furthest below its target share of the
            tokens written so far supplies it (ties go to text). No randomness, so a
            rebuild reproduces byte-identical shards.
- code_val  Code from held-out parquet files that train never opens, deduplicated by
            exact content hash against every code document written to train.

Validation comes off the head of the text stream and train text continues from
exactly where validation stopped, through ONE shared iterator. Iterating a streaming
HF dataset a second time restarts it from the beginning, so passing the dataset
object itself to both builds would make train re-read the validation documents.

Code is read straight from github-code-clean's parquet files (its loading script is
not used), one row group at a time, keeping only the configured languages and
licences. HTML is capped: see CodeDocs.

A killed build is re-run from scratch: shard writes are atomic, both streams are
deterministic, and each split directory is cleared of old shards first.

Run: uv run python scripts/build_shards.py --config configs/quipu-114m.toml

Data v2 (quipu-moe; any config with code_language_weights) is a different build,
build_mix below: the config's tokenizer, per-language code weights, English plus nine
FineWeb-2 languages, --workers tokenising processes, the stepbuild leakage guard and,
with --lid-filter, the owner's language-ID model. See the "data v2" section and
quipu/shard_mix.py. Unlike the build above, it resumes after a crash or preemption
(rerun the same command) and fails early; scripts/remote/build_shards_box.md is the
runbook for building it on a rented CPU box.

Run: uv run python scripts/build_shards.py --config configs/quipu-moe.toml \
         --lid-filter --train-tokens 8.3e9 --preflight     # metadata only, minutes
     uv run python scripts/build_shards.py --config configs/quipu-moe.toml \
         --lid-filter --train-tokens 8.3e9                 # the build (resumable)
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import numpy as np
from tqdm import tqdm

from quipu import shard_mix as sm
from quipu.config import load_config
from quipu.data import encode_document, write_shard
from quipu.fsio import replace_with_retry
from quipu.tokenizer import Tokenizer

MANIFEST = "manifest.json"
TEXT = "text"
CODE_PREFIX = "code:"  # document labels are "text" or "code:<language>"

# The HTML cap is relative to the code tokens written so far, which is zero at the
# start, so a strict cap would refuse the first HTML documents and skew the opening
# of the stream. Until this many code tokens have been written HTML is always
# admitted; after that the cap holds exactly (see CodeDocs). The overshoot this
# allows is bounded by the warmup itself and is paid back as the cap then skips HTML
# until HTML is under html_cap of the code written.
HTML_CAP_WARMUP_TOKENS = 1_000_000
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
CODE_STAT_KEYS = ("rows_scanned", "dropped_language", "dropped_license",
                  "dropped_as_duplicate", "skipped_blank", "skipped_html_cap")

# Generated, vendored and bundled code: third-party libraries, minified builds and
# data blobs teach the model nothing about writing code and, at hundreds of
# thousands of tokens each, would crowd out real source. A document is skipped for
# the first rule its path matches (case-insensitive; "/" or "\" separators):
#   minified  the file name ends .min.js / .min.css, or it is a source map (.map)
#   vendored  a DIRECTORY segment is exactly one of VENDORED_DIRS (a segment match,
#             so src/vendor_utils.py is kept), or the file name contains "bundle"
#             or ".pack."
# Separately, any document longer than code_max_doc_tokens is skipped as too_long.
CODE_PATH_SKIP_RULES = {
    "minified": {"file_name_patterns": [r"\.min\.(js|css)$", r"\.map$"]},
    "vendored": {"directory_segments": ["vendor", "node_modules", "dist", "build",
                                        "bower_components", "third_party", "external"],
                 "file_name_contains": ["bundle", ".pack."]},
}
_MINIFIED_RE = re.compile("|".join(CODE_PATH_SKIP_RULES["minified"]["file_name_patterns"]),
                          re.IGNORECASE)
_VENDORED_DIRS = frozenset(CODE_PATH_SKIP_RULES["vendored"]["directory_segments"])
_VENDORED_NAME_PARTS = tuple(CODE_PATH_SKIP_RULES["vendored"]["file_name_contains"])
SKIP_REASONS = ("minified", "vendored", "too_long")
# Code tokens are also bucketed by document length, to show how much sits in long files.
DOC_LENGTH_BUCKETS = (1_000, 4_000, 8_000, 16_000)


def path_skip_reason(path: str | None) -> str | None:
    """"minified", "vendored" or None for a code file's repository path."""
    if not path:
        return None
    parts = [p for p in path.replace("\\", "/").lower().split("/") if p]
    if not parts:
        return None
    name = parts[-1]
    if _MINIFIED_RE.search(name):
        return "minified"
    if _VENDORED_DIRS.intersection(parts[:-1]) or any(k in name for k in _VENDORED_NAME_PARTS):
        return "vendored"
    return None


def _length_bucket(n: int) -> str:
    for edge in DOC_LENGTH_BUCKETS:
        if n <= edge:
            return f"<={edge}"
    return f">{DOC_LENGTH_BUCKETS[-1]}"


def clear_split_dir(out_dir: Path) -> None:
    """Remove shards and temp files left by an earlier (possibly longer) build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in list(out_dir.glob("shard_*.bin")) + list(out_dir.glob("*.tmp")):
        p.unlink()


def write_split(out_dir: Path, target_tokens: int, shard_tokens: int,
                pieces: Iterator[tuple[np.ndarray, str]], *,
                allow_short: bool = False) -> dict[str, Any]:
    """Write exactly target_tokens tokens of labelled documents into shard_000.bin, ...

    Every shard holds shard_tokens tokens except the last, which holds the
    remainder. Consumption stops the moment the target is met; the rest of the
    document in progress is discarded, never carried into the next split.
    Raises RuntimeError if pieces runs dry first (a silently short dataset would
    make training wrap around), unless allow_short, in which case whatever was
    written is kept and "short" is recorded.

    by_label counts, per label, the tokens actually written (a truncated final
    document counts only its written part) and the documents begun.
    """
    if target_tokens <= 0 or shard_tokens <= 0:
        raise ValueError("target_tokens and shard_tokens must be positive")
    out_dir = Path(out_dir)
    clear_split_dir(out_dir)

    # A preallocated uint16 buffer rather than a Python list: a 100M-token list of
    # int objects would cost several GB.
    buf = np.empty(min(shard_tokens, target_tokens), dtype=np.uint16)
    fill = 0
    written = 0
    documents = 0
    by_label: dict[str, dict[str, int]] = {}
    shards: list[dict[str, Any]] = []
    progress = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc=out_dir.name,
                    mininterval=5.0)

    def flush(size: int) -> None:
        nonlocal fill, written
        name = f"shard_{len(shards):03d}.bin"
        write_shard(out_dir / name, buf[:size])
        shards.append({"file": name, "tokens": size})
        written += size
        progress.update(size)
        fill = 0

    for arr, label in pieces:
        documents += 1
        counts = by_label.setdefault(label, {"tokens": 0, "documents": 0})
        counts["documents"] += 1
        pos = 0
        while pos < len(arr) and written < target_tokens:
            size = min(shard_tokens, target_tokens - written)
            take = min(size - fill, len(arr) - pos)
            buf[fill:fill + take] = arr[pos:pos + take]
            fill += take
            pos += take
            counts["tokens"] += take
            if fill == size:
                flush(size)
        if written >= target_tokens:
            break

    short = written + fill < target_tokens
    if short and allow_short and fill:
        flush(fill)  # keep the partial last shard
    progress.close()
    if short and not allow_short:
        raise RuntimeError(
            f"{out_dir.name}: document stream ran out after {written + fill:,} of "
            f"{target_tokens:,} tokens"
        )
    out: dict[str, Any] = {"tokens": written, "documents": documents,
                           "by_label": by_label, "shards": shards}
    if allow_short:
        out["short"] = short
    return out


class TextDocs:
    """Rows {"text": ...} -> (uint16 ids, "text"); blank rows are skipped."""

    def __init__(self, rows: Iterator[dict], tok: Tokenizer) -> None:
        self._rows = rows
        self._tok = tok
        self.rows_consumed = 0

    def __iter__(self) -> Iterator[tuple[np.ndarray, str]]:
        return self

    def __next__(self) -> tuple[np.ndarray, str]:
        while True:
            row = next(self._rows)
            self.rows_consumed += 1
            ids = encode_document(row.get("text") or "", self._tok)
            if ids is not None:
                return np.asarray(ids, dtype=np.uint16), TEXT


def build(out_dir: Path, target_tokens: int, shard_tokens: int,
          docs_iter: Iterator[dict], tok: Tokenizer) -> dict[str, Any]:
    """A text-only split (the FineWeb val): exactly target_tokens tokens of docs_iter.

    docs_iter is not advanced past the document that completes the target, so the
    caller's iterator continues right after it. Raises RuntimeError if it runs dry.
    """
    docs = TextDocs(iter(docs_iter), tok)
    res = write_split(out_dir, target_tokens, shard_tokens, docs)
    return {"tokens": res["tokens"], "rows_consumed": docs.rows_consumed,
            "documents": res["documents"], "shards": res["shards"]}


def content_hash(code: str) -> int:
    """Exact-content identity for dedup: blake2b-8 of the raw UTF-8 source."""
    digest = hashlib.blake2b(code.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class CodeDocs:
    """Code rows {"code", "language", "license"[, "path", "file"]} -> (ids, "code:<language>").

    A row is dropped, in this order, if its language is not kept; if its licence is
    not kept; (code val) if its content hash is in `exclude`; if it is blank; if its
    path is minified or vendored/bundled (CODE_PATH_SKIP_RULES); if it is longer than
    max_doc_tokens tokens (counting its EOT); or if it is HTML over the cap. Skipped
    documents are still tokenized, so .skipped records the tokens each reason
    removed. The HTML cap is hard-coded to the language name "HTML", so that name
    must be among the kept languages. The HTML cap: once at least `html_warmup` code tokens
    have been yielded, an HTML document of n tokens is skipped when
    html + n > html_cap * (code + n), i.e. when yielding it would take HTML above
    html_cap of the code tokens yielded. The consumer writes every yielded
    document (the last one possibly truncated), so yielded == written up to that one
    document.

    With record_hashes, the content hash of every yielded document is kept in
    .hashes (train records them; code val excludes them).
    """

    def __init__(self, rows: Iterator[dict], tok: Tokenizer, *, languages: Iterable[str],
                 licenses: Iterable[str], html_cap: float, max_doc_tokens: int,
                 html_warmup: int = HTML_CAP_WARMUP_TOKENS,
                 exclude: set[int] | frozenset[int] = frozenset(),
                 record_hashes: bool = False, leak_guard: Any = None,
                 decontam: Any = None) -> None:
        self._rows = rows
        # Data v2: a stepbuild LeakageGuard; a row copying a benchmark reference file
        # is dropped (stats "dropped_leakage", not among CODE_STAT_KEYS). And a
        # quipu.decontam.Decontaminator: a row containing a HumanEval / MBPP problem is
        # dropped (stats "dropped_contamination", "dropped_contamination_as:<name>").
        self._leak_guard = leak_guard
        self._decontam = decontam
        self._tok = tok
        self.languages = frozenset(languages)
        if "HTML" not in self.languages:
            raise ValueError("code_languages must include 'HTML': the HTML cap is keyed on "
                             "that exact name, so a renamed or missing entry disables it")
        self.max_doc_tokens = max_doc_tokens
        self.skipped = {r: {"documents": 0, "tokens": 0} for r in SKIP_REASONS}
        self.length_buckets: Counter[str] = Counter()
        self.licenses = frozenset(licenses)
        self.html_cap = html_cap
        self.html_warmup = html_warmup
        self._exclude = exclude
        self.record_hashes = record_hashes
        self.hashes: set[int] = set()
        self.tokens = 0
        self.html_tokens = 0
        self.stats: Counter[str] = Counter()
        self.last_file: int | None = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, str]]:
        return self

    def __next__(self) -> tuple[np.ndarray, str]:
        stats = self.stats
        while True:
            row = next(self._rows)
            stats["rows_scanned"] += 1
            language = row.get("language")
            if language not in self.languages:
                stats["dropped_language"] += 1
                continue
            if row.get("license") not in self.licenses:
                stats["dropped_license"] += 1
                continue
            code = row.get("code") or ""
            h = content_hash(code)
            if h in self._exclude:
                stats["dropped_as_duplicate"] += 1
                continue
            if self._leak_guard is not None and self._leak_guard.find_file(code) is not None:
                stats["dropped_leakage"] += 1
                continue
            if self._decontam is not None:
                hit = self._decontam.find(code)
                if hit is not None:
                    stats["dropped_contamination"] += 1
                    stats[f"dropped_contamination_as:{hit[0]}"] += 1
                    continue
            ids = encode_document(code, self._tok)
            if ids is None:
                stats["skipped_blank"] += 1
                continue
            n = len(ids)
            reason = path_skip_reason(row.get("path"))
            if reason is None and n > self.max_doc_tokens:
                reason = "too_long"
            if reason is not None:
                self.skipped[reason]["documents"] += 1
                self.skipped[reason]["tokens"] += n
                continue
            if language == "HTML":
                if (self.tokens >= self.html_warmup
                        and self.html_tokens + n > self.html_cap * (self.tokens + n)):
                    stats["skipped_html_cap"] += 1
                    continue
                self.html_tokens += n
            self.tokens += n
            self.length_buckets[_length_bucket(n)] += n
            if self.record_hashes:
                self.hashes.add(h)
            if "file" in row:
                self.last_file = row["file"]
            return np.asarray(ids, dtype=np.uint16), CODE_PREFIX + language


def interleave(text: Iterator[tuple[np.ndarray, str]],
               code: Iterator[tuple[np.ndarray, str]],
               code_share: float) -> Iterator[tuple[np.ndarray, str]]:
    """Merge two document streams so that code holds code_share of the tokens.

    Before each document, take from the source furthest below its target share of
    the tokens taken so far: code when code_share * total - code_tokens exceeds
    (1 - code_share) * total - text_tokens, text otherwise (ties included).
    Deterministic. A source running dry is an error, not a silent change of mix.
    """
    taken = {"text": 0, "code": 0}
    sources = {"text": text, "code": code}
    while True:
        total = taken["text"] + taken["code"]
        code_deficit = code_share * total - taken["code"]
        text_deficit = (1 - code_share) * total - taken["text"]
        name = "code" if code_deficit > text_deficit else "text"
        try:
            arr, label = next(sources[name])
        except StopIteration:
            raise RuntimeError(
                f"train: {name} stream ran out after {taken[name]:,} {name} tokens"
            ) from None
        taken[name] += len(arr)
        yield arr, label


def prefetch(items: Iterable[Any], depth: int) -> Iterator[Any]:
    """Iterate `items` in a background thread, up to `depth` items ahead.

    Order is preserved exactly, so determinism is unaffected; network reads just
    overlap with tokenizing. An exception in the producer is re-raised here. When
    the consumer stops early (closes this generator, or drops it), the producer
    thread stops after its current item and closes `items` in its own thread, so a
    finished collection leaves no reader or download behind.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def put(entry: tuple[bool, Any]) -> bool:
        while not stop.is_set():
            try:
                q.put(entry, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        it = iter(items)
        try:
            for item in it:
                if not put((True, item)):
                    return
            put((True, done))
        except BaseException as exc:  # handed to the consumer
            put((False, exc))
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()

    threading.Thread(target=produce, daemon=True, name="prefetch").start()
    try:
        while True:
            ok, item = q.get()
            if not ok:
                raise item
            if item is done:
                return
            yield item
    finally:
        stop.set()
        while True:  # unblock a producer waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                break


def _close(it: Any) -> None:
    """Close a generator (or anything with close()); a no-op for other iterables."""
    close = getattr(it, "close", None)
    if close is not None:
        close()


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    it = iter(items)
    try:
        batch: list[Any] = []
        for item in it:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        _close(it)


def flatten(batches: Iterable[list[Any]]) -> Iterator[Any]:
    it = iter(batches)
    try:
        for batch in it:
            yield from batch
    finally:
        _close(it)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    replace_with_retry(tmp, path)


def dataset_revision(stream: Any) -> str | None:
    """The commit sha a streaming HF dataset resolved to, if it can be found.

    datasets exposes no public field for it, but the resolved file URLs carry it
    (hf://datasets/<repo>@<sha>/...). Anything unexpected yields None.
    """
    try:
        # Private datasets attribute: a library upgrade can silently make this null.
        files = stream._ex_iterable.kwargs["files"]
        m = re.search(r"@([0-9a-f]{40})/", str(files[0]))
        return m.group(1) if m else None
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CodeSpec(NamedTuple):
    """Everything build_all needs for the code half of the mix.

    train_rows is consumed while train is written. val_rows is called only after
    train is complete (so its reads start then) and must yield held-out files only.
    """
    share: float
    languages: tuple[str, ...]
    licenses: tuple[str, ...]
    html_cap: float
    max_doc_tokens: int
    val_tokens: int
    train_rows: Iterable[dict]
    val_rows: Callable[[], Iterable[dict]]
    dataset: str = "codeparrot/github-code-clean"
    revision: str | None = None
    heldout_first_file: int | None = None
    files_total: int | None = None
    html_warmup: int = HTML_CAP_WARMUP_TOKENS


def _share(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def _code_summary(split: dict[str, Any], docs: CodeDocs) -> dict[str, Any]:
    """Per-language tokens/documents actually written, plus the filter counts."""
    langs = {label[len(CODE_PREFIX):]: counts
             for label, counts in split["by_label"].items() if label.startswith(CODE_PREFIX)}
    tokens = sum(c["tokens"] for c in langs.values())
    html = langs.get("HTML", {"tokens": 0})["tokens"]
    return {
        "tokens": tokens,
        "documents": sum(c["documents"] for c in langs.values()),
        "html_share_of_code": _share(html, tokens),
        "by_language": {lang: {**langs[lang], "share_of_code": _share(langs[lang]["tokens"],
                                                                      tokens)}
                        for lang in sorted(langs)},
        **{k: docs.stats.get(k, 0) for k in CODE_STAT_KEYS},
        "skipped": docs.skipped,
        # Tokens per document-length bucket, counted as yielded (a truncated final
        # document counts in full), so these sum to within one document of "tokens".
        "tokens_by_doc_length": {b: docs.length_buckets.get(b, 0) for b in
                                 [f"<={e}" for e in DOC_LENGTH_BUCKETS]
                                 + [f">{DOC_LENGTH_BUCKETS[-1]}"]},
        "last_file_read": docs.last_file,
    }


def _build_code(root: Path, it: Iterator[dict], *, train_tokens: int, shard_tokens: int,
                tok: Tokenizer, code: CodeSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mixed train from the text iterator `it` plus code, then the deduplicated code val."""
    text_docs = TextDocs(it, tok)
    code_docs = CodeDocs(iter(code.train_rows), tok, languages=code.languages,
                         licenses=code.licenses, html_cap=code.html_cap,
                         max_doc_tokens=code.max_doc_tokens, html_warmup=code.html_warmup,
                         record_hashes=True)
    mixed = write_split(root / "train", train_tokens, shard_tokens,
                        interleave(text_docs, code_docs, code.share))
    text_counts = mixed["by_label"].get(TEXT, {"tokens": 0, "documents": 0})
    code_counts = _code_summary(mixed, code_docs)
    train = {
        "tokens": mixed["tokens"],
        "rows_consumed": text_docs.rows_consumed + code_docs.stats["rows_scanned"],
        "documents": mixed["documents"],
        "shards": mixed["shards"],
        "sources": {"text": {**text_counts, "rows_consumed": text_docs.rows_consumed},
                    "code": code_counts},
        "target_code_share": code.share,
        "achieved_code_share": _share(code_counts["tokens"], mixed["tokens"]),
    }
    print(f"train: {train['tokens']:,} tokens, code share "
          f"{train['achieved_code_share']:.4f}", flush=True)

    # Code val strictly after code train: every train hash is known by now.
    val_docs = CodeDocs(iter(code.val_rows()), tok, languages=code.languages,
                        licenses=code.licenses, html_cap=code.html_cap,
                        max_doc_tokens=code.max_doc_tokens, html_warmup=code.html_warmup,
                        exclude=code_docs.hashes)
    cv = write_split(root / "code_val", code.val_tokens, shard_tokens, val_docs,
                     allow_short=True)
    if cv["tokens"] == 0:
        raise RuntimeError("code_val: no held-out code document was written")
    summary = _code_summary(cv, val_docs)
    code_val = {**summary, "target_tokens": code.val_tokens,
                "short_of_target": cv["short"], "shards": cv["shards"]}
    print(f"code_val: {code_val['tokens']:,} tokens, "
          f"{code_val['dropped_as_duplicate']:,} dropped as duplicates", flush=True)
    return train, code_val


def build_all(root: Path, stream: Iterable[dict], *, val_tokens: int, train_tokens: int,
              shard_tokens: int, tok: Tokenizer, dataset: str, subset: str,
              revision: str | None, tokenizer_name: str = "gpt2",
              code: CodeSpec | None = None) -> dict[str, Any]:
    """FineWeb val, then train (text only, or the text/code mix when `code` is given),
    then code val. The manifest is written last, atomically."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # A manifest from an earlier build must not survive to describe new shards.
    (root / MANIFEST).unlink(missing_ok=True)
    started, t0 = _now(), time.monotonic()

    # ONE iterator for val and train text, so train continues where val stopped.
    it = iter(stream)
    val = build(root / "val", val_tokens, shard_tokens, it, tok)
    print(f"val:   {val['tokens']:,} tokens", flush=True)

    splits: dict[str, Any] = {"val": val}
    if code is None:
        splits["train"] = build(root / "train", train_tokens, shard_tokens, it, tok)
        if (root / "code_val").exists():  # never leave an older code_val behind
            clear_split_dir(root / "code_val")
    else:
        splits["train"], splits["code_val"] = _build_code(
            root, it, train_tokens=train_tokens, shard_tokens=shard_tokens, tok=tok,
            code=code)
    train = splits["train"]

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "subset": subset,
        "dataset_revision": revision,
        "tokenizer": tokenizer_name,
        "eot": tok.eot,
        "dtype": "uint16 little-endian",
        "splits": splits,
        # The trainer's data (val + train); code_val is for post-run evaluation only.
        "total_tokens": val["tokens"] + train["tokens"],
    }
    if code is not None:
        held = code.heldout_first_file
        manifest["code"] = {
            "dataset": code.dataset,
            "dataset_revision": code.revision,
            "languages": list(code.languages),
            "licenses": list(code.licenses),
            "html_cap": code.html_cap,
            "html_cap_warmup_tokens": code.html_warmup,
            "max_doc_tokens": code.max_doc_tokens,
            "path_skip_rules": CODE_PATH_SKIP_RULES,
            "skipped_train": train["sources"]["code"]["skipped"],
            "skipped_code_val": splits["code_val"]["skipped"],
            "files_total": code.files_total,
            "train_files": None if held is None else [0, held - 1],
            "heldout_files": None if held is None else [held, code.files_total - 1],
            "target_shares": {"text": round(1 - code.share, 6), "code": code.share},
            "achieved_shares": {
                "text": _share(train["sources"]["text"]["tokens"], train["tokens"]),
                "code": train["achieved_code_share"]},
            "html_achieved_share_of_code": train["sources"]["code"]["html_share_of_code"],
            "dedup": "exact content (blake2b-8 of the raw source); near-duplicates remain",
            "dropped_as_duplicate": splits["code_val"]["dropped_as_duplicate"],
        }
    manifest.update({
        "build_started": started,
        "build_finished": _now(),
        "build_seconds": round(time.monotonic() - t0, 1),
    })
    write_manifest(root / MANIFEST, manifest)
    return manifest


# ------------------------------------------------------------------ data v2 (quipu-moe)
#
# Used when the config has code_language_weights (quipu-moe); quipu-114m's configs
# have none and take build_all above, unchanged. The machinery (weighted sampling,
# parallel tokenisation, language ID) is in quipu/shard_mix.py; this is the build.
#
# Train is every bucket (code language, English, each other text language; cmn_Hani
# as zh-Hans + zh-Hant under --lid-filter) interleaved by its final share:
#   code        code_share of train, split by code_language_weights (HTML capped);
#   English     (1 - code_share) * text_language_weights["eng_Latn"];
#   the others  the rest of text, split by their weights (short ones redistributed
#               among the others, never into code or English).
# Each bucket is first collected to disk (quotas + slack, see shard_mix), then the
# final shares are fixed with allocate() and the train shards written. If any code
# language of weight >= 5% (or text language of weight >= 5% of text) ends more than
# 2 percentage points off its weight, the build stops with ShareError before writing
# train (details in collect_report.json).
#
# Failing early (the box bills by the hour): --preflight reads only the code files'
# language/licence/size/path columns and says whether the files can give the code
# mix at all; during the build the code mix is projected at every code file from
# the 20th (what each language holds plus its rate so far, at the file cap; the build
# stops when it misses at PROJECTION_STREAK file ends in a row) and the code shares
# are checked the moment code collection ends, before any text is read. A download
# or read error that no retry can fix (is_permanent) fails at once.
#
# Resume: <shard_dir>/_work/state.json (written atomically) records a fingerprint of
# everything that shapes the collected data (tokenizer and LID hashes, the LID
# threshold, the pinned dataset revisions and file lists, the sampling settings, the
# weights) and a checkpoint per phase: the English val (done), code (after every
# code file, in decision order: bucket files fsynced and their lengths, the
# collector, the stats, the hash count and the next file) and text (after every
# file of every language, and when a language is done). A rerun of the same command
# checks the saved files, deletes partial downloads, truncates the bucket files back
# to the checkpoint and carries on from there, so a preempted box loses at most one
# code file or one text file, and the shards are byte-identical to an uninterrupted
# build. A different fingerprint is refused (--fresh starts over); --from-work
# accepts changed weights / token target and re-allocates from what was collected.
# The old manifest.json is deleted only once the saved build has been accepted. A
# finished build (manifest.json and no state.json; the cleanup deletes state.json
# before the rest of _work) is left alone: its manifest records the fingerprint, and
# a rerun with the same sources and mix prints "already built"; any difference (or a
# manifest without a fingerprint) is a ResumeError (exit 4) naming what differs.
#
# Held out, as for the tokenizer and its gate: val = the last sample-10BT file
# (English, the trainer's loss); val_lang/<bucket> = FineWeb-2's test split; code_val
# = github-code-clean files code_heldout_first_file.. (exact-dedup against train code).
#
# Decontamination (quipu/decontam.py): every code document (train and code_val) and
# every text document (train, val, val_lang) that contains a HumanEval or MBPP
# problem (whitespace-collapsed substring of >= 60 chars, or >= 50% and >= 2 of the
# 13-grams of its solution, HumanEval prompt or a docstring in that prompt) is
# dropped, in the workers, before tokenisation. The benchmarks are read
# at the config's pinned commits; the index's fingerprint is part of the resume
# fingerprint, and the manifest's "decontamination" counts the drops per benchmark.

TEXT_PREFIX = "text:"
MIX_WORK = "_work"
STATE = "state.json"
STATE_VERSION = 1
HASHES = "hashes.u64"
COLLECT_REPORT = "collect_report.json"
PREFLIGHT_REPORT = "preflight.json"
SHARE_CHECK_MIN_WEIGHT = 0.05
SHARE_CHECK_MAX_OFF = 0.02
PROJECTION_MIN_FILES = 20
# The projection is noisy over the first files (a language can be clustered), so it
# stops the build only when it has missed on PROJECTION_STREAK file ends in a row, and
# before file PROJECTION_EARLY_FILES it allows PROJECTION_EARLY_MARGIN more than the
# final check's SHARE_CHECK_MAX_OFF.
PROJECTION_STREAK = 5
PROJECTION_EARLY_FILES = 60
PROJECTION_EARLY_MARGIN = 0.01
LID_WARN_DROP = 0.40         # warn when LID drops more than this of a language's first file
DOWNLOAD_ATTEMPTS = 12       # attempts in all (the first + 11 retries)
BACKOFF_CAP_S = 300
DOWNLOAD_WORKERS = 6


def retry_wait(attempt: int) -> float:
    """Seconds to wait after failed attempt `attempt` (1-based): 5, 10, 20, ... <= 300."""
    return min(BACKOFF_CAP_S, 5 * 2 ** (attempt - 1))


def projection_max_off(files_read: int) -> float:
    """How far off (share of code) a projected language may be after files_read files."""
    early = PROJECTION_EARLY_MARGIN if files_read < PROJECTION_EARLY_FILES else 0.0
    return SHARE_CHECK_MAX_OFF + early


class TransientError(RuntimeError):
    """A read or download failure that a retry can fix (e.g. a truncated download)."""


def is_permanent(exc: BaseException) -> bool:
    """True when retrying the download or read that raised `exc` cannot help, so it
    fails at once instead of backing off for ~30 minutes: the repo, revision or file
    does not exist or is not ours to read (Hub 401/403/404 and their named errors), a
    local file is missing, unreadable or corrupt (FileNotFoundError, PermissionError,
    pyarrow ArrowInvalid), the disk is full (ENOSPC), or a programming error.

    Retried (False): timeouts, connection errors, HTTP 408/429/5xx, TransientError,
    huggingface_hub's LocalEntryNotFoundError (what hf_hub_download raises when the
    connection fails and nothing is cached), other OSErrors (the network stack's
    failures are OSErrors) and anything unrecognised: an unattended build would rather
    back off than stop on an error it cannot name.
    """
    if isinstance(exc, TransientError):
        return False
    try:
        from huggingface_hub import errors as hub
    except ImportError:  # pragma: no cover - huggingface_hub is a dependency
        hub = None
    if hub is not None:
        if isinstance(exc, hub.LocalEntryNotFoundError):
            return False
        if isinstance(exc, (hub.RepositoryNotFoundError, hub.RevisionNotFoundError,
                            hub.EntryNotFoundError, hub.GatedRepoError)):
            return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):  # HfHubHTTPError, httpx / requests HTTP status errors
        return not (status in (408, 429) or status >= 500)
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError,
                        NotADirectoryError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC))
    # pyarrow.ArrowInvalid is a ValueError: a corrupt or unreadable parquet file.
    return isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, AssertionError,
                            NotImplementedError, ImportError, MemoryError))


def backoff(exc: BaseException, attempt: int, attempts: int, what: str,
            sleep: Callable[[float], None] = time.sleep) -> bool:
    """After failed attempt `attempt` (1-based) of `attempts`: False when the caller
    should give up and raise (a permanent error, or the last attempt); otherwise
    prints the error, waits retry_wait(attempt) and returns True."""
    if is_permanent(exc):
        print(f"\n{what} failed and cannot succeed on retry: {exc!r}", file=sys.stderr,
              flush=True)
        return False
    if attempt >= attempts:
        return False
    wait = retry_wait(attempt)
    print(f"\n{what} failed: {exc!r}; retry {attempt}/{attempts - 1} in {wait}s",
          file=sys.stderr, flush=True)
    sleep(wait)
    return True


def sweep_incomplete(directory: Path) -> int:
    """Delete the partial downloads (*.incomplete, huggingface_hub's temp files) a
    killed build left under `directory`; returns how many."""
    n = 0
    for p in sorted(Path(directory).rglob("*.incomplete")) if Path(directory).exists() else ():
        p.unlink(missing_ok=True)
        n += 1
    return n


class MixSpec(NamedTuple):
    """The data v2 build's numbers (main() fills them from the config and flags)."""
    train_tokens: int
    val_tokens: int
    shard_tokens: int
    code_share: float
    code_weights: dict[str, float]
    text_weights: dict[str, float]  # eng_Latn + FineWeb-2 subsets; {} = English only
    licenses: tuple[str, ...]
    html_cap: float
    max_doc_tokens: int
    code_val_tokens: int
    lang_val_tokens: int = 1_000_000
    slack: float = sm.DEFAULT_SLACK
    window: int = sm.DEFAULT_WINDOW
    stall_windows: int = sm.DEFAULT_STALL_WINDOWS
    stall_gain: float = sm.DEFAULT_STALL_GAIN
    workers: int = 1
    batch_docs: int = 256
    text_order: tuple[str, ...] | None = None  # FineWeb-2 collection order; None: config
    keep_work: bool = False
    # Code train files (the file cap), numbered from 0: the projection's horizon.
    # None: no projection (the share check after code collection still runs).
    code_files: int | None = None
    projection_min_files: int = PROJECTION_MIN_FILES


class MixSources(NamedTuple):
    """Row sources for build_mix.

    code_train(start_file): rows {"code", "language", "license", "path", "file"} of
    code files start_file, start_file + 1, ... in file order ("file" is the file's
    index, from 0). A resumed build passes the file after its last checkpoint. A
    real source that filters rows before they become dicts attaches what it dropped
    to the next row it yields, as row[FILTERED] = {"rows_scanned": n, ...}, and ends
    every file with a marker row {FILE_END: index, FILTERED: {...}} (what the file's
    last row groups dropped); without markers, a change of "file" ends a file.
    code_val(): the held-out code rows. text_train: language -> rows(start_file)
    {"text"[, "file"]}, the language's rows from its start_file-th source file on (a
    resumed build passes the files it has read; a source without "file" is only ever
    called with 0). text_val: language -> rows(). Every *_val source must yield
    held-out rows only, never rows its train source can reach.
    """
    code_train: Callable[[int], Iterable[dict]]
    code_val: Callable[[], Iterable[dict]]
    text_train: dict[str, Callable[[int], Iterable[dict]]]
    text_val: dict[str, Callable[[], Iterable[dict]]]


FILTERED = "_filtered"
FILE_END = "_file_end"


def uses_mix(data_cfg: Any) -> bool:
    """True when a DataConfig asks for the data v2 build (it has code weights)."""
    return bool(getattr(data_cfg, "code_language_weights", None))


def _file_mark(counts: Counter, file: Any) -> sm.Offer:
    return sm.Offer(sm.MARK, "", "", {"counts": counts, "file_done": file})


def code_offers(rows: Iterable[dict], languages: Iterable[str],
                licenses: Iterable[str]) -> Iterator[sm.Offer]:
    """The cheap code filters, in the main process: language, licence, blank, path.

    The rows read and filtered on the way to each offer travel with it (meta) and
    are counted when the Runner decides it, not when they are read: the Runner reads
    ahead by a worker-dependent amount, and counting at read time would make the
    manifest depend on the number of workers. The end of each file becomes a MARK
    offer carrying the file's trailing counts, so a checkpoint there is exact.
    """
    languages, licenses = frozenset(languages), frozenset(licenses)
    counts: Counter = Counter()
    current: Any = None
    it = iter(rows)
    try:
        for row in it:
            if FILE_END in row:
                counts.update(row.get(FILTERED) or {})
                yield _file_mark(counts, row[FILE_END])
                counts, current = Counter(), None
                continue
            file = row.get("file")
            if current is not None and file != current:
                yield _file_mark(counts, current)
                counts = Counter()
            current = file
            counts.update(row.get(FILTERED) or {})
            counts["rows_scanned"] += 1
            language = row.get("language")
            if language not in languages:
                counts["dropped_language"] += 1
                continue
            if row.get("license") not in licenses:
                counts["dropped_license"] += 1
                continue
            code = row.get("code") or ""
            if not code.strip():
                counts["skipped_blank"] += 1
                continue
            reason = path_skip_reason(row.get("path"))
            if reason is not None:
                counts[f"skipped_{reason}"] += 1
                continue
            meta = {"counts": counts}
            if file is not None:
                meta["last_file"] = file
            yield sm.Offer(sm.CODE, language, code, meta)
            counts = Counter()
        if current is not None:
            yield _file_mark(counts, current)
    finally:
        _close(it)


def text_offers(rows: Iterable[dict], source: str) -> Iterator[sm.Offer]:
    """Non-blank texts; filter counts travel with the offers as in code_offers, and a
    change of the rows' "file" (when they have one) becomes a MARK offer."""
    counts: Counter = Counter()
    current: Any = None
    it = iter(rows)
    try:
        for row in it:
            file = row.get("file")
            if current is not None and file != current:
                yield _file_mark(counts, current)
                counts = Counter()
            current = file
            counts["rows_scanned"] += 1
            text = row.get("text") or ""
            if not text.strip():
                counts["skipped_blank"] += 1
                continue
            yield sm.Offer(sm.TEXT, source, text, {"counts": counts})
            counts = Counter()
        if current is not None:
            yield _file_mark(counts, current)
    finally:
        _close(it)


def _counted(rows: Iterable[dict], counts: Counter) -> Iterator[dict]:
    """Rows as they are consumed, with any attached filter counts added to counts
    (file-end markers are counted and dropped)."""
    it = iter(rows)
    try:
        for row in it:
            counts.update(row.pop(FILTERED, None) or {})
            if FILE_END not in row:
                yield row
    finally:
        _close(it)


def text_buckets(language: str, lid: bool) -> dict[str, float]:
    """A text source's buckets. With the LID filter, cmn_Hani becomes zh-Hans and
    zh-Hant in EQUAL halves (the specialist labels them zho_Hans / zho_Hant); if one
    script stalls, its remainder goes to the other. Without LID, cmn_Hani is one."""
    if lid and language == sm.ZH_SOURCE:
        return {b: 1 / len(sm.ZH_BUCKETS) for b in sm.ZH_BUCKETS}
    return {language: 1.0}


def _collect_text(runner: sm.Runner, rows: Iterable[dict], language: str, total: float,
                  store: sm.BucketStore, spec: MixSpec, *, lid: bool, slack: float,
                  stats: Counter | None = None,
                  on_mark: Callable[[dict], None] | None = None,
                  check_text: bool = False) -> tuple[sm.Collector, Counter]:
    """check_text: drop documents containing a benchmark problem (validation splits)."""
    stats = Counter() if stats is None else stats
    col = sm.Collector(text_buckets(language, lid), total, slack=slack, window=spec.window,
                       stall_windows=spec.stall_windows, stall_gain=spec.stall_gain,
                       store=store)
    runner.collect(text_offers(rows, language), col, stats, on_mark=on_mark,
                   check_text=check_text)
    return col, stats


def _plain(stats: Counter) -> dict[str, Any]:
    """A stats Counter as manifest JSON: "<name>:<sub>" keys nested (lid_dropped_as,
    dropped_contamination_as)."""
    out: dict[str, Any] = {}
    for k in sorted(stats):
        if k == "last_file":
            continue
        if ":" in k:
            name, sub = k.split(":", 1)
            out.setdefault(name, {})[sub] = stats[k]
        else:
            out[k] = stats[k]
    return out


def _dropped_by_benchmark(stats: Any, benchmarks: Iterable[str]) -> dict[str, int]:
    return {b: int(stats.get(f"dropped_contamination_as:{b}", 0)) for b in benchmarks}


def _text_summary(col: sm.Collector, stats: Counter) -> dict[str, Any]:
    s = _plain(stats)
    offers = stats["offers"]
    return {**s, "lid_kept": offers - stats["lid_dropped"],
            "lid_dropped": stats["lid_dropped"],
            "lid_kept_unsure": stats["lid_kept_unsure"],
            "tokens": sum(col.taken.values()), "collection": col.summary()}


class LidWatch:
    """Warns (loudly, never fails) when language ID drops more than LID_WARN_DROP of a
    language's documents in its first file (or, for a source without files, in all
    it read): a sign that the source or the threshold is wrong for that language."""

    def __init__(self, language: str, stats: Counter) -> None:
        self.language, self.stats = language, stats
        self.checked = False
        self.warning: str | None = None

    def state(self) -> dict[str, Any]:
        return {"checked": self.checked, "warning": self.warning}

    @classmethod
    def from_state(cls, language: str, stats: Counter, state: dict[str, Any]) -> "LidWatch":
        watch = cls(language, stats)
        watch.checked, watch.warning = bool(state["checked"]), state["warning"]
        return watch

    def on_mark(self, meta: dict) -> None:
        if not self.checked:
            self.check("its first file")

    def finish(self) -> str | None:
        if not self.checked:
            self.check("everything read")
        return self.warning

    def check(self, where: str) -> None:
        self.checked = True
        offers, dropped = self.stats["offers"], self.stats["lid_dropped"]
        if not offers or dropped / offers <= LID_WARN_DROP:
            return
        as_ = Counter({k.split(":", 1)[1]: v for k, v in self.stats.items()
                       if k.startswith("lid_dropped_as:")})
        top = ", ".join(f"{label} {n:,}" for label, n in as_.most_common(3))
        self.warning = (f"{self.language}: language ID dropped {dropped / offers:.0%} of the "
                        f"documents in {where} ({dropped:,} of {offers:,}; as {top}). "
                        "Check the source language and --lid-threshold; the build goes on.")
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: {self.warning}\n{bar}\n", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ resume state

def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x))


def _write_json_durable(path: Path, obj: Any) -> None:
    """Write JSON atomically (temp file, fsync, rename): a crash leaves the old file
    or the new one, never a torn one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    replace_with_retry(tmp, path)


def mix_fingerprint(spec: MixSpec, setup: sm.DocSetup,
                    provenance: dict[str, Any] | None) -> dict[str, Any]:
    """What a resumed build must share with the saved one. "sources": everything that
    changes which documents are collected (a mismatch needs --fresh); "mix": the
    weights and token target (a mismatch is allowed with --from-work)."""
    prov = provenance or {}
    lid = setup.lid
    sources = {
        "val_tokens": spec.val_tokens, "licenses": list(spec.licenses),
        "max_doc_tokens": spec.max_doc_tokens, "slack": spec.slack, "window": spec.window,
        "stall_windows": spec.stall_windows, "stall_gain": spec.stall_gain,
        "text_order": None if spec.text_order is None else list(spec.text_order),
        "code_files": spec.code_files, "path_skip_rules": CODE_PATH_SKIP_RULES,
        "tokenizer_sha256": (prov.get("tokenizer") or {}).get("sha256"),
        "lid": None if lid is None else {
            "threshold": setup.lid_threshold,
            "max_chars": getattr(lid, "max_chars", sm.LID_MAX_CHARS),
            "model_sha256": (prov.get("lid") or {}).get("sha256")},
        "leakage_guard_files": getattr(setup.guard, "reference_files", None),
        "decontamination": None if setup.decontam is None else setup.decontam.fingerprint(),
        "datasets": prov.get("sources"),
    }
    mix = {"train_tokens": spec.train_tokens, "code_share": spec.code_share,
           "code_weights": spec.code_weights, "text_weights": spec.text_weights,
           "html_cap": spec.html_cap}
    return _jsonable({"sources": sources, "mix": mix})


def _diff(old: dict, new: dict) -> list[str]:
    def short(v: Any) -> str:
        s = json.dumps(v, sort_keys=True)
        return s if len(s) <= 100 else s[:97] + "..."
    return [f"{k}: {short(old.get(k))} -> {short(new.get(k))}"
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def _finished_diff(old: dict | None, new: dict[str, Any]) -> list[str]:
    """How a finished build's manifest fingerprint (sources + mix) differs from the
    current one; a manifest without one (older builder) always differs."""
    if not isinstance(old, dict):
        return ["fingerprint: the manifest records none (built by an older builder), so "
                "its settings cannot be checked"]
    return [f"{part}.{d}" for part in ("sources", "mix")
            for d in _diff(old.get(part) or {}, new[part])]


class BuildState:
    """<shard_dir>/_work/state.json: the fingerprint and each phase's checkpoint."""

    def __init__(self, work: Path, fingerprint: dict[str, Any], *, fresh: bool,
                 from_work: bool) -> None:
        self.path = Path(work) / STATE
        self.mix_changed: list[str] = []
        if fresh and work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        self.resumed = self.path.exists()
        if not self.resumed:
            if from_work:
                raise sm.ResumeError(f"--from-work needs the saved state of an earlier build "
                                     f"({self.path}); there is none")
            for p in list(work.iterdir()):  # left by a build that never saved: stale
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            self.data = {"version": STATE_VERSION, "fingerprint": fingerprint, "phases": {}}
            self.save()
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != STATE_VERSION:
            raise sm.ResumeError(f"{self.path} is from another builder version; rerun with "
                                 "--fresh")
        old = data.get("fingerprint", {})
        diff = _diff(old.get("sources", {}), fingerprint["sources"])
        if diff:
            raise sm.ResumeError(
                f"the saved build in {work} collected different data, so it cannot be "
                "resumed:\n  " + "\n  ".join(diff)
                + "\nrerun with --fresh to discard it (everything is read again), or restore "
                  "the settings it was started with")
        mix_diff = _diff(old.get("mix", {}), fingerprint["mix"])
        if mix_diff and not from_work:
            raise sm.ResumeError(
                f"the saved build in {work} has different mix settings:\n  "
                + "\n  ".join(mix_diff)
                + "\nrerun with --from-work to re-allocate the mix from the collected buckets "
                  "(nothing is downloaded again; needs its code collection finished), or "
                  "--fresh to start over")
        if from_work and not (data["phases"].get("code") or {}).get("done"):
            raise sm.ResumeError(f"--from-work needs a finished code collection in the saved "
                                 f"state {self.path}; rerun without it (same settings) to "
                                 "finish collecting, or --fresh")
        self.mix_changed = mix_diff
        data["fingerprint"] = fingerprint
        self.data = data
        self.save()

    def phase(self, name: str) -> Any:
        return self.data["phases"].get(name)

    def set_phase(self, name: str, value: Any) -> None:
        self.data["phases"][name] = _jsonable(value)
        self.save()

    def save(self) -> None:
        _write_json_durable(self.path, self.data)


class HashLog:
    """Append-only uint64 file of the content hashes of admitted code documents (the
    code_val dedup set), truncatable to a checkpoint's count like the bucket files."""

    def __init__(self, path: Path, count: int) -> None:
        self.path = Path(path)
        if count == 0:
            self.path.write_bytes(b"")
            self.hashes: set[int] = set()
        else:
            size = self.path.stat().st_size if self.path.exists() else -1
            if size < 8 * count:
                raise sm.ResumeError(f"{self.path} holds {size} bytes, less than its "
                                     f"checkpoint's {count} hashes; rerun with --fresh")
            os.truncate(self.path, 8 * count)
            self.hashes = set(np.fromfile(self.path, dtype="<u8").tolist())
        self.count = count
        self._f = open(self.path, "ab")

    def add(self, h: int) -> None:
        self.hashes.add(h)
        self._f.write(np.uint64(h).tobytes())
        self.count += 1

    def sync(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def _verify_work(work: Path, state: BuildState, manifest: Path) -> None:
    """Check, before anything is truncated or deleted, that every checkpointed file
    in `work` is still there and long enough; a damaged _work is a ResumeError that
    leaves the shards and any manifest.json untouched."""
    try:
        for name in ("val", "code", "text"):
            ph = state.phase(name)
            if ph and ph.get("store"):
                sm.BucketStore.verify(work / name, ph["store"])
        code = state.phase("code")
        if code and code.get("hashes"):
            path = work / HASHES
            size = path.stat().st_size if path.exists() else -1
            if size < 8 * code["hashes"]:
                raise sm.ResumeError(f"{path} holds {size} bytes, less than its "
                                     f"checkpoint's {code['hashes']} hashes")
    except sm.ResumeError as exc:
        kept = (f"; {manifest} and the shards are untouched: if that build had finished, "
                f"delete {work} and use them" if manifest.exists() else "")
        raise sm.ResumeError(f"the saved build in {work} is damaged ({exc}){kept}; "
                             "otherwise rerun with --fresh") from exc


# ------------------------------------------------------------------ build phases

def _val_phase(runner: sm.Runner, sources: MixSources, spec: MixSpec, work: Path,
               state: BuildState, lid: bool) -> tuple[sm.Collector, Counter, sm.BucketStore]:
    """The trainer's validation text (English, held out); done once, then reused."""
    ph = state.phase("val")
    store = sm.BucketStore(work / "val", fresh=False)
    if ph and ph["done"]:
        store.restore(ph["store"])
        col, stats = sm.Collector.from_snapshot(ph["collector"]), Counter(ph["stats"])
    else:
        store.restore({})
        col, stats = _collect_text(runner, sources.text_val[sm.ENGLISH](), sm.ENGLISH,
                                   spec.val_tokens, store, spec, lid=lid, slack=0.0,
                                   check_text=True)
        state.set_phase("val", {"done": True, "store": store.snapshot(),
                                "collector": col.snapshot(), "stats": stats})
    store.close()
    return col, stats, store


def _code_phase(runner: sm.Runner, sources: MixSources, spec: MixSpec, work: Path,
                state: BuildState, *, weights: dict[str, float], total: float,
                caps: dict[str, float], report: Path, resume: dict[str, Any]
                ) -> tuple[sm.Collector, Counter, sm.BucketStore, set[int]]:
    """Code collection over the files in order, checkpointed at every file end."""
    ph = state.phase("code")
    store = sm.BucketStore(work / "code", fresh=False)
    if ph and ph["done"]:
        store.restore(ph["store"])
        store.close()
        log = HashLog(work / HASHES, ph["hashes"])
        log.close()
        return (sm.Collector.from_snapshot(ph["collector"]), Counter(ph["stats"]), store,
                log.hashes)
    if ph:
        store.restore(ph["store"])
        col = sm.Collector.from_snapshot(ph["collector"], store)
        stats, start = Counter(ph["stats"]), int(ph["next_file"])
        log = HashLog(work / HASHES, ph["hashes"])
        resume["code_from_file"] = start
        print(f"code: resuming at file {start} ({sum(col.taken.values()):,} tokens held)",
              flush=True)
    else:
        store.restore({})
        col = sm.Collector(weights, total, caps=caps, slack=spec.slack, window=spec.window,
                           stall_windows=spec.stall_windows, stall_gain=spec.stall_gain,
                           store=store)
        stats, start = Counter(), 0
        log = HashLog(work / HASHES, 0)

    # File ends in a row at which the projection missed (saved with each checkpoint,
    # so a resumed build stops exactly where an uninterrupted one would).
    streak = int(ph.get("projection_streak", 0)) if ph else 0

    def checkpoint(next_file: int | None, done: bool) -> None:
        snap = store.snapshot()
        log.sync()
        state.set_phase("code", {"done": done, "next_file": next_file, "store": snap,
                                 "collector": col.snapshot(), "stats": stats,
                                 "hashes": log.count, "projection_streak": streak})

    def project(files_read: int) -> None:
        """Stop the build when the code mix projected at the file cap has missed its
        target (by more than projection_max_off) at PROJECTION_STREAK file ends in a
        row. The projection only decides whether the build stops, never which
        documents are collected, so --projection-min-files is not in the resume
        fingerprint: a stopped build can be resumed with a later (or no) projection."""
        nonlocal streak
        files = spec.code_files
        if files is None or files_read < spec.projection_min_files or files_read >= files:
            return
        violations, avail = sm.projection_violations(
            col, files_read, files, min_weight=SHARE_CHECK_MIN_WEIGHT,
            max_off=projection_max_off(files_read))
        if not violations:
            streak = 0
            return
        streak += 1
        if streak < PROJECTION_STREAK:
            return
        write_manifest(report, {"stage": f"code projection after file {files_read} of {files}",
                                "violations": violations,
                                "projected_available_tokens": avail,
                                "missed_file_ends_in_a_row": streak,
                                "collection": col.summary(), "stats": _plain(stats)})
        raise sm.ShareError(
            f"code: after file {files_read} of {files}, the mix projected at the file cap "
            f"has missed its target at {streak} file ends in a row, so the build stops now "
            f"rather than read {files - files_read} more files:\n  " + "\n  ".join(violations)
            + f"\n(projected: what each language holds plus its rate so far; see {report}.) "
              "Two ways on:\n"
              "  - the mix cannot be met: change the weights in the config and rerun with "
              f"--fresh (the code read so far, about {files_read} files, is downloaded "
              "again: changed weights cannot resume a code collection); --preflight shows "
              "what the files hold;\n"
              "  - the projection is wrong (a language clustered in later files): rerun the "
              f"same command with --projection-min-files N, N > {files_read} (N >= {files} "
              f"turns it off); the build resumes after file {files_read}, and the share "
              "check after code collection still applies.")

    def on_mark(meta: dict) -> None:
        files_read = int(meta["file_done"]) + 1
        checkpoint(files_read, False)
        project(files_read)

    if start:
        project(start)
    runner.collect(code_offers(sources.code_train(start), weights, spec.licenses), col, stats,
                   on_admit=lambda offer, res: log.add(sm.content_hash(offer.text)),
                   on_mark=on_mark)
    checkpoint(None, True)
    log.close()
    store.close()
    return col, stats, store, log.hashes


def _text_phase(runner: sm.Runner, sources: MixSources, spec: MixSpec, work: Path,
                state: BuildState, *, lid: bool, english_total: float, group_total: float,
                others: dict[str, float], order: list[str]
                ) -> tuple[dict[str, tuple[sm.Collector, Counter]], dict[str, str],
                           sm.BucketStore]:
    """English, then the other languages in order; each checkpointed at every end of
    one of its source files (ph["partial"]: the files read, the collector, the stats,
    the LID warning state) and when done. A resumed language restarts at the file
    after its checkpoint, so a crash loses at most one text file."""
    ph = state.phase("text") or {"done": [], "store": {}, "langs": {}}
    ph.setdefault("partial", None)
    store = sm.BucketStore(work / "text", fresh=False)
    store.restore(ph["store"])
    cols: dict[str, tuple[sm.Collector, Counter]] = {}
    warnings: dict[str, str] = {}
    for lang in [sm.ENGLISH] + order:
        saved = ph["langs"].get(lang)
        if saved is not None:
            cols[lang] = (sm.Collector.from_snapshot(saved["collector"]),
                          Counter(saved["stats"]))
            if saved.get("lid_warning"):
                warnings[lang] = saved["lid_warning"]
            continue
        if lang == sm.ENGLISH:
            total, slack = english_total, 0.0
        else:
            # Each language's target is set when its turn comes, from what the earlier
            # ones actually held.
            held = {x: float(sum(cols[x][0].taken.values())) for x in order if x in cols}
            now, _ = sm.allocate(group_total, others, held)
            total, slack = now[lang], spec.slack
        partial = ph["partial"] if (ph["partial"] or {}).get("lang") == lang else None
        if partial is not None:
            col = sm.Collector.from_snapshot(partial["collector"], store)
            stats: Counter = Counter(partial["stats"])
            files = int(partial["files"])
            watch = LidWatch.from_state(lang, stats, partial["lid"]) if lid else None
            print(f"{lang}: resuming at its file {files} "
                  f"({sum(col.taken.values()):,} tokens held)", flush=True)
        else:
            col = sm.Collector(text_buckets(lang, lid), total, slack=slack, window=spec.window,
                               stall_windows=spec.stall_windows, stall_gain=spec.stall_gain,
                               store=store)
            stats, files = Counter(), 0
            watch = LidWatch(lang, stats) if lid else None

        def on_mark(meta: dict, lang: str = lang, col: sm.Collector = col,
                    stats: Counter = stats, watch: LidWatch | None = watch) -> None:
            nonlocal files
            if watch is not None:
                watch.on_mark(meta)
            files += 1
            ph["partial"] = {"lang": lang, "files": files, "collector": col.snapshot(),
                             "stats": stats, "lid": watch.state() if watch else None}
            ph["store"] = store.snapshot()
            state.set_phase("text", ph)

        # Train text is decontaminated too (check_text): web pages quote benchmark
        # problems and their answers.
        runner.collect(text_offers(sources.text_train[lang](files), lang), col, stats,
                       on_mark=on_mark, check_text=True)
        warning = watch.finish() if watch else None
        if warning:
            warnings[lang] = warning
        cols[lang] = (col, stats)
        ph["langs"][lang] = {"collector": col.snapshot(), "stats": stats,
                             "lid_warning": warning}
        ph["done"].append(lang)
        ph["partial"] = None
        ph["store"] = store.snapshot()
        state.set_phase("text", ph)
        print(f"{lang}: {sum(col.taken.values()):,} tokens (target {total:,.0f})", flush=True)
    store.close()
    return cols, warnings, store


def build_mix(root: Path, spec: MixSpec, sources: MixSources, setup: sm.DocSetup,
              provenance: dict[str, Any] | None = None, *, fresh: bool = False,
              from_work: bool = False) -> dict[str, Any]:
    """val, train, code_val and val_lang for the data v2 mix; manifest written last.

    Resumes from <root>/_work when an earlier run of the same build was interrupted
    (see "Resume" above); fresh=True discards it; from_work=True re-allocates from it
    with changed weights or token target. Raises ShareError (after writing
    collect_report.json, before writing train) when the collected data cannot give
    the configured mix within the tolerance, RuntimeError when a source ran short of
    what the mix needs, ResumeError when _work cannot be resumed.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    provenance = provenance or {}
    started, t0 = _now(), time.monotonic()
    work = root / MIX_WORK
    manifest_path = root / MANIFEST
    fingerprint = mix_fingerprint(spec, setup, provenance)
    if manifest_path.exists() and not fresh and not (work / STATE).exists():
        # A finished build (its cleanup deletes state.json before anything else, so a
        # crash part way through it lands here too). Never rebuild it by accident, and
        # never call it "already built" for settings it was not built with.
        if from_work:
            raise sm.ResumeError(f"{root} is a finished build and its _work is gone, so "
                                 "--from-work has nothing to re-allocate from; rerun with "
                                 "--fresh to build it again (everything is read again)")
        done = json.loads(manifest_path.read_text(encoding="utf-8"))
        diff = _finished_diff(done.get("fingerprint"), fingerprint)
        if diff:
            raise sm.ResumeError(
                f"{root} holds a finished build made with other settings:\n  "
                + "\n  ".join(diff)
                + "\nrerun with --fresh to rebuild it with these settings (everything is "
                  "read again), or restore the settings it was built with")
        shutil.rmtree(work, ignore_errors=True)  # what a crashed cleanup left
        print(f"{root} is already built (use --fresh to rebuild)", flush=True)
        return done
    lid = setup.lid is not None
    cw = dict(spec.code_weights)
    tw = dict(spec.text_weights) or {sm.ENGLISH: 1.0}
    if "HTML" not in cw:
        raise ValueError("code weights must include 'HTML': the HTML cap is keyed on it")
    if sm.ENGLISH not in tw:
        raise ValueError(f"text weights must include {sm.ENGLISH} (FineWeb-Edu)")
    others = {lang: w for lang, w in tw.items() if lang != sm.ENGLISH}
    order = list(spec.text_order) if spec.text_order is not None else list(others)
    if sorted(order) != sorted(others):
        raise ValueError(f"text_order {order} must list exactly {sorted(others)}")
    T = spec.train_tokens
    C = spec.code_share * T
    text_total = T - C
    E = text_total * tw[sm.ENGLISH]
    G = text_total - E
    caps = {"HTML": spec.html_cap}
    state = BuildState(work, fingerprint, fresh=fresh, from_work=from_work)
    if (state.data.get("finished") and not state.mix_changed and not from_work
            and manifest_path.exists()):
        print(f"{root} is already built (its _work was kept; use --fresh to rebuild)",
              flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if state.resumed:
        _verify_work(work, state, manifest_path)
        swept = sweep_incomplete(work / "dl")
        if swept:
            print(f"deleted {swept} partial download(s) left in {work / 'dl'}", flush=True)
    # Only now, with the saved build accepted and intact, may the old manifest go: it
    # must not survive to describe the shards about to be rewritten.
    manifest_path.unlink(missing_ok=True)
    (root / COLLECT_REPORT).unlink(missing_ok=True)
    if state.data.get("finished"):
        state.data["finished"] = False
        state.save()
    resume: dict[str, Any] = {"resumed": state.resumed, "from_work": from_work,
                              "mix_changed": state.mix_changed, "code_from_file": None}
    if state.resumed:
        print(f"resuming the build saved in {work}"
              + (" (re-allocating: " + "; ".join(state.mix_changed) + ")"
                 if state.mix_changed else ""), flush=True)

    with sm.Runner(setup, spec.workers, batch_docs=spec.batch_docs) as runner:
        eot = runner.tok.eot
        # 1. The trainer's validation: English, from the held-out file.
        val_col, val_stats, val_store = _val_phase(runner, sources, spec, work, state, lid)
        val = write_split(root / "val", spec.val_tokens, spec.shard_tokens,
                          ((a, TEXT_PREFIX + sm.ENGLISH) for a in val_store.docs(sm.ENGLISH)))
        val["source"] = _text_summary(val_col, val_stats)
        print(f"val:   {val['tokens']:,} tokens", flush=True)

        # 2. Code: one pass over the mixed stream, per-language quotas.
        code_col, code_stats, code_store, hashes = _code_phase(
            runner, sources, spec, work, state, weights=cw, total=C, caps=caps,
            report=root / COLLECT_REPORT, resume=resume)
        print(f"code collected: {sum(code_col.taken.values()):,} tokens, "
              f"{code_col.windows} windows, exhausted {sorted(code_col.exhausted)}, "
              f"last file {code_stats.get('last_file')}", flush=True)

        # The code mix is known now: check it before any text is read. The HTML cap
        # is re-applied inside allocate after every redistribution.
        code_have = {b: float(code_store.tokens[b]) for b in cw}
        code_final, code_short = sm.allocate(C, cw, code_have, caps)
        planned_code = {b: v / C for b, v in code_final.items()}
        code_violations = _violations(planned_code, cw, {}, {})
        if code_violations:
            write_manifest(root / COLLECT_REPORT, {
                "stage": "after code collection", "violations": code_violations,
                "code": {"collection": code_col.summary(), "stats": _plain(code_stats),
                         "planned_share_of_code": planned_code}})
            raise sm.ShareError(
                "code is off target right after code collection, before any text was read, "
                "so no train shards were written:\n  " + "\n  ".join(code_violations)
                + f"\n(collected per language: {root / COLLECT_REPORT}; a common language "
                  "short at the file cap needs more files (--max-code-files) or a lower "
                  "weight; the collected code is kept in _work: rerun with --from-work "
                  "after changing the weights)")

        # 3. English, then the other languages one at a time.
        text_cols, lid_warnings, text_store = _text_phase(
            runner, sources, spec, work, state, lid=lid, english_total=E, group_total=G,
            others=others, order=order)

        # 4. Final shares.
        eng_have = float(text_store.tokens[sm.ENGLISH])
        grp_have = {x: float(sum(text_cols[x][0].taken.values())) for x in others}
        grp_final, grp_short = sm.allocate(G, others, grp_have) if others else ({}, 0.0)
        final: dict[str, float] = {CODE_PREFIX + b: v for b, v in code_final.items()}
        final[TEXT_PREFIX + sm.ENGLISH] = min(E, eng_have)
        for x in others:
            col = text_cols[x][0]
            if len(col.weights) > 1:
                split, _ = sm.allocate(grp_final[x], col.weights,
                                       {b: float(col.taken[b]) for b in col.weights})
                final.update({TEXT_PREFIX + b: v for b, v in split.items()})
            else:
                final[TEXT_PREFIX + x] = grp_final[x]
        planned_text = {sm.ENGLISH: final[TEXT_PREFIX + sm.ENGLISH] / text_total,
                        **{x: grp_final[x] / text_total for x in others}}
        violations = _violations(planned_code, cw, planned_text, tw)
        collection = {
            "code": {"collection": code_col.summary(), "stats": _plain(code_stats),
                     "last_file_read": code_stats.get("last_file"),
                     "planned_share_of_code": planned_code},
            "text": {x: _text_summary(c, s) for x, (c, s) in text_cols.items()},
            "planned_share_of_text": planned_text,
        }
        for x, warning in lid_warnings.items():
            collection["text"][x]["lid_warning"] = warning
        short = []
        if code_short > 0.5:
            short.append(f"code: {code_short:,.0f} tokens short of {C:,.0f}")
        if eng_have < E:
            short.append(f"{sm.ENGLISH}: {eng_have:,.0f} tokens of {E:,.0f}")
        if grp_short > 0.5:
            short.append(f"other languages: {grp_short:,.0f} tokens short of {G:,.0f}")
        if violations or short:
            write_manifest(root / COLLECT_REPORT, {"violations": violations, "short": short,
                                                   **collection})
            if violations:
                raise sm.ShareError(
                    "the mix is off target, so no train shards were written:\n  "
                    + "\n  ".join(violations)
                    + f"\n(collected per language: {root / COLLECT_REPORT}; everything "
                      "collected is kept in _work: change the weights or --train-tokens and "
                      "rerun with --from-work, nothing is downloaded again)")
            raise RuntimeError("sources ran short: " + "; ".join(short)
                               + f" (see {root / COLLECT_REPORT}; rerun with --from-work "
                                 "and a smaller --train-tokens)")

        # 5. Train: every bucket interleaved by its final share.
        shares = {b: v / T for b, v in final.items()}
        streams = {b: (code_store if b.startswith(CODE_PREFIX) else text_store)
                   .docs(b.split(":", 1)[1]) for b in final}
        train = write_split(root / "train", T, spec.shard_tokens,
                            sm.weighted_interleave(streams, shares))
        del streams
        print(f"train: {train['tokens']:,} tokens", flush=True)

        # 6. Code validation (held-out files), deduplicated against train code.
        cv_read: Counter = Counter()
        cv_rows = _counted(sources.code_val(), cv_read)
        cv_docs = CodeDocs(cv_rows, runner.tok, languages=tuple(cw),
                           licenses=spec.licenses, html_cap=spec.html_cap,
                           max_doc_tokens=spec.max_doc_tokens, exclude=hashes,
                           leak_guard=setup.guard, decontam=setup.decontam)
        try:
            cv = write_split(root / "code_val", spec.code_val_tokens, spec.shard_tokens,
                             cv_docs, allow_short=True)
        finally:
            _close(cv_rows)  # stops the held-out downloads at once
        if cv["tokens"] == 0:
            raise RuntimeError("code_val: no held-out code document was written")
        code_val = {**_code_summary(cv, cv_docs), "target_tokens": spec.code_val_tokens,
                    "short_of_target": cv["short"], "shards": cv["shards"],
                    "dropped_leakage": cv_docs.stats.get("dropped_leakage", 0),
                    "read_filters": dict(cv_read)}

        # 7. Per-language validation (FineWeb-2 test split), one split per bucket.
        shutil.rmtree(root / "val_lang", ignore_errors=True)
        lv_store = sm.BucketStore(work / "val_lang")
        lv_cols = {}
        for x in others:
            n = len(text_buckets(x, lid))
            lv_cols[x] = _collect_text(runner, sources.text_val[x](), x,
                                       spec.lang_val_tokens * n, lv_store, spec, lid=lid,
                                       slack=0.0, check_text=True)
        lv_store.close()
        val_lang = {}
        for x in others:
            for b in text_buckets(x, lid):
                res = write_split(root / "val_lang" / b, spec.lang_val_tokens, spec.shard_tokens,
                                  ((a, TEXT_PREFIX + b) for a in lv_store.docs(b)),
                                  allow_short=True)
                val_lang[b] = {"tokens": res["tokens"], "documents": res["documents"],
                               "short": res["short"], "shards": res["shards"],
                               "source": x}
            val_lang_source = _text_summary(*lv_cols[x])
            for b in text_buckets(x, lid):
                val_lang[b]["source_stats"] = val_lang_source

    # Achieved, from what write_split actually wrote.
    by_label = train["by_label"]
    tok_of = {b: by_label.get(b, {"tokens": 0})["tokens"] for b in final}
    code_tokens = sum(v for b, v in tok_of.items() if b.startswith(CODE_PREFIX))
    text_tokens = train["tokens"] - code_tokens
    zh = {b for b in final if b[len(TEXT_PREFIX):] in sm.ZH_BUCKETS}

    def text_lang(b: str) -> str:
        return sm.ZH_SOURCE if b in zh else b[len(TEXT_PREFIX):]

    achieved_code = {b[len(CODE_PREFIX):]: _share(v, code_tokens)
                     for b, v in tok_of.items() if b.startswith(CODE_PREFIX)}
    achieved_text: dict[str, float] = {}
    for b, v in tok_of.items():
        if b.startswith(TEXT_PREFIX):
            achieved_text[text_lang(b)] = achieved_text.get(text_lang(b), 0.0) + v
    achieved_text = {x: _share(int(v), text_tokens) for x, v in achieved_text.items()}
    final_violations = _violations(achieved_code, cw, achieved_text, tw)

    code_summary = {
        "weights": cw, "html_cap": spec.html_cap, "licenses": list(spec.licenses),
        "max_doc_tokens": spec.max_doc_tokens, "path_skip_rules": CODE_PATH_SKIP_RULES,
        "tokens": code_tokens,
        "by_language": {
            b: {"target_share": cw[b], "collected_tokens": code_store.tokens[b],
                "collected_documents": code_store.documents(b),
                "allocated_tokens": round(code_final[b], 1),
                "tokens": tok_of[CODE_PREFIX + b],
                "documents": by_label.get(CODE_PREFIX + b, {"documents": 0})["documents"],
                "achieved_share": achieved_code[b],
                "exhausted": b in code_col.exhausted} for b in cw},
        "html_achieved_share_of_code": achieved_code.get("HTML", 0.0),
        **collection["code"],
        "leakage": {"reference_files": getattr(setup.guard, "reference_files", None),
                    "rule": "whitespace-collapsed exact match with a stepbuild benchmark "
                            "reference file (LeakageGuard.find_file)",
                    "dropped_train": code_stats.get("dropped_leakage", 0),
                    "dropped_code_val": code_val["dropped_leakage"]}
        if setup.guard is not None else None,
        "dropped_contamination": code_stats.get("dropped_contamination", 0),
        "dedup": "code_val: exact content (blake2b-8) against every collected train "
                 "code document; near-duplicates remain",
        "dropped_as_duplicate": code_val["dropped_as_duplicate"],
    }
    text_summary = {
        "weights": tw,
        "collection_order": [sm.ENGLISH] + order,
        "by_language": {
            x: {"target_share_of_text": tw[x], "achieved_share_of_text": achieved_text.get(x, 0.0),
                **collection["text"][x],
                "buckets": {b[len(TEXT_PREFIX):]: {"tokens": tok_of[b],
                                                   "documents": by_label.get(b, {"documents": 0})[
                                                       "documents"],
                                                   "allocated_tokens": round(final[b], 1)}
                            for b in final if b.startswith(TEXT_PREFIX) and text_lang(b) == x}}
            for x in tw},
    }
    lid_info = None
    if lid:
        lid_info = {
            **(provenance.get("lid") or {}),
            "threshold": setup.lid_threshold,
            "max_chars": getattr(setup.lid, "max_chars", sm.LID_MAX_CHARS),
            "rule": "a text document is dropped when the classifier's top label is not its "
                    "source language and that label's probability >= threshold (at 0.0, "
                    "every mismatch is dropped); a mismatch below the threshold is kept "
                    "and counted as kept_while_unsure. The first max_chars characters are "
                    "classified. cmn_Hani goes to zho_Hans or zho_Hant by label (the likelier "
                    "of the two when unsure), equal halves",
            "by_language": {x: {"kept": s["lid_kept"], "dropped": s["lid_dropped"],
                                "kept_while_unsure": s["lid_kept_unsure"],
                                "dropped_as": s.get("lid_dropped_as", {})}
                            for x, s in collection["text"].items()},
            "warnings": lid_warnings,
        }
    decontam_info = None
    if setup.decontam is not None:
        benches = setup.decontam.benchmarks
        lv_stats: Counter = Counter()
        for _, s in lv_cols.values():
            lv_stats.update(s)
        tt_stats: Counter = Counter()
        for _, s in text_cols.values():
            tt_stats.update(s)
        decontam_info = {
            "index": setup.decontam.summary(),
            "benchmarks": provenance.get("decontamination"),
            "applied_to": "every code document (train and code_val), the train text "
                          "(FineWeb-Edu, FineWeb-2) and the validation text (val, val_lang)",
            "dropped": {"train_code": _dropped_by_benchmark(code_stats, benches),
                        "train_text": _dropped_by_benchmark(tt_stats, benches),
                        "code_val": _dropped_by_benchmark(cv_docs.stats, benches),
                        "val": _dropped_by_benchmark(val_stats, benches),
                        "val_lang": _dropped_by_benchmark(lv_stats, benches)},
        }
    manifest: dict[str, Any] = {
        "format": "data v2 (scripts/build_shards.py build_mix)",
        "tokenizer": provenance.get("tokenizer"),
        "eot": eot,
        "dtype": "uint16 little-endian",
        "sources": provenance.get("sources", {}),
        "lid": lid_info,
        "decontamination": decontam_info,
        "splits": {"val": val, "train": train, "code_val": code_val, "val_lang": val_lang},
        # The trainer's data (val + train); code_val and val_lang are for evaluation.
        "total_tokens": val["tokens"] + train["tokens"],
        "mix": {"target": {"code": spec.code_share,
                           sm.ENGLISH: round(text_total * tw[sm.ENGLISH] / T, 6),
                           "other_languages": round(G / T, 6)},
                "achieved": {"code": _share(code_tokens, train["tokens"]),
                             sm.ENGLISH: _share(tok_of[TEXT_PREFIX + sm.ENGLISH],
                                                train["tokens"]),
                             "other_languages": _share(
                                 text_tokens - tok_of[TEXT_PREFIX + sm.ENGLISH],
                                 train["tokens"])}},
        "code": code_summary,
        "text": text_summary,
        "share_check": {"min_weight": SHARE_CHECK_MIN_WEIGHT, "max_off": SHARE_CHECK_MAX_OFF,
                        "violations": final_violations},
        "sampling": {"slack": spec.slack, "window": spec.window,
                     "stall_windows": spec.stall_windows, "stall_gain": spec.stall_gain,
                     "projection": None if spec.code_files is None else {
                         "code_files": spec.code_files,
                         "from_file": spec.projection_min_files}},
        "workers": spec.workers,
        "resume": resume,
        # What a rerun is compared with before it may say "already built".
        "fingerprint": fingerprint,
        "build_started": started,
        "build_finished": _now(),
        "build_seconds": round(time.monotonic() - t0, 1),
    }
    write_manifest(root / MANIFEST, manifest)
    if final_violations:  # keep _work: --from-work can re-allocate without downloading
        raise sm.ShareError("train was written but its mix is off target:\n  "
                            + "\n  ".join(final_violations))
    if spec.keep_work:
        state.data["finished"] = True  # a rerun is then a no-op, as without _work
        state.save()
    else:
        # state.json first: a crash while _work is being deleted then leaves a
        # finished build (manifest, no state), never a half-deleted resumable one.
        state.path.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def _violations(code_shares: dict[str, float], code_weights: dict[str, float],
                text_shares: dict[str, float], text_weights: dict[str, float]) -> list[str]:
    kw = {"min_weight": SHARE_CHECK_MIN_WEIGHT, "max_off": SHARE_CHECK_MAX_OFF}
    return (sm.share_violations(code_shares, code_weights, what="code", **kw)
            + sm.share_violations(text_shares, text_weights, what="text", **kw))


# ------------------------------------------------------------------ real sources

def code_file_name(index: int, total: int) -> str:
    return f"data/train-{index:05d}-of-{total:05d}.parquet"


def code_file_path(repo: str, revision: str, index: int, total: int) -> str:
    return f"datasets/{repo}@{revision}/{code_file_name(index, total)}"


def iter_code_row_groups(fs: Any, repo: str, revision: str, files: range, total: int,
                         columns: tuple[str, ...] = ("code", "language", "license", "path"),
                         retries: int = DOWNLOAD_ATTEMPTS,
                         transform: Callable[[Any, int], Any] | None = None) -> Iterator[Any]:
    """The rows of each parquet row group, in file order, one row group at a time.

    Every row also carries "file" (its parquet index). A failed read is retried
    from the same (file, row group), so a network blip changes nothing in the output.
    With `transform`, each row group's pyarrow table and file index are passed to it
    and whatever it returns is yielded instead.
    """
    import pyarrow.parquet as pq

    for index in files:
        path = code_file_path(repo, revision, index, total)
        rg, n_groups, attempt = 0, None, 0
        while n_groups is None or rg < n_groups:
            try:
                with fs.open(path, "rb", block_size=16 * 1024 * 1024) as f:
                    pf = pq.ParquetFile(f)
                    n_groups = pf.metadata.num_row_groups
                    while rg < n_groups:
                        table = pf.read_row_group(rg, columns=list(columns))
                        if transform is not None:
                            batch = transform(table, index)
                            del table
                            yield batch
                            rg += 1
                            attempt = 0
                            continue
                        cols = [table.column(c).to_pylist() for c in columns]
                        del table
                        batch = [dict(zip(columns, values), file=index)
                                 for values in zip(*cols)]
                        del cols
                        yield batch
                        rg += 1
                        attempt = 0
            except Exception as exc:
                attempt += 1
                if not backoff(exc, attempt, retries, f"read ({path}, row group {rg})"):
                    raise


def iter_text_files(fs: Any, paths: Iterable[str], attempts: int = DOWNLOAD_ATTEMPTS,
                    sleep: Callable[[float], None] = time.sleep) -> Iterator[dict]:
    """{"text", "file"} rows of each parquet file's "text" column, one row group at a
    time; a failed read is retried from the same (file, row group) with capped
    backoff, `attempts` in all, unless the error is permanent (is_permanent).

    These files are remote: bytes garbled or cut short in transit surface as a
    pyarrow.ArrowInvalid or another ValueError (UnicodeDecodeError, ...), which
    is_permanent would call permanent (right for a local file). Here they are raised
    as TransientError, so they get the normal bounded retries."""
    import pyarrow.parquet as pq

    def garbled(what: str, exc: BaseException) -> TransientError:
        return TransientError(f"{path} ({what}) read garbled: {exc!r}")

    for path in paths:
        rg, n_groups, attempt = 0, None, 0
        while n_groups is None or rg < n_groups:
            try:
                with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f:
                    try:  # pyarrow.ArrowInvalid is a ValueError
                        pf = pq.ParquetFile(f)
                        n_groups = pf.metadata.num_row_groups
                    except ValueError as exc:
                        raise garbled("footer", exc) from exc
                    while rg < n_groups:
                        try:
                            texts = pf.read_row_group(rg, columns=["text"]).column(
                                "text").to_pylist()
                        except ValueError as exc:
                            raise garbled(f"row group {rg}", exc) from exc
                        rg += 1
                        attempt = 0
                        for t in texts:
                            yield {"text": t, "file": path}
            except Exception as exc:
                attempt += 1
                if not backoff(exc, attempt, attempts, f"read ({path}, row group {rg})",
                               sleep):
                    raise


def fetch_ahead(indices: Iterable[int], fetch: Callable[[int], Any], ahead: int, *,
                attempts: int = DOWNLOAD_ATTEMPTS,
                sleep: Callable[[float], None] = time.sleep) -> Iterator[tuple[int, Path]]:
    """(index, local path) for each index, IN ORDER, downloading `ahead` files in
    parallel threads. Each file is deleted as soon as the consumer asks for the next
    one (or stops), so at most ahead + 1 files are on disk. A failed fetch is retried
    (attempts in all, waits 5, 10, 20, ... capped at 300 s) and then raises; a
    permanent error (is_permanent: a 404, a full disk, ...) raises at once. Closing
    the generator stops further retries and deletes what finishes later. The download
    threads are daemons, so a download still hanging when the build fails never keeps
    the process alive."""
    from collections import deque
    from concurrent.futures import Future

    if ahead < 1:
        raise ValueError("ahead must be at least 1")
    todo = list(indices)
    stop = threading.Event()

    def fetch_retry(i: int) -> Path:
        attempt = 0
        while True:
            try:
                return Path(fetch(i))
            except Exception as exc:
                attempt += 1
                if stop.is_set() or not backoff(exc, attempt, attempts,
                                                f"download (file {i})", sleep):
                    why = ("cannot succeed on retry" if is_permanent(exc)
                           else f"after {attempt} attempts")
                    raise RuntimeError(f"download of code file {i} failed {why}: "
                                       f"{exc!r}") from exc
            if stop.is_set():
                raise RuntimeError(f"download of code file {i} abandoned")

    def start(i: int) -> Future:
        fut: Future = Future()
        fut.set_running_or_notify_cancel()

        def run() -> None:
            try:
                fut.set_result(fetch_retry(i))
            except BaseException as exc:  # handed to the consumer
                fut.set_exception(exc)
        threading.Thread(target=run, daemon=True, name=f"download-{i}").start()
        return fut

    def discard(fut: Future) -> None:
        if fut.exception() is None:
            Path(fut.result()).unlink(missing_ok=True)

    # One daemon thread per file in flight: at most `ahead` at once, as a pool of
    # `ahead` threads would run them, but none can block interpreter exit.
    pending: deque = deque()
    pos = 0

    def submit() -> None:
        nonlocal pos
        if pos < len(todo):
            pending.append((todo[pos], start(todo[pos])))
            pos += 1

    try:
        for _ in range(ahead):
            submit()
        while pending:
            index, fut = pending.popleft()
            path = fut.result()
            submit()
            try:
                yield index, path
            finally:
                path.unlink(missing_ok=True)
    finally:
        stop.set()
        for _, fut in pending:
            fut.add_done_callback(discard)


def hf_code_fetcher(repo: str, revision: str, total: int, dl_dir: Path
                    ) -> Callable[[int], Path]:
    """Download one github-code-clean parquet file whole into dl_dir (never the HF
    cache) and check its footer, so a truncated download fails (and is retried)."""
    def fetch(index: int) -> Path:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, code_file_name(index, total), repo_type="dataset",
                                    revision=revision, local_dir=dl_dir))
        try:
            pq.read_metadata(path)
        except Exception as exc:
            # A corrupt local parquet is permanent by is_permanent's rule, but one
            # just downloaded is most likely truncated: download it again.
            path.unlink(missing_ok=True)
            raise TransientError(f"{path.name} failed its parquet check after download "
                                 f"(truncated?): {exc!r}") from exc
        return path
    return fetch


def lower_priority(kernel32: Any = None) -> bool:
    """Drop this process to BELOW_NORMAL priority on Windows; a no-op elsewhere."""
    if kernel32 is None:
        if sys.platform != "win32":
            return False
        import ctypes
        from ctypes import wintypes
        # A private handle to kernel32 with explicit types: without restype,
        # GetCurrentProcess's pseudo-handle (-1) is truncated to a 32-bit int and
        # SetPriorityClass fails on 64-bit Python.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                          BELOW_NORMAL_PRIORITY_CLASS))


LID_FILE = "lid-specialist-fasttext.ftz"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(items: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def mix_code_rows(fetch: Callable[[int], Any], files: range, languages: Iterable[str],
                  licenses: Iterable[str], *, ahead: int = DOWNLOAD_WORKERS,
                  attempts: int = DOWNLOAD_ATTEMPTS,
                  sleep: Callable[[float], None] = time.sleep
                  ) -> Callable[..., Iterator[dict]]:
    """A MixSources code source: rows(start_file) yields the rows of files
    max(start_file, files.start) .. files.stop - 1 in order. Files are downloaded
    whole, `ahead` at a time (fetch_ahead), read one row group at a time and deleted
    once read. Language and licence are filtered in pyarrow before any row becomes a
    Python object (the rest of the filters are code_offers'); what a row group drops
    is attached to its next kept row (row[FILTERED]), and each file ends with a marker
    row {FILE_END: index, FILTERED: what its last row groups dropped}."""
    columns = ("code", "language", "license", "path")
    langs, lics = sorted(languages), sorted(licenses)

    def transform(table: Any, index: int) -> tuple[list[dict], dict[str, int]]:
        import pyarrow as pa
        import pyarrow.compute as pc

        lang_ok = pc.fill_null(pc.is_in(table.column("language"), value_set=pa.array(langs)),
                               False)
        lic_ok = pc.fill_null(pc.is_in(table.column("license"), value_set=pa.array(lics)),
                              False)
        keep = pc.and_(lang_ok, lic_ok)
        n = table.num_rows
        n_lang = pc.sum(lang_ok).as_py() or 0
        n_keep = pc.sum(keep).as_py() or 0
        kept = table.filter(keep)
        cols = [kept.column(c).to_pylist() for c in columns]
        rows = [dict(zip(columns, values), file=index) for values in zip(*cols)]
        return rows, {"rows_scanned": n - n_keep, "dropped_language": n - n_lang,
                      "dropped_license": n_lang - n_keep}

    def groups(start: int) -> Iterator[tuple[list[dict], dict[str, int], int | None]]:
        import pyarrow.parquet as pq

        dl = fetch_ahead(range(max(start, files.start), files.stop), fetch, ahead,
                         attempts=attempts, sleep=sleep)
        try:
            for index, path in dl:
                with open(path, "rb") as f:
                    pf = pq.ParquetFile(f)
                    for rg in range(pf.metadata.num_row_groups):
                        yield (*transform(pf.read_row_group(rg, columns=list(columns)), index),
                               None)
                yield [], {}, index
        finally:
            dl.close()

    def rows(start: int = 0) -> Iterator[dict]:
        carry: Counter = Counter()
        it = prefetch(groups(start), depth=4)
        try:
            for batch, counts, file_end in it:
                carry.update(counts)
                if batch:
                    batch[0][FILTERED] = dict(carry)
                    carry = Counter()
                    yield from batch
                if file_end is not None:
                    yield {FILE_END: file_end, FILTERED: dict(carry)}
                    carry = Counter()
        finally:
            it.close()

    return rows


def _fw2_listing(fs: Any, dataset: str, revision: str, lang: str,
                 split: str) -> tuple[list[str], int]:
    listed = fs.ls(f"datasets/{dataset}@{revision}/data/{lang}/{split}", detail=True)
    files = sorted((p["name"], p.get("size") or 0) for p in listed
                   if p["name"].endswith(".parquet"))
    if not files:
        raise RuntimeError(f"{dataset} {lang}/{split} has no parquet files")
    return [name for name, _ in files], sum(size for _, size in files)


# ------------------------------------------------------------------ preflight

def code_file_counts(table: Any, languages: Iterable[str], licenses: Iterable[str],
                     max_bytes: float) -> dict[str, float]:
    """Bytes of source per language in one code file's {language, license, size,
    path} columns that pass the build's cheap filters (language, licence, path) and
    the size cap (max_bytes, about max_doc_tokens tokens)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    keep = pc.and_(
        pc.and_(pc.fill_null(pc.is_in(table.column("language"),
                                      value_set=pa.array(sorted(languages))), False),
                pc.fill_null(pc.is_in(table.column("license"),
                                      value_set=pa.array(sorted(licenses))), False)),
        pc.fill_null(pc.less_equal(table.column("size"), max_bytes), False))
    kept = table.filter(keep)
    out: Counter = Counter()
    for lang, size, path in zip(kept.column("language").to_pylist(),
                                kept.column("size").to_pylist(),
                                kept.column("path").to_pylist()):
        if path_skip_reason(path) is None:
            out[lang] += size
    return {k: float(v) for k, v in out.items()}


def read_code_file_counts(fs: Any, path: str, languages: Iterable[str],
                          licenses: Iterable[str], max_bytes: float, *,
                          attempts: int = DOWNLOAD_ATTEMPTS,
                          sleep: Callable[[float], None] = time.sleep) -> dict[str, float]:
    """code_file_counts of one code file, read (metadata columns only) from `fs`;
    retried like every other read (backoff), a permanent error raised at once."""
    import pyarrow.parquet as pq

    attempt = 0
    while True:
        try:
            with fs.open(path, "rb", block_size=1 << 20) as f:
                table = pq.ParquetFile(f).read(columns=["language", "license", "size", "path"])
            return code_file_counts(table, languages, licenses, max_bytes)
        except Exception as exc:
            attempt += 1
            if not backoff(exc, attempt, attempts, f"preflight read ({path})", sleep):
                raise


def preflight(read_counts: Callable[[int], dict[str, float]], files: Iterable[int],
              weights: dict[str, float], code_tokens: float, *, caps: dict[str, float],
              chars_per_token: float = sm.DEFAULT_CHARS_PER_TOKEN,
              file_bytes: list[int] | None = None, workers: int = 16, header: str = "",
              report_path: Path | None = None) -> int:
    """Read every file's counts (in parallel, results in file order), print the
    report (shard_mix.format_preflight) and return the exit status: 0 when the files
    can give the code mix, 3 when they cannot."""
    from concurrent.futures import ThreadPoolExecutor

    files = list(files)
    with ThreadPoolExecutor(workers) as pool:
        per_file = list(tqdm(pool.map(read_counts, files), total=len(files), desc="preflight",
                             unit="file", mininterval=5.0))
    report = sm.preflight_report(per_file, weights, code_tokens, caps=caps,
                                 chars_per_token=chars_per_token, file_bytes=file_bytes,
                                 min_weight=SHARE_CHECK_MIN_WEIGHT,
                                 max_off=SHARE_CHECK_MAX_OFF)
    if header:
        print(header)
    print(sm.format_preflight(report), flush=True)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        write_manifest(Path(report_path), {"header": header, **report})
        print(f"(report: {report_path})")
    return 0 if report["ok"] else 3


def _pinned(pinned: str, resolve: Callable[[], str], what: str) -> str:
    if pinned:
        return pinned
    rev = resolve()
    print(f"WARNING: data.{what} is not pinned in the config; using the Hub's current "
          f"commit {rev}", file=sys.stderr, flush=True)
    return rev


def _code_train_files(d: Any, args: argparse.Namespace) -> int:
    return min(args.max_code_files or d.code_heldout_first_file, d.code_heldout_first_file)


def run_preflight(cfg: Any, args: argparse.Namespace) -> int:
    """--preflight: read only the metadata columns of the code train files and report
    whether they can fill the code mix at the configured token target."""
    from huggingface_hub import HfApi, HfFileSystem

    d = cfg.data
    api, fs = HfApi(), HfFileSystem()
    rev = _pinned(d.code_revision, lambda: api.dataset_info(d.code_dataset).sha,
                  "code_revision")
    listed = {p["name"].rsplit("/", 1)[-1]: p.get("size") or 0
              for p in fs.ls(f"datasets/{d.code_dataset}@{rev}/data", detail=True)
              if p["name"].endswith(".parquet")}
    if len(listed) != d.code_files_total:
        raise RuntimeError(f"{d.code_dataset}@{rev} has {len(listed)} parquet files; "
                           f"config says code_files_total = {d.code_files_total}")
    n = _code_train_files(d, args)
    file_bytes = [listed[code_file_name(i, d.code_files_total).rsplit("/", 1)[-1]]
                  for i in range(n)]
    T = int(args.train_tokens or cfg.train.total_tokens)
    C = d.code_share * T
    cpt = args.chars_per_token
    max_bytes = d.code_max_doc_tokens * cpt
    langs, lics = tuple(d.code_language_weights), d.code_licenses

    def read(index: int) -> dict[str, float]:
        return read_code_file_counts(
            fs, code_file_path(d.code_dataset, rev, index, d.code_files_total), langs, lics,
            max_bytes)

    header = (f"preflight: {d.code_dataset}@{rev[:10]}, code train files 0..{n - 1} "
              f"({sum(file_bytes) / 1e9:,.0f} GB in all), code target {C:,.0f} tokens "
              f"({d.code_share:.0%} of {T:,}), {cpt} chars/token; filters: languages, "
              f"licences, vendored/minified paths, size <= {max_bytes:,.0f} bytes. Estimates "
              "only: blank, leaked and too-long-in-tokens files are not seen here.")
    return preflight(read, range(n), dict(d.code_language_weights), C,
                     caps={"HTML": d.html_cap}, chars_per_token=cpt, file_bytes=file_bytes,
                     workers=args.preflight_workers, header=header,
                     report_path=Path(d.shard_dir) / PREFLIGHT_REPORT)


def check_vocab(tokenizer_vocab: int, model_vocab: int, name: str) -> None:
    """The shards' ids must fit the model's embedding and uint16."""
    if tokenizer_vocab != model_vocab:
        raise SystemExit(f"tokenizer {name} has a vocabulary of {tokenizer_vocab:,} but "
                         f"model.vocab_size is {model_vocab:,}: shards built with it would not "
                         "fit the model. Fix data.tokenizer or model.vocab_size.")
    if tokenizer_vocab > 65536:
        raise SystemExit(f"vocab {tokenizer_vocab} does not fit uint16 shards")


def run_mix(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """The data v2 build from the real sources (Hugging Face), per the config and flags.

    Everything that can fail without the network is checked first, before any
    dataset is touched: the tokenizer (and its vocabulary against the model), the
    LID model (downloaded and loaded, so a broken fastText install fails in seconds).
    """
    import functools

    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    from quipu.tokenizer import make_tokenizer

    d = cfg.data
    tok_factory = functools.partial(make_tokenizer, d.tokenizer)
    tok = tok_factory()
    check_vocab(tok.vocab_size, cfg.model.vocab_size, d.tokenizer)
    tokenizer = {"name": d.tokenizer, "vocab_size": tok.vocab_size, "eot": tok.eot,
                 "path": None if d.tokenizer == "gpt2" else d.tokenizer,
                 "sha256": None if d.tokenizer == "gpt2" else _sha256_file(d.tokenizer)}

    api = HfApi()
    tw = dict(d.text_language_weights) or {sm.ENGLISH: 1.0}
    lid = lid_info = None
    if args.lid_filter:
        lid_rev = d.lid_revision or api.model_info(d.lid_model).sha
        path = hf_hub_download(d.lid_model, LID_FILE, revision=lid_rev)
        lid = sm.FastTextLid(path)
        labels = lid.labels()  # loads the model here, so a missing fasttext fails early
        missing = sorted(sm.lid_labels_needed(tw) - set(labels))
        if missing:
            raise SystemExit(f"{d.lid_model}@{lid_rev} has no label for {missing}; "
                             f"its labels: {labels}")
        lid_info = {"model": d.lid_model, "revision": lid_rev, "file": LID_FILE,
                    "sha256": _sha256_file(path), "labels": labels}

    import train_tokenizer as tt
    from stepbuild.bench.run import LeakageGuard

    if (d.dataset, d.subset) != (tt.TEXT_DATASET, "sample-10BT"):
        raise SystemExit(f"data v2 reads English from {tt.TEXT_DATASET} sample-10BT; "
                         f"the config says {d.dataset} {d.subset}")
    others = [x for x in tw if x != sm.ENGLISH]
    guard = LeakageGuard()
    # HumanEval and MBPP (~0.2 MB), pinned: no code document or validation split may
    # contain one of their problems (quipu/decontam.py).
    from quipu import decontam as dc

    bench_revs = {
        dc.HUMANEVAL: _pinned(d.humaneval_revision,
                              lambda: api.dataset_info(dc.HUMANEVAL_DATASET).sha,
                              "humaneval_revision"),
        dc.MBPP: _pinned(d.mbpp_revision, lambda: api.dataset_info(dc.MBPP_DATASET).sha,
                         "mbpp_revision")}
    decontam, bench_info = dc.load_benchmarks(bench_revs, hf_hub_download)
    print(f"decontamination: {decontam.summary()['problems']} problems", flush=True)
    setup = sm.DocSetup(tokenizer=tok_factory, max_doc_tokens=d.code_max_doc_tokens,
                        guard=guard, lid=lid, lid_threshold=args.lid_threshold,
                        decontam=decontam)

    # Every dataset pinned to one commit (the config's), so every read agrees.
    fs = HfFileSystem()
    code_rev = _pinned(d.code_revision, lambda: api.dataset_info(d.code_dataset).sha,
                       "code_revision")
    listed = [p for p in fs.ls(f"datasets/{d.code_dataset}@{code_rev}/data", detail=False)
              if p.endswith(".parquet")]
    if len(listed) != d.code_files_total:
        raise RuntimeError(f"{d.code_dataset}@{code_rev} has {len(listed)} parquet files; "
                           f"config says code_files_total = {d.code_files_total}")
    max_files = _code_train_files(d, args)
    langs, lics = tuple(d.code_language_weights), d.code_licenses
    root = Path(d.shard_dir)
    dl = root / MIX_WORK / "dl"
    fetch = hf_code_fetcher(d.code_dataset, code_rev, d.code_files_total, dl)
    code_train = mix_code_rows(fetch, range(0, max_files), langs, lics,
                               ahead=args.download_workers)
    code_val = mix_code_rows(fetch, range(d.code_heldout_first_file, d.code_files_total),
                             langs, lics, ahead=2)

    def text_source(paths: list[str]) -> Callable[..., Iterator[dict]]:
        # start: files of this language already read (a resumed build skips them).
        return lambda start=0: flatten(prefetch(batched(iter_text_files(fs, paths[start:]),
                                                        1000), depth=8))

    text_rev = _pinned(d.text_revision, lambda: api.dataset_info(tt.TEXT_DATASET).sha,
                       "text_revision")
    names = tt.text_files(fs, text_rev)
    full = [f"datasets/{tt.TEXT_DATASET}@{text_rev}/{n}" for n in names]
    text_train = {sm.ENGLISH: text_source(full[:-1])}
    text_val = {sm.ENGLISH: text_source(full[-1:])}
    fw2_rev = (_pinned(d.fineweb2_revision,
                       lambda: api.dataset_info(tt.FINEWEB2_DATASET).sha, "fineweb2_revision")
               if others else None)
    fw2: dict[str, Any] = {}
    for x in others:
        train_files, train_bytes = _fw2_listing(fs, tt.FINEWEB2_DATASET, fw2_rev, x, "train")
        test_files, _ = _fw2_listing(fs, tt.FINEWEB2_DATASET, fw2_rev, x, "test")
        text_train[x] = text_source(train_files)
        text_val[x] = text_source(test_files)
        fw2[x] = {"train_files": len(train_files), "train_bytes": train_bytes,
                  "train_files_sha256": _sha256_text(train_files),
                  "test_files": [f.split("/data/", 1)[-1] for f in test_files]}
    # Smallest language first: one that runs dry is then known before the larger
    # ones are collected, and they are given its remainder.
    order = tuple(sorted(others, key=lambda x: (fw2[x]["train_bytes"], x)))

    spec = MixSpec(
        train_tokens=int(args.train_tokens or cfg.train.total_tokens),
        val_tokens=d.val_tokens, shard_tokens=d.shard_tokens, code_share=d.code_share,
        code_weights=dict(d.code_language_weights), text_weights=tw, licenses=lics,
        html_cap=d.html_cap, max_doc_tokens=d.code_max_doc_tokens,
        code_val_tokens=d.code_val_tokens, lang_val_tokens=args.lang_val_tokens,
        slack=args.slack, window=args.window, stall_windows=args.stall_windows,
        stall_gain=args.stall_gain, workers=args.workers, batch_docs=args.batch_docs,
        text_order=order, keep_work=args.keep_work, code_files=max_files,
        projection_min_files=args.projection_min_files)
    provenance = {
        "tokenizer": tokenizer,
        "lid": lid_info,
        "decontamination": bench_info,
        "sources": {
            "code": {"dataset": d.code_dataset, "revision": code_rev,
                     "files_total": d.code_files_total, "train_files": [0, max_files - 1],
                     "heldout_files": [d.code_heldout_first_file, d.code_files_total - 1]},
            "english": {"dataset": tt.TEXT_DATASET, "revision": text_rev,
                        "train_files": len(names) - 1, "heldout_files": names[-1:],
                        "files_sha256": _sha256_text(names)},
            "fineweb2": {"dataset": tt.FINEWEB2_DATASET, "revision": fw2_rev,
                         "collection_order": list(order), "by_language": fw2},
        },
    }
    return build_mix(root, spec, MixSources(code_train, code_val, text_train, text_val),
                     setup, provenance, fresh=args.fresh, from_work=args.from_work)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--low-priority", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run at BELOW_NORMAL process priority (Windows; default on)")
    v2 = parser.add_argument_group("data v2 (configs with code_language_weights)")
    v2.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)),
                    help="tokenising processes (default: all cores but one, at most 8; the "
                         "downloads, not the CPU, set the pace)")
    v2.add_argument("--download-workers", type=int, default=DOWNLOAD_WORKERS,
                    help="code parquet files downloaded ahead in parallel (whole files into "
                         "<shard_dir>/_work/dl, each deleted once read; at most N+1 on disk)")
    v2.add_argument("--lid-filter", action=argparse.BooleanOptionalAction, default=False,
                    help="filter text with the config's lid_model (needs fasttext: Linux)")
    v2.add_argument("--lid-threshold", type=float, default=sm.DEFAULT_LID_THRESHOLD,
                    help="drop a text document when another language is the top label with "
                         "at least this probability (default 0.0: every mismatch is dropped; "
                         "t > 0 keeps mismatches the classifier is less sure of, counted as "
                         "kept_while_unsure)")
    v2.add_argument("--train-tokens", type=float, default=None,
                    help="train split size (default: train.total_tokens)")
    v2.add_argument("--max-code-files", type=int, default=None,
                    help="hard cap on code train files (default and maximum: "
                         "code_heldout_first_file)")
    v2.add_argument("--lang-val-tokens", type=int, default=1_000_000,
                    help="tokens per val_lang/<bucket> split")
    v2.add_argument("--slack", type=float, default=sm.DEFAULT_SLACK)
    v2.add_argument("--window", type=int, default=sm.DEFAULT_WINDOW)
    v2.add_argument("--stall-windows", type=int, default=sm.DEFAULT_STALL_WINDOWS)
    v2.add_argument("--stall-gain", type=float, default=sm.DEFAULT_STALL_GAIN)
    v2.add_argument("--batch-docs", type=int, default=256)
    v2.add_argument("--projection-min-files", type=int, default=PROJECTION_MIN_FILES,
                    help="project the code mix at every code file from this one on (not in "
                         "the resume fingerprint: it never changes what is collected, so a "
                         "build stopped by the projection resumes with a larger value)")
    v2.add_argument("--keep-work", action="store_true",
                    help="keep <shard_dir>/_work (the collected buckets) after the build")
    v2.add_argument("--fresh", action="store_true",
                    help="discard a saved build in <shard_dir>/_work and start over")
    v2.add_argument("--from-work", action="store_true",
                    help="re-allocate and write from the buckets a saved build collected "
                         "(after a share error; the weights / --train-tokens may differ); "
                         "nothing collected is downloaded again")
    v2.add_argument("--preflight", action="store_true",
                    help="only read the code files' language/licence/size/path columns and "
                         "report whether the build can meet the code mix (exit 3 if not)")
    v2.add_argument("--chars-per-token", type=float, default=sm.DEFAULT_CHARS_PER_TOKEN,
                    help="bytes of code per token, for the preflight's estimates")
    v2.add_argument("--preflight-workers", type=int, default=32,
                    help="files whose metadata the preflight reads at once")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.low_priority:
        print(f"below-normal priority set: {lower_priority()}", flush=True)

    cfg = load_config(args.config)
    if uses_mix(cfg.data):
        if args.preflight:
            raise SystemExit(run_preflight(cfg, args))
        try:
            run_mix(cfg, args)
        except sm.ShareError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(2) from None
        except sm.ResumeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(4) from None
        except sm.WorkerDied as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(5) from None
        return

    # Heavy imports; the tests never need them.
    from datasets import load_dataset
    from huggingface_hub import HfApi, HfFileSystem

    d = cfg.data
    stream = load_dataset(d.dataset, name=d.subset, split="train", streaming=True)
    revision = dataset_revision(stream)
    # Only "text" is used; skipping the other nine columns lightens the stream.
    stream = stream.select_columns(["text"])
    text_rows = flatten(prefetch(batched(stream, 1000), depth=16))

    # Pin the code dataset to one commit so every file read agrees.
    code_rev = HfApi().dataset_info(d.code_dataset).sha
    fs = HfFileSystem()
    listed = [p for p in fs.ls(f"datasets/{d.code_dataset}@{code_rev}/data", detail=False)
              if p.endswith(".parquet")]
    if len(listed) != d.code_files_total:
        raise RuntimeError(f"{d.code_dataset}@{code_rev} has {len(listed)} parquet files; "
                           f"config says code_files_total = {d.code_files_total}")

    def rows(files: range) -> Iterator[dict]:
        return flatten(prefetch(iter_code_row_groups(fs, d.code_dataset, code_rev, files,
                                                     d.code_files_total), depth=3))

    code = CodeSpec(
        share=d.code_share, languages=d.code_languages, licenses=d.code_licenses,
        html_cap=d.html_cap, max_doc_tokens=d.code_max_doc_tokens,
        val_tokens=d.code_val_tokens,
        train_rows=rows(range(0, d.code_heldout_first_file)),
        val_rows=lambda: rows(range(d.code_heldout_first_file, d.code_files_total)),
        dataset=d.code_dataset, revision=code_rev,
        heldout_first_file=d.code_heldout_first_file, files_total=d.code_files_total)

    build_all(Path(d.shard_dir), text_rows,
              val_tokens=d.val_tokens, train_tokens=cfg.train.total_tokens,
              shard_tokens=d.shard_tokens, tok=Tokenizer(), dataset=d.dataset,
              subset=d.subset, revision=revision, code=code)


if __name__ == "__main__":
    main()
