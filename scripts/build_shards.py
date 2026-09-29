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
quipu/shard_mix.py.

Run: uv run python scripts/build_shards.py --config configs/quipu-moe.toml \
         --workers 47 --lid-filter --train-tokens 8.3e9
"""
from __future__ import annotations

import argparse
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
                 record_hashes: bool = False, leak_guard: Any = None) -> None:
        self._rows = rows
        # Data v2: a stepbuild LeakageGuard; a row copying a benchmark reference file
        # is dropped (stats "dropped_leakage", not among CODE_STAT_KEYS).
        self._leak_guard = leak_guard
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
    overlap with tokenizing. An exception in the producer is re-raised here.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def produce() -> None:
        try:
            for item in items:
                q.put((True, item))
        except BaseException as exc:  # handed to the consumer
            q.put((False, exc))
            return
        q.put((True, done))

    threading.Thread(target=produce, daemon=True, name="prefetch").start()
    while True:
        ok, item = q.get()
        if not ok:
            raise item
        if item is done:
            return
        yield item


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def flatten(batches: Iterable[list[Any]]) -> Iterator[Any]:
    for batch in batches:
        yield from batch


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
# Held out, as for the tokenizer and its gate: val = the last sample-10BT file
# (English, the trainer's loss); val_lang/<bucket> = FineWeb-2's test split; code_val
# = github-code-clean files code_heldout_first_file.. (exact-dedup against train code).

TEXT_PREFIX = "text:"
MIX_WORK = "_work"
COLLECT_REPORT = "collect_report.json"
SHARE_CHECK_MIN_WEIGHT = 0.05
SHARE_CHECK_MAX_OFF = 0.02


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


class MixSources(NamedTuple):
    """Row sources for build_mix: zero-argument callables returning rows.

    code_train / code_val: rows {"code", "language", "license", "path"[, "file"]}; a
    real source that filters rows before they become dicts attaches what it dropped
    to the next row it yields, as row[FILTERED] = {"rows_scanned": n, ...}.
    text_train / text_val: language -> rows {"text"[, "file"]}. Every *_val source
    must yield held-out rows only, never rows its train source can reach.
    """
    code_train: Callable[[], Iterable[dict]]
    code_val: Callable[[], Iterable[dict]]
    text_train: dict[str, Callable[[], Iterable[dict]]]
    text_val: dict[str, Callable[[], Iterable[dict]]]


FILTERED = "_filtered"


def uses_mix(data_cfg: Any) -> bool:
    """True when a DataConfig asks for the data v2 build (it has code weights)."""
    return bool(getattr(data_cfg, "code_language_weights", None))


def code_offers(rows: Iterable[dict], languages: Iterable[str],
                licenses: Iterable[str]) -> Iterator[sm.Offer]:
    """The cheap code filters, in the main process: language, licence, blank, path.

    The rows read and filtered on the way to each offer travel with it (meta) and
    are counted when the Runner decides it, not when they are read: the Runner reads
    ahead by a worker-dependent amount, and counting at read time would make the
    manifest depend on the number of workers.
    """
    languages, licenses = frozenset(languages), frozenset(licenses)
    counts: Counter = Counter()
    for row in rows:
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
        if "file" in row:
            meta["last_file"] = row["file"]
        yield sm.Offer(sm.CODE, language, code, meta)
        counts = Counter()


def text_offers(rows: Iterable[dict], source: str) -> Iterator[sm.Offer]:
    """Non-blank texts; filter counts travel with the offers as in code_offers."""
    counts: Counter = Counter()
    for row in rows:
        counts["rows_scanned"] += 1
        text = row.get("text") or ""
        if not text.strip():
            counts["skipped_blank"] += 1
            continue
        yield sm.Offer(sm.TEXT, source, text, {"counts": counts})
        counts = Counter()


def _counted(rows: Iterable[dict], counts: Counter) -> Iterator[dict]:
    """Rows as they are consumed, with any attached filter counts added to counts."""
    for row in rows:
        counts.update(row.pop(FILTERED, None) or {})
        yield row


def text_buckets(language: str, lid: bool) -> dict[str, float]:
    """A text source's buckets. With the LID filter, cmn_Hani becomes zh-Hans and
    zh-Hant in EQUAL halves (the specialist labels them zho_Hans / zho_Hant); if one
    script stalls, its remainder goes to the other. Without LID, cmn_Hani is one."""
    if lid and language == sm.ZH_SOURCE:
        return {b: 1 / len(sm.ZH_BUCKETS) for b in sm.ZH_BUCKETS}
    return {language: 1.0}


def _collect_text(runner: sm.Runner, rows: Iterable[dict], language: str, total: float,
                  store: sm.BucketStore, spec: MixSpec, *, lid: bool,
                  slack: float) -> tuple[sm.Collector, Counter]:
    stats: Counter = Counter()
    col = sm.Collector(text_buckets(language, lid), total, slack=slack, window=spec.window,
                       stall_windows=spec.stall_windows, stall_gain=spec.stall_gain,
                       store=store)
    runner.collect(text_offers(rows, language), col, stats)
    return col, stats


def _plain(stats: Counter) -> dict[str, Any]:
    """A stats Counter as manifest JSON: "lid_dropped_as:<label>" keys nested."""
    out: dict[str, Any] = {}
    for k in sorted(stats):
        if k == "last_file":
            continue
        if k.startswith("lid_dropped_as:"):
            out.setdefault("lid_dropped_as", {})[k.split(":", 1)[1]] = stats[k]
        else:
            out[k] = stats[k]
    return out


def _text_summary(col: sm.Collector, stats: Counter) -> dict[str, Any]:
    s = _plain(stats)
    offers = stats["offers"]
    return {**s, "lid_kept": offers - stats["lid_dropped"],
            "lid_dropped": stats["lid_dropped"],
            "tokens": sum(col.taken.values()), "collection": col.summary()}


def build_mix(root: Path, spec: MixSpec, sources: MixSources, setup: sm.DocSetup,
              provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """val, train, code_val and val_lang for the data v2 mix; manifest written last.

    Raises ShareError (after writing collect_report.json, before writing train) when
    the collected data cannot give the configured mix within the tolerance, and
    RuntimeError when a source ran short of what the mix needs.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).unlink(missing_ok=True)
    (root / COLLECT_REPORT).unlink(missing_ok=True)
    provenance = provenance or {}
    started, t0 = _now(), time.monotonic()
    work = root / MIX_WORK
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
    kw = {"window": spec.window, "stall_windows": spec.stall_windows,
          "stall_gain": spec.stall_gain}

    with sm.Runner(setup, spec.workers, batch_docs=spec.batch_docs) as runner:
        eot = runner.tok.eot
        # 1. The trainer's validation: English, from the held-out file.
        val_store = sm.BucketStore(work / "val")
        val_col, val_stats = _collect_text(runner, sources.text_val[sm.ENGLISH](), sm.ENGLISH,
                                           spec.val_tokens, val_store, spec, lid=lid, slack=0.0)
        val_store.close()
        val = write_split(root / "val", spec.val_tokens, spec.shard_tokens,
                          ((a, TEXT_PREFIX + sm.ENGLISH) for a in val_store.docs(sm.ENGLISH)))
        val["source"] = _text_summary(val_col, val_stats)
        print(f"val:   {val['tokens']:,} tokens", flush=True)

        # 2. Code: one pass over the mixed stream, per-language quotas.
        code_store = sm.BucketStore(work / "code")
        code_col = sm.Collector(cw, C, caps=caps, slack=spec.slack, store=code_store, **kw)
        code_stats: Counter = Counter()
        hashes: set[int] = set()
        runner.collect(code_offers(sources.code_train(), cw, spec.licenses),
                       code_col, code_stats,
                       on_admit=lambda offer, res: hashes.add(sm.content_hash(offer.text)))
        code_store.close()
        print(f"code collected: {sum(code_col.taken.values()):,} tokens, "
              f"{code_col.windows} windows, exhausted {sorted(code_col.exhausted)}, "
              f"last file {code_stats.get('last_file')}", flush=True)

        # 3. English, then the other languages one at a time. Each language's target is
        # set when its turn comes, from what the earlier ones actually held.
        text_store = sm.BucketStore(work / "text")
        text_cols: dict[str, tuple[sm.Collector, Counter]] = {}
        text_cols[sm.ENGLISH] = _collect_text(runner, sources.text_train[sm.ENGLISH](),
                                              sm.ENGLISH, E, text_store, spec, lid=lid,
                                              slack=0.0)
        for lang in order:
            held = {x: float(sum(text_cols[x][0].taken.values())) for x in order
                    if x in text_cols}
            now, _ = sm.allocate(G, others, held)
            text_cols[lang] = _collect_text(runner, sources.text_train[lang](), lang,
                                            now[lang], text_store, spec, lid=lid,
                                            slack=spec.slack)
            print(f"{lang}: {sum(text_cols[lang][0].taken.values()):,} tokens "
                  f"(target {now[lang]:,.0f})", flush=True)
        text_store.close()

        # 4. Final shares. The HTML cap is re-applied inside allocate after every
        # redistribution of a short language's remainder.
        code_have = {b: float(code_store.tokens[b]) for b in cw}
        code_final, code_short = sm.allocate(C, cw, code_have, caps)
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
        planned_code = {b: v / C for b, v in code_final.items()}
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
                    + f"\n(collected per language: {root / COLLECT_REPORT}; a common "
                      "language short at the file cap needs more files (--max-code-files) "
                      "or a lower weight)")
            raise RuntimeError("sources ran short: " + "; ".join(short)
                               + f" (see {root / COLLECT_REPORT})")

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
        cv_docs = CodeDocs(_counted(sources.code_val(), cv_read), runner.tok,
                           languages=tuple(cw),
                           licenses=spec.licenses, html_cap=spec.html_cap,
                           max_doc_tokens=spec.max_doc_tokens, exclude=hashes,
                           leak_guard=setup.guard)
        cv = write_split(root / "code_val", spec.code_val_tokens, spec.shard_tokens, cv_docs,
                         allow_short=True)
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
                                       slack=0.0)
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
    lid_info = provenance.get("lid")
    if lid_info is not None:
        lid_info = {**lid_info, "threshold": setup.lid_threshold,
                    "rule": "drop when the top label is not the source language and its "
                            "probability >= threshold; cmn_Hani goes to zho_Hans or zho_Hant "
                            "by label (the likelier of the two when unsure), equal halves"}
    manifest: dict[str, Any] = {
        "format": "data v2 (scripts/build_shards.py build_mix)",
        "tokenizer": provenance.get("tokenizer"),
        "eot": eot,
        "dtype": "uint16 little-endian",
        "sources": provenance.get("sources", {}),
        "lid": lid_info,
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
                     "stall_windows": spec.stall_windows, "stall_gain": spec.stall_gain},
        "workers": spec.workers,
        "build_started": started,
        "build_finished": _now(),
        "build_seconds": round(time.monotonic() - t0, 1),
    }
    write_manifest(root / MANIFEST, manifest)
    if not spec.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    if final_violations:
        raise sm.ShareError("train was written but its mix is off target:\n  "
                            + "\n  ".join(final_violations))
    return manifest


def _violations(code_shares: dict[str, float], code_weights: dict[str, float],
                text_shares: dict[str, float], text_weights: dict[str, float]) -> list[str]:
    kw = {"min_weight": SHARE_CHECK_MIN_WEIGHT, "max_off": SHARE_CHECK_MAX_OFF}
    return (sm.share_violations(code_shares, code_weights, what="code", **kw)
            + sm.share_violations(text_shares, text_weights, what="text", **kw))


# ------------------------------------------------------------------ real sources

def code_file_path(repo: str, revision: str, index: int, total: int) -> str:
    return f"datasets/{repo}@{revision}/data/train-{index:05d}-of-{total:05d}.parquet"


def iter_code_row_groups(fs: Any, repo: str, revision: str, files: range, total: int,
                         columns: tuple[str, ...] = ("code", "language", "license", "path"),
                         retries: int = 5,
                         transform: Callable[[Any, int], Any] | None = None) -> Iterator[Any]:
    """The rows of each parquet row group, in file order, one row group at a time.

    Every row also carries "file" (its parquet index). A failed read is retried
    from the same (file, row group), so a network blip changes nothing in the output.
    With `transform`, each row group's pyarrow table and file index are passed to it
    and whatever it returns is yielded instead (the data v2 build filters there).
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
                if attempt > retries:
                    raise
                wait = 5 * 2 ** (attempt - 1)
                print(f"\nread failed ({path}, row group {rg}): {exc!r}; "
                      f"retry {attempt}/{retries} in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)


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


def mix_code_rows(fs: Any, repo: str, revision: str, files: range, total: int,
                  languages: Iterable[str], licenses: Iterable[str]
                  ) -> Callable[[], Iterator[dict]]:
    """A MixSources code source: the files' rows, language and licence filtered in
    pyarrow before any row becomes a Python object (the rest of the filters are
    code_offers'). What a row group drops is attached to its next kept row
    (row[FILTERED]), so it is counted when that row is consumed."""
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

    def rows() -> Iterator[dict]:
        groups = iter_code_row_groups(fs, repo, revision, files, total, columns=columns,
                                      transform=transform)
        carry: Counter = Counter()
        for batch, counts in prefetch(groups, depth=4):
            carry.update(counts)
            if batch:
                batch[0][FILTERED] = dict(carry)
                carry = Counter()
                yield from batch

    return rows


def _fw2_listing(fs: Any, dataset: str, revision: str, lang: str,
                 split: str) -> tuple[list[str], int]:
    listed = fs.ls(f"datasets/{dataset}@{revision}/data/{lang}/{split}", detail=True)
    files = sorted((p["name"], p.get("size") or 0) for p in listed
                   if p["name"].endswith(".parquet"))
    if not files:
        raise RuntimeError(f"{dataset} {lang}/{split} has no parquet files")
    return [name for name, _ in files], sum(size for _, size in files)


def run_mix(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """The data v2 build from the real sources (Hugging Face), per the config and flags."""
    import functools

    from huggingface_hub import HfApi, HfFileSystem

    import train_tokenizer as tt
    from quipu.tokenizer import make_tokenizer

    d = cfg.data
    if (d.dataset, d.subset) != (tt.TEXT_DATASET, "sample-10BT"):
        raise SystemExit(f"data v2 reads English from {tt.TEXT_DATASET} sample-10BT; "
                         f"the config says {d.dataset} {d.subset}")
    tw = dict(d.text_language_weights) or {sm.ENGLISH: 1.0}
    others = [x for x in tw if x != sm.ENGLISH]
    api, fs = HfApi(), HfFileSystem()

    tok_factory = functools.partial(make_tokenizer, d.tokenizer)
    tok = tok_factory()
    tokenizer = {"name": d.tokenizer, "vocab_size": tok.vocab_size, "eot": tok.eot,
                 "path": None if d.tokenizer == "gpt2" else d.tokenizer,
                 "sha256": None if d.tokenizer == "gpt2" else _sha256_file(d.tokenizer)}
    if tok.vocab_size > 65536:
        raise SystemExit(f"vocab {tok.vocab_size} does not fit uint16 shards")

    lid = lid_info = None
    if args.lid_filter:
        from huggingface_hub import hf_hub_download

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

    from stepbuild.bench.run import LeakageGuard
    guard = LeakageGuard()
    setup = sm.DocSetup(tokenizer=tok_factory, max_doc_tokens=d.code_max_doc_tokens,
                        guard=guard, lid=lid, lid_threshold=args.lid_threshold)

    # Pin every dataset to one commit so every read agrees.
    code_rev = api.dataset_info(d.code_dataset).sha
    listed = [p for p in fs.ls(f"datasets/{d.code_dataset}@{code_rev}/data", detail=False)
              if p.endswith(".parquet")]
    if len(listed) != d.code_files_total:
        raise RuntimeError(f"{d.code_dataset}@{code_rev} has {len(listed)} parquet files; "
                           f"config says code_files_total = {d.code_files_total}")
    max_files = min(args.max_code_files or d.code_heldout_first_file,
                    d.code_heldout_first_file)
    langs, lics = tuple(d.code_language_weights), d.code_licenses
    code_train = mix_code_rows(fs, d.code_dataset, code_rev, range(0, max_files),
                               d.code_files_total, langs, lics)
    code_val = mix_code_rows(fs, d.code_dataset, code_rev,
                             range(d.code_heldout_first_file, d.code_files_total),
                             d.code_files_total, langs, lics)

    def text_source(paths: list[str]) -> Callable[[], Iterator[dict]]:
        return lambda: flatten(prefetch(batched(tt.iter_parquet_text(fs, paths), 1000),
                                        depth=8))

    text_rev = api.dataset_info(tt.TEXT_DATASET).sha
    names = tt.text_files(fs, text_rev)
    full = [f"datasets/{tt.TEXT_DATASET}@{text_rev}/{n}" for n in names]
    text_train = {sm.ENGLISH: text_source(full[:-1])}
    text_val = {sm.ENGLISH: text_source(full[-1:])}
    fw2_rev = api.dataset_info(tt.FINEWEB2_DATASET).sha if others else None
    fw2: dict[str, Any] = {}
    for x in others:
        train_files, train_bytes = _fw2_listing(fs, tt.FINEWEB2_DATASET, fw2_rev, x, "train")
        test_files, _ = _fw2_listing(fs, tt.FINEWEB2_DATASET, fw2_rev, x, "test")
        text_train[x] = text_source(train_files)
        text_val[x] = text_source(test_files)
        fw2[x] = {"train_files": len(train_files), "train_bytes": train_bytes,
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
        text_order=order, keep_work=args.keep_work)
    provenance = {
        "tokenizer": tokenizer,
        "lid": lid_info,
        "sources": {
            "code": {"dataset": d.code_dataset, "revision": code_rev,
                     "files_total": d.code_files_total, "train_files": [0, max_files - 1],
                     "heldout_files": [d.code_heldout_first_file, d.code_files_total - 1]},
            "english": {"dataset": tt.TEXT_DATASET, "revision": text_rev,
                        "train_files": len(names) - 1, "heldout_files": names[-1:]},
            "fineweb2": {"dataset": tt.FINEWEB2_DATASET, "revision": fw2_rev,
                         "collection_order": list(order), "by_language": fw2},
        },
    }
    return build_mix(Path(d.shard_dir), spec, MixSources(code_train, code_val, text_train,
                                                         text_val),
                     setup, provenance)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--low-priority", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run at BELOW_NORMAL process priority (Windows; default on)")
    v2 = parser.add_argument_group("data v2 (configs with code_language_weights)")
    v2.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="tokenising processes (default: all cores but one)")
    v2.add_argument("--lid-filter", action=argparse.BooleanOptionalAction, default=False,
                    help="filter text with the config's lid_model (needs fasttext: Linux)")
    v2.add_argument("--lid-threshold", type=float, default=sm.DEFAULT_LID_THRESHOLD,
                    help="drop a document when another language wins with at least this "
                         "probability")
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
    v2.add_argument("--keep-work", action="store_true",
                    help="keep <shard_dir>/_work (the collected buckets) after the build")
    args = parser.parse_args(argv)
    if args.low_priority:
        print(f"below-normal priority set: {lower_priority()}", flush=True)

    cfg = load_config(args.config)
    if uses_mix(cfg.data):
        try:
            run_mix(cfg, args)
        except sm.ShareError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(2) from None
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
              shard_tokens=d.shard_tokens, tok=Tokenizer(),
              dataset=d.dataset, subset=d.subset, revision=revision, code=code)


if __name__ == "__main__":
    main()
