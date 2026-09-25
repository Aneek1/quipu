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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import numpy as np
from tqdm import tqdm

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
    """Code rows {"code", "language", "license"[, "file"]} -> (ids, "code:<language>").

    A row is dropped, in this order, if its language is not kept; if its licence is
    not kept; (code val) if its content hash is in `exclude`; if it is blank; or if
    it is HTML over the cap. The HTML cap: once at least `html_warmup` code tokens
    have been yielded, an HTML document of n tokens is skipped when
    html + n > html_cap * (code + n), i.e. when yielding it would take HTML above
    html_cap of the code tokens yielded. The consumer writes every yielded
    document (the last one possibly truncated), so yielded == written up to that one
    document.

    With record_hashes, the content hash of every yielded document is kept in
    .hashes (train records them; code val excludes them).
    """

    def __init__(self, rows: Iterator[dict], tok: Tokenizer, *, languages: Iterable[str],
                 licenses: Iterable[str], html_cap: float,
                 html_warmup: int = HTML_CAP_WARMUP_TOKENS,
                 exclude: set[int] | frozenset[int] = frozenset(),
                 record_hashes: bool = False) -> None:
        self._rows = rows
        self._tok = tok
        self.languages = frozenset(languages)
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
            ids = encode_document(code, self._tok)
            if ids is None:
                stats["skipped_blank"] += 1
                continue
            n = len(ids)
            if language == "HTML":
                if (self.tokens >= self.html_warmup
                        and self.html_tokens + n > self.html_cap * (self.tokens + n)):
                    stats["skipped_html_cap"] += 1
                    continue
                self.html_tokens += n
            self.tokens += n
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
        "last_file_read": docs.last_file,
    }


def _build_code(root: Path, it: Iterator[dict], *, train_tokens: int, shard_tokens: int,
                tok: Tokenizer, code: CodeSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mixed train from the text iterator `it` plus code, then the deduplicated code val."""
    text_docs = TextDocs(it, tok)
    code_docs = CodeDocs(iter(code.train_rows), tok, languages=code.languages,
                         licenses=code.licenses, html_cap=code.html_cap,
                         html_warmup=code.html_warmup, record_hashes=True)
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
                        html_warmup=code.html_warmup, exclude=code_docs.hashes)
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


# ------------------------------------------------------------------ real sources

def code_file_path(repo: str, revision: str, index: int, total: int) -> str:
    return f"datasets/{repo}@{revision}/data/train-{index:05d}-of-{total:05d}.parquet"


def iter_code_row_groups(fs: Any, repo: str, revision: str, files: range, total: int,
                         columns: tuple[str, ...] = ("code", "language", "license"),
                         retries: int = 5) -> Iterator[list[dict]]:
    """The rows of each parquet row group, in file order, one row group at a time.

    Every row also carries "file" (its parquet index). A failed read is retried
    from the same (file, row group), so a network blip changes nothing in the output.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--low-priority", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run at BELOW_NORMAL process priority (Windows; default on)")
    args = parser.parse_args()
    if args.low_priority:
        print(f"below-normal priority set: {lower_priority()}", flush=True)

    # Heavy imports; the tests never need them.
    from datasets import load_dataset
    from huggingface_hub import HfApi, HfFileSystem

    cfg = load_config(args.config)
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
        html_cap=d.html_cap, val_tokens=d.code_val_tokens,
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
