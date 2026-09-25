"""Download FineWeb-Edu, tokenize, and write uint16 shards.

Streams rather than downloading the whole set: sample-10BT is far larger than the
tokens this run needs, and there is no reason to store the remainder.

Validation comes off the head of the stream and training continues from exactly
where validation stopped, through ONE shared iterator. Iterating a streaming HF
dataset a second time restarts it from the beginning, so passing the dataset
object itself to both builds would make train re-read the validation documents.

A killed build is re-run from scratch: shard writes are atomic, the stream order
is deterministic, and each split directory is cleared of old shards first.

Run: uv run python scripts/build_shards.py --config configs/quipu-114m.toml
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from tqdm import tqdm

from quipu.config import load_config
from quipu.data import encode_document, write_shard
from quipu.fsio import replace_with_retry
from quipu.tokenizer import Tokenizer

MANIFEST = "manifest.json"


def clear_split_dir(out_dir: Path) -> None:
    """Remove shards and temp files left by an earlier (possibly longer) build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in list(out_dir.glob("shard_*.bin")) + list(out_dir.glob("*.tmp")):
        p.unlink()


def build(out_dir: Path, target_tokens: int, shard_tokens: int,
          docs_iter: Iterator[dict], tok: Tokenizer) -> dict[str, Any]:
    """Write exactly target_tokens tokens from docs_iter into shard_000.bin, ...

    Every shard holds shard_tokens tokens except the last, which holds the
    remainder. Consumption stops the moment the target is met; the rest of the
    document in progress is discarded, never carried into the next split.
    Raises RuntimeError if docs_iter runs dry first: a silently short dataset
    would make training wrap around.
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
    rows = 0
    documents = 0
    shards: list[dict[str, Any]] = []
    progress = tqdm(total=target_tokens, unit="tok", unit_scale=True, desc=out_dir.name)

    for row in docs_iter:
        rows += 1
        ids = encode_document(row.get("text") or "", tok)
        if ids is None:
            continue
        documents += 1
        arr = np.asarray(ids, dtype=np.uint16)
        pos = 0
        while pos < len(arr) and written < target_tokens:
            size = min(shard_tokens, target_tokens - written)
            take = min(size - fill, len(arr) - pos)
            buf[fill:fill + take] = arr[pos:pos + take]
            fill += take
            pos += take
            if fill == size:
                name = f"shard_{len(shards):03d}.bin"
                write_shard(out_dir / name, buf[:size])
                shards.append({"file": name, "tokens": size})
                written += size
                fill = 0
                progress.update(size)
        if written >= target_tokens:
            break
    progress.close()

    if written < target_tokens:
        raise RuntimeError(
            f"{out_dir.name}: document stream ran out after {written:,} of "
            f"{target_tokens:,} tokens"
        )
    return {"tokens": written, "rows_consumed": rows, "documents": documents,
            "shards": shards}


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


def build_all(root: Path, stream: Iterable[dict], *, val_tokens: int, train_tokens: int,
              shard_tokens: int, tok: Tokenizer, dataset: str, subset: str,
              revision: str | None, tokenizer_name: str = "gpt2") -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # A manifest from an earlier build must not survive to describe new shards.
    (root / MANIFEST).unlink(missing_ok=True)
    started, t0 = _now(), time.monotonic()

    # ONE iterator for both splits, so train continues where val stopped.
    it = iter(stream)
    val = build(root / "val", val_tokens, shard_tokens, it, tok)
    print(f"val:   {val['tokens']:,} tokens")
    train = build(root / "train", train_tokens, shard_tokens, it, tok)
    print(f"train: {train['tokens']:,} tokens")

    manifest = {
        "dataset": dataset,
        "subset": subset,
        "dataset_revision": revision,
        "tokenizer": tokenizer_name,
        "eot": tok.eot,
        "dtype": "uint16 little-endian",
        "splits": {"val": val, "train": train},
        "total_tokens": val["tokens"] + train["tokens"],
        "build_started": started,
        "build_finished": _now(),
        "build_seconds": round(time.monotonic() - t0, 1),
    }
    write_manifest(root / MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    args = parser.parse_args()

    from datasets import load_dataset  # heavy import; the tests never need it

    cfg = load_config(args.config)
    stream = load_dataset(cfg.data.dataset, name=cfg.data.subset, split="train",
                          streaming=True)
    revision = dataset_revision(stream)
    # Only "text" is used; skipping the other nine columns lightens the stream.
    stream = stream.select_columns(["text"])

    build_all(Path(cfg.data.shard_dir), stream,
              val_tokens=cfg.data.val_tokens, train_tokens=cfg.train.total_tokens,
              shard_tokens=cfg.data.shard_tokens, tok=Tokenizer(),
              dataset=cfg.data.dataset, subset=cfg.data.subset, revision=revision)


if __name__ == "__main__":
    main()
