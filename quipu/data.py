"""Shard format: a flat little-endian uint16 array of token ids, nothing else.

No header, no index, no compression. The file length divided by two is the token
count, which makes a shard trivially memory-mappable and impossible to misparse.
The vocabulary is 50257, so uint16 is exact; tokenizer.py asserts that.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from quipu.tokenizer import Tokenizer


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        raise ValueError(f"shards are uint16, got {tokens.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first and swap it in atomically: Task 11 runs for
    # hours, and a kill mid-write must not leave a truncated file at the final
    # path, since the loader's shard_*.bin glob can't tell a truncated shard
    # from a complete one. The .tmp suffix also doesn't match that glob.
    tmp = path.with_name(path.name + ".tmp")
    try:
        # The "<u2" cast is what guarantees little-endian on disk; the dtype
        # guard above means the cast itself is a no-op on this machine.
        tokens.astype("<u2", copy=False).tofile(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def read_shard(path: str | Path) -> np.ndarray:
    path = Path(path)
    size = path.stat().st_size
    if size % 2:
        raise ValueError(f"{path}: odd byte count, truncated shard")
    return np.fromfile(path, dtype="<u2")


def encode_document(text: str, tok: Tokenizer) -> list[int] | None:
    """Encode one document, terminated by one EOT.

    Returns None for empty/whitespace-only text: a bare EOT would teach the
    model that EOT follows EOT, which never happens in the corpus.
    """
    if not text or not text.strip():
        return None
    return tok.encode(text) + [tok.eot]


def tokenize_documents(docs: Iterable[str], tok: Tokenizer) -> np.ndarray:
    """Concatenate documents, each terminated by one EOT.

    Blank documents are dropped via encode_document.
    """
    out: list[int] = []
    for doc in docs:
        encoded = encode_document(doc, tok)
        if encoded is None:
            continue
        out.extend(encoded)
    return np.array(out, dtype=np.uint16)
