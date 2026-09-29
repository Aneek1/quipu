"""Shard format: a flat little-endian uint16 array of token ids, nothing else.

No header, no index, no compression. The file length divided by two is the token
count, which makes a shard trivially memory-mappable and impossible to misparse.
The vocabulary is 50257, so uint16 is exact; tokenizer.py asserts that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from quipu.fsio import replace_with_retry
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
    replace_with_retry(tmp, path)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    """The chat fine-tune's loss mask beside a shard (shard_NNN.mask next to
    shard_NNN.bin): one uint8 per token, 1 where the token is a training target.
    Written atomically like write_shard."""
    if mask.dtype != np.uint8:
        raise ValueError(f"masks are uint8, got {mask.dtype}")
    if mask.size and int(mask.max()) > 1:
        raise ValueError("mask values must be 0 or 1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        mask.tofile(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    replace_with_retry(tmp, path)


def shard_token_count(path: str | Path) -> int:
    """Validate a shard's byte count and return its token count.

    The single source of truth for "is this a well-formed shard": both the
    eager reader below and the memmap-based loader route through this before
    touching file contents, so a truncated shard is rejected the same way
    everywhere.
    """
    path = Path(path)
    size = path.stat().st_size
    if size % 2:
        raise ValueError(f"{path}: odd byte count, truncated shard")
    return size // 2


def read_shard(path: str | Path) -> np.ndarray:
    path = Path(path)
    shard_token_count(path)  # validates; raises ValueError on a truncated shard
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
