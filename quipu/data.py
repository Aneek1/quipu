"""Shard format: a flat little-endian uint16 array of token ids, nothing else.

No header, no index, no compression. The file length divided by two is the token
count, which makes a shard trivially memory-mappable and impossible to misparse.
The vocabulary is 50257, so uint16 is exact; tokenizer.py asserts that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from quipu.tokenizer import Tokenizer


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        raise ValueError(f"shards are uint16, got {tokens.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tofile writes raw little-endian on every platform we target.
    tokens.astype("<u2", copy=False).tofile(path)


def read_shard(path: str | Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u2")


def tokenize_documents(docs: Iterable[str], tok: Tokenizer) -> np.ndarray:
    """Concatenate documents, each terminated by one EOT.

    Blank documents are dropped: a bare EOT would teach the model that EOT follows
    EOT, which never happens in the corpus.
    """
    out: list[int] = []
    for doc in docs:
        if not doc or not doc.strip():
            continue
        out.extend(tok.encode(doc))
        out.append(tok.eot)
    return np.array(out, dtype=np.uint16)
