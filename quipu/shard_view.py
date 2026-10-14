"""A shard directory the trainer can read, assembled from shards that live elsewhere.

The trainer reads <shard_dir>/train/shard_*.bin and <shard_dir>/val/shard_*.bin and
checks <shard_dir>/manifest.json's tokenizer sha256 (quipu.train.check_shard_tokenizer).
The data v3 pool (quipu/pool.py) has another layout: <pool>/mixed/shard_NNNNN.bin and
a pool manifest whose tokenizer sits under provenance.tokenizer. make_shard_view builds
the trainer's layout from links, so neither the pool nor the old validation shards are
copied or rewritten:

    <view>/train/shard_NNNNN.bin  -> <pool>/mixed/shard_NNNNN.bin     (every shard, name order)
    <view>/val/shard_NNN.bin      -> <val dir>/shard_NNN.bin
    <view>/manifest.json          {"format": "shard view ...", "tokenizer": {"sha256": ...}, ...}

The tokenizer is checked here, not trusted: the pool manifest must record the sha256
the caller expects (the hash of the tokenizer file the trainer will load), so the view
never claims a hash the pool was not built with. The validation shards carry no
manifest of their own on the Modal Volume; their tokenizer is the caller's to vouch for
(val_tokenizer_sha256, recorded in the view's manifest as given).

Links: symlinks, falling back to hard links where the OS refuses symlinks (Windows
without developer mode). A copy only with allow_copy (tests, small data): copying a
20 GB pool by accident must not happen silently. The view directory is rebuilt from
scratch on every call; nothing outside it is written.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from quipu.data import shard_token_count

VIEW_FORMAT = "shard view (quipu/shard_view.py)"
POOL_FORMAT_PREFIX = "data v3 pool"


class ShardViewError(ValueError):
    """The pool or the validation shards cannot back a view (missing, wrong tokenizer)."""


def _link(src: Path, dst: Path, allow_copy: bool) -> str:
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        if not allow_copy:
            raise
    shutil.copyfile(src, dst)
    return "copy"


def _shards(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).glob("shard_*.bin") if p.is_file())


def pool_tokenizer(manifest: dict[str, Any]) -> dict[str, Any]:
    """The tokenizer record of a pool manifest (provenance.tokenizer), {} if none."""
    tok = (manifest.get("provenance") or {}).get("tokenizer")
    return tok if isinstance(tok, dict) else {}


def make_shard_view(view_dir: str | Path, pool_dir: str | Path, val_dir: str | Path,
                    tokenizer_sha256: str, *, val_tokenizer_sha256: str | None = None,
                    max_train_shards: int | None = None, allow_copy: bool = False,
                    ) -> dict[str, Any]:
    """Build the trainer's shard layout at view_dir from the pool at pool_dir (its
    manifest.json and mixed/shard_*.bin) and the validation shards in val_dir.
    Returns the manifest written to view_dir/manifest.json.

    tokenizer_sha256 is the hash of the tokenizer the trainer will use; the pool
    manifest must record the same (ShardViewError otherwise). max_train_shards keeps
    only the first N training shards (a smoke run). Every shard's size is checked
    (whole uint16 tokens) and every listed pool shard must exist."""
    view, pool, val = Path(view_dir), Path(pool_dir), Path(val_dir)
    manifest_path = pool / "manifest.json"
    try:
        pm = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ShardViewError(f"{manifest_path}: no readable pool manifest ({exc})") from exc
    if not str(pm.get("format", "")).startswith(POOL_FORMAT_PREFIX):
        raise ShardViewError(f"{manifest_path}: format {pm.get('format')!r} is not a "
                             f"{POOL_FORMAT_PREFIX!r} manifest")
    tok = pool_tokenizer(pm)
    if tok.get("sha256") != tokenizer_sha256:
        raise ShardViewError(
            f"the pool at {pool} was built with tokenizer sha256 {tok.get('sha256')!r}, "
            f"not {tokenizer_sha256!r}; training on it would read every token id wrong")
    if pm.get("dtype") != "uint16 little-endian":
        raise ShardViewError(f"{manifest_path}: dtype {pm.get('dtype')!r}, expected uint16")
    mixed = pool / str(pm.get("shard_dir", "mixed"))
    train = _shards(mixed)
    listed = [s["file"] for s in pm.get("shards", []) if isinstance(s, dict)]
    missing = sorted(set(listed) - {p.name for p in train})
    if missing:
        raise ShardViewError(f"{mixed}: {len(missing)} shard(s) the manifest lists are "
                             f"missing, e.g. {missing[:3]}")
    if not train:
        raise ShardViewError(f"{mixed}: no shard_*.bin")
    if max_train_shards is not None:
        if max_train_shards < 1:
            raise ValueError(f"max_train_shards must be >= 1, got {max_train_shards}")
        train = train[:max_train_shards]
    val_shards = _shards(val)
    if not val_shards:
        raise ShardViewError(f"{val}: no validation shard_*.bin")

    if view.exists():
        shutil.rmtree(view)
    (view / "train").mkdir(parents=True)
    (view / "val").mkdir()
    kinds = set()
    train_tokens = 0
    for p in train:
        train_tokens += shard_token_count(p)            # raises on an odd byte count
        kinds.add(_link(p.resolve(), view / "train" / p.name, allow_copy))
    val_tokens = 0
    for p in val_shards:
        val_tokens += shard_token_count(p)
        kinds.add(_link(p.resolve(), view / "val" / p.name, allow_copy))

    out = {
        "format": VIEW_FORMAT,
        # What check_shard_tokenizer reads; checked against the pool manifest above.
        "tokenizer": {**{k: tok[k] for k in ("path", "vocab_size", "eot") if k in tok},
                      "sha256": tokenizer_sha256},
        "eot": tok.get("eot"),
        "dtype": "uint16 little-endian",
        "links": sorted(kinds),
        "train": {"source": str(mixed), "shards": len(train),
                  "first": train[0].name, "last": train[-1].name, "tokens": train_tokens},
        "val": {"source": str(val), "shards": len(val_shards), "tokens": val_tokens,
                "tokenizer_sha256": val_tokenizer_sha256 or "not recorded"},
        "pool": {k: pm.get(k) for k in ("format", "version", "config_sha256",
                                        "total_tokens", "written")},
    }
    (view / "manifest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
