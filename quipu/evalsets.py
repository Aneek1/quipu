"""What the post-run evaluation tools share: the evaluation splits of a shard set (one
per language plus code), fixed non-wrapping batches from them, and loading a
checkpoint of either model kind.

Splits (spec 7 and 11). A shard set built by scripts/build_shards.py holds
    val/                English (FineWeb-Edu), the trainer's own validation loss
    val_lang/<bucket>/  every other language, FineWeb-2's test split (data v2 only;
                        cmn_Hani is split into zho_Hans / zho_Hant under the LID filter)
    code_val/           held-out code files
eval_splits() names them "eng_Latn", the bucket names, and "code", in LANGUAGES order.
quipu-114m's shard set has only val/ and code_val/, so it gets English and code.

Batches: split_batches() reads the first tokens of a split into rows of `context`
(contiguous, targets shifted by one, never wrapping), so every checkpoint sees the same
tokens and a short split is evaluated once rather than repeated. The per-language
validation splits can be short (allow_short in the builder).

Bits per byte over these batches (quipu.eval.nll_tokens_bytes): summed next-token
loss in bits / the UTF-8 bytes of the target tokens (tokenizer.token_byte_lengths).
The splits keep their <|endoftext|> document separators, and a separator is a target
like any other token: its loss is counted but it is 0 bytes of text. So bpb here is
slightly higher (more pessimistic) than a separator-free definition would give, by a
share that depends on the split's document length; compare numbers only across models
evaluated on the same splits.

Checkpoints: a milestone is a bare bf16 state_dict; a full checkpoint is
{"model": fp32 state_dict, "optimizers": ..., ...}. load_model() takes either, builds
the configured model (quipu.model_factory) and loads strictly, so a mismatch fails
loudly instead of evaluating half-random weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from quipu.config import ModelConfig
from quipu.data import read_shard

ENGLISH = "eng_Latn"
CODE = "code"
# The ten languages of spec 11 (Chinese as its two scripts, or unsplit without the LID
# filter), then code. Display names for tables.
LANGUAGES: dict[str, str] = {
    ENGLISH: "English",
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Malay",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "cmn_Hani": "Chinese (script not split)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "tam_Taml": "Tamil",
    "hin_Deva": "Hindi (Devanagari)",
    "hin_Latn": "Hindi (romanised)",
    "urd_Latn": "Urdu (romanised)",
    CODE: "Code",
}


def _has_shards(d: Path) -> bool:
    return d.is_dir() and any(d.glob("shard_*.bin"))


def _order(label: str) -> tuple[int, str]:
    keys = list(LANGUAGES)
    return (keys.index(label), label) if label in keys else (keys.index(CODE) - 0.5, label)


def eval_splits(shard_dir: str | Path) -> dict[str, Path]:
    """label -> split directory, for every split that has shards, in LANGUAGES order
    (an unknown val_lang bucket sorts just before code)."""
    root = Path(shard_dir)
    found: dict[str, Path] = {}
    if _has_shards(root / "val"):
        found[ENGLISH] = root / "val"
    lang_root = root / "val_lang"
    if lang_root.is_dir():
        for d in lang_root.iterdir():
            if d.name != ENGLISH and _has_shards(d):
                found[d.name] = d
    if _has_shards(root / "code_val"):
        found[CODE] = root / "code_val"
    return {k: found[k] for k in sorted(found, key=_order)}


# val_lang buckets that are one script of a source language (build_shards.py's
# text_buckets under --lid-filter; quipu.shard_mix.ZH_BUCKETS), for a shard set whose
# manifest does not name each bucket's source.
_BUCKET_SOURCE = {"zho_Hans": "cmn_Hani", "zho_Hant": "cmn_Hani"}


def mix_weights(data_cfg: Any, shard_dir: str | Path) -> dict[str, float]:
    """The data mix as weights over the evaluation splits (eval_splits labels, in its
    order), for the mix-weighted validation loss: code = data.code_share on code_val,
    English = (1 - code_share) x text_language_weights["eng_Latn"] on val, and each
    val_lang bucket (1 - code_share) x its source language's text weight, shared
    equally between the buckets of one source (zho_Hans / zho_Hant split cmn_Hani's).
    A bucket's source is the manifest's (splits.val_lang.<bucket>.source) when the
    shard set has one, else the bucket name (zho_Hans / zho_Hant -> cmn_Hani). A
    bucket whose source has no text weight gets none (left out).

    {} (no mix: the trainer evaluates English only, as before) unless the config has
    text_language_weights (data v2) AND the shard set has code_val and val_lang:
    quipu-114m has code_val but neither of the others, so it is unaffected. The
    weights are not renormalised; weighted_loss divides by those present."""
    text = dict(getattr(data_cfg, "text_language_weights", None) or {})
    splits = eval_splits(shard_dir)
    buckets = [k for k in splits if k not in (ENGLISH, CODE)]
    if not text or ENGLISH not in text or CODE not in splits or not buckets:
        return {}
    sources: dict[str, str] = {}
    manifest = Path(shard_dir) / "manifest.json"
    if manifest.is_file():
        try:
            lang = json.loads(manifest.read_text(encoding="utf-8"))["splits"]["val_lang"]
            sources = {b: str(v["source"]) for b, v in lang.items()
                       if isinstance(v, dict) and "source" in v}
        except (OSError, ValueError, KeyError, TypeError):
            sources = {}
    source_of = {b: sources.get(b, _BUCKET_SOURCE.get(b, b)) for b in buckets}
    code_share = float(data_cfg.code_share)
    per_source: dict[str, int] = {}
    for b, src in source_of.items():
        per_source[src] = per_source.get(src, 0) + 1
    out: dict[str, float] = {}
    for label in splits:
        if label == CODE:
            out[label] = code_share
        elif label == ENGLISH:
            out[label] = (1 - code_share) * text[ENGLISH]
        elif source_of[label] in text and source_of[label] != ENGLISH:
            out[label] = (1 - code_share) * text[source_of[label]] / per_source[source_of[label]]
    return out


def weighted_loss(losses: dict[str, float], weights: dict[str, float]) -> float:
    """sum(weight x loss) / sum(weight) over the splits in both (the weights of the
    splits that were evaluated, renormalised)."""
    keys = [k for k in losses if weights.get(k, 0) > 0]
    total = sum(weights[k] for k in keys)
    if total <= 0:
        raise ValueError("weighted_loss: no split with a weight")
    return sum(weights[k] * losses[k] for k in keys) / total


def read_split(split_dir: str | Path, max_tokens: int) -> np.ndarray:
    """The first max_tokens tokens of a split (its shards in name order), int64."""
    parts: list[np.ndarray] = []
    have = 0
    for p in sorted(Path(split_dir).glob("shard_*.bin")):
        if have >= max_tokens:
            break
        t = read_shard(p)[: max_tokens - have]
        parts.append(t)
        have += len(t)
    if not parts:
        raise FileNotFoundError(f"no shards in {split_dir}")
    return np.concatenate(parts).astype(np.int64)


def split_batches(split_dir: str | Path, micro_batch: int, context: int,
                  max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Up to max_batches (x, y) batches of micro_batch rows x context tokens from the
    start of the split, never wrapping: a short split gives fewer rows (the last batch
    may be smaller), and one shorter than context + 1 tokens gives one shorter row."""
    if micro_batch < 1 or context < 1 or max_batches < 1:
        raise ValueError("micro_batch, context and max_batches must be positive")
    t = read_split(split_dir, max_batches * micro_batch * context + 1)
    n = len(t) - 1
    if n < 1:
        raise ValueError(f"{split_dir}: fewer than 2 tokens")
    width = min(context, n)
    rows = n // width
    x = torch.from_numpy(t[: rows * width]).view(rows, width)
    y = torch.from_numpy(t[1 : rows * width + 1]).view(rows, width)
    return [(x[i : i + micro_batch], y[i : i + micro_batch]) for i in range(0, rows, micro_batch)]


def checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    """The model state_dict of a milestone (bare) or a full training checkpoint."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def latest_checkpoint(ckpt_dir: str | Path) -> Path:
    """The full checkpoint latest.pt points at."""
    from quipu.train import LATEST  # the trainer module is heavy; only when needed

    ckpt_dir = Path(ckpt_dir)
    pointer = torch.load(ckpt_dir / LATEST, map_location="cpu", weights_only=True)
    return ckpt_dir / pointer["file"]


def load_model(cfg: ModelConfig, path: str | Path | None = None, device: str = "cpu",
               state: dict[str, torch.Tensor] | None = None) -> nn.Module:
    """The configured model (dense or MoE) with a checkpoint's weights, strictly, in
    eval mode on `device`; fp32 whatever the file's dtype. Pass `state` instead of
    `path` when the state_dict is already in memory."""
    from quipu.model_factory import build_model

    if state is None:
        if path is None:
            raise ValueError("load_model needs a path or a state")
        state = checkpoint_state(path)
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    assert model.lm_head.weight is model.embed.weight, "lm_head/embed tie was broken"
    return model.to(device).eval()
