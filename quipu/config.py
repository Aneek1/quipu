"""One TOML file fully determines a run. Everything derived is computed here, once,
so no two call sites can disagree about how many steps there are.

steps floors total_tokens / batch_tokens, so up to batch_tokens-1 tokens are unused
(194,816 for the shipped config)."""
from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any

TOP_LEVEL_KEYS = {"name", "model", "data", "train"}


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    ffn_hidden: int
    context: int
    rope_base: float
    norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclasses.dataclass(frozen=True)
class DataConfig:
    dataset: str
    subset: str
    shard_dir: str
    shard_tokens: int
    val_tokens: int


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    total_tokens: int
    batch_tokens: int
    micro_batch: int
    context: int
    lr: float
    lr_min: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    seed: int
    ckpt_dir: str
    ckpt_every: int
    eval_every: int
    eval_batches: int

    @property
    def steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def grad_accum(self) -> int:
        return self.batch_tokens // (self.micro_batch * self.context)


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate_int_fields(instance: Any, allow_zero: frozenset[str] = frozenset()) -> None:
    """Every int-annotated field must actually be an int (not bool, not float) and
    strictly positive, so a typo or a 0 doesn't silently reach a division below.
    Fields named in allow_zero may additionally be 0 (e.g. seed)."""
    for f in dataclasses.fields(instance):
        # f.type is the raw annotation; it's the string "int" under `from __future__
        # import annotations` and the type int otherwise, so cover both.
        if f.type not in (int, "int"):
            continue
        value = getattr(instance, f.name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{f.name} must be an int, got {value!r}")
        minimum = 0 if f.name in allow_zero else 1
        if value < minimum:
            raise ValueError(f"{f.name} must be positive, got {value!r}")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        raw = _merge(raw, overrides)

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level key(s): {sorted(unknown)}")

    if "context" in raw.get("train", {}):
        raise ValueError("context belongs in [model]; [train] inherits it")

    model = ModelConfig(**raw["model"])
    data = DataConfig(**raw["data"])
    train = TrainConfig(context=model.context, **raw["train"])

    # Type/positivity checks must run before any division below (e.g. grad_accum's
    # micro_batch * context), so a bad value raises a clear ValueError instead of a
    # ZeroDivisionError or a silently-wrong result. seed may be 0; everything else in
    # TrainConfig, including warmup_steps (Task 10's lr_at divides by it), must stay
    # strictly positive.
    _validate_int_fields(model)
    _validate_int_fields(data)
    _validate_int_fields(train, allow_zero=frozenset({"seed"}))

    if model.d_model % model.n_head != 0:
        raise ValueError("d_model must divide evenly by n_head")
    if model.n_head % model.n_kv_head != 0:
        raise ValueError("n_head must be a multiple of n_kv_head for GQA")
    if model.head_dim % 2 != 0:
        raise ValueError(f"head_dim ({model.head_dim}) must be even for RoPE")

    if train.batch_tokens % (train.micro_batch * model.context) != 0:
        raise ValueError(
            f"batch_tokens ({train.batch_tokens}) must be a multiple of "
            f"micro_batch x context ({train.micro_batch} x {model.context})"
        )

    if train.lr <= 0:
        raise ValueError(f"lr must be positive, got {train.lr!r}")
    if not (0 <= train.lr_min <= train.lr):
        raise ValueError(f"lr_min ({train.lr_min}) must be between 0 and lr ({train.lr})")

    if train.steps < 1:
        raise ValueError(
            f"steps ({train.steps}) must be at least 1; total_tokens "
            f"({train.total_tokens}) must be >= batch_tokens ({train.batch_tokens})"
        )
    if train.warmup_steps >= train.steps:
        raise ValueError(
            f"warmup_steps ({train.warmup_steps}) must be less than steps ({train.steps})"
        )

    return Config(name=raw["name"], model=model, data=data, train=train)
