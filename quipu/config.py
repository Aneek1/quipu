"""One TOML file fully determines a run. Everything derived is computed here, once,
so no two call sites can disagree about how many steps there are."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    subset: str
    shard_dir: str
    shard_tokens: int
    val_tokens: int


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        raw = _merge(raw, overrides)

    model = ModelConfig(**raw["model"])
    data = DataConfig(**raw["data"])
    train = TrainConfig(context=model.context, **raw["train"])

    if train.batch_tokens % (train.micro_batch * model.context) != 0:
        raise ValueError(
            f"batch_tokens ({train.batch_tokens}) must be a multiple of "
            f"micro_batch x context ({train.micro_batch} x {model.context})"
        )
    if model.d_model % model.n_head != 0:
        raise ValueError("d_model must divide evenly by n_head")
    if model.n_head % model.n_kv_head != 0:
        raise ValueError("n_head must be a multiple of n_kv_head for GQA")

    return Config(name=raw["name"], model=model, data=data, train=train)
