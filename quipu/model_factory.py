"""build_model(cfg): the one place that turns a ModelConfig into a model.

kind "dense" is quipu.model.Quipu, unchanged (quipu-114m's checkpoints and export
keep loading); kind "moe" is quipu.model_moe.QuipuMoE.
"""
from __future__ import annotations

import torch.nn as nn

from quipu.config import ModelConfig
from quipu.model import Quipu
from quipu.model_moe import QuipuMoE


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.kind == "dense":
        return Quipu(cfg)
    if cfg.kind == "moe":
        return QuipuMoE(cfg)
    raise ValueError(f"unknown model kind {cfg.kind!r}")
