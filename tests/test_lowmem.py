"""Low-memory training on T4 (spec docs/superpowers/specs/2026-10-20-t4-low-memory-training-design.md)."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


def smoke_cfg(tmp_path, model=None, **train):
    over = {"data": {"shard_dir": str(tmp_path / "shards")},
            "train": {"ckpt_dir": str(tmp_path / "ckpt"), **train}}
    if model:
        over["model"] = model
    return load_config(SMOKE, over)


# ---- config ----------------------------------------------------------------------

def test_new_train_fields_default_to_todays_behaviour(tmp_path):
    t = smoke_cfg(tmp_path).train
    assert t.precision == "bf16"
    assert t.optimizer_state_dtype == "fp32"
    assert t.activation_checkpoint == "none"
    assert t.loss_chunk_tokens == 0


def test_t4_settings_load(tmp_path):
    t = smoke_cfg(tmp_path, model={"attnres_blocks": 0}, precision="fp16",
                  optimizer_state_dtype="bf16", activation_checkpoint="block",
                  loss_chunk_tokens=300).train
    assert (t.precision, t.optimizer_state_dtype, t.activation_checkpoint,
            t.loss_chunk_tokens) == ("fp16", "bf16", "block", 300)


@pytest.mark.parametrize("field,value", [
    ("precision", "fp32"), ("optimizer_state_dtype", "fp16"),
    ("activation_checkpoint", "layer"), ("loss_chunk_tokens", -1),
])
def test_bad_values_are_refused(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        smoke_cfg(tmp_path, model={"attnres_blocks": 0}, **{field: value})


def test_block_checkpointing_refuses_attnres(tmp_path):
    # The smoke config has attnres_blocks = 2; AttnRes has its own checkpoint option.
    with pytest.raises(ValueError, match="activation_checkpoint"):
        smoke_cfg(tmp_path, activation_checkpoint="block")


def test_apply_precision_leaves_the_model_alone_for_fp16():
    from quipu.fp8 import apply_precision
    m = torch.nn.Linear(4, 4)
    assert apply_precision(m, "fp16", "cpu") is m
