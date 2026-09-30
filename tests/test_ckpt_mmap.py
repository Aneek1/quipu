"""Laptop-side checkpoint loads memory-map the file (torch.load(..., mmap=True)), so a
~12 GB quipu-moe training checkpoint (weights + optimizer states) never has to fit in
RAM: only the pages of the model weights that are read come in."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import quipu.train as train_mod
from quipu import evalsets
from quipu.config import load_config
from quipu.model_factory import build_model

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_mmap_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ckpts(tmp_path):
    cfg = load_config(SMOKE, {"train": {"ckpt_dir": str(tmp_path / "ck")}})
    torch.manual_seed(0)
    model = build_model(cfg.model)
    ck = tmp_path / "ck"
    (ck / "milestones").mkdir(parents=True)
    full = ck / "step_000010.pt"
    train_mod._atomic_save({"step": 10, "model": model.state_dict(), "optimizers": []}, full)
    train_mod._atomic_save({"file": full.name}, ck / train_mod.LATEST)
    ms = ck / "milestones" / "step_000010.pt"
    train_mod._atomic_save(train_mod._bf16_state_dict(model), ms)
    return cfg, full, ms


@pytest.fixture
def spy(monkeypatch):
    """Records torch.load calls (path name, mmap flag); still loads for real."""
    calls = []
    real = torch.load

    def load(f, *a, **kw):
        calls.append((Path(f).name, kw.get("mmap", False)))
        return real(f, *a, **kw)

    monkeypatch.setattr(torch, "load", load)
    return calls


def test_evalsets_maps_full_checkpoints_and_milestones(ckpts, spy):
    cfg, full, ms = ckpts
    for path in (full, ms):
        model = evalsets.load_model(cfg.model, path)
        assert isinstance(model, torch.nn.Module)
    assert (full.name, True) in spy and ("step_000010.pt", True) in spy
    assert all(m for name, m in spy if name != train_mod.LATEST)


def test_milestone_eval_maps_every_checkpoint_it_reads(ckpts, spy):
    cfg, full, ms = ckpts
    me = _script("milestone_eval")
    found = me.discover_checkpoints(full.parent)
    assert found
    me.load_milestone_model(cfg, ms, "cpu")
    me.load_final_model(cfg, full, "cpu")
    assert spy and all(m for name, m in spy if name != train_mod.LATEST)


def test_the_dense_export_maps_its_checkpoints(tmp_path, spy, monkeypatch):
    ex = _script("export_hf")
    src = Path(ex.__file__).read_text(encoding="utf-8")
    # export_dense runs a full parity check on quipu-114m's GPT-2 model; here only
    # its loads are checked, in the source.
    body = src.split("def export_dense", 1)[1].split("\ndef ", 1)[0]
    assert body.count("mmap=True") == 2
