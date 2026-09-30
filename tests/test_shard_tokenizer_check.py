"""The trainer refuses shards built with another tokenizer than the config's (exit 2)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import quipu.train as train_mod
from quipu.bpe import train_bpe
from quipu.data import write_shard

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


def setup(tmp_path, manifest_sha: str | None, make_tokenizer: bool = True):
    shards = tmp_path / "shards"
    for split in ("train", "val"):
        write_shard(shards / split / "shard_000.bin",
                    np.random.RandomState(0).randint(0, 300, 20_000).astype(np.uint16))
    if manifest_sha is not None:
        (shards / "manifest.json").write_text(json.dumps(
            {"tokenizer": {"path": "artifacts/tokenizer/tokenizer.json",
                           "sha256": manifest_sha}}), encoding="utf-8")
    tok = tmp_path / "tok" / "tokenizer.json"
    if make_tokenizer:
        tok.parent.mkdir()
        train_bpe(["the quipu was a recording device of knotted cords.\n"] * 50, 300, tok)
    return shards, tok


def run(tmp_path, shards, tok, monkeypatch):
    # The run never gets past the check (or is stopped right after it).
    monkeypatch.setattr(train_mod.Trainer, "run", lambda self: None)
    return train_mod.run_main([
        "--config", str(SMOKE), "--run-id", "t", "--run-dir", str(tmp_path / "runs"),
        "--device", "cpu", "--override", f"data.shard_dir={shards}",
        "--override", f"data.tokenizer={tok}", "--override", "model.vocab_size=300",
        "--override", f"train.ckpt_dir={tmp_path / 'ck'}"])


def test_a_different_tokenizer_is_a_usage_error(tmp_path, monkeypatch, capsys):
    shards, tok = setup(tmp_path, "0" * 64)
    assert run(tmp_path, shards, tok, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "built with another tokenizer" in err and "0" * 12 in err
    assert not (tmp_path / "runs" / "t.json").exists()       # nothing left behind


def test_the_same_tokenizer_passes(tmp_path, monkeypatch):
    shards, tok = setup(tmp_path, "")
    (shards / "manifest.json").write_text(json.dumps(
        {"tokenizer": {"sha256": hashlib.sha256(tok.read_bytes()).hexdigest()}}))
    assert run(tmp_path, shards, tok, monkeypatch) == 0


def test_a_missing_tokenizer_file_cannot_be_checked(tmp_path, monkeypatch, capsys):
    shards, tok = setup(tmp_path, "0" * 64, make_tokenizer=False)
    assert run(tmp_path, shards, tok, monkeypatch) == 2
    assert "cannot check" in capsys.readouterr().err


@pytest.mark.parametrize("manifest", [None, "no-sha"])
def test_shards_without_a_recorded_tokenizer_hash_are_not_checked(tmp_path, monkeypatch,
                                                                  manifest):
    shards, tok = setup(tmp_path, None, make_tokenizer=False)
    if manifest == "no-sha":
        (shards / "manifest.json").write_text(json.dumps({"tokenizer": "gpt2"}))
    assert run(tmp_path, shards, tok, monkeypatch) == 0
