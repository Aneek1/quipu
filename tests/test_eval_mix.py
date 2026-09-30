"""The mix-weighted validation loss (the A/B decision metric): evalsets.mix_weights,
eval.mean_loss, and the trainer's per-split evaluation (CPU only, smoke sizes)."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import quipu.train as train_mod
from quipu import evalsets
from quipu.config import load_config
from quipu.data import write_shard
from quipu.eval import mean_loss
from quipu.train import Trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"
DENSE = ROOT / "configs" / "quipu-114m.toml"


def shard(d: Path, tokens: int = 4_000, seed: int = 0) -> Path:
    write_shard(d / "shard_000.bin",
                np.random.RandomState(seed).randint(0, 512, tokens).astype(np.uint16))
    return d


def mix_shards(root: Path, langs=("ind_Latn", "zho_Hans", "zho_Hant")) -> Path:
    shard(root / "train", 20_000)
    shard(root / "val", seed=1)
    shard(root / "code_val", seed=2)
    for i, lang in enumerate(langs):
        shard(root / "val_lang" / lang, seed=3 + i)
    return root


def smoke(tmp_path, **train):
    return load_config(SMOKE, {"data": {"shard_dir": str(tmp_path / "shards")},
                               "train": {"ckpt_dir": str(tmp_path / "ckpt"),
                                         "optimizer": "adamw", **train}})


# ---- the weights -------------------------------------------------------------------------

def test_weights_follow_the_data_mix(tmp_path):
    cfg = smoke(tmp_path)
    w = evalsets.mix_weights(cfg.data, mix_shards(tmp_path / "shards"))
    assert w == pytest.approx({"eng_Latn": 0.28, "ind_Latn": 0.4 / 30,
                               "zho_Hans": 0.4 / 60, "zho_Hant": 0.4 / 60, "code": 0.6})
    assert list(w) == list(evalsets.eval_splits(tmp_path / "shards"))


def test_all_nine_languages_and_code_sum_to_one(tmp_path):
    cfg = smoke(tmp_path)
    langs = [k for k in cfg.data.text_language_weights if k not in ("eng_Latn", "cmn_Hani")]
    w = evalsets.mix_weights(cfg.data, mix_shards(tmp_path / "shards",
                                                  (*langs, "zho_Hans", "zho_Hant")))
    assert sum(w.values()) == pytest.approx(1.0)
    assert sum(v for k, v in w.items() if k not in ("eng_Latn", "code")) == pytest.approx(0.12)


def test_the_manifest_names_each_buckets_source(tmp_path):
    cfg = smoke(tmp_path)
    root = mix_shards(tmp_path / "shards", ("kor_Hang", "mystery"))
    (root / "manifest.json").write_text(json.dumps({"splits": {"val_lang": {
        "kor_Hang": {"source": "kor_Hang"}, "mystery": {"source": "tam_Taml"}}}}))
    w = evalsets.mix_weights(cfg.data, root)
    assert w["mystery"] == pytest.approx(0.4 / 30) and w["kor_Hang"] == pytest.approx(0.4 / 30)


def test_a_bucket_of_no_weighted_language_gets_no_weight(tmp_path):
    cfg = smoke(tmp_path)
    w = evalsets.mix_weights(cfg.data, mix_shards(tmp_path / "shards", ("ind_Latn", "xyz")))
    assert "xyz" not in w and "ind_Latn" in w


def test_no_mix_without_val_lang_or_text_weights(tmp_path):
    # quipu-114m: code_val exists, but no val_lang and no text weights -> English only.
    root = tmp_path / "shards"
    shard(root / "val")
    shard(root / "code_val")
    assert evalsets.mix_weights(load_config(DENSE).data, root) == {}
    assert evalsets.mix_weights(smoke(tmp_path).data, root) == {}
    mix_shards(root)
    assert evalsets.mix_weights(load_config(DENSE).data, root) == {}


def test_weighted_loss_renormalises_over_the_splits_it_has():
    w = {"eng_Latn": 0.28, "code": 0.6}
    got = evalsets.weighted_loss({"eng_Latn": 2.0, "code": 1.0}, w)
    assert got == pytest.approx((0.28 * 2.0 + 0.6 * 1.0) / 0.88)


# ---- mean_loss -----------------------------------------------------------------------------

def test_mean_loss_is_the_token_mean_and_restores_training_mode(tmp_path):
    cfg = smoke(tmp_path)
    torch.manual_seed(0)
    model = train_mod.build_model(cfg.model).train()
    batches = evalsets.split_batches(shard(tmp_path / "s", 1_500), 2, 256, 3)
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]      # a short last batch
    got = mean_loss(model, batches, "cpu")
    assert model.training
    model.eval()
    with torch.no_grad():
        nll = sum(torch.nn.functional.cross_entropy(
            model(x).view(-1, cfg.model.vocab_size), y.reshape(-1), reduction="sum").item()
            for x, y in batches)
    assert got == pytest.approx(nll / sum(y.numel() for _, y in batches), rel=1e-5)


# ---- the trainer ---------------------------------------------------------------------------

def test_the_trainer_logs_every_split_and_the_weighted_loss(tmp_path, capsys):
    cfg = smoke(tmp_path, eval_every=1, eval_batches=4)
    root = mix_shards(tmp_path / "shards")
    weights = evalsets.mix_weights(cfg.data, root)
    splits = evalsets.eval_splits(root)
    trainer = Trainer(model_cfg=cfg.model, train_cfg=dataclasses.replace(
                          cfg.train, total_tokens=2 * cfg.train.batch_tokens),
                      shard_dir=root / "train", val_dir=root / "val", device="cpu",
                      run_dir=tmp_path / "runs", run_id="t", mix_splits=splits,
                      mix_weights=weights)
    trainer.run()
    log = json.loads((tmp_path / "runs" / "t.json").read_text(encoding="utf-8"))
    assert log["eval_mix"]["weights"] == pytest.approx(weights)
    # English and code read eval_batches; each other language a quarter of them.
    assert log["eval_mix"]["batches"] == {"eng_Latn": 4, "code": 4, "ind_Latn": 1,
                                          "zho_Hans": 1, "zho_Hant": 1}
    assert [e["step"] for e in log["evals"]] == [1, 2]
    for e in log["evals"]:
        assert set(e["split_losses"]) == set(weights)
        assert e["split_losses"]["eng_Latn"] == e["val_loss"]
        want = sum(weights[k] * v for k, v in e["split_losses"].items()) / sum(weights.values())
        assert e["weighted_loss"] == pytest.approx(want)
        assert all(math.isfinite(v) for v in e["split_losses"].values())
    out = capsys.readouterr().out
    assert "weighted" in out and "code" in out


def test_without_a_mix_the_eval_entry_is_english_only(tmp_path):
    cfg = smoke(tmp_path, eval_every=1)
    root = mix_shards(tmp_path / "shards")
    trainer = Trainer(model_cfg=cfg.model, train_cfg=dataclasses.replace(
                          cfg.train, total_tokens=cfg.train.batch_tokens),
                      shard_dir=root / "train", val_dir=root / "val", device="cpu",
                      run_dir=tmp_path / "runs", run_id="t")
    trainer.run()
    log = json.loads((tmp_path / "runs" / "t.json").read_text(encoding="utf-8"))
    assert set(log["evals"][0]) == {"step", "val_loss"} and "eval_mix" not in log


def test_main_turns_the_mix_on_for_data_v2_shards(tmp_path, monkeypatch):
    root = mix_shards(tmp_path / "shards")
    seen = {}
    real = Trainer.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        real(self, *a, **kw)

    monkeypatch.setattr(Trainer, "__init__", spy)
    monkeypatch.setattr(Trainer, "run", lambda self: None)
    train_mod.main(["--config", str(SMOKE), "--run-id", "m", "--run-dir", str(tmp_path / "r"),
                    "--device", "cpu",
                    "--override", f"data.shard_dir={root}",
                    "--override", f"train.ckpt_dir={tmp_path / 'ck'}"])
    assert set(seen["mix_weights"]) == {"eng_Latn", "ind_Latn", "zho_Hans", "zho_Hant", "code"}
    assert set(seen["mix_splits"]) == set(seen["mix_weights"])
