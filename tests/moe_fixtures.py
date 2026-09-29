"""Tiny quipu-moe fixtures for the M10 evaluation and export tests (CPU only).

tiny_moe(tmp_path) trains a small BPE on a multilingual fixture corpus, points the
smoke config at it (vocab_size set to what the tokenizer holds), and returns
(cfg, tokenizer). write_eval_shards() writes val / val_lang / code_val splits encoded
with that tokenizer; save_milestone() / save_final() write checkpoints the way
quipu.train does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import quipu.train as train_mod
from quipu.bpe import train_bpe
from quipu.config import Config, load_config
from quipu.data import write_shard
from quipu.model_factory import build_model
from quipu.train import LATEST

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"

TEXTS = {
    "eng_Latn": "The quipu was a recording device made of knotted cords. Each cord "
                "could carry several knots, and the position of a knot mattered.\n",
    "ind_Latn": "Sejarah kota Jakarta dimulai dari pelabuhan kecil. Fotosintesis adalah "
                "proses yang dilakukan tumbuhan.\n",
    "zho_Hans": "光合作用是植物利用阳光制造养分的过程。北京是中国的首都。\n",
    "jpn_Jpan": "日本の四季は美しい。東京で一番有名な場所はどこですか。\n",
    "tam_Taml": "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று. சென்னை நகரம் பெரியது.\n",
    "code": "def area(width, height):\n    return width * height\n\n"
            "for i in range(10):\n    print(area(i, i + 1))\n",
}


def corpus() -> list[str]:
    return [t for t in TEXTS.values()] * 30


def tiny_moe(tmp_path: Path, **model_over) -> tuple[Config, object]:
    tok = train_bpe(corpus(), 400, tmp_path / "tokenizer.json")
    over = {
        "model": {"vocab_size": tok.vocab_size, "context": 64, **model_over},
        "data": {"tokenizer": str(tmp_path / "tokenizer.json"),
                 "shard_dir": str(tmp_path / "shards")},
        "train": {"ckpt_dir": str(tmp_path / "ckpt"), "micro_batch": 2},
    }
    return load_config(SMOKE, over), tok


def encode_split(tok, text: str, tokens: int) -> np.ndarray:
    ids: list[int] = []
    while len(ids) < tokens:
        ids += tok.encode(text) + [tok.eot]
    return np.array(ids[:tokens], dtype=np.uint16)


def write_eval_shards(cfg: Config, tok, tokens: int = 300,
                      langs: tuple[str, ...] = ("ind_Latn", "zho_Hans", "tam_Taml")) -> Path:
    root = Path(cfg.data.shard_dir)
    write_shard(root / "val" / "shard_000.bin", encode_split(tok, TEXTS["eng_Latn"], tokens))
    write_shard(root / "code_val" / "shard_000.bin", encode_split(tok, TEXTS["code"], tokens))
    for lang in langs:
        write_shard(root / "val_lang" / lang / "shard_000.bin",
                    encode_split(tok, TEXTS[lang], tokens))
    return root


def fresh_model(cfg: Config, seed: int = 0):
    torch.manual_seed(seed)
    model = build_model(cfg.model)
    # Non-zero balancer biases, so a checkpoint that drops them would be caught.
    for block in model.blocks:
        block.moe.balancer.bias.copy_(torch.randn_like(block.moe.balancer.bias) * 0.01)
    return model


def save_milestone(cfg: Config, step: int, model) -> Path:
    path = Path(cfg.train.ckpt_dir) / "milestones" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    train_mod._atomic_save(train_mod._bf16_state_dict(model), path)
    return path


def save_final(cfg: Config, step: int, model) -> Path:
    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    train_mod._atomic_save({"step": step, "model": model.state_dict(), "optimizers": []}, path)
    train_mod._atomic_save({"file": path.name}, ckpt_dir / LATEST)
    return path
