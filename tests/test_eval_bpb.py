"""M10: evaluation splits, bits per byte per language, and milestone_eval on quipu-moe.

CPU only (device="cpu" everywhere): CUDA_VISIBLE_DEVICES="" does not hide the GPU on
the development laptop.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from quipu import evalsets
from quipu.data import write_shard
from quipu.eval import nll_tokens_bytes
from tests.moe_fixtures import (fresh_model, save_final, save_milestone, tiny_moe,
                                write_eval_shards)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("milestone_eval_m10",
                                               ROOT / "scripts" / "milestone_eval.py")
milestone_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["milestone_eval_m10"] = milestone_eval
_SPEC.loader.exec_module(milestone_eval)


# ---- splits and batches --------------------------------------------------------------

def test_eval_splits_names_and_orders_every_split(tmp_path):
    for sub in ["val", "code_val", "val_lang/tam_Taml", "val_lang/ind_Latn",
                "val_lang/zho_Hant", "val_lang/xyz_Test"]:
        write_shard(tmp_path / sub / "shard_000.bin", np.arange(10, dtype=np.uint16))
    (tmp_path / "val_lang" / "empty_Dir").mkdir()
    got = evalsets.eval_splits(tmp_path)
    assert list(got) == ["eng_Latn", "ind_Latn", "zho_Hant", "tam_Taml", "xyz_Test", "code"]
    assert got["eng_Latn"] == tmp_path / "val" and got["code"] == tmp_path / "code_val"


def test_eval_splits_of_a_dense_shard_set_are_english_and_code(tmp_path):
    for sub in ["val", "code_val", "train"]:
        write_shard(tmp_path / sub / "shard_000.bin", np.arange(10, dtype=np.uint16))
    assert list(evalsets.eval_splits(tmp_path)) == ["eng_Latn", "code"]


def test_split_batches_are_contiguous_shifted_and_never_wrap(tmp_path):
    write_shard(tmp_path / "s" / "shard_000.bin", np.arange(0, 30, dtype=np.uint16))
    write_shard(tmp_path / "s" / "shard_001.bin", np.arange(30, 50, dtype=np.uint16))
    batches = evalsets.split_batches(tmp_path / "s", micro_batch=2, context=8, max_batches=10)
    xs = torch.cat([x for x, _ in batches])
    ys = torch.cat([y for _, y in batches])
    assert xs.shape == (6, 8)                       # 49 targets // 8 = 6 rows, no wrap
    assert xs.flatten().tolist() == list(range(48))
    assert ys.flatten().tolist() == list(range(1, 49))
    assert [x.shape[0] for x, _ in batches] == [2, 2, 2]
    few = evalsets.split_batches(tmp_path / "s", micro_batch=2, context=8, max_batches=1)
    assert len(few) == 1 and few[0][0].flatten().tolist() == list(range(16))


def test_split_batches_of_a_short_split_give_one_short_row(tmp_path):
    write_shard(tmp_path / "s" / "shard_000.bin", np.arange(5, dtype=np.uint16))
    [(x, y)] = evalsets.split_batches(tmp_path / "s", micro_batch=4, context=64, max_batches=3)
    assert x.tolist() == [[0, 1, 2, 3]] and y.tolist() == [[1, 2, 3, 4]]


# ---- bits per byte per language --------------------------------------------------------

def test_bits_per_byte_per_language_matches_a_hand_computation(tmp_path):
    cfg, tok = tiny_moe(tmp_path)
    write_eval_shards(cfg, tok, tokens=200)
    model = fresh_model(cfg, 1).eval()
    lens = tok.token_byte_lengths()
    got = milestone_eval.evaluate_bpb(model, cfg, "cpu", batches=2, byte_lengths=lens)
    assert list(got) == ["eng_Latn", "ind_Latn", "zho_Hans", "tam_Taml", "code"]

    for label, split in evalsets.eval_splits(cfg.data.shard_dir).items():
        # By hand: the first 2 x micro_batch x context + 1 tokens, rows of `context`,
        # summed cross-entropy in bits over the targets' UTF-8 bytes.
        ctx, mb = cfg.model.context, cfg.train.micro_batch
        t = np.fromfile(split / "shard_000.bin", dtype="<u2").astype(np.int64)[: 2 * mb * ctx + 1]
        rows = (len(t) - 1) // ctx
        x = torch.from_numpy(t[: rows * ctx]).view(rows, ctx)
        y = torch.from_numpy(t[1 : rows * ctx + 1]).view(rows, ctx)
        model.set_dispatch("loop")
        with torch.no_grad():
            nll = sum(F.cross_entropy(model(x[i : i + 1])[0], y[i], reduction="sum").item()
                      for i in range(rows))
        text_bytes = sum(len(tok.decode([i]).encode("utf-8")) for i in y.flatten().tolist()
                         if i != tok.eot)
        # Byte-level BPE: a token can hold part of a character, so decode() of one
        # token is not always its raw bytes; this fixture's tokens are checked below.
        n_bytes = sum(lens[i] for i in y.flatten().tolist())
        assert got[label]["tokens"] == rows * ctx
        assert got[label]["bytes"] == n_bytes
        assert got[label]["bpb"] == pytest.approx(nll / (n_bytes * math.log(2)), rel=1e-5)
        assert got[label]["loss"] == pytest.approx(nll / (rows * ctx), rel=1e-5)
        if label in ("eng_Latn", "code"):   # ASCII: every token decodes to its bytes
            assert text_bytes == n_bytes


def test_nll_tokens_bytes_restores_the_padded_dispatch(tmp_path):
    cfg, tok = tiny_moe(tmp_path, moe_dispatch="padded")
    model = fresh_model(cfg, 2)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    nll, n, b = nll_tokens_bytes(model, [(x, x)], tok.token_byte_lengths(), "cpu")
    assert n == 32 and nll > 0 and b > 0
    assert all(block.moe.dispatch == "padded" for block in model.blocks)


# ---- milestone_eval on a tiny quipu-moe ------------------------------------------------

@pytest.fixture
def moe_run(tmp_path, monkeypatch):
    monkeypatch.setattr(milestone_eval, "MAX_NEW_TOKENS", 6)
    cfg, tok = tiny_moe(tmp_path)
    write_eval_shards(cfg, tok, tokens=300)
    save_milestone(cfg, 10, fresh_model(cfg, 1))
    save_final(cfg, 20, fresh_model(cfg, 2))
    return cfg, tok


def test_milestone_eval_runs_on_a_tiny_moe_and_writes_bpb_per_language(tmp_path, moe_run):
    cfg, _ = moe_run
    out = tmp_path / "out"
    assert milestone_eval.run(cfg, device="cpu", eval_batches=2, out_dir=out) is True
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["checkpoints"]
    assert [m["label"] for m in metrics] == ["step_000010", "final"]
    for m in metrics:
        assert "error" not in m
        assert math.isfinite(m["text_val_loss"]) and math.isfinite(m["code_val_loss"])
        assert set(m["bpb"]) == {"eng_Latn", "ind_Latn", "zho_Hans", "tam_Taml", "code"}
        assert all(v["bpb"] > 0 and v["bytes"] > 0 for v in m["bpb"].values())
    table = (out / "bpb.md").read_text(encoding="utf-8")
    assert "| checkpoint | step | English | Indonesian | Chinese (Simplified) | Tamil | Code |" in table
    assert "| final | 20 |" in table


def test_milestone_eval_samples_two_prompts_per_language_greedy(tmp_path, moe_run):
    cfg, _ = moe_run
    out = tmp_path / "out"
    milestone_eval.run(cfg, device="cpu", eval_batches=1, out_dir=out)
    samples = (out / "samples.md").read_text(encoding="utf-8")
    assert len(milestone_eval.LANG_PROMPTS) == 10
    assert all(len(v) == 2 for v in milestone_eval.LANG_PROMPTS.values())
    for prompt in milestone_eval.ALL_PROMPTS + milestone_eval.GREEDY_ONLY_PROMPTS:
        assert f"## Prompt: `{prompt}`" in samples
    # Language prompts are greedy only; the quipu-114m prompts keep their sample.
    lang_section = samples[samples.index(f"`{milestone_eval.GREEDY_ONLY_PROMPTS[0]}`"):]
    assert "**Sampled" not in lang_section
    assert samples.count("**Sampled") == 2 * len(milestone_eval.ALL_PROMPTS)


def test_milestone_eval_greedy_is_reproducible_and_leaves_padded_dispatch_alone(tmp_path, moe_run):
    cfg, _ = moe_run
    a, b = tmp_path / "a", tmp_path / "b"
    milestone_eval.run(cfg, device="cpu", eval_batches=1, out_dir=a)
    milestone_eval.run(cfg, device="cpu", eval_batches=1, out_dir=b)
    assert (a / "samples.md").read_text(encoding="utf-8") == (b / "samples.md").read_text(encoding="utf-8")


def test_milestone_eval_greedy_uses_loop_dispatch(tmp_path, moe_run):
    cfg, tok = moe_run
    model = fresh_model(cfg, 3)
    model.set_dispatch("padded")
    seen = []
    orig = type(model.blocks[0].moe.experts).forward_loop

    def spy(self, *a, **k):
        seen.append("loop")
        return orig(self, *a, **k)

    type(model.blocks[0].moe.experts).forward_loop = spy
    try:
        milestone_eval.greedy_generate(model, torch.tensor([tok.encode("def area")]), 2, "cpu")
    finally:
        type(model.blocks[0].moe.experts).forward_loop = orig
    assert seen and all(b.moe.dispatch == "padded" for b in model.blocks)


def test_milestone_eval_main_on_a_run_config_file(tmp_path, moe_run):
    """run_moe.py calls: milestone_eval.py --config run_config.toml --out-dir D --device X."""
    cfg, _ = moe_run
    raw = (ROOT / "configs" / "quipu-moe-smoke.toml").read_text(encoding="utf-8")
    lines = []
    section = None
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s
        key = s.split("=")[0].strip() if "=" in s else None
        if section == "[model]" and key == "vocab_size":
            line = f"vocab_size = {cfg.model.vocab_size}"
        elif section == "[model]" and key == "context":
            line = "context = 64"
        elif section == "[data]" and key == "tokenizer":
            line = f"tokenizer = {json.dumps(cfg.data.tokenizer)}"
        elif section == "[data]" and key == "shard_dir":
            line = f"shard_dir = {json.dumps(cfg.data.shard_dir)}"
        elif section == "[train]" and key == "ckpt_dir":
            line = f"ckpt_dir = {json.dumps(cfg.train.ckpt_dir)}"
        elif section == "[train]" and key == "micro_batch":
            line = "micro_batch = 2"
        lines.append(line)
    run_config = tmp_path / "run_config.toml"
    run_config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = milestone_eval.main(["--config", str(run_config), "--out-dir", str(tmp_path / "m"),
                                "--device", "cpu", "--eval-batches", "1"])
    assert code == 0
    assert (tmp_path / "m" / "bpb.md").is_file()


# ---- needle_eval on a tiny quipu-moe -------------------------------------------------------

def test_needle_eval_runs_on_a_tiny_moe_with_the_bpe_tokenizer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("needle_eval_m10", ROOT / "scripts" / "needle_eval.py")
    needle_eval = importlib.util.module_from_spec(spec)
    sys.modules["needle_eval_m10"] = needle_eval
    spec.loader.exec_module(needle_eval)
    # The fixture's 400-token BPE spends ~36 tokens on a needle sentence (the real 49k
    # tokenizer about as many as GPT-2), so the reserve is raised for this test only.
    monkeypatch.setattr(needle_eval, "NEEDLE_RESERVE", 64)
    cfg, tok = tiny_moe(tmp_path, context=256, moe_dispatch="padded")
    write_eval_shards(cfg, tok, tokens=600)
    save_final(cfg, 3, fresh_model(cfg, 1))
    monkeypatch.setattr(needle_eval, "load_config", lambda _p: cfg)
    out = tmp_path / "needle"
    code = needle_eval.main(["--config", "x.toml", "--sizes", "128", "--depths", "0,100",
                             "--trials", "1", "--conditions", "off,bm25", "--device", "cpu",
                             "--out", str(out), "--chunk-tokens", "16"])
    assert code == 0
    assert (out / "text_128.json").is_file() and (out / "summary.md").is_file()
