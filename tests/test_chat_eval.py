"""The chat card's samples (scripts/chat_eval.py) and the chat card itself: samples
from the chat model at temperature 0, never the base model's; held-out loss per
source; the cost row's wording (CPU only, tiny model)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from quipu import model_card
from tests.moe_fixtures import fresh_model, save_final, tiny_moe

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "hf" / "README_moe.md").read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("chat_eval", ROOT / "scripts" / "chat_eval.py")
chat_eval = importlib.util.module_from_spec(_spec)
sys.modules["chat_eval"] = chat_eval
_spec.loader.exec_module(chat_eval)


def test_every_language_has_two_fixed_prompts():
    assert len(chat_eval.CHAT_PROMPTS) == 11
    assert all(len(p) == 2 and all(q.strip() for q in p) for p in chat_eval.CHAT_PROMPTS.values())
    assert {"eng_Latn", "zho_Hans", "zho_Hant", "urd_Latn"} <= set(chat_eval.CHAT_PROMPTS)


def test_chat_eval_writes_greedy_samples_and_the_held_out_loss(tmp_path):
    cfg, tok = tiny_moe(tmp_path)
    path = save_final(cfg, 7, fresh_model(cfg))
    held = tmp_path / "val_by_source.json"
    held.write_text(json.dumps({"step": 7, "sources": {"aya": {"loss": 2.5}}}))
    prompts = {"eng_Latn": ("Hi?", "Why?"), "ind_Latn": ("Apa?", "Kenapa?")}
    out = tmp_path / "sft" / "chat_samples.json"
    a = chat_eval.run(cfg, path, "cpu", out, held, prompts=prompts, max_new_tokens=4)
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got == json.loads(json.dumps(a, ensure_ascii=False))
    assert got["model"] == "chat" and got["decoding"]["temperature"] == 0.0
    assert [r["prompt"] for r in got["samples"]["eng_Latn"]] == ["Hi?", "Why?"]
    for rows in got["samples"].values():
        for r in rows:
            assert r["finish"] in ("stop", "length") and r["reply_tokens"] <= 4
    assert got["val_by_source"]["sources"]["aya"]["loss"] == 2.5
    # Greedy: the same checkpoint gives the same replies.
    b = chat_eval.run(cfg, path, "cpu", tmp_path / "again.json", None, prompts=prompts,
                      max_new_tokens=4)
    assert b["samples"] == a["samples"] and b["val_by_source"] is None


def _card(tmp_path, cfg, variant, **inp):
    f = model_card.fields(cfg, model_card.CardInputs(**inp), variant=variant)
    return f, model_card.render(TEMPLATE, f)


def _base_samples(tmp_path) -> Path:
    ms = tmp_path / "milestones"
    ms.mkdir()
    (ms / "samples.json").write_text(json.dumps({"def fibonacci(n):": [
        {"label": "final", "step": 9, "greedy": "BASE MODEL CONTINUATION", "sampled": None}]}))
    return ms


def test_the_chat_card_never_shows_the_base_models_samples(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    f, card = _card(tmp_path, cfg, "chat", milestones_dir=_base_samples(tmp_path),
                    sft_dir=tmp_path / "no-sft")
    assert "BASE MODEL CONTINUATION" not in card
    assert f["samples"].startswith(model_card.NOT_MEASURED)
    # The base card still shows them.
    _, base = _card(tmp_path / "b", cfg, "base", milestones_dir=tmp_path / "milestones")
    assert "BASE MODEL CONTINUATION" in base


def test_the_chat_card_shows_the_chat_samples_and_loss_per_source(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "chat_samples.json").write_text(json.dumps({
        "model": "chat", "checkpoint": "checkpoints/quipu-moe-sft/step_000120.pt",
        "decoding": {"temperature": 0.0, "max_new_tokens": 200},
        "samples": {"eng_Latn": [{"prompt": "What is photosynthesis?",
                                  "reply": "It is how plants make food.", "finish": "stop",
                                  "reply_tokens": 8}],
                    "zho_Hans": [{"prompt": "中国的首都是哪里？", "reply": "北京。",
                                  "finish": "length", "reply_tokens": 200}]},
        "val_by_source": {"step": 120, "sources": {"aya": {"loss": 2.3456,
                                                           "assistant_tokens": 1000},
                                                   "oasst2": {"loss": 1.9,
                                                              "assistant_tokens": 500}}}},
        ensure_ascii=False), encoding="utf-8")
    f, card = _card(tmp_path, cfg, "chat", milestones_dir=_base_samples(tmp_path), sft_dir=sft)
    assert "BASE MODEL CONTINUATION" not in card
    assert "It is how plants make food." in card and "北京。" in card
    assert "What is photosynthesis?" in card and "temperature 0" in card
    assert "| aya | 2.3456 | 1,000 |" in card and "| oasst2 | 1.9000 | 500 |" in card
    assert "cut at 200 tokens" in card
    assert "### Chat samples" in card


def test_the_held_out_loss_alone_is_read_from_val_by_source(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "val_by_source.json").write_text(json.dumps(
        {"sources": {"stepbuild": {"loss": 1.25, "assistant_tokens": 42}}}))
    f, card = _card(tmp_path, cfg, "chat", sft_dir=sft)
    assert "| stepbuild | 1.2500 | 42 |" in card
    assert "Chat samples: " + model_card.NOT_MEASURED in card


def test_the_cost_row_says_hourly_gpu_session_without_bandwidth(tmp_path):
    from quipu.spend import Ledger

    cfg, _ = tiny_moe(tmp_path)
    clock = [1000.0]
    led = Ledger.load(tmp_path / "spend.json", clock=lambda: clock[0])
    led.ensure_session(0.6)
    clock[0] += 3600
    led.tick()
    _, card = _card(tmp_path, cfg, "base", ledger=tmp_path / "spend.json")
    row = next(ln for ln in card.splitlines() if ln.startswith("| Cost |"))
    assert "$0.60" in row
    assert "GPU session (hourly)" in row and "excludes Vast bandwidth charges" in row
