"""M12: the chat fine-tune (spec 13) -- template, loss mask, masked data, the SFT
trainer path (--init-from, max_epochs, the budget backstop), the data builder and the
terminal chat. CPU only, fixtures only, no network: every Trainer is built with
device="cpu" and every run_main call passes --device cpu (CUDA_VISIBLE_DEVICES=""
does not hide the laptop's GPU)."""
from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import quipu.train as train_mod
from quipu import chat
from quipu.bpe import train_bpe
from quipu.config import load_config
from quipu.data import write_mask, write_shard
from quipu.decontam import Decontaminator, Problem
from quipu.eval import estimate_loss
from quipu.loader import IGNORE_INDEX, MaskedBlockStream
from quipu.train import EXIT_BUDGET, EXIT_OK, EXIT_USAGE, LATEST, Trainer, run_main

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"
CONTEXT = 64


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


bcd = _load("build_chat_data_m12", ROOT / "scripts" / "build_chat_data.py")
chat_script = _load("chat_script_m12", ROOT / "scripts" / "chat.py")

REPLY_FILE = ("=== FILE: app.py ===\nfrom flask import Flask\napp = Flask(__name__)\n"
              "=== END FILE ===\n")
CONVERSATION = [
    {"role": "system", "content": "You are quipu."},
    {"role": "user", "content": "Apa ibu kota Indonesia?"},
    {"role": "assistant", "content": "Jakarta adalah ibu kota Indonesia."},
    {"role": "user", "content": "东京在哪里？ And write app.py."},
    {"role": "assistant", "content": REPLY_FILE},
]
QA = [("What is two plus two?", "Two plus two is four."),
      ("Apa ibu kota Indonesia?", "Ibu kotanya Jakarta."),
      ("東京はどこですか。", "日本にあります。"),
      ("Say hello.", "Hello there!")]


def corpus() -> list[str]:
    texts = [f"{q}\n{a}\n" for q, a in QA] + [m["content"] for m in CONVERSATION]
    texts += ["def add(a, b):\n    return a + b\n", "The quick brown fox jumps.\n",
              "தமிழ் மொழி. सबसे अच्छा। Aap kaise ho? 你好，世界。\n"]
    return texts * 20


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return train_bpe(corpus(), 512, tmp_path_factory.mktemp("tok") / "tokenizer.json")


# ---- template ----------------------------------------------------------------------------

def test_render_is_the_export_template_and_the_standalone_loaders():
    jinja2 = pytest.importorskip("jinja2")
    export_hf = _load("export_hf_m12", ROOT / "scripts" / "export_hf.py")
    mq = _load("modeling_quipu_moe_m12", ROOT / "hf" / "modeling_quipu_moe.py")
    for gen in (False, True):
        hf = jinja2.Template(export_hf.CHAT_TEMPLATE).render(messages=CONVERSATION,
                                                             add_generation_prompt=gen)
        assert chat.render(CONVERSATION, add_generation_prompt=gen) == hf
    assert chat.render(CONVERSATION[:4], add_generation_prompt=True) == mq.render_chat(
        CONVERSATION[:4])


def test_render_tokenize_parse_round_trips(tok):
    ids, mask = chat.encode(tok, CONVERSATION)
    # The per-turn tokens are the tokenizer's tokens of the whole rendered string, so
    # training sees what the HF template and the standalone loader produce.
    assert ids == tok.encode_with_special(chat.render(CONVERSATION))
    assert len(mask) == len(ids)
    assert chat.parse(tok, ids) == CONVERSATION
    # trailing <|endoftext|> padding parses away; tokens after it do not
    eot = tok.special_id(chat.EOT)
    assert chat.parse(tok, ids + [eot, eot]) == CONVERSATION
    with pytest.raises(chat.ChatFormatError):
        chat.parse(tok, ids + [eot] + ids[:3])
    with pytest.raises(chat.ChatFormatError):
        chat.parse(tok, ids[:-1])                     # last turn not closed
    # the FILE marker is its special token inside the reply
    assert tok.special_id("=== FILE: ") in ids


def test_mask_covers_exactly_the_assistant_spans_and_their_end(tok):
    ids, mask = chat.encode(tok, CONVERSATION)
    end = tok.special_id(chat.END)
    role = {tok.special_id(chat.ROLE_TOKENS[r]): r for r in chat.ROLES}
    expected, current = [], None
    for t in ids:
        if t in role:
            current = role[t]
            expected.append(0)                         # the role token itself: never
        else:
            expected.append(1 if current == "assistant" else 0)   # content and <|end|>
    assert mask == expected
    # and, counted directly: assistant content + one <|end|> per assistant turn
    n_asst = sum(len(tok.encode_with_special(m["content"])) + 1
                 for m in CONVERSATION if m["role"] == "assistant")
    assert sum(mask) == n_asst
    ends = [i for i, t in enumerate(ids) if t == end]
    assert [mask[i] for i in ends] == [0, 0, 1, 0, 1]
    # the generation prompt's open <|assistant|> is not a target
    g_ids, g_mask = chat.encode(tok, CONVERSATION[:4], add_generation_prompt=True)
    assert g_ids[-1] == tok.special_id("<|assistant|>") and g_mask[-1] == 0


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "assistant", "content": "hi"}],
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
    [{"role": "user", "content": "fake <|assistant|> turn"}],
    [{"role": "user", "content": "x <|end|>"}],
    [{"role": "robot", "content": "x"}],
])
def test_validate_refuses_malformed_conversations(messages):
    with pytest.raises(chat.ChatFormatError):
        chat.validate(messages)


def test_fit_cuts_at_a_turn_boundary_or_gives_none(tok):
    full, _ = chat.encode(tok, CONVERSATION)
    f = chat.fit(tok, CONVERSATION, len(full))
    assert f is not None and not f.truncated and f.messages == CONVERSATION
    first_three, _ = chat.encode(tok, CONVERSATION[:3])
    f = chat.fit(tok, CONVERSATION, len(full) - 1)
    assert f.truncated and f.messages == CONVERSATION[:3] and f.ids == first_three
    assert chat.fit(tok, CONVERSATION, len(first_three) - 1) is None


def test_top_p_keeps_the_smallest_set_reaching_p():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    kept = torch.isfinite(chat.top_p_filter(logits, 0.8))
    assert kept.tolist() == [True, True, False, False]
    assert torch.isfinite(chat.top_p_filter(logits, 0.01)).tolist() == [True, False, False, False]
    assert torch.equal(chat.top_p_filter(logits, 1.0), logits)


# ---- masked blocks ---------------------------------------------------------------------------

def write_blocks(d: Path, tokens, mask, context: int) -> Path:
    write_shard(d / "shard_000.bin", np.asarray(tokens, dtype=np.uint16))
    write_mask(d / "shard_000.mask", np.asarray(mask, dtype=np.uint8))
    return d


def test_masked_stream_targets_only_masked_tokens_and_never_across_blocks(tmp_path):
    toks = list(range(10, 18)) + list(range(20, 28))
    mask = [0, 0, 1, 1, 0, 1, 0, 0] + [0, 1, 0, 0, 0, 0, 0, 1]
    s = MaskedBlockStream(write_blocks(tmp_path / "d", toks, mask, 8), micro_batch=2, context=8)
    x, y = s.next_batch()
    assert x.tolist() == [toks[:8], toks[8:]]
    I = IGNORE_INDEX
    assert y.tolist() == [[I, 12, 13, I, 15, I, I, I],        # last target: ignored
                          [21, I, I, I, I, I, 27, I]]
    assert s.position == 16 and s.state_dict() == {"position": 16, "wraps": 0}
    s.next_batch()
    assert s.wraps == 1                                        # wrapped to block 0


@pytest.mark.parametrize("problem", ["no_mask", "short_mask", "partial_block", "empty_block",
                                     "manifest_context"])
def test_masked_stream_refuses_bad_data(tmp_path, problem):
    d = tmp_path / "chat" / "train"
    toks, mask = list(range(16)), [0, 1] * 8
    if problem == "empty_block":
        mask = [0, 1] * 4 + [1] + [0] * 7                      # block 2: only position 0 set
    if problem == "partial_block":
        toks, mask = toks[:12], mask[:12]
    write_blocks(d, toks, mask, 8)
    if problem == "no_mask":
        (d / "shard_000.mask").unlink()
    if problem == "short_mask":
        write_mask(d / "shard_000.mask", np.zeros(15, dtype=np.uint8))
    if problem == "manifest_context":
        (d.parent / "manifest.json").write_text(json.dumps({"context": 16}), encoding="utf-8")
    with pytest.raises(ValueError):
        MaskedBlockStream(d, micro_batch=1, context=8)


def test_cross_entropy_ignores_masked_positions_by_hand():
    # A bigram "model": logits for the next token are a fixed row per current token.
    table = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    x = torch.tensor([[0, 1, 2, 1]])
    y = torch.tensor([[1, IGNORE_INDEX, 1, IGNORE_INDEX]])
    loss = F.cross_entropy(table[x].view(-1, 3), y.view(-1), ignore_index=IGNORE_INDEX)
    # by hand: position 0 (token 0 -> 1): -log(e^0 / (e^2 + 2)); position 2 (2 -> 1):
    # -log(e^0 / (e^3 + 2)); the mean of those two, the masked two not counted at all
    hand = (math.log(math.e ** 2 + 2) + math.log(math.e ** 3 + 2)) / 2
    assert loss.item() == pytest.approx(hand, rel=1e-6)


# ---- fixture chat data and the SFT trainer ------------------------------------------------------

def qa_rows(n: int = 24) -> list[dict]:
    return [{"inputs": QA[i % len(QA)][0] + f" ({i})", "targets": QA[i % len(QA)][1],
             "language": "English", "language_code": "eng"} for i in range(n)]


def build_sft_data(tmp_path: Path, tok, context: int = CONTEXT, val_fraction: float = 0.0,
                   rows=None) -> dict:
    return bcd.build({bcd.AYA: lambda st: bcd.aya_examples(rows or qa_rows(), st)},
                     tmp_path / "chat", tok=tok, context=context, val_fraction=val_fraction,
                     shard_tokens=10_000)


def sft_cfg(tmp_path: Path, tok, **train):
    over = {
        "model": {"vocab_size": tok.vocab_size, "context": CONTEXT},
        "data": {"tokenizer": str(tok.path), "shard_dir": str(tmp_path / "chat")},
        "train": {"ckpt_dir": str(tmp_path / "ckpt-sft"), "micro_batch": 2,
                  "batch_tokens": 2 * 2 * CONTEXT, "mode": "sft", "optimizer": "adamw",
                  "milestones": [], **train},
    }
    return load_config(SMOKE, over)


def sft_trainer(tmp_path, cfg, run_id="sft", val=True, resume=False):
    shard = Path(cfg.data.shard_dir)
    return Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=shard / "train",
                   val_dir=(shard / "val") if val else None, device="cpu",
                   run_dir=tmp_path / "runs", run_id=run_id, resume=resume)


def test_the_trainer_loss_is_the_mean_over_assistant_targets_only(tmp_path, tok):
    build_sft_data(tmp_path, tok)
    cfg = sft_cfg(tmp_path, tok, batch_tokens=2 * CONTEXT)       # grad_accum 1
    tr = sft_trainer(tmp_path, cfg, val=False)
    blocks = np.fromfile(tmp_path / "chat" / "train" / "shard_000.bin", dtype="<u2")
    masks = np.fromfile(tmp_path / "chat" / "train" / "shard_000.mask", dtype=np.uint8)
    x = torch.from_numpy(blocks[:2 * CONTEXT].astype(np.int64)).view(2, CONTEXT)
    m = torch.from_numpy(masks[:2 * CONTEXT].astype(bool)).view(2, CONTEXT)
    with torch.no_grad():
        logp = torch.log_softmax(tr.model(x).double(), -1)
    # by hand: -log p(next token) at every position whose NEXT token is an assistant
    # token, averaged; every other position (prompts, padding, the last) left out
    nll = [-logp[b, t, x[b, t + 1]] for b in range(2) for t in range(CONTEXT - 1)
           if m[b, t + 1]]
    assert 0 < len(nll) < 2 * (CONTEXT - 1)
    assert tr.train_step() == pytest.approx(float(sum(nll) / len(nll)), rel=1e-5)


def _pretrained_checkpoint(tmp_path, tok) -> Path:
    """A 'pretraining' checkpoint of the same model: a few plain steps on tokens."""
    cfg = sft_cfg(tmp_path, tok, mode="pretrain", ckpt_dir=str(tmp_path / "ckpt-pre"))
    data = tmp_path / "pre"
    write_shard(data / "shard_000.bin",
                np.random.RandomState(0).randint(0, tok.vocab_size, 5000).astype(np.uint16))
    tr = Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data, device="cpu",
                 run_dir=tmp_path / "runs", run_id="pre")
    for _ in range(3):
        tr.train_step()
    return tr.save_checkpoint()


@pytest.mark.parametrize("form", ["file", "dir", "pointer", "milestone"])
def test_init_from_loads_weights_with_a_fresh_optimizer_and_step_0(tmp_path, tok, form):
    ckpt = _pretrained_checkpoint(tmp_path, tok)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if form == "milestone":
        path = tmp_path / "milestone.pt"
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v
                    for k, v in saved["model"].items()}, path)       # as save_milestone
    else:
        path = {"file": ckpt, "dir": ckpt.parent, "pointer": ckpt.parent / LATEST}[form]
    build_sft_data(tmp_path, tok)
    tr = sft_trainer(tmp_path, sft_cfg(tmp_path, tok, warmup_steps=5), val=False)
    tr.init_weights_from(path)
    for k, v in tr.model.state_dict().items():
        want = saved["model"][k]
        if form == "milestone":
            want = want.to(torch.bfloat16).to(v.dtype) if v.is_floating_point() else want
        assert torch.equal(v, want), k
    assert tr.step == 0 and tr.stream.position == 0
    assert all(not opt.state for opt in tr.optimizers)          # no moments carried over
    tr.train_step()
    assert tr.step == 1
    log = json.loads((tmp_path / "runs" / "sft.json").read_text(encoding="utf-8"))
    assert log["steps"][0]["lr"] == pytest.approx(tr.train_cfg.lr / 5)   # schedule step 0


def test_init_from_refuses_weights_of_another_model(tmp_path, tok):
    ckpt = _pretrained_checkpoint(tmp_path, tok)
    build_sft_data(tmp_path, tok)
    cfg = sft_cfg(tmp_path, tok)
    other = dataclasses.replace(cfg.model, n_experts=4)
    tr = Trainer(model_cfg=other, train_cfg=cfg.train, shard_dir=tmp_path / "chat" / "train",
                 device="cpu", run_dir=tmp_path / "runs", run_id="x")
    with pytest.raises(train_mod.UsageError):
        tr.init_weights_from(ckpt)


def test_smoke_sft_of_30_steps_lowers_the_assistant_loss(tmp_path, tok):
    build_sft_data(tmp_path, tok, rows=qa_rows(40), val_fraction=0.0)
    # the held-out measure: the training blocks themselves through a second stream
    cfg = sft_cfg(tmp_path, tok, lr=3e-3, warmup_steps=3)
    tr = sft_trainer(tmp_path, cfg, val=False)
    probe = MaskedBlockStream(tmp_path / "chat" / "train", 2, CONTEXT)
    n = probe.n_blocks // 2
    before = estimate_loss(tr.model, probe, n, "cpu")
    losses = [tr.train_step() for _ in range(30)]
    after = estimate_loss(tr.model, probe, n, "cpu")
    assert all(math.isfinite(v) for v in losses)
    assert after < before - 1.0, (before, after)


def test_max_epochs_caps_the_run_to_the_data(tmp_path, tok):
    manifest = build_sft_data(tmp_path, tok)
    blocks = manifest["splits"]["train"]["blocks"]
    cfg = sft_cfg(tmp_path, tok, max_epochs=3.0, warmup_steps=2)
    tr = sft_trainer(tmp_path, cfg, val=False)
    # 3 passes over `blocks` blocks at 4 blocks a step
    assert tr.train_cfg.steps == math.floor(3 * blocks / 4) < cfg.train.steps
    note = json.loads((tmp_path / "runs" / "sft.json").read_text(encoding="utf-8"))
    assert note["config"]["train"]["total_tokens"] == tr.train_cfg.total_tokens
    with pytest.raises(train_mod.UsageError):
        sft_trainer(tmp_path, sft_cfg(tmp_path, tok, max_epochs=3.0,
                                      warmup_steps=100), run_id="too-short", val=False)


# ---- the process: exit codes, --init-from, the budget backstop ---------------------------------

def _sft_toml(tmp_path, tok) -> Path:
    """A real config file that inherits the smoke config (the SFT config's pattern)."""
    build_sft_data(tmp_path, tok)
    path = tmp_path / "sft.toml"
    path.write_text(
        f'name = "sft-test"\ninherit = "{SMOKE.as_posix()}"\n'
        f'inherit_if_present = ["winners-not-here.toml"]\n\n'
        f"[model]\nvocab_size = {tok.vocab_size}\ncontext = {CONTEXT}\n\n"
        f'[data]\ntokenizer = "{Path(tok.path).as_posix()}"\n'
        f'shard_dir = "{(tmp_path / "chat").as_posix()}"\n\n'
        f'[train]\nmode = "sft"\noptimizer = "adamw"\nbase_lr_scale = 0.1\nlr_min = 0.0\n'
        f"warmup_steps = 2\ntotal_tokens = {40 * 4 * CONTEXT}\nbatch_tokens = {4 * CONTEXT}\n"
        f"micro_batch = 2\nmilestones = []\nckpt_every = 5\neval_every = 1000\n"
        f'ckpt_dir = "{(tmp_path / "ckpt-sft").as_posix()}"\n',
        encoding="utf-8")
    return path


def _args(config: Path, *extra: str) -> list[str]:
    return ["--config", str(config), "--run-id", "sft", "--device", "cpu",
            "--run-dir", str(config.parent / "runs"), *extra]


def test_sft_without_init_from_is_a_usage_error_that_leaves_no_run_log(tmp_path, tok, capsys):
    config = _sft_toml(tmp_path, tok)
    assert run_main(_args(config)) == EXIT_USAGE
    assert "--init-from" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "sft.json").exists()
    assert run_main(_args(config, "--init-from", str(tmp_path / "missing"))) == EXIT_USAGE
    assert not (tmp_path / "runs" / "sft.json").exists()


class StepClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_sft_respects_the_budget_backstop_like_pretraining(tmp_path, tok, monkeypatch, capsys):
    ckpt = _pretrained_checkpoint(tmp_path, tok)
    config = _sft_toml(tmp_path, tok)
    clock = StepClock()
    monkeypatch.setattr(train_mod, "_clock", clock)
    real_step = Trainer.train_step

    def timed_step(self):
        loss = real_step(self)
        clock.t += 1.0
        return loss
    monkeypatch.setattr(Trainer, "train_step", timed_step)
    budget = ["--override", "train.budget_usd=0.0065", "--override", "train.usd_per_hour=3.6"]
    assert run_main(_args(config, "--init-from", str(ckpt.parent), *budget)) == EXIT_BUDGET
    record = json.loads((tmp_path / "runs" / "sft.json").read_text(encoding="utf-8"))
    assert record["status"] == "stopped_budget"
    assert [s["step"] for s in record["steps"]][-1] == 7
    assert (tmp_path / "ckpt-sft" / "step_000007.pt").exists()
    assert "budget backstop" in capsys.readouterr().err
    # the lr is 0.1x the inherited smoke lr, and the log says which files were merged
    assert record["config"]["train"]["lr"] == pytest.approx(0.1 * 1e-3)
    assert record["config_layers"][-1] == str(config)
    # A retry resumes the SFT's own checkpoint; --init-from is then ignored.
    out = capsys.readouterr()
    assert run_main(_args(config, "--init-from", str(ckpt.parent), "--resume",
                          "--override", "train.budget_usd=1.0",
                          "--override", "train.usd_per_hour=3.6")) == EXIT_OK
    assert "ignored" in capsys.readouterr().out
    record = json.loads((tmp_path / "runs" / "sft.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed" and record["resumes"][-1]["from_step"] == 7
    del out


# ---- config: inherit, base_lr_scale, mode -------------------------------------------------------

def test_the_sft_config_inherits_the_full_model_and_scales_the_lr(tmp_path):
    full = load_config(ROOT / "configs" / "quipu-moe.toml")
    sft = load_config(ROOT / "configs" / "quipu-moe-sft.toml")
    assert sft.model == full.model
    assert sft.data.tokenizer == full.data.tokenizer
    assert sft.train.precision == full.train.precision
    assert sft.train.optimizer == full.train.optimizer
    assert sft.train.lr == pytest.approx(0.1 * full.train.lr)
    assert sft.train.muon_lr == pytest.approx(0.1 * full.train.muon_lr)
    assert sft.train.lr_min == 0.0 and sft.train.mode == "sft" and sft.train.max_epochs == 3.0
    assert sft.train.total_tokens == 100_000_000 and sft.train.budget_usd > 0
    assert sft.data.aya_revision and sft.data.oasst2_revision


def test_winners_are_merged_when_present(tmp_path):
    base = tmp_path / "base.toml"
    shutil.copy(ROOT / "configs" / "quipu-moe.toml", base)
    sft = tmp_path / "sft.toml"
    text = (ROOT / "configs" / "quipu-moe-sft.toml").read_text(encoding="utf-8")
    sft.write_text(text.replace('inherit = "quipu-moe.toml"', 'inherit = "base.toml"')
                   .replace('"../results/ab/winners.toml"', '"winners.toml"'), encoding="utf-8")
    assert load_config(sft).train.optimizer == "adamw"
    (tmp_path / "winners.toml").write_text(
        '[train]\noptimizer = "muon"\nlr = 1.2e-3\nmuon_lr = 0.04\n\n'
        '[model]\nactivation = "situ_glu"\n', encoding="utf-8")
    cfg = load_config(sft)
    assert cfg.train.optimizer == "muon" and cfg.model.activation == "situ_glu"
    assert cfg.train.lr == pytest.approx(1.2e-4) and cfg.train.muon_lr == pytest.approx(4e-3)
    assert [Path(p).name for p in cfg.layers] == ["base.toml", "winners.toml", "sft.toml"]


def test_base_lr_scale_conflicts_and_mode_rules(tmp_path):
    sft = ROOT / "configs" / "quipu-moe-sft.toml"
    with pytest.raises(ValueError, match="base_lr_scale"):
        load_config(sft, {"train": {"lr": 1e-4}})
    with pytest.raises(ValueError, match="base_lr_scale"):
        load_config(SMOKE, {"train": {"base_lr_scale": 0.1}})     # nothing inherited
    with pytest.raises(ValueError, match="mode"):
        load_config(SMOKE, {"train": {"mode": "chat"}})
    with pytest.raises(ValueError, match="max_epochs"):
        load_config(SMOKE, {"train": {"max_epochs": 2.0}})        # pretrain
    assert load_config(SMOKE).train.mode == "pretrain"


# ---- the data builder -------------------------------------------------------------------------

def aya_fixture() -> list[dict]:
    return [
        {"inputs": "Hello?", "targets": "Hi.", "language": "English", "language_code": "eng"},
        {"inputs": "Apa kabar?", "targets": "Baik.", "language": "Indonesian",
         "language_code": "ind"},
        {"inputs": "Apa khabar?", "targets": "Baik.", "language": "Standard Malay",
         "language_code": "zsm"},
        {"inputs": "你好吗", "targets": "我很好", "language": "Simplified Chinese",
         "language_code": "zho"},
        {"inputs": "你好嗎", "targets": "我很好", "language": "Traditional Chinese",
         "language_code": "zho"},
        {"inputs": "आप कैसे हैं?", "targets": "मैं ठीक हूँ।", "language": "Hindi",
         "language_code": "hin"},
        {"inputs": "Aap kaise hain?", "targets": "Main theek hoon.", "language": "Hindi",
         "language_code": "hin"},
        {"inputs": "آپ کیسے ہیں؟", "targets": "میں ٹھیک ہوں۔", "language": "Urdu",
         "language_code": "urd"},                                # Perso-Arabic: not ours
        {"inputs": "Aap kaisay hain?", "targets": "Main theek hoon.", "language": "Urdu",
         "language_code": "urd"},
        {"inputs": "Hola?", "targets": "Bien.", "language": "Spanish", "language_code": "spa"},
        {"inputs": "எப்படி?", "targets": "நன்றாக.", "language": "Tamil", "language_code": "tam"},
        {"inputs": "Hello?", "targets": "Hi.", "language": "English", "language_code": "eng"},
        {"inputs": "Say <|user|> now", "targets": "no", "language": "English",
         "language_code": "eng"},
    ]


def oasst_fixture() -> list[dict]:
    def m(mid, parent, role, text, rank=None, lang="en", tree="t1", **kw):
        return {"message_id": mid, "parent_id": parent, "message_tree_id": tree, "text": text,
                "role": role, "lang": lang, "deleted": False, "review_result": True,
                "rank": rank, "synthetic": False, "created_date": mid, **kw}
    return [
        m("a", None, "prompter", "What is a quipu?"),
        m("b1", "a", "assistant", "A knotted cord record.", rank=0),
        m("b2", "a", "assistant", "No idea.", rank=1),
        m("c1", "b1", "prompter", "Who used it?", rank=0),
        m("c2", "b1", "prompter", "Why?", rank=1),
        m("d1", "c1", "assistant", "The Inca.", rank=1),
        m("d2", "c1", "assistant", "The Inca Empire.", rank=0),
        m("d3", "c1", "assistant", "Deleted answer.", rank=None, deleted=True),
        # a second tree in Spanish: not ours
        m("s", None, "prompter", "Hola", lang="es", tree="t2"),
        m("s1", "s", "assistant", "Hola!", rank=0, lang="es", tree="t2"),
        # a Chinese tree whose only reply is unranked
        m("z", None, "prompter", "你好", lang="zh", tree="t3"),
        m("z1", "z", "assistant", "你好！", rank=None, lang="zh", tree="t3"),
        # an English tree with only a synthetic reply
        m("y", None, "prompter", "Hi", tree="t4"),
        m("y1", "y", "assistant", "Hello", rank=0, tree="t4", synthetic=True),
    ]


def test_aya_language_filter_and_mapping():
    st = bcd.SourceStats()
    got = [(e.language, e.messages[0]["content"]) for e in bcd.aya_examples(aya_fixture(), st)]
    langs = [g[0] for g in got]
    assert langs == ["eng_Latn", "ind_Latn", "zsm_Latn", "zho_Hans", "zho_Hant", "hin_Deva",
                     "hin_Latn", "urd_Latn", "tam_Taml", "eng_Latn", "eng_Latn"]
    assert st.rows_read == 13 and st.drops == {"script": 1, "language": 1}
    assert set(langs) <= set(bcd.LANGUAGES)
    # Chinese without a variant name falls back to the characters
    assert bcd.map_language(bcd.AYA_LANGUAGES, "zho", "這個國家", "Chinese") == ("zho_Hant", None)
    assert bcd.map_language(bcd.AYA_LANGUAGES, "zho", "这个国家", None) == ("zho_Hans", None)


def test_oasst2_takes_the_top_ranked_path_in_our_languages():
    st = bcd.SourceStats()
    ex = list(bcd.oasst2_examples(oasst_fixture(), st))
    by_lang = {e.language: e.messages for e in ex}
    assert by_lang["eng_Latn"] == [
        {"role": "user", "content": "What is a quipu?"},
        {"role": "assistant", "content": "A knotted cord record."},     # rank 0, not b2
        {"role": "user", "content": "Who used it?"},                    # best-ranked follow-up
        {"role": "assistant", "content": "The Inca Empire."},           # rank 0 (d2)
    ]
    assert by_lang["zho_Hans"][1]["content"] == "你好！"                 # an only, unranked reply
    assert len(ex) == 2
    assert st.drops["tree_language"] == 1                               # Spanish
    assert st.drops["tree_no_ranked_reply"] == 1                        # synthetic only
    assert st.drops["message_unusable"] == 2                            # deleted + synthetic


def _step_row(repo, split, reply, commit="c0"):
    return {"repo": repo, "licence": "MIT", "tag": "flask", "commit": commit, "split": split,
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "STEP: add the app"},
                         {"role": "assistant", "content": reply}]}


def test_stepbuild_train_split_only_with_both_leakage_guards():
    from stepbuild.bench.acceptance import list_apps, load_reference
    from stepbuild.bench.run import LeakageGuard
    bench_reply = load_reference(list_apps()[0])[0]
    test_reply = ("=== FILE: backend/app.py ===\n" + "\n".join(
        f"def handler_{i}(request):\n    return {{'id': {i}, 'ok': True}}" for i in range(12))
        + "\n=== END FILE ===\n")
    clean = "=== FILE: notes.py ===\nNOTES = []\n=== END FILE ===\n"
    rows = [_step_row("o/a", "train", clean, "c1"),
            _step_row("o/b", "train", bench_reply, "c2"),               # a bench reference
            _step_row("o/c", "train", test_reply.replace("    ", "  "), "c3"),  # re-indented copy
            _step_row("o/d", "test", clean, "c4"),                      # not train
            _step_row("o/e", "validation", clean, "c5")]
    guard = bcd.HeldOutSplitGuard([_step_row("o/t", "test", test_reply)])
    st = bcd.SourceStats()
    ex = list(bcd.stepbuild_examples(rows, st, bench_guard=LeakageGuard(), test_guard=guard))
    assert [e.key for e in ex] == ["o/a@c1"] and ex[0].language == bcd.CODE
    assert ex[0].licence == "MIT"
    assert st.drops == {"leakage_bench": 1, "leakage_test_split": 1, "not_train_split": 2}


def test_build_decontaminates_dedupes_fits_and_counts_everything(tmp_path, tok):
    solution = ("def is_palindrome_number(value):\n    text = str(value)\n"
                "    return text == text[::-1] and len(text) > 0\n")
    decontam = Decontaminator([Problem("humaneval", "HumanEval/999",
                                       ("Check whether a number reads the same backwards.",
                                        solution), solution)])
    long_reply = " ".join(["panjang"] * 400)
    aya = aya_fixture() + [
        {"inputs": "Write it", "targets": "Here:\n" + solution, "language": "English",
         "language_code": "eng"},                                     # contaminated
        {"inputs": "Ceritakan", "targets": long_reply, "language": "Indonesian",
         "language_code": "ind"},                                     # too long
    ]
    rows = [_step_row("o/a", "train", REPLY_FILE, "c1")]
    manifest = bcd.build(
        {bcd.AYA: lambda st: bcd.aya_examples(aya, st),
         bcd.OASST2: lambda st: bcd.oasst2_examples(oasst_fixture(), st),
         bcd.STEPBUILD: lambda st: bcd.stepbuild_examples(rows, st, bench_guard=None,
                                                          test_guard=None)},
        tmp_path / "chat", tok=tok, context=CONTEXT, decontam=decontam, val_fraction=0.0,
        shard_tokens=10_000, provenance={bcd.AYA: {"dataset": bcd.AYA_DATASET, "revision": "r"}})
    aya_m = manifest["sources"]["aya"]
    assert aya_m["dataset"] == bcd.AYA_DATASET and aya_m["license"] == "Apache-2.0"
    assert aya_m["rows_read"] == 15
    assert aya_m["drops"] == {"decontam_humaneval": 1, "duplicate": 1, "language": 1,
                              "script": 1, "special_tokens": 1, "too_long": 1}
    assert aya_m["kept"]["train_conversations"] == 9
    assert aya_m["languages"]["zho_Hant"]["train_conversations"] == 1
    assert manifest["decontamination"]["dropped"] == {"humaneval": 1}
    assert manifest["sources"]["oasst2"]["kept"]["train_conversations"] == 2
    assert manifest["sources"]["stepbuild"]["licences"] == {"MIT": 1}
    kept = sum(s["kept"]["train_conversations"] for s in manifest["sources"].values())
    train = manifest["splits"]["train"]
    assert train["conversations"] == kept == 12
    per_source_tokens = sum(s["kept"]["train_tokens"] for s in manifest["sources"].values())
    assert train["conversation_tokens"] == per_source_tokens
    assert train["assistant_tokens"] == sum(s["kept"]["train_assistant_tokens"]
                                            for s in manifest["sources"].values())
    assert train["tokens"] == train["blocks"] * CONTEXT
    assert manifest["context"] == CONTEXT
    assert set(manifest["language_map"]) == {"aya", "oasst2", "stepbuild"}
    # the shards parse back into exactly the kept conversations (packing is lossless)
    ids = np.fromfile(tmp_path / "chat" / "train" / "shard_000.bin", dtype="<u2")
    mask = np.fromfile(tmp_path / "chat" / "train" / "shard_000.mask", dtype=np.uint8)
    eot = tok.special_id(chat.EOT)
    convs = []
    for b in range(len(ids) // CONTEXT):
        block = ids[b * CONTEXT:(b + 1) * CONTEXT].tolist()
        bm = mask[b * CONTEXT:(b + 1) * CONTEXT].tolist()
        cur, cm = [], []
        for t, k in zip(block + [eot], bm + [0]):
            if t == eot:
                if cur:
                    msgs = chat.parse(tok, cur)
                    assert chat.encode(tok, msgs) == (cur, cm)      # mask as encode made it
                    convs.append(msgs)
                cur, cm = [], []
            else:
                cur.append(t)
                cm.append(k)
    assert len(convs) == 12
    assert MaskedBlockStream(tmp_path / "chat" / "train", 1, CONTEXT).n_blocks == train["blocks"]


def test_val_split_and_per_source_val_dirs(tmp_path, tok):
    manifest = build_sft_data(tmp_path, tok, rows=qa_rows(60), val_fraction=0.3)
    val = manifest["splits"]["val"]
    assert 0 < val["conversations"] < 60
    assert val["conversations"] + manifest["splits"]["train"]["conversations"] == 60
    assert manifest["splits"]["val_by_source"]["aya"]["conversations"] == val["conversations"]
    assert (tmp_path / "chat" / "val_by_source" / "aya" / "shard_000.mask").is_file()
    # the split is a function of the example alone: a rebuild puts the same ones there
    again = build_sft_data(tmp_path / "again", tok, rows=qa_rows(60), val_fraction=0.3)
    assert again["splits"]["val"]["conversations"] == val["conversations"]


def test_no_pack_puts_one_conversation_per_block(tmp_path, tok):
    manifest = bcd.build({bcd.AYA: lambda st: bcd.aya_examples(qa_rows(8), st)},
                         tmp_path / "chat", tok=tok, context=CONTEXT, val_fraction=0.0,
                         packed=False)
    assert manifest["splits"]["train"]["blocks"] == 8
    packed = bcd.build({bcd.AYA: lambda st: bcd.aya_examples(qa_rows(8), st)},
                       tmp_path / "chat2", tok=tok, context=CONTEXT, val_fraction=0.0)
    assert packed["splits"]["train"]["blocks"] < 8


# ---- the terminal chat ------------------------------------------------------------------------

class ScriptedModel(nn.Module):
    """Emits a fixed token sequence, one per call, whatever the input."""

    def __init__(self, script: list[int], vocab: int) -> None:
        super().__init__()
        self.script, self.vocab, self.calls = script, vocab, 0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab)
        logits[:, -1, self.script[min(self.calls, len(self.script) - 1)]] = 10.0
        self.calls += 1
        return logits


def test_generation_stops_at_end(tok):
    hello = tok.encode("Hello there!")
    after = tok.encode(" garbage after the end")
    model = ScriptedModel(hello + [tok.special_id(chat.END)] + after, tok.vocab_size)
    prompt, _ = chat.encode(tok, [{"role": "user", "content": "Say hello."}],
                            add_generation_prompt=True)
    seen = []
    ids, why = chat.generate_reply(model, prompt, chat.stop_ids(tok), max_new_tokens=50,
                                   context=CONTEXT, on_token=seen.append)
    assert ids == hello == seen and why == "stop"
    assert model.calls == len(hello) + 1                # nothing generated after <|end|>
    model = ScriptedModel(hello * 10, tok.vocab_size)
    ids, why = chat.generate_reply(model, prompt, chat.stop_ids(tok), max_new_tokens=5,
                                   context=CONTEXT)
    assert len(ids) == 5 and why == "length"


def test_chat_loop_prints_the_reply_up_to_end_and_keeps_the_history(tok):
    hello = tok.encode("Hello there!")
    model = ScriptedModel(hello + [tok.special_id(chat.END)] + tok.encode(" junk"),
                          tok.vocab_size)
    out = io.StringIO()
    messages = chat_script.run_chat(model, tok, context=CONTEXT, lines=["Say hello.", "/quit",
                                                                         "never read"],
                                    out=out, temperature=0.0, max_new_tokens=20,
                                    system="Be brief.")
    assert "quipu> Hello there!\n" in out.getvalue() and "junk" not in out.getvalue()
    assert messages == [{"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello there!"}]


def test_chat_drops_the_oldest_turns_to_fit(tok):
    msgs = [{"role": "system", "content": "S"}]
    for q, a in QA:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "last question"})
    full, _ = chat.encode(tok, msgs, add_generation_prompt=True)
    ids = chat_script.prompt_ids(tok, msgs, len(full) - 1)
    back = chat.parse(tok, ids[:-1])               # minus the open <|assistant|>
    assert back[0]["content"] == "S" and back[-1]["content"] == "last question"
    assert len(back) < len(msgs) and len(ids) <= len(full) - 1


def test_chat_answers_the_same_from_a_checkpoint_and_the_hf_export(tmp_path, tok):
    from safetensors.torch import save_file
    from quipu.eval import loop_dispatch
    ckpt = _pretrained_checkpoint(tmp_path, tok)
    cfg = sft_cfg(tmp_path, tok)
    ckpt_model, ckpt_tok, context = chat_script.load_checkpoint(str(_cfg_file(tmp_path, cfg)),
                                                                str(ckpt), "cpu")
    folder = tmp_path / "hf"
    folder.mkdir()
    shutil.copy(ROOT / "hf" / "modeling_quipu_moe.py", folder)
    shutil.copy(tok.path, folder / "tokenizer.json")
    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(cfg.model)),
                                        encoding="utf-8")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    save_file({k: v.float().contiguous().clone() for k, v in state.items()
               if k != "lm_head.weight"}, str(folder / "model.safetensors"))
    hf_model, hf_tok, hf_context = chat_script.load_hf(folder, "cpu")
    assert hf_context == context == CONTEXT
    replies = []
    for model, t, wrap in ((ckpt_model, ckpt_tok, lambda: loop_dispatch(ckpt_model)),
                           (hf_model, hf_tok, None)):
        kw = {"wrap": wrap} if wrap else {}
        msgs = chat_script.run_chat(model, t, context=context, lines=["Say hello."],
                                    out=io.StringIO(), temperature=0.0, max_new_tokens=8, **kw)
        replies.append(msgs[-1]["content"])
    assert replies[0] == replies[1]


def _cfg_file(tmp_path: Path, cfg) -> Path:
    """A config file holding `cfg` (smoke + the test's overrides)."""
    path = tmp_path / "cfg.toml"
    path.write_text(
        f'name = "c"\ninherit = "{SMOKE.as_posix()}"\n\n'
        f"[model]\nvocab_size = {cfg.model.vocab_size}\ncontext = {cfg.model.context}\n\n"
        f'[data]\ntokenizer = "{Path(cfg.data.tokenizer).as_posix()}"\n\n'
        f'[train]\nckpt_dir = "{Path(cfg.train.ckpt_dir).as_posix()}"\nmicro_batch = 2\n'
        f"batch_tokens = {cfg.train.batch_tokens}\n", encoding="utf-8")
    return path
