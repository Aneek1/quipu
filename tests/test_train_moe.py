"""M6: the trainer on quipu-moe (and the dense model through the same path).

Every Trainer here is built with device="cpu" explicitly: CUDA_VISIBLE_DEVICES=""
does not hide the GPU on the development laptop, and these tests must not touch it.
"""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import quipu.train as train_mod
from quipu.config import ModelConfig, TrainConfig, load_config
from quipu.data import write_shard
from quipu.eval import bits_per_byte, estimate_loss
from quipu.loader import TokenStream
from quipu.model_moe import QuipuMoE
from quipu.moe import MoELayer, QuantileBalancer
from quipu.optim import Muon
from quipu.train import Trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"


# ---- helpers ---------------------------------------------------------------------

def smoke(tmp_path, optimizer="muon", model=None, **train):
    """The smoke config, pointed at tmp_path. The tokenizer path is left as it is
    (it does not exist in the repo, so the vocab check is skipped)."""
    over = {
        "data": {"shard_dir": str(tmp_path / "shards")},
        "train": {"ckpt_dir": str(tmp_path / "ckpt"), "optimizer": optimizer, **train},
    }
    if model:
        over["model"] = model
    return load_config(SMOKE, over)


def make_data(tmp_path, tokens=20_000, name="data", seed=0):
    d = tmp_path / name
    write_shard(d / "shard_000.bin",
                np.random.RandomState(seed).randint(0, 512, tokens).astype(np.uint16))
    return d


def build(tmp_path, cfg, data, resume=False, run_id="t", val=None):
    return Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=data,
                   device="cpu", run_dir=tmp_path / "runs", run_id=run_id,
                   resume=resume, val_dir=val)


def logged(tmp_path, run_id="t") -> dict:
    return json.loads((tmp_path / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))


def biases(model):
    return [b.moe.balancer.bias.clone() for b in model.blocks]


# ---- training ---------------------------------------------------------------------

@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_smoke_config_trains_30_steps_with_loss_decreasing(tmp_path, optimizer):
    # 3000 tokens hold under three 1025-token micro-batches: the stream wraps and the
    # model sees the same text again and again, so there is something to learn.
    cfg = smoke(tmp_path, optimizer)
    trainer = build(tmp_path, cfg, make_data(tmp_path, tokens=3000))
    assert isinstance(trainer.model, QuipuMoE)
    kinds = [type(o) for o in trainer.optimizers]
    assert kinds == ([Muon, torch.optim.AdamW] if optimizer == "muon" else [torch.optim.AdamW])
    losses = [trainer.train_step() for _ in range(30)]
    assert all(math.isfinite(x) for x in losses)
    assert trainer.stream.wraps > 0
    assert np.mean(losses[-5:]) < losses[0] - 0.5, losses


def test_every_optimizer_follows_the_same_schedule_factor(tmp_path):
    cfg = smoke(tmp_path, "muon")
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    for _ in range(3):
        trainer.train_step()
    factor = trainer.lr_at(2) / cfg.train.lr        # the lr of the last step taken
    muon, adam = trainer.optimizers
    assert all(g["lr"] == pytest.approx(cfg.train.muon_lr * factor) for g in muon.param_groups)
    assert all(g["lr"] == pytest.approx(cfg.train.lr * factor) for g in adam.param_groups)
    assert logged(tmp_path)["steps"][-1]["muon_lr"] == pytest.approx(cfg.train.muon_lr * factor)


def test_the_balancer_moves_after_each_optimizer_step(tmp_path):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    before = biases(trainer.model)
    trainer.train_step()
    after = biases(trainer.model)
    assert all(not torch.equal(a, b) for a, b in zip(before, after))
    # The bias is a buffer: no optimizer holds it.
    held = {id(p) for o in trainer.optimizers for g in o.param_groups for p in g["params"]}
    assert all(id(b.moe.balancer.bias) not in held for b in trainer.model.blocks)


# ---- grad accumulation: the whole step, not the last micro-batch --------------------

def _record_micro_batches(monkeypatch):
    """Record every MoELayer forward's counts, drops and router scores."""
    seen = []
    real = MoELayer.forward

    def recording(self, x):
        y, stats = real(self, x)
        seen.append((self, stats.counts.clone(), stats.dropped.clone(), self._last_scores.clone()))
        return y, stats

    monkeypatch.setattr(MoELayer, "forward", recording)
    return seen


def test_expert_counts_over_a_step_are_the_sum_of_its_micro_batches(tmp_path, monkeypatch):
    cfg = smoke(tmp_path, model={"moe_dispatch": "padded", "capacity_factor": 1.0})
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    assert cfg.train.grad_accum == 4
    trainer.train_step()                       # let the router move off its init
    seen = _record_micro_batches(monkeypatch)
    trainer.train_step()
    layers = [b.moe for b in trainer.model.blocks]
    for l, layer in enumerate(layers):
        mine = [s for s in seen if s[0] is layer]
        assert len(mine) == cfg.train.grad_accum
        assert torch.equal(trainer.last_step_counts[l], sum(c for _, c, _, _ in mine))
        assert torch.equal(trainer.last_step_dropped[l], sum(d for _, _, d, _ in mine))
    assert int(trainer.last_step_counts.sum()) == (
        cfg.model.n_layer * cfg.train.batch_tokens * cfg.model.top_k)


def test_the_balancer_update_sees_every_micro_batch_of_the_step(tmp_path, monkeypatch):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    trainer.train_step()
    before = biases(trainer.model)
    seen = _record_micro_batches(monkeypatch)
    trainer.train_step()
    for l, block in enumerate(trainer.model.blocks):
        ref = QuantileBalancer(cfg.model.n_experts, cfg.model.top_k, cfg.model.balance_update_rate)
        ref.bias.copy_(before[l])
        ref.update(torch.cat([s for layer, _, _, s in seen if layer is block.moe]))
        assert torch.equal(block.moe.balancer.bias, ref.bias)


def test_balance_scores_are_sampled_evenly_when_capped():
    cfg = load_config(SMOKE).model
    torch.manual_seed(0)
    model = QuipuMoE(cfg)
    stash = []
    for _ in range(3):
        model(torch.randint(0, cfg.vocab_size, (2, 16)))          # 32 tokens a forward
        stash.append(model.blocks[0].moe._last_scores.clone())
        model.accumulate_balance_scores(max_rows=10)
    kept = model.blocks[0].moe._step_scores
    assert [s.shape[0] for s in kept] == [8, 8, 8]                 # stride 4, <= 10 rows
    # Offsets rotate so every micro-batch does not sample the same positions.
    for i, (s, full) in enumerate(zip(kept, stash)):
        assert torch.equal(s, full[i % 4::4])
        assert s.untyped_storage().nbytes() == s.numel() * s.element_size()   # not a view
    model.clear_balance_scores()
    before = biases(model)
    model.update_balance()                                         # nothing stashed
    assert all(torch.equal(a, b) for a, b in zip(before, biases(model)))


# ---- checkpoints and resume ----------------------------------------------------------

@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_resume_at_step_10_is_bit_identical_to_the_uninterrupted_run(tmp_path, optimizer):
    cfg = smoke(tmp_path, optimizer)
    data = make_data(tmp_path)
    a = build(tmp_path, cfg, data)
    for _ in range(10):
        a.train_step()
    a.save_checkpoint()
    # Two steps: the loss of step 11 is computed before its update, so only step 12's
    # loss (and the weights after 11) can see lost optimizer or balancer state.
    expected = [a.train_step() for _ in range(2)]

    b = build(tmp_path, cfg, data, resume=True)
    b.resume_from_latest()
    assert b.step == 10
    got = [b.train_step() for _ in range(2)]

    assert got == expected
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_balancer_biases_are_saved_and_restored(tmp_path):
    cfg = smoke(tmp_path)
    data = make_data(tmp_path)
    a = build(tmp_path, cfg, data)
    for _ in range(3):
        a.train_step()
    path = a.save_checkpoint()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for l, bias in enumerate(biases(a.model)):
        assert bias.abs().sum() > 0
        assert torch.equal(saved["model"][f"blocks.{l}.moe.balancer.bias"], bias)
    assert len(saved["optimizers"]) == 2 and "optimizer" not in saved

    b = build(tmp_path, cfg, data, resume=True)
    assert all(bias.abs().sum() == 0 for bias in biases(b.model))
    b.resume_from_latest()
    assert all(torch.equal(x, y) for x, y in zip(biases(a.model), biases(b.model)))


def test_resume_takes_learning_rates_and_decay_from_the_config(tmp_path):
    cfg = smoke(tmp_path)
    data = make_data(tmp_path)
    a = build(tmp_path, cfg, data)
    a.train_step()
    a.save_checkpoint()
    cfg2 = smoke(tmp_path, muon_lr=0.01, lr=5e-4, lr_min=5e-5, muon_weight_decay=0.0)
    b = build(tmp_path, cfg2, data, resume=True)
    b.resume_from_latest()
    muon, adam = b.optimizers
    assert all(g["base_lr"] == 0.01 and g["weight_decay"] == 0.0 for g in muon.param_groups)
    assert all(g["base_lr"] == 5e-4 for g in adam.param_groups)


def test_a_checkpoint_from_another_optimizer_setup_is_refused(tmp_path):
    data = make_data(tmp_path)
    a = build(tmp_path, smoke(tmp_path, "adamw"), data)
    a.train_step()
    a.save_checkpoint()
    b = build(tmp_path, smoke(tmp_path, "muon"), data, resume=True)
    with pytest.raises(train_mod.UsageError, match="optimizer"):
        b.resume_from_latest()


def _tiny_dense() -> ModelConfig:
    return ModelConfig(vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
                       ffn_hidden=128, context=16, rope_base=10000.0, norm_eps=1e-6)


def _tiny_dense_train(tmp_path) -> TrainConfig:
    return TrainConfig(
        total_tokens=16 * 2 * 20, batch_tokens=16 * 2, micro_batch=2, context=16,
        lr=1e-3, lr_min=1e-4, warmup_steps=2, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, seed=7, ckpt_dir=str(tmp_path / "ckpt"), ckpt_every=5,
        ckpt_keep=3, eval_every=1000, eval_batches=1,
    )


def test_an_old_quipu_114m_checkpoint_still_loads(tmp_path):
    """The pre-M6 format: one AdamW under "optimizer", its groups without base_lr or
    the decay tag, and no skipped_steps. Built here the way the old trainer did."""
    from quipu.model import Quipu

    data = tmp_path / "data"
    write_shard(data / "shard_000.bin",
                np.random.RandomState(0).randint(0, 128, 8192).astype(np.uint16))
    mcfg, tcfg = _tiny_dense(), _tiny_dense_train(tmp_path)
    torch.manual_seed(tcfg.seed)
    model = Quipu(mcfg)
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
        lr=tcfg.lr, betas=(0.9, 0.95))
    stream = TokenStream(data, 2, 16)
    for _ in range(3):
        x, y = stream.next_batch()
        F.cross_entropy(model(x).view(-1, 128), y.reshape(-1)).backward()
        opt.step()
        opt.zero_grad()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save({"step": 3, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "stream": stream.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": None}, ckpt / "step_000003.pt")
    torch.save({"file": "step_000003.pt"}, ckpt / "latest.pt")

    trainer = Trainer(model_cfg=mcfg, train_cfg=tcfg, shard_dir=data, device="cpu",
                      run_dir=tmp_path / "runs", run_id="old", resume=False)
    trainer.load_checkpoint()
    assert trainer.step == 3 and trainer.skipped_steps == 0
    for p, q in zip(model.parameters(), trainer.model.parameters()):
        assert torch.equal(p, q)
    (adam,) = trainer.optimizers
    assert trainer.opt is adam
    assert [g["decay"] for g in adam.param_groups] == [True, False]
    assert all(g["base_lr"] == tcfg.lr for g in adam.param_groups)
    assert [g["weight_decay"] for g in adam.param_groups] == [0.1, 0.0]
    assert math.isfinite(trainer.train_step()) and trainer.step == 4


def test_checkpoints_hold_the_uncompiled_state_dict(tmp_path, monkeypatch):
    # Real dynamo, the "eager" backend: no Triton or C++ compiler needed.
    monkeypatch.setattr(train_mod, "COMPILE_BACKEND", "eager")
    cfg = smoke(tmp_path, compile=True)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    assert trainer.compiled
    assert trainer.forward_model is not trainer.model
    assert logged(tmp_path)["compile"] == "on (eager)"
    trainer.train_step()
    path = trainer.save_checkpoint()
    keys = torch.load(path, map_location="cpu", weights_only=False)["model"].keys()
    assert not any("_orig_mod" in k for k in keys)
    assert keys == trainer.model.state_dict().keys()


def test_compile_falls_back_to_eager_with_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_mod, "_compile_unavailable", lambda device: "no Triton here")
    cfg = smoke(tmp_path, compile=True)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    assert not trainer.compiled and trainer.forward_model is trainer.model
    assert "torch.compile unavailable (no Triton here)" in capsys.readouterr().err
    assert logged(tmp_path)["compile"] == "skipped: no Triton here"
    assert math.isfinite(trainer.train_step())


@pytest.mark.skipif(train_mod._compile_unavailable("cpu") is not None,
                    reason=f"torch.compile unavailable: {train_mod._compile_unavailable('cpu')}")
def test_compiled_logits_match_eager_on_the_smoke_config(tmp_path):
    cfg = smoke(tmp_path, compile=True)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    assert trainer.compiled
    idx = torch.randint(0, cfg.model.vocab_size, (2, 64))
    with torch.no_grad():
        torch.testing.assert_close(trainer.forward_model(idx), trainer.model(idx),
                                   atol=1e-3, rtol=0)


# ---- expert-load logging and the health alert -----------------------------------------

def test_expert_load_is_logged_every_eval_interval(tmp_path):
    cfg = smoke(tmp_path, eval_every=2,
                model={"moe_dispatch": "padded", "capacity_factor": 1.0})
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    for _ in range(4):
        trainer.train_step()
    moe = logged(tmp_path)["moe"]
    assert [m["step"] for m in moe] == [2, 4]
    for entry in moe:
        assert len(entry["layers"]) == cfg.model.n_layer
        for layer in entry["layers"]:
            assert set(layer) == {"load_min", "load_max", "load_cv", "dead", "drop_rate"}
            assert 0 <= layer["load_min"] <= 1 <= layer["load_max"]
            assert layer["load_cv"] >= 0 and layer["dead"] >= 0
        # capacity 1.0: any imbalance at all drops tokens.
        assert any(layer["drop_rate"] > 0 for layer in entry["layers"])


def test_interval_load_statistics_are_computed_from_the_summed_counts(tmp_path):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    counts = torch.tensor([[0, 10, 20, 30, 40, 50, 60, 110]] * 2)     # sum 320, target 40
    stats = trainer._load_stats(counts, torch.tensor([32, 0]))
    assert stats[0]["load_min"] == 0.0 and stats[0]["load_max"] == pytest.approx(110 / 40)
    assert stats[0]["dead"] == 1
    c = counts[0].double()
    assert stats[0]["load_cv"] == pytest.approx(float(c.std(unbiased=False) / c.mean()))
    assert stats[0]["drop_rate"] == pytest.approx(0.1) and stats[1]["drop_rate"] == 0.0


def _skewed(n_layer=2, n=8):
    # Expert 0 gets nothing (0% of target), the rest share the load evenly.
    c = torch.full((n_layer, n), 100, dtype=torch.long)
    c[:, 0] = 0
    return c


def test_health_alert_fires_after_the_window_of_skewed_steps(tmp_path, capsys):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    trainer.moe_health_window = 3
    for _ in range(3):
        trainer._track_moe_health(_skewed())
    assert "moe_health" not in logged(tmp_path)
    assert "MoE health" not in capsys.readouterr().err
    trainer._track_moe_health(_skewed())             # 4 > 3 consecutive steps
    err = capsys.readouterr().err
    assert "MoE health" in err and "layer 0 expert 0" in err
    alerts = logged(tmp_path)["moe_health"]
    assert {(a["layer"], a["expert"]) for a in alerts} == {(0, 0), (1, 0)}
    assert all(a["load"] == 0.0 and a["streak"] == 4 for a in alerts)
    trainer._track_moe_health(_skewed())             # one alert per episode
    assert len(logged(tmp_path)["moe_health"]) == 2


def test_health_alert_fires_for_an_overloaded_expert(tmp_path):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    trainer.moe_health_window = 2
    c = torch.full((2, 8), 10, dtype=torch.long)
    c[1, 5] = 400                                     # 400 / (470 / 8) = 6.8x target
    for _ in range(3):
        trainer._track_moe_health(c)
    assert [(a["layer"], a["expert"]) for a in logged(tmp_path)["moe_health"]] == [(1, 5)]


def test_health_alert_stays_quiet_on_balanced_or_interrupted_skew(tmp_path, capsys):
    cfg = smoke(tmp_path)
    trainer = build(tmp_path, cfg, make_data(tmp_path))
    trainer.moe_health_window = 3
    balanced = torch.full((2, 8), 100, dtype=torch.long)
    for i in range(10):
        trainer._track_moe_health(_skewed() if i % 3 else balanced)   # never 4 in a row
    for _ in range(10):
        trainer._track_moe_health(balanced)
    assert "moe_health" not in logged(tmp_path)
    assert "MoE health" not in capsys.readouterr().err


def test_health_window_default_is_500_steps():
    assert train_mod.MOE_HEALTH_WINDOW == 500
    assert (train_mod.MOE_HEALTH_LOW, train_mod.MOE_HEALTH_HIGH) == (0.1, 3.0)


# ---- evaluation --------------------------------------------------------------------------

def _record_dispatch(monkeypatch):
    seen = []
    real = MoELayer.forward

    def recording(self, x):
        seen.append(self.dispatch)
        return real(self, x)

    monkeypatch.setattr(MoELayer, "forward", recording)
    return seen


def test_estimate_loss_runs_loop_dispatch_and_restores_padded(tmp_path, monkeypatch):
    cfg = load_config(SMOKE, {"model": {"moe_dispatch": "padded"}}).model
    model = QuipuMoE(cfg)
    seen = _record_dispatch(monkeypatch)
    stream = TokenStream(make_data(tmp_path), micro_batch=2, context=32)
    estimate_loss(model, stream, batches=2, device="cpu")
    assert seen and set(seen) == {"loop"}
    assert all(b.moe.dispatch == "padded" for b in model.blocks)


def test_trainer_eval_uses_loop_dispatch_and_training_stays_padded(tmp_path, monkeypatch):
    cfg = smoke(tmp_path, eval_every=1, model={"moe_dispatch": "padded"})
    trainer = build(tmp_path, cfg, make_data(tmp_path), val=make_data(tmp_path, name="val", seed=1))
    seen = _record_dispatch(monkeypatch)
    trainer.train_cfg = dataclasses.replace(cfg.train, total_tokens=cfg.train.batch_tokens)
    trainer.run()
    n_layer, accum = cfg.model.n_layer, cfg.train.grad_accum
    assert seen[: n_layer * accum] == ["padded"] * (n_layer * accum)
    assert set(seen[n_layer * accum:]) == {"loop"}
    assert all(b.moe.dispatch == "padded" for b in trainer.model.blocks)
    assert logged(tmp_path)["evals"]


class _TableModel(nn.Module):
    """logits[b, t] = table[x[b, t]]: an exactly known distribution per input."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table)

    def forward(self, x):
        return self.table[x]


class _PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_bits_per_byte_equals_a_hand_computation():
    pieces = ["a", "é", "日本", "xyz"]                     # 1, 2, 6, 3 UTF-8 bytes
    table = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                          [1.0, 0.0, -1.0, 0.5],
                          [2.0, 2.0, 0.0, 0.0],
                          [-1.0, 0.0, 1.0, 4.0]])
    batches = [
        (torch.tensor([[0, 1, 2], [3, 3, 0]]), torch.tensor([[1, 2, 3], [3, 0, 2]])),
        (torch.tensor([[2, 0, 1]]), torch.tensor([[0, 0, 1]])),
    ]
    nll, n_bytes = 0.0, 0
    for x, y in batches:
        for xr, yr in zip(x.tolist(), y.tolist()):
            for xi, yi in zip(xr, yr):
                row = table[xi].tolist()
                log_z = math.log(sum(math.exp(v) for v in row))
                nll += log_z - row[yi]
            n_bytes += len("".join(pieces[i] for i in yr).encode("utf-8"))
    expected = nll / (n_bytes * math.log(2))
    got = bits_per_byte(_TableModel(table), batches, _PieceTokenizer(pieces))
    assert got == pytest.approx(expected, rel=1e-6)


def test_bits_per_byte_counts_bytes_of_the_decoded_row_not_of_each_token():
    # Two byte-level tokens that only form one character together: decoding them one
    # at a time would give two replacement characters (6 bytes), not 2 bytes.
    class ByteTokenizer:
        def decode(self, ids):
            return bytes(ids).decode("utf-8", errors="replace")

    table = torch.zeros(256, 256)
    y = torch.tensor([[0xC3, 0xA9]])                            # "é"
    got = bits_per_byte(_TableModel(table), [(torch.zeros_like(y), y)], ByteTokenizer())
    assert got == pytest.approx(2 * math.log(256) / (2 * math.log(2)))


def test_bits_per_byte_runs_moe_with_loop_dispatch(monkeypatch):
    cfg = load_config(SMOKE, {"model": {"moe_dispatch": "padded"}}).model
    model = QuipuMoE(cfg)
    seen = _record_dispatch(monkeypatch)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    class Tok:
        def decode(self, ids):
            return "ab" * len(ids)

    bpb = bits_per_byte(model, [(x, y)], Tok())
    assert set(seen) == {"loop"} and all(b.moe.dispatch == "padded" for b in model.blocks)
    model.set_dispatch("loop")
    with torch.no_grad():
        nll = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.reshape(-1), reduction="sum")
    assert bpb == pytest.approx(float(nll) / (64 * math.log(2)), rel=1e-5)
