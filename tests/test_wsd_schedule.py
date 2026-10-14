"""Warmup-stable-decay (train.lr_schedule "wsd") and the continued-pretraining starts:
--init-from with --init-optimizers (weights and optimizer moments, new schedule and
stream) and --branch-from (a run's whole state under a new ckpt_dir: decay branches).
The cosine schedule must be exactly what it was."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from quipu.config import TrainConfig, load_config, parse_overrides
from quipu.data import write_shard
from quipu.train import Trainer, UsageError, lr_at, resolve_checkpoint, run_main
from tests.test_checkpoint_resume import tiny_model

ROOT = Path(__file__).resolve().parents[1]
CPT = ROOT / "configs" / "quipu-moe-cpt.toml"


def cfg(**over) -> TrainConfig:
    base = dict(total_tokens=32 * 1000, batch_tokens=32, micro_batch=2, context=16,
                lr=1e-3, lr_min=1e-4, warmup_steps=10, weight_decay=0.1, beta1=0.9,
                beta2=0.95, grad_clip=1.0, seed=7, ckpt_dir="unused", ckpt_every=5,
                ckpt_keep=2, eval_every=1000, eval_batches=1)
    base.update(over)
    return TrainConfig(**base)


def old_cosine(c: TrainConfig, step: int) -> float:
    """quipu.train.Trainer.lr_at before data v3, verbatim."""
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    if step >= c.steps:
        return c.lr_min
    progress = (step - c.warmup_steps) / max(1, c.steps - c.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return c.lr_min + (c.lr - c.lr_min) * cosine


def test_the_cosine_schedule_is_unchanged():
    c = cfg()
    assert c.lr_schedule == "cosine"
    for step in list(range(0, 40)) + list(range(900, 1010)):
        assert lr_at(c, step) == old_cosine(c, step)


def test_wsd_rewarms_from_the_start_lr_then_holds_the_stable_lr_with_no_end():
    c = cfg(lr_schedule="wsd", warmup_start_lr=1e-4, warmup_steps=10)
    assert lr_at(c, 0) == pytest.approx(1e-4)                 # where the pretrain ended
    assert lr_at(c, 5) == pytest.approx(1e-4 + 0.5 * (1e-3 - 1e-4))
    assert lr_at(c, 10) == pytest.approx(1e-3)
    for step in (11, 999, 1000, 10**7):                       # past `steps`: still stable
        assert lr_at(c, step) == pytest.approx(1e-3)
    lrs = [lr_at(c, s) for s in range(11)]
    assert lrs == sorted(lrs)


@pytest.mark.parametrize("shape", ["linear", "1-sqrt"])
def test_a_decay_branch_falls_from_the_stable_lr_to_lr_min(shape):
    c = cfg(lr_schedule="wsd", warmup_start_lr=1e-4, decay_start_step=500, decay_steps=100,
            decay_shape=shape)
    assert lr_at(c, 499) == pytest.approx(1e-3)
    assert lr_at(c, 500) == pytest.approx(1e-3)
    mid = 0.5 if shape == "linear" else 1 - math.sqrt(0.5)
    assert lr_at(c, 550) == pytest.approx(1e-4 + (1e-3 - 1e-4) * mid)
    assert lr_at(c, 600) == pytest.approx(1e-4)
    assert lr_at(c, 10**6) == pytest.approx(1e-4)
    lrs = [lr_at(c, s) for s in range(500, 601)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


@pytest.mark.parametrize("over, match", [
    ({"warmup_start_lr": 1e-5}, "apply to lr_schedule 'wsd'"),
    ({"decay_steps": 5, "decay_start_step": 20}, "apply to lr_schedule 'wsd'"),
    ({"decay_shape": "1-sqrt"}, "apply to lr_schedule 'wsd'"),
    ({"lr_schedule": "wsd", "decay_steps": 5}, "both be 0"),
    ({"lr_schedule": "wsd", "decay_start_step": 5}, "both be 0"),
    ({"lr_schedule": "wsd", "decay_start_step": 3, "decay_steps": 5}, "end of the warmup"),
    ({"lr_schedule": "step"}, "lr_schedule must be one of"),
    ({"lr_schedule": "wsd", "decay_shape": "cosine"}, "decay_shape must be one of"),
    ({"lr_schedule": "wsd", "warmup_start_lr": -1e-4}, "warmup_start_lr"),
])
def test_bad_schedule_keys_are_refused(tmp_path, over, match):
    smoke = ROOT / "configs" / "quipu-moe-smoke.toml"
    with pytest.raises(ValueError, match=match):
        load_config(smoke, {"train": over})


def test_schedule_keys_come_through_overrides():
    o = parse_overrides(["train.lr_schedule=wsd", "train.decay_start_step=8000",
                         "train.decay_steps=600", "train.warmup_start_lr=6e-5",
                         "train.decay_shape=1-sqrt"])
    assert o["train"] == {"lr_schedule": "wsd", "decay_start_step": 8000, "decay_steps": 600,
                          "warmup_start_lr": 6e-5, "decay_shape": "1-sqrt"}


def test_the_cpt_config_is_wsd_from_the_pretrains_final_lr():
    c = load_config(CPT)
    t = c.train
    assert t.lr_schedule == "wsd" and t.optimizer == "muon"
    assert t.warmup_start_lr == pytest.approx(6e-5)           # run_config: lr_min 6e-5
    assert lr_at(t, 0) == pytest.approx(6e-5) and lr_at(t, t.warmup_steps) == t.lr
    assert t.decay_steps == 0 and t.milestones == ()
    assert t.steps > 10**6                                     # no fixed end in practice
    assert t.batch_tokens % (t.micro_batch * c.model.context) == 0


# ---- --init-optimizers and --branch-from ---------------------------------------------

def _data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    write_shard(d / "shard_000.bin",
                np.random.RandomState(0).randint(0, 128, 8192).astype(np.uint16))
    return d


def _trainer(tmp_path, data, run_id, ckpt, **over) -> Trainer:
    c = cfg(total_tokens=32 * 40, warmup_steps=2, ckpt_dir=str(tmp_path / ckpt), **over)
    return Trainer(model_cfg=tiny_model(), train_cfg=c, shard_dir=data, device="cpu",
                   run_dir=tmp_path / "runs", run_id=run_id)


def _pretrained(tmp_path, data) -> Path:
    a = _trainer(tmp_path, data, "pre", "pre")
    for _ in range(4):
        a.train_step()
    return a.save_checkpoint()


def test_init_optimizers_keeps_the_moments_but_starts_a_new_schedule_and_stream(tmp_path):
    data = _data(tmp_path)
    ckpt = _pretrained(tmp_path, data)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    b = _trainer(tmp_path, data, "cpt", "cpt", lr_schedule="wsd", warmup_start_lr=1e-4,
                 lr=5e-4)
    b.init_weights_from(ckpt.parent, optimizers=True)
    assert b.step == 0 and b.stream.position == 0
    state = b.opt.state_dict()
    want = saved["optimizers"][-1]
    assert state["state"].keys() == want["state"].keys() and state["state"]
    for k, v in state["state"].items():
        assert torch.equal(v["exp_avg"], want["state"][k]["exp_avg"])
    # the config's LR wins over the checkpoint's
    assert all(g["base_lr"] == 5e-4 for g in b.opt.param_groups)
    b.train_step()
    assert b.log.record["steps"][0]["lr"] == pytest.approx(1e-4)   # wsd step 0
    assert b.log.record["init_from"]["optimizers"] is True


def test_init_optimizers_needs_a_full_checkpoint_and_the_same_optimizers(tmp_path):
    data = _data(tmp_path)
    ckpt = _pretrained(tmp_path, data)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    milestone = tmp_path / "milestone.pt"
    torch.save(saved["model"], milestone)
    b = _trainer(tmp_path, data, "m", "m")
    with pytest.raises(UsageError, match="no optimizer state"):
        b.init_weights_from(milestone, optimizers=True)
    c = _trainer(tmp_path, data, "mu", "mu", optimizer="muon")
    with pytest.raises(UsageError, match="optimizer state for"):
        c.init_weights_from(ckpt, optimizers=True)


def test_resolve_checkpoint_follows_a_directory_and_its_pointer(tmp_path):
    data = _data(tmp_path)
    ckpt = _pretrained(tmp_path, data)
    assert resolve_checkpoint(ckpt.parent, "--x") == ckpt
    assert resolve_checkpoint(ckpt.parent / "latest.pt", "--x") == ckpt
    assert resolve_checkpoint(ckpt, "--x") == ckpt
    with pytest.raises(UsageError):
        resolve_checkpoint(tmp_path / "nothing", "--x")


def _cli_config(tmp_path: Path, data: Path) -> Path:
    """The smoke config on a tokenizer-free tiny dense model over `data`."""
    p = tmp_path / "cli.toml"
    p.write_text(
        f'inherit = "{(ROOT / "configs" / "quipu-moe-smoke.toml").as_posix()}"\n'
        "[model]\nkind = \"dense\"\nvocab_size = 128\nd_model = 64\nn_layer = 2\n"
        "n_head = 4\nn_kv_head = 2\nffn_hidden = 128\ncontext = 16\nn_experts = 0\n"
        "top_k = 0\nexpert_hidden = 0\nshared_experts = 0\nshared_hidden = 0\n"
        "attnres_blocks = 0\nactivation = \"swiglu\"\n"
        f'[data]\ntokenizer = "gpt2"\nshard_dir = "{data.as_posix()}"\n'
        "[train]\nmicro_batch = 2\nbatch_tokens = 32\ntotal_tokens = 320\nwarmup_steps = 2\n"
        "ckpt_every = 4\neval_every = 1000\nmilestones = []\ncompile = false\n"
        "optimizer = \"adamw\"\nlr_schedule = \"wsd\"\nlr_min = 1e-5\n",
        encoding="utf-8")
    return p


def _shards(tmp_path: Path) -> Path:
    root = tmp_path / "shards"
    rng = np.random.RandomState(1)
    write_shard(root / "train" / "shard_000.bin", rng.randint(0, 128, 8192).astype(np.uint16))
    write_shard(root / "val" / "shard_000.bin", rng.randint(0, 128, 2048).astype(np.uint16))
    return root


def _run(config, run_id, ckpt, *extra):
    return run_main(["--config", str(config), "--run-id", run_id, "--device", "cpu",
                     "--run-dir", str(Path(ckpt).parent / "runs"),
                     "--override", f"train.ckpt_dir={ckpt}", *extra])


def test_a_decay_branch_continues_the_stable_runs_state_in_its_own_directory(tmp_path):
    import json

    config = _cli_config(tmp_path, _shards(tmp_path))
    stable = tmp_path / "stable"
    assert _run(config, "stable", stable) == 0                  # 10 steps, stable lr
    src = torch.load(stable / "step_000010.pt", map_location="cpu", weights_only=False)
    decay = tmp_path / "decay"
    rc = _run(config, "decay", decay, "--branch-from", str(stable),
              "--override", "train.decay_start_step=10", "--override", "train.decay_steps=4",
              "--override", "train.total_tokens=448")
    assert rc == 0
    rec = json.loads((tmp_path / "runs" / "decay.json").read_text(encoding="utf-8"))
    assert [s["step"] for s in rec["steps"]] == [11, 12, 13, 14]
    lrs = [s["lr"] for s in rec["steps"]]
    assert lrs[0] == pytest.approx(1e-3) and all(a > b for a, b in zip(lrs, lrs[1:]))
    assert rec["branch_from"]["step"] == 10
    end = torch.load(decay / "step_000014.pt", map_location="cpu", weights_only=False)
    # the branch read on from where the stable run's stream stopped
    assert end["stream"]["position"] == src["stream"]["position"] + 4 * 32
    assert (decay / "milestones" / "step_000014.pt").exists()      # the release weights
    assert sorted(p.name for p in stable.glob("step_*.pt")) == ["step_000008.pt",
                                                                "step_000010.pt"]


@pytest.mark.parametrize("args, match", [
    (["--init-optimizers"], "needs --init-from"),
    (["--init-from", "x", "--branch-from", "y"], "exclude each other"),
])
def test_contradictory_start_flags_are_usage_errors(tmp_path, args, match, capsys):
    config = _cli_config(tmp_path, _shards(tmp_path))
    assert _run(config, "bad", tmp_path / "bad", *args) == 2
    assert match in capsys.readouterr().err


def test_branching_into_the_source_directory_is_refused(tmp_path, capsys):
    config = _cli_config(tmp_path, _shards(tmp_path))
    stable = tmp_path / "stable"
    assert _run(config, "stable", stable) == 0
    assert _run(config, "again", stable, "--branch-from", str(stable)) == 2
    assert "own ckpt_dir" in capsys.readouterr().err
