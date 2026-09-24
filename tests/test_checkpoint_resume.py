import json

import numpy as np
import torch

from quipu.config import ModelConfig, TrainConfig
from quipu.data import write_shard
from quipu.train import Trainer


def tiny_model() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=16, rope_base=10000.0, norm_eps=1e-6,
    )


def tiny_train(tmp_path, ckpt_keep: int = 3) -> TrainConfig:
    return TrainConfig(
        total_tokens=16 * 2 * 20, batch_tokens=16 * 2, micro_batch=2, context=16,
        lr=1e-3, lr_min=1e-4, warmup_steps=2, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, seed=7, ckpt_dir=str(tmp_path / "ckpt"), ckpt_every=5,
        ckpt_keep=ckpt_keep, eval_every=1000, eval_batches=1,
    )


def make_data(tmp_path):
    write_shard(tmp_path / "shard_000.bin",
                np.random.RandomState(0).randint(0, 128, 8192).astype(np.uint16))
    return tmp_path


def build(tmp_path, data_dir, resume=False, ckpt_keep=3):
    return Trainer(
        model_cfg=tiny_model(), train_cfg=tiny_train(tmp_path, ckpt_keep),
        shard_dir=data_dir, device="cpu", run_dir=tmp_path / "runs", run_id="t",
        resume=resume,
    )


def logged(tmp_path) -> dict:
    return json.loads((tmp_path / "runs" / "t.json").read_text(encoding="utf-8"))


def test_loss_decreases_on_repeated_data(tmp_path):
    # 100 tokens holds three 33-token micro-batches, so the stream wraps every third
    # step and the model sees the same data over and over. (Uniform-random data that
    # never repeats has nothing to learn: its loss floor is ln(128) from step one.)
    data = tmp_path / "small"
    write_shard(data / "shard_000.bin",
                np.random.RandomState(0).randint(0, 128, 100).astype(np.uint16))
    trainer = build(tmp_path, data)
    first = trainer.train_step()
    for _ in range(30):
        last = trainer.train_step()
    assert trainer.stream.wraps > 0
    assert last < first - 0.5


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    """Save at step N, reload, and the next step must match what an uninterrupted
    run would have produced. Anything less and a 20-hour run silently diverges."""
    data = make_data(tmp_path)

    a = build(tmp_path, data)
    for _ in range(6):
        a.train_step()
    a.save_checkpoint()
    # Two steps, not one: the loss train_step returns comes from the forward pass
    # BEFORE the optimizer update, so a single step cannot see lost Adam moments.
    # The second step's loss (and the weights after it) can.
    expected = [a.train_step() for _ in range(2)]

    b = build(tmp_path, data, resume=True)
    b.load_checkpoint()
    got = [b.train_step() for _ in range(2)]

    for e, g in zip(expected, got):
        assert abs(e - g) < 1e-5, f"resume diverged: {expected} vs {got}"
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.allclose(p, q, atol=1e-6), "weights diverged after resume"


def test_checkpoint_restores_every_piece_of_state(tmp_path):
    data = make_data(tmp_path)
    a = build(tmp_path, data)
    for _ in range(6):
        a.train_step()
    a.save_checkpoint()

    b = build(tmp_path, data, resume=True)
    b.load_checkpoint()

    assert b.step == a.step
    assert b.stream.position == a.stream.position
    assert b.stream.wraps == a.stream.wraps
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(p, q)


def test_learning_rate_warms_up_then_decays(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    cfg = trainer.train_cfg
    assert trainer.lr_at(0) < cfg.lr                 # warming up
    assert abs(trainer.lr_at(cfg.warmup_steps) - cfg.lr) < 1e-9   # peak
    assert abs(trainer.lr_at(cfg.steps) - cfg.lr_min) < 1e-9      # floor
    assert trainer.lr_at(cfg.steps // 2) < cfg.lr                 # decaying


def test_resume_truncates_the_log_so_no_step_appears_twice(tmp_path):
    data = make_data(tmp_path)
    a = build(tmp_path, data)
    for _ in range(6):
        a.train_step()
        if a.step == 5:
            a.save_checkpoint()
    # Step 6 was logged after the step-5 checkpoint, so it is about to be re-run.

    b = build(tmp_path, data, resume=True)
    b.resume_from_latest()
    assert b.step == 5
    b.train_step()

    steps = [s["step"] for s in logged(tmp_path)["steps"]]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert all(x < y for x, y in zip(steps, steps[1:]))


def test_log_step_extra_fields_record_wraps_and_grad_norm(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    trainer.train_step()
    entry = logged(tmp_path)["steps"][0]
    assert entry["wraps"] == 0
    assert isinstance(entry["grad_norm"], float) and entry["grad_norm"] > 0


def test_config_is_logged_as_a_copy_with_derived_values(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    cfg = logged(tmp_path)["config"]
    assert cfg["derived"] == {"steps": 20, "grad_accum": 1}
    assert cfg["train"]["ckpt_keep"] == 3
    trainer.log.record["config"]["train"]["lr"] = 99.0
    assert trainer.train_cfg.lr == 1e-3


def test_a_failed_log_write_does_not_stop_training(tmp_path, capsys):
    trainer = build(tmp_path, make_data(tmp_path))

    def broken(**kwargs):
        raise OSError("disk hiccup")

    trainer.log.log_step = broken
    loss = trainer.train_step()
    trainer.train_step()

    assert isinstance(loss, float)
    assert trainer.step == 2
    assert trainer.log_failures == 2
    assert "disk hiccup" in capsys.readouterr().err


def test_latest_pointer_holds_a_bare_file_name(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    for _ in range(3):
        trainer.train_step()
    trainer.save_checkpoint()
    pointer = torch.load(tmp_path / "ckpt" / "latest.pt", weights_only=False)
    assert pointer == {"file": "step_000003.pt"}


def test_checkpoint_dir_can_be_moved(tmp_path):
    data = make_data(tmp_path)
    a = build(tmp_path, data)
    for _ in range(3):
        a.train_step()
    a.save_checkpoint()
    moved = tmp_path / "elsewhere"
    (tmp_path / "ckpt").rename(moved)

    b = build(tmp_path, data, resume=True)
    b.load_checkpoint(moved / "step_000003.pt")
    assert b.step == 3
    # Loading via the pointer after moving the directory back into ckpt_dir's place.
    moved.rename(tmp_path / "ckpt")
    c = build(tmp_path, data, resume=True)
    c.load_checkpoint()
    assert c.step == 3


def test_checkpoint_writes_leave_no_tmp_behind(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    for _ in range(2):
        trainer.train_step()
        trainer.save_checkpoint()
    assert not list((tmp_path / "ckpt").glob("*.tmp"))


def test_retention_keeps_only_the_newest_checkpoints(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path), ckpt_keep=2)
    for _ in range(5):
        trainer.train_step()
        trainer.save_checkpoint()
    names = sorted(p.name for p in (tmp_path / "ckpt").glob("step_*.pt"))
    assert names == ["step_000004.pt", "step_000005.pt"]
    pointer = torch.load(tmp_path / "ckpt" / "latest.pt", weights_only=False)
    assert pointer["file"] in names


def test_run_completes_and_marks_the_log(tmp_path):
    trainer = build(tmp_path, make_data(tmp_path))
    trainer.run()
    record = logged(tmp_path)
    assert record["status"] == "completed"
    assert [s["step"] for s in record["steps"]] == list(range(1, 21))
    names = sorted(p.name for p in (tmp_path / "ckpt").glob("step_*.pt"))
    assert names == ["step_000010.pt", "step_000015.pt", "step_000020.pt"]
