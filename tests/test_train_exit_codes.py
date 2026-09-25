"""The exit-code contract of `python -m quipu.train`, which the weekend launcher
uses to decide whether a failed attempt is worth retrying:
0 completed, 1 any other crash, 2 usage/config error, 3 non-finite stop, 130 interrupt.

Everything but one test goes through run_main in-process; the last one spawns the
real module to prove the code survives to the process exit status."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import quipu.train as train_mod
from quipu.data import write_shard
from quipu.train import (
    EXIT_CRASH, EXIT_INTERRUPTED, EXIT_NONFINITE, EXIT_OK, EXIT_USAGE,
    NonFiniteStop, Trainer, UsageError, run_main,
)

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "exit-test"

TINY_TOML = """
name = "tiny"

[model]
vocab_size = 128
d_model = 64
n_layer = 2
n_head = 4
n_kv_head = 2
ffn_hidden = 128
context = 16
rope_base = 10000.0
norm_eps = 1e-6

[data]
dataset = "x"
subset = "x"
shard_dir = "{shards}"
shard_tokens = 4096
val_tokens = 4096
code_dataset = "x"
code_share = 0.2
code_languages = ["Python"]
code_licenses = ["mit"]
html_cap = 0.1
code_val_tokens = 1
code_heldout_first_file = 1
code_files_total = 2

[train]
total_tokens = 640
batch_tokens = 32
micro_batch = 2
lr = {lr!r}
lr_min = 1e-4
warmup_steps = 1
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
seed = 7
ckpt_dir = "{ckpt}"
ckpt_every = 5
ckpt_keep = 2
eval_every = 1000
eval_batches = 1
milestones = [3, 7]
"""

# An lr of 1e30 takes the weights past float32 range on the first step, so every
# later loss is non-finite and the guard stops the run after three in a row.
EXPLODING_LR = 1e30


def tiny_config(tmp_path: Path, lr: float = 1e-3) -> Path:
    """A complete config for a 20-step CPU run on random tokens, with its shards."""
    shards = tmp_path / "shards"
    for split in ("train", "val"):
        write_shard(shards / split / "shard_000.bin",
                    np.random.RandomState(0).randint(0, 128, 4096).astype(np.uint16))
    path = tmp_path / "tiny.toml"
    path.write_text(
        TINY_TOML.format(shards=shards.as_posix(), lr=lr, ckpt=(tmp_path / "ckpt").as_posix()),
        encoding="utf-8",
    )
    return path


def args(config: Path, *extra: str) -> list[str]:
    return ["--config", str(config), "--run-id", RUN_ID, "--device", "cpu", *extra]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    # main() writes its run log under ./results/runs.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_log(tmp_path) -> Path:
    return tmp_path / "results" / "runs" / f"{RUN_ID}.json"


def test_a_completed_run_exits_0(in_tmp):
    assert run_main(args(tiny_config(in_tmp))) == EXIT_OK
    assert json.loads(run_log(in_tmp).read_text(encoding="utf-8"))["status"] == "completed"
    milestones = sorted(p.name for p in (in_tmp / "ckpt" / "milestones").iterdir())
    assert milestones == ["step_000003.pt", "step_000007.pt", "step_000020.pt"]


def test_the_non_finite_stop_exits_3(in_tmp, capsys):
    assert run_main(args(tiny_config(in_tmp, lr=EXPLODING_LR))) == EXIT_NONFINITE
    assert "non-finite" in capsys.readouterr().err


def test_an_interrupt_exits_130(in_tmp, monkeypatch):
    def interrupted(self):
        raise KeyboardInterrupt
    monkeypatch.setattr(Trainer, "train_step", interrupted)
    assert run_main(args(tiny_config(in_tmp))) == EXIT_INTERRUPTED


def test_any_other_crash_exits_1_with_a_traceback(in_tmp, monkeypatch, capsys):
    def broken(self):
        raise RuntimeError("cuda fell over")
    monkeypatch.setattr(Trainer, "train_step", broken)
    assert run_main(args(tiny_config(in_tmp))) == EXIT_CRASH
    err = capsys.readouterr().err
    assert "Traceback" in err and "cuda fell over" in err


def test_a_reused_run_id_exits_2(in_tmp, capsys):
    config = tiny_config(in_tmp)
    run_log(in_tmp).parent.mkdir(parents=True)
    run_log(in_tmp).write_text("{}", encoding="utf-8")
    assert run_main(args(config)) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "already has a log" in err and "--resume" in err
    assert "Traceback" not in err
    assert run_log(in_tmp).read_text(encoding="utf-8") == "{}"   # not clobbered


def test_resume_without_a_run_log_exits_2(in_tmp, capsys):
    assert run_main(args(tiny_config(in_tmp), "--resume")) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "cannot resume" in err and "Traceback" not in err


def test_resume_with_a_corrupt_run_log_exits_2(in_tmp, capsys):
    config = tiny_config(in_tmp)
    run_log(in_tmp).parent.mkdir(parents=True)
    run_log(in_tmp).write_text("{ truncated", encoding="utf-8")
    assert run_main(args(config, "--resume")) == EXIT_USAGE
    assert "cannot resume" in capsys.readouterr().err


def test_a_bad_config_exits_2(in_tmp, capsys):
    bad = in_tmp / "bad.toml"
    bad.write_text(tiny_config(in_tmp).read_text(encoding="utf-8")
                   .replace("milestones = [3, 7]", "milestones = [7, 3]"), encoding="utf-8")
    assert run_main(args(bad)) == EXIT_USAGE
    assert "milestones" in capsys.readouterr().err


def test_a_missing_config_exits_2(in_tmp):
    assert run_main(args(in_tmp / "nope.toml")) == EXIT_USAGE


def test_resume_refuses_to_restart_when_checkpoints_exist_but_latest_is_gone(in_tmp, monkeypatch):
    # Crash at step 17 with step_10 and step_15 on disk, then latest.pt deleted:
    # restarting from 0 would silently throw away 15 steps and truncate the log.
    config = tiny_config(in_tmp)
    real_step = Trainer.train_step

    def dies_at_17(self):
        if self.step == 17:
            raise RuntimeError("transient")
        return real_step(self)
    monkeypatch.setattr(Trainer, "train_step", dies_at_17)
    assert run_main(args(config)) == EXIT_CRASH
    monkeypatch.setattr(Trainer, "train_step", real_step)
    ckpt = in_tmp / "ckpt"
    assert sorted(p.name for p in ckpt.glob("step_*.pt")) == ["step_000010.pt", "step_000015.pt"]
    (ckpt / "latest.pt").unlink()
    before = run_log(in_tmp).read_bytes()

    assert run_main(args(config, "--resume")) == EXIT_USAGE
    assert run_log(in_tmp).read_bytes() == before              # untouched


def test_resume_refuses_to_restart_when_the_log_is_past_the_first_checkpoint(
        in_tmp, monkeypatch, capsys):
    # The checkpoint folder was moved (or ckpt_dir edited): no files at all, but the
    # log shows progress past the first checkpoint interval.
    config = tiny_config(in_tmp)
    real_step = Trainer.train_step

    def dies_at_17(self):
        if self.step == 17:
            raise RuntimeError("transient")
        return real_step(self)
    monkeypatch.setattr(Trainer, "train_step", dies_at_17)
    assert run_main(args(config)) == EXIT_CRASH
    monkeypatch.setattr(Trainer, "train_step", real_step)
    (in_tmp / "ckpt").rename(in_tmp / "ckpt_moved")
    before = run_log(in_tmp).read_bytes()
    capsys.readouterr()

    assert run_main(args(config, "--resume")) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "ckpt" in err and "step 17" in err and "latest.pt" in err
    assert run_log(in_tmp).read_bytes() == before


def test_a_typo_in_device_is_a_usage_error():
    assert run_main(["--device", "gpu"]) == 2


def test_resume_after_a_crash_before_the_first_checkpoint_restarts_at_0(in_tmp, monkeypatch):
    # The launcher passes --resume whenever a run log exists. If the first attempt
    # died before its first checkpoint, that resume must start over, not crash on
    # the missing latest.pt on every retry.
    config = tiny_config(in_tmp)
    real_step = Trainer.train_step

    def dies_at_2(self):
        if self.step == 2:
            raise RuntimeError("transient")
        return real_step(self)
    monkeypatch.setattr(Trainer, "train_step", dies_at_2)
    assert run_main(args(config)) == EXIT_CRASH
    monkeypatch.setattr(Trainer, "train_step", real_step)
    assert run_main(args(config, "--resume")) == EXIT_OK
    record = json.loads(run_log(in_tmp).read_text(encoding="utf-8"))
    assert [s["step"] for s in record["steps"]] == list(range(1, 21))


@pytest.mark.parametrize("exc, code", [
    (NonFiniteStop("x"), EXIT_NONFINITE),
    (KeyboardInterrupt(), EXIT_INTERRUPTED),
    (UsageError("x"), EXIT_USAGE),
    (FileNotFoundError("x"), EXIT_CRASH),
    (RuntimeError("x"), EXIT_CRASH),
])
def test_run_main_maps_exceptions_by_type(monkeypatch, exc, code):
    def main(argv=None):
        raise exc
    monkeypatch.setattr(train_mod, "main", main)
    assert run_main([]) == code


def test_non_finite_stop_is_identified_by_type_not_message(monkeypatch):
    # A plain RuntimeError that merely mentions non-finite values is an ordinary crash.
    def main(argv=None):
        raise RuntimeError("non-finite loss/grad for 3 consecutive steps")
    monkeypatch.setattr(train_mod, "main", main)
    assert run_main([]) == EXIT_CRASH
    assert issubclass(NonFiniteStop, RuntimeError)


def test_argparse_errors_keep_their_own_codes():
    assert run_main(["--no-such-flag"]) == 2
    assert run_main(["--help"]) == 0


def test_the_real_process_exits_3_on_the_non_finite_stop(tmp_path):
    config = tiny_config(tmp_path, lr=EXPLODING_LR)
    pythonpath = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONPATH=pythonpath)
    proc = subprocess.run(
        [sys.executable, "-m", "quipu.train", *args(config)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == EXIT_NONFINITE, proc.stderr
    assert "non-finite" in proc.stderr
