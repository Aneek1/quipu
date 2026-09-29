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


STOP_SIGNALS = [s for s in ("SIGTERM", "SIGBREAK") if hasattr(__import__("signal"), s)]


@pytest.mark.parametrize("name", STOP_SIGNALS)
def test_sigterm_checkpoints_and_exits_130_like_an_interrupt(in_tmp, monkeypatch, name):
    # The A/B orchestrator and the box launcher stop a run with SIGINT and then, if it
    # is still alive, SIGTERM (Windows: CTRL_BREAK -> SIGBREAK). Either must take the
    # interrupt path: checkpoint at the current step, log "interrupted", exit 130.
    import signal

    sig = getattr(signal, name)
    before = signal.getsignal(sig)
    real_step = Trainer.train_step

    def step_then_signal(self):
        loss = real_step(self)
        if self.step == 3:
            signal.raise_signal(sig)
        return loss

    monkeypatch.setattr(Trainer, "train_step", step_then_signal)
    assert run_main(args(tiny_config(in_tmp))) == EXIT_INTERRUPTED
    record = json.loads(run_log(in_tmp).read_text(encoding="utf-8"))
    assert record["status"] == "interrupted"
    assert (in_tmp / "ckpt" / "step_000003.pt").exists()
    assert signal.getsignal(sig) == before          # run_main restores the handler


def test_every_checkpoint_prints_its_save_time(in_tmp, capsys):
    # The A/B runner and the box launcher size their stop grace from this line.
    import re
    assert run_main(args(tiny_config(in_tmp))) == EXIT_OK
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("checkpoint step")]
    assert [int(l.split()[2]) for l in lines] == [5, 10, 15, 20]
    assert all(re.fullmatch(r"checkpoint step \d+ saved in \d+\.\d s \(\d+\.\d GB\)", l)
               for l in lines)


def test_a_second_sigint_during_the_interrupt_checkpoint_is_ignored(in_tmp, monkeypatch, capsys):
    # Ctrl+C at step 3, then Ctrl+C again while the (slow) interrupt checkpoint is
    # being written: the save must complete, the run log says interrupted, exit 130.
    import signal
    import time as time_mod

    before = signal.getsignal(signal.SIGINT)
    real_step, real_save = Trainer.train_step, train_mod._atomic_save
    state = {"interrupted": False, "repeats": 0}

    def step_then_sigint(self):
        loss = real_step(self)
        if self.step == 3:
            state["interrupted"] = True
            signal.raise_signal(signal.SIGINT)
        return loss

    def slow_save(obj, path):
        if state["interrupted"] and path.name.startswith("step_"):
            time_mod.sleep(0.2)
            for _ in range(2):          # two more Ctrl+C mid-save
                state["repeats"] += 1
                signal.raise_signal(signal.SIGINT)
        real_save(obj, path)

    monkeypatch.setattr(Trainer, "train_step", step_then_sigint)
    monkeypatch.setattr(train_mod, "_atomic_save", slow_save)
    assert run_main(args(tiny_config(in_tmp))) == EXIT_INTERRUPTED
    assert state["repeats"] == 2
    ckpt = in_tmp / "ckpt"
    assert (ckpt / "step_000003.pt").exists() and (ckpt / "latest.pt").exists()
    assert json.loads(run_log(in_tmp).read_text(encoding="utf-8"))["status"] == "interrupted"
    out = capsys.readouterr()
    assert "checkpoint step 3 saved in" in out.out
    assert out.err.count("again; still saving the interrupt checkpoint") == 2
    assert signal.getsignal(signal.SIGINT) is before       # restored


def test_an_ignored_sigint_is_left_alone(in_tmp, monkeypatch):
    # A process started with SIGINT ignored (nohup-style) keeps ignoring it.
    import signal

    before = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        seen = []

        def step(self):
            seen.append(signal.getsignal(signal.SIGINT))
            raise KeyboardInterrupt
        monkeypatch.setattr(Trainer, "train_step", step)
        assert run_main(args(tiny_config(in_tmp))) == EXIT_INTERRUPTED
        assert seen == [signal.SIG_IGN]
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, before)


SLOW_SAVE_CHILD = r'''
import os, signal, sys, time
from pathlib import Path
import quipu.train as t
flag = Path(sys.argv[1])
real_save = t._atomic_save
def slow_save(obj, path):
    if path.name.startswith("step_") and t._INTERRUPTED_FOR_TEST:
        flag.write_text("saving")
        time.sleep(3.0)
    real_save(obj, path)
t._atomic_save = slow_save
t._INTERRUPTED_FOR_TEST = False
real_step = t.Trainer.train_step
def step(self):
    loss = real_step(self)
    if self.step == 3:
        t._INTERRUPTED_FOR_TEST = True
        os.kill(os.getpid(), signal.SIGINT)
    return loss
t.Trainer.train_step = step
sys.exit(t.run_main(sys.argv[2:]))
'''


@pytest.mark.skipif(os.name == "nt", reason="POSIX: a real SIGINT from outside the process")
def test_the_real_process_survives_a_second_sigint_during_its_checkpoint(tmp_path):
    import signal
    import time as time_mod

    config = tiny_config(tmp_path)
    script = tmp_path / "slow_child.py"
    script.write_text(SLOW_SAVE_CHILD, encoding="utf-8")
    flag = tmp_path / "saving.flag"
    pythonpath = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONPATH=pythonpath)
    proc = subprocess.Popen([sys.executable, str(script), str(flag), *args(config)],
                            cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        deadline = time_mod.monotonic() + 240
        while not flag.exists() and proc.poll() is None and time_mod.monotonic() < deadline:
            time_mod.sleep(0.05)
        assert flag.exists(), "the interrupt checkpoint never started"
        proc.send_signal(signal.SIGINT)          # a second Ctrl+C mid-save
        out, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == EXIT_INTERRUPTED, err
    assert "again; still saving the interrupt checkpoint" in err
    assert "checkpoint step 3 saved in" in out
    assert (tmp_path / "ckpt" / "step_000003.pt").exists()
