"""scripts/weekend.py against a fake child-process runner and fake guard probes.

No real training, no real nvidia-smi/GetSystemPowerStatus/disk calls: every
probe and the child-process runner are injected.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "weekend", Path(__file__).resolve().parents[1] / "scripts" / "weekend.py"
)
weekend = importlib.util.module_from_spec(_spec)
sys.modules["weekend"] = weekend  # dataclasses needs the module registered to resolve annotations
_spec.loader.exec_module(weekend)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class ScriptedRunner:
    """Returns exit codes from a fixed list, one per call, then repeats the last."""

    def __init__(self, codes: list[int]) -> None:
        self.codes = codes
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], log_path: Path) -> int:
        self.calls.append(cmd)
        idx = min(len(self.calls) - 1, len(self.codes) - 1)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"fake run of {cmd}\n", encoding="utf-8")
        return self.codes[idx]


def _sleeps() -> tuple[list[float], object]:
    calls: list[float] = []

    def sleep(s: float) -> None:
        calls.append(s)

    return calls, sleep


# --------------------------------------------------------------------------
# training retry logic
# --------------------------------------------------------------------------


def test_retries_on_exit_1_and_adds_resume(tmp_path):
    """A failed first attempt normally does leave a run log behind (the trainer
    writes RunLog before doing any real work), so the retry picks it up and
    adds --resume -- via the same fresh run_log_exists probe every attempt
    uses, not by assuming it."""
    runner = ScriptedRunner([1, 0])
    attempts = []
    sleeps, sleep = _sleeps()
    log_exists = {"value": False}

    def run_log_exists(rid):
        return log_exists["value"]

    def runner_then_flip(cmd, log_path):
        code = runner(cmd, log_path)
        log_exists["value"] = True  # the failed attempt left a run log behind
        return code

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=run_log_exists,
        on_attempt=attempts.append, run_child=runner_then_flip, sleep=sleep,
    )

    assert code == 0
    assert len(attempts) == 2
    assert attempts[0].returncode == 1 and attempts[0].retried is True
    assert attempts[1].returncode == 0 and attempts[1].retried is False
    assert "--resume" not in runner.calls[0]
    assert "--resume" in runner.calls[1]
    assert sleeps == [0]


def test_no_retry_on_exit_3_nonfinite(tmp_path):
    runner = ScriptedRunner([3])
    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert code == 3
    assert len(attempts) == 1
    assert attempts[0].retried is False


def test_no_retry_on_exit_130_interrupt(tmp_path):
    runner = ScriptedRunner([130])
    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert code == 130
    assert len(attempts) == 1
    assert attempts[0].retried is False


def test_retry_cap_reached_exit_is_last_code(tmp_path):
    runner = ScriptedRunner([1, 1, 1, 1, 1])  # more failures than max_retries allows
    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    # 1 initial attempt + 3 retries = 4 attempts, all exit 1, last one gives up.
    assert len(attempts) == 4
    assert code == 1
    assert attempts[-1].retried is False
    assert all(a.retried for a in attempts[:-1])


def test_resume_added_on_first_attempt_when_run_log_exists(tmp_path):
    runner = ScriptedRunner([0])
    attempts = []
    _, sleep = _sleeps()

    weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: True,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert "--resume" in runner.calls[0]


def test_no_resume_on_first_attempt_when_no_run_log(tmp_path):
    runner = ScriptedRunner([0])
    attempts = []
    _, sleep = _sleeps()

    weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert "--resume" not in runner.calls[0]


def test_no_retry_on_exit_2_usage_error(tmp_path):
    runner = ScriptedRunner([2])
    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert code == 2
    assert len(attempts) == 1
    assert attempts[0].retried is False
    assert "usage" in attempts[0].note.lower()


@pytest.mark.parametrize("windows_ctrl_c_code", [-1073741510, 3221225786])
def test_no_retry_on_windows_ctrl_c_exit_codes(tmp_path, windows_ctrl_c_code):
    runner = ScriptedRunner([windows_ctrl_c_code])
    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert code == windows_ctrl_c_code
    assert len(attempts) == 1
    assert attempts[0].retried is False


def test_keyboard_interrupt_from_runner_stops_without_retry(tmp_path):
    def runner(cmd, log_path):
        raise KeyboardInterrupt

    attempts = []
    _, sleep = _sleeps()

    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert code == weekend.EXIT_INTERRUPT
    assert len(attempts) == 1
    assert attempts[0].retried is False
    assert "ctrl+c" in attempts[0].note.lower()


def test_keyboard_interrupt_writes_owner_stop_summary(tmp_path):
    summary_path = tmp_path / "results" / "weekend_summary.md"
    state = weekend.WeekendState()

    def runner(cmd, log_path):
        raise KeyboardInterrupt

    def on_attempt(a):
        state.train_attempts.append(a)
        weekend.write_summary(state, summary_path)

    _, sleep = _sleeps()
    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=on_attempt, run_child=runner, sleep=sleep,
    )
    assert code == 130
    # main() sets the owner-stop final_status; simulate what it does so the
    # summary contract is checked end to end.
    state.final_status = "stopped by owner (Ctrl+C)"
    weekend.write_summary(state, summary_path)
    text = summary_path.read_text(encoding="utf-8")
    assert "stopped by owner (Ctrl+C)" in text


def test_resume_rechecked_before_every_attempt_not_forced_true(tmp_path):
    """A first attempt that crashes before ever writing a run log (e.g. exit 1
    from a usage-ish failure that still left no results/runs/<id>.json) must
    not get --resume on the retry just because *some* attempt failed."""
    runner = ScriptedRunner([1, 1, 0])
    attempts = []
    _, sleep = _sleeps()

    # No run log ever appears until after the 2nd attempt.
    log_appears_after = {"n": 0}

    def run_log_exists(rid):
        log_appears_after["n"] += 0  # probed multiple times per attempt is fine
        return len(attempts) >= 2  # simulates: log exists only after attempt 2

    weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=run_log_exists,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )

    assert "--resume" not in runner.calls[0]   # no log yet
    assert "--resume" not in runner.calls[1]   # still no log after attempt 1
    assert "--resume" in runner.calls[2]       # log appeared after attempt 2


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------


def _shard_dir_with(tmp_path, train=True, val=True):
    d = tmp_path / "shards"
    if train:
        (d / "train").mkdir(parents=True)
        (d / "train" / "shard0.bin").write_bytes(b"x")
    if val:
        (d / "val").mkdir(parents=True)
        (d / "val" / "shard0.bin").write_bytes(b"x")
    return d


def test_gpu_vram_guard_trips_and_force_bypasses(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)

    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 2.8, probe_battery=lambda: False,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    vram = next(g for g in results if g.name == "gpu-vram")
    assert vram.ok is False
    assert "fix" in vram.__dict__ and vram.fix

    forced = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=True,
        probe_vram=lambda: 2.8, probe_battery=lambda: False,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    assert next(g for g in forced if g.name == "gpu-vram").ok is True


def test_battery_guard_trips_and_force_bypasses(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)

    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 0.0, probe_battery=lambda: True,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    power = next(g for g in results if g.name == "power")
    assert power.ok is False

    forced = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=True,
        probe_vram=lambda: 0.0, probe_battery=lambda: True,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    assert next(g for g in forced if g.name == "power").ok is True


def test_disk_guard_trips_and_force_bypasses(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)

    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=40.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 0.0, probe_battery=lambda: False,
        probe_disk=lambda p: 10.0, probe_shards=lambda d: (True, True),
    )
    disk = next(g for g in results if g.name == "disk")
    assert disk.ok is False

    forced = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=40.0, shard_dir=shard_dir, force=True,
        probe_vram=lambda: 0.0, probe_battery=lambda: False,
        probe_disk=lambda p: 10.0, probe_shards=lambda d: (True, True),
    )
    assert next(g for g in forced if g.name == "disk").ok is True


def test_shards_guard_trips_and_force_bypasses(tmp_path):
    shard_dir = _shard_dir_with(tmp_path, train=True, val=False)

    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 0.0, probe_battery=lambda: False, probe_disk=lambda p: 100.0,
    )
    shards = next(g for g in results if g.name == "shards")
    assert shards.ok is False
    assert "val" in shards.detail

    forced = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=True,
        probe_vram=lambda: 0.0, probe_battery=lambda: False, probe_disk=lambda p: 100.0,
    )
    assert next(g for g in forced if g.name == "shards").ok is True


def test_missing_nvidia_smi_warns_and_continues(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)
    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: None, probe_battery=lambda: False,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    vram = next(g for g in results if g.name == "gpu-vram")
    assert vram.ok is True


def test_all_guards_pass_when_healthy(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)
    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 0.1, probe_battery=lambda: False,
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    assert all(g.ok for g in results)


def test_power_unknown_fails_guard_and_force_bypasses(tmp_path):
    shard_dir = _shard_dir_with(tmp_path)

    results = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=False,
        probe_vram=lambda: 0.0, probe_battery=lambda: "unknown",
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    power = next(g for g in results if g.name == "power")
    assert power.ok is False
    assert "unknown" in power.detail.lower()
    assert power.fix

    forced = weekend.run_guards(
        max_other_vram_gb=1.5, min_free_gb=1.0, shard_dir=shard_dir, force=True,
        probe_vram=lambda: 0.0, probe_battery=lambda: "unknown",
        probe_disk=lambda p: 100.0, probe_shards=lambda d: (True, True),
    )
    assert next(g for g in forced if g.name == "power").ok is True


def test_system_power_status_acline_field_is_unsigned():
    """Would read back as -1, not 255, if the field were still c_byte."""
    status = weekend._SYSTEM_POWER_STATUS()
    status.ACLineStatus = 255
    assert status.ACLineStatus == 255


def test_probe_on_battery_returns_unknown_for_ac_line_status_255(monkeypatch):
    """The real probe (not a fake) must read ACLineStatus 255 as "unknown", not
    as -1 (which c_byte would have given) or silently as AC."""
    monkeypatch.setattr(weekend.platform, "system", lambda: "Windows")

    def fake_get_status(ptr):
        real = weekend.ctypes.cast(ptr, weekend.ctypes.POINTER(weekend._SYSTEM_POWER_STATUS)).contents
        real.ACLineStatus = 255
        return 1

    class FakeKernel32:
        GetSystemPowerStatus = staticmethod(fake_get_status)

    class FakeWindll:
        kernel32 = FakeKernel32()

    monkeypatch.setattr(weekend.ctypes, "windll", FakeWindll(), raising=False)
    assert weekend.probe_on_battery() == "unknown"


# --------------------------------------------------------------------------
# evaluation steps
# --------------------------------------------------------------------------


def test_evals_run_only_after_success_is_orchestrated_by_main(tmp_path, monkeypatch):
    """main() only calls run_evals when training exits 0; exercised via run_training
    return code directly, since that's the contract main() relies on."""
    runner = ScriptedRunner([1])
    attempts = []
    _, sleep = _sleeps()
    code = weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=0, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=attempts.append, run_child=runner, sleep=sleep,
    )
    assert code == 1  # main() would not proceed to run_evals with this code


def test_missing_scripts_are_skipped_not_errors(tmp_path):
    runner = ScriptedRunner([0])
    attempts = []
    weekend.run_evals(
        log_dir=tmp_path, on_attempt=attempts.append, run_child=runner,
        script_exists=lambda p: False,
    )
    by_kind = {a.kind: a for a in attempts}
    assert by_kind["milestone_eval"].returncode is None
    assert by_kind["milestone_eval"].note == "not present (skipped)"
    assert by_kind["needle_eval"].returncode is None
    assert by_kind["needle_eval"].note == "not present (skipped)"
    # results_table always runs, it's not an optional script
    assert by_kind["results_table"].returncode == 0


def test_present_scripts_run_and_a_failure_does_not_stop_later_steps(tmp_path):
    # milestone_eval fails (exit 1), needle_eval and results_table still run.
    runner = ScriptedRunner([1, 0, 0])
    attempts = []
    weekend.run_evals(
        log_dir=tmp_path, on_attempt=attempts.append, run_child=runner,
        script_exists=lambda p: True,
    )
    by_kind = {a.kind: a for a in attempts}
    assert by_kind["milestone_eval"].returncode == 1
    assert by_kind["milestone_eval"].note == "failed"
    assert by_kind["needle_eval"].returncode == 0
    assert by_kind["results_table"].returncode == 0
    assert len(attempts) == 3


def test_results_table_always_runs_even_with_no_optional_scripts(tmp_path):
    runner = ScriptedRunner([0])
    attempts = []
    weekend.run_evals(
        log_dir=tmp_path, on_attempt=attempts.append, run_child=runner,
        script_exists=lambda p: False,
    )
    assert attempts[-1].kind == "results_table"
    assert runner.calls[-1][:3] == [runner.calls[-1][0], "-m", "quipu.results_table"]


# --------------------------------------------------------------------------
# summary file
# --------------------------------------------------------------------------


def test_summary_written_after_each_train_attempt(tmp_path):
    summary_path = tmp_path / "results" / "weekend_summary.md"
    state = weekend.WeekendState()
    seen_after_first_write = []

    runner = ScriptedRunner([1, 0])

    def on_attempt(a):
        state.train_attempts.append(a)
        weekend.write_summary(state, summary_path)
        seen_after_first_write.append(summary_path.read_text(encoding="utf-8"))

    _, sleep = _sleeps()
    weekend.run_training(
        config="cfg.toml", run_id="run-a", max_retries=3, retry_wait_s=0,
        log_dir=tmp_path, run_log_exists=lambda rid: False,
        on_attempt=on_attempt, run_child=runner, sleep=sleep,
    )

    assert summary_path.exists()
    assert len(seen_after_first_write) == 2
    assert "exit 1" in seen_after_first_write[0]
    final = summary_path.read_text(encoding="utf-8")
    assert "exit 0" in final
    assert "exit 1" in final  # both attempts remembered, not just the last


def test_summary_includes_guard_results():
    state = weekend.WeekendState()
    state.guard_results = [
        weekend.GuardResult("disk", False, "3.0 GB free", fix="free up space"),
    ]
    text = weekend.render_summary(state)
    assert "disk" in text
    assert "free up space" in text


def test_summary_is_written_atomically(tmp_path, monkeypatch):
    """It must go through quipu.fsio.replace_with_retry (tmp then swap), not a
    direct write to the final path."""
    summary_path = tmp_path / "results" / "weekend_summary.md"
    calls = []
    real = weekend.replace_with_retry

    def spy(src, dst, *a, **kw):
        calls.append((Path(src).name, Path(dst).name))
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(weekend, "replace_with_retry", spy)
    weekend.write_summary(weekend.WeekendState(), summary_path)
    assert calls and calls[0][0].endswith(".tmp")
    assert summary_path.exists()


# --------------------------------------------------------------------------
# default_run_child (real subprocess machinery, Popen mocked)
# --------------------------------------------------------------------------


def test_default_run_child_sets_pythonunbuffered(tmp_path, monkeypatch):
    captured_kwargs = {}

    class FakeProc:
        stdout = iter([])
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(weekend.subprocess, "Popen", fake_popen)
    code = weekend.default_run_child(["prog"], tmp_path / "attempt.log")

    assert code == 0
    assert captured_kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured_kwargs["stderr"] == weekend.subprocess.STDOUT


def test_default_run_child_stops_the_child_on_keyboard_interrupt(tmp_path, monkeypatch):
    events = []

    class FakeStdout:
        def __iter__(self):
            events.append("reading")
            raise KeyboardInterrupt

    class FakeProc:
        stdout = FakeStdout()

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    def fake_popen(cmd, **kwargs):
        return FakeProc()

    monkeypatch.setattr(weekend.subprocess, "Popen", fake_popen)

    with pytest.raises(KeyboardInterrupt):
        weekend.default_run_child(["prog"], tmp_path / "attempt.log")

    assert "reading" in events
    assert any(e == ("wait", 10.0) for e in events)
    assert "terminate" not in events  # the fake process "exits" within the wait


# --------------------------------------------------------------------------
# main() reads the shard directory from --config, not a hard-coded path
# --------------------------------------------------------------------------


def test_main_passes_config_shard_dir_to_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(weekend, "REPO_ROOT", tmp_path)
    captured = {}

    class FakeData:
        shard_dir = "custom_shards"

    class FakeCfg:
        data = FakeData()

    monkeypatch.setattr(weekend, "load_config", lambda path: FakeCfg())

    def fake_run_guards(**kwargs):
        captured.update(kwargs)
        return [weekend.GuardResult("shards", True, "present")]

    monkeypatch.setattr(weekend, "run_guards", fake_run_guards)

    code = weekend.main(["--config", "whatever.toml", "--dry-run"])

    assert code == 0
    assert captured["shard_dir"] == tmp_path / "custom_shards"


def test_resolve_leaves_absolute_paths_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(weekend, "REPO_ROOT", tmp_path / "repo")
    absolute = tmp_path / "elsewhere" / "shards"
    assert weekend._resolve(str(absolute)) == absolute


def test_resolve_anchors_relative_paths_at_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(weekend, "REPO_ROOT", tmp_path)
    assert weekend._resolve("data/shards") == tmp_path / "data" / "shards"


# --------------------------------------------------------------------------
# CLI wiring (parse_args)
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    args = weekend.parse_args([])
    assert args.config == "configs/quipu-114m.toml"
    assert args.run_id == "quipu-114m-weekend"
    assert args.force is False
    assert args.max_retries == 3
    assert args.max_other_vram_gb == 1.5
    assert args.min_free_gb == 40.0
    assert args.dry_run is False


def test_parse_args_overrides():
    args = weekend.parse_args([
        "--config", "c.toml", "--run-id", "rid", "--force",
        "--retry-wait-s", "0", "--max-retries", "1",
        "--max-other-vram-gb", "2.0", "--min-free-gb", "5", "--dry-run",
    ])
    assert args.config == "c.toml"
    assert args.run_id == "rid"
    assert args.force is True
    assert args.retry_wait_s == 0
    assert args.max_retries == 1
    assert args.dry_run is True
