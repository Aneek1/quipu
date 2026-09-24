import json
from pathlib import Path

import pytest

from quipu.runlog import RunLog


def test_writes_a_record_with_config_and_metrics(tmp_path):
    log = RunLog(tmp_path, run_id="test-run", config={"name": "quipu-114m"})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=524288)
    log.log_step(step=2, train_loss=4.5, lr=2e-4, tokens=1048576)
    log.finish(status="completed")

    record = json.loads((tmp_path / "test-run.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "test-run"
    assert record["config"]["name"] == "quipu-114m"
    assert record["status"] == "completed"
    assert len(record["steps"]) == 2
    assert record["steps"][-1]["train_loss"] == 4.5


def test_survives_being_killed_mid_run(tmp_path):
    # A 20-hour run that loses its log because the laptop slept is not acceptable.
    log = RunLog(tmp_path, run_id="killed", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    record = json.loads((tmp_path / "killed.json").read_text(encoding="utf-8"))
    assert record["status"] == "running"
    assert len(record["steps"]) == 1


def test_records_validation_loss_separately(tmp_path):
    log = RunLog(tmp_path, run_id="val", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    log.log_eval(step=1, val_loss=4.9)
    record = json.loads((tmp_path / "val.json").read_text(encoding="utf-8"))
    assert record["evals"] == [{"step": 1, "val_loss": 4.9}]


def test_failed_flush_never_destroys_the_last_good_record(tmp_path, monkeypatch):
    log = RunLog(tmp_path, run_id="atomic", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)

    before = (tmp_path / "atomic.json").read_text(encoding="utf-8")
    before_record = json.loads(before)
    assert len(before_record["steps"]) == 1

    monkeypatch.setattr("quipu.fsio.os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        log.log_step(step=2, train_loss=4.0, lr=1e-4, tokens=2)

    after = (tmp_path / "atomic.json").read_text(encoding="utf-8")
    assert after == before
    after_record = json.loads(after)
    assert len(after_record["steps"]) == 1
    assert after_record["steps"][-1]["train_loss"] == 5.0


def test_config_with_path_is_json_serializable(tmp_path):
    log = RunLog(tmp_path, run_id="path-cfg", config={"data_dir": Path("/some/data")})
    log.finish(status="completed")

    record = json.loads((tmp_path / "path-cfg.json").read_text(encoding="utf-8"))
    assert record["config"]["data_dir"] == str(Path("/some/data"))


def test_resume_preserves_prior_steps_and_records_a_resume_event(tmp_path):
    log = RunLog(tmp_path, run_id="resumable", config={"name": "quipu-114m"})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    log.log_step(step=2, train_loss=4.5, lr=1e-4, tokens=2)

    resumed = RunLog(tmp_path, run_id="resumable", config={"name": "quipu-114m"}, resume=True)

    record = json.loads((tmp_path / "resumable.json").read_text(encoding="utf-8"))
    assert len(record["steps"]) == 2
    assert record["status"] == "running"
    assert len(record["resumes"]) == 1
    assert record["resumes"][0]["from_step"] == 2
    assert "at" in record["resumes"][0]

    resumed.log_step(step=3, train_loss=4.0, lr=1e-4, tokens=3)
    record = json.loads((tmp_path / "resumable.json").read_text(encoding="utf-8"))
    assert len(record["steps"]) == 3


def test_resume_on_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        RunLog(tmp_path, run_id="never-started", config={}, resume=True)


def test_resume_on_unparseable_file_raises(tmp_path):
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        RunLog(tmp_path, run_id="corrupt", config={}, resume=True)


def test_non_resume_construction_over_existing_file_raises(tmp_path):
    RunLog(tmp_path, run_id="dup", config={})
    with pytest.raises(FileExistsError):
        RunLog(tmp_path, run_id="dup", config={})


def test_non_resume_construction_over_existing_file_does_not_touch_it(tmp_path):
    log = RunLog(tmp_path, run_id="dup2", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    before = (tmp_path / "dup2.json").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        RunLog(tmp_path, run_id="dup2", config={})

    after = (tmp_path / "dup2.json").read_text(encoding="utf-8")
    assert after == before


def test_truncate_to_drops_later_steps_and_evals(tmp_path):
    log = RunLog(tmp_path, run_id="trunc", config={})
    log.log_step(step=1, train_loss=5.0, lr=1e-4, tokens=1)
    log.log_step(step=2, train_loss=4.5, lr=1e-4, tokens=2)
    log.log_step(step=3, train_loss=4.0, lr=1e-4, tokens=3)
    log.log_eval(step=1, val_loss=4.9)
    log.log_eval(step=3, val_loss=3.9)

    log.truncate_to(2)

    record = json.loads((tmp_path / "trunc.json").read_text(encoding="utf-8"))
    assert [s["step"] for s in record["steps"]] == [1, 2]
    assert [e["step"] for e in record["evals"]] == [1]
