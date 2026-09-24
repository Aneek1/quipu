import json

from quipu.results_table import build_table


def write_run(tmp_path, run_id, final_train, final_val, status="completed"):
    record = {
        "run_id": run_id,
        "status": status,
        "config": {"model": {"d_model": 768, "n_layer": 12}, "train": {"total_tokens": 2_500_000_000}},
        "steps": [{"step": 1, "train_loss": 10.0, "lr": 1e-4, "tokens": 1},
                  {"step": 2, "train_loss": final_train, "lr": 1e-4, "tokens": 2}],
        "evals": [{"step": 2, "val_loss": final_val}],
    }
    (tmp_path / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")


def test_table_has_one_row_per_run(tmp_path):
    write_run(tmp_path, "run-a", 3.9, 4.0)
    write_run(tmp_path, "run-b", 3.5, 3.6)
    table = build_table(tmp_path)
    assert "run-a" in table and "run-b" in table
    assert table.count("\n|") >= 3          # header, separator, two rows


def test_table_reports_the_final_losses(tmp_path):
    write_run(tmp_path, "run-a", 3.9123, 4.0456)
    table = build_table(tmp_path)
    assert "3.9123" in table and "4.0456" in table


def test_unfinished_runs_are_marked_not_hidden(tmp_path):
    # A run that died at hour 19 is still evidence and must not vanish from the table.
    write_run(tmp_path, "killed", 5.0, 5.1, status="interrupted")
    assert "interrupted" in build_table(tmp_path)


def test_a_run_with_no_evals_does_not_crash_the_table(tmp_path):
    record = {"run_id": "noeval", "status": "running", "config": {},
              "steps": [{"step": 1, "train_loss": 9.0, "lr": 1e-4, "tokens": 1}], "evals": []}
    (tmp_path / "noeval.json").write_text(json.dumps(record), encoding="utf-8")
    assert "noeval" in build_table(tmp_path)


def test_steps_column_shows_progress_against_plan(tmp_path):
    record = {
        "run_id": "planned",
        "status": "running",
        "config": {"derived": {"steps": 2861, "grad_accum": 4}},
        "steps": [{"step": 2861, "train_loss": 3.1, "lr": 1e-4, "tokens": 100}],
        "evals": [],
    }
    (tmp_path / "planned.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    assert "2,861 / 2,861" in table


def test_steps_column_falls_back_to_last_step_without_plan(tmp_path):
    write_run(tmp_path, "run-a", 3.9, 4.0)
    table = build_table(tmp_path)
    # no config.derived.steps in write_run's record, so just the last step number
    assert "| 2 |" in table


def test_tokens_column_is_formatted_with_thousands_separators(tmp_path):
    record = {
        "run_id": "big",
        "status": "running",
        "config": {},
        "steps": [{"step": 1, "train_loss": 3.0, "lr": 1e-4, "tokens": 1_234_567}],
        "evals": [],
    }
    (tmp_path / "big.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    assert "1,234,567" in table


def test_tokens_column_is_dash_when_no_steps(tmp_path):
    record = {"run_id": "nostep", "status": "running", "config": {}, "steps": [], "evals": []}
    (tmp_path / "nostep.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    lines = [l for l in table.splitlines() if l.startswith("| nostep")]
    assert lines and lines[0].count(" - ") >= 1


def test_resumes_column_counts_resumes(tmp_path):
    record = {
        "run_id": "resumed",
        "status": "running",
        "config": {},
        "steps": [{"step": 1, "train_loss": 3.0, "lr": 1e-4, "tokens": 1}],
        "evals": [],
        "resumes": [{"at": "2026-01-01T00:00:00+00:00", "from_step": 0},
                    {"at": "2026-01-02T00:00:00+00:00", "from_step": 500}],
    }
    (tmp_path / "resumed.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    lines = [l for l in table.splitlines() if l.startswith("| resumed")]
    assert lines and "| 2 |" in lines[0]


def test_resumes_column_defaults_to_zero_when_absent(tmp_path):
    # Older records may not have a "resumes" key at all.
    record = {
        "run_id": "old-record",
        "status": "completed",
        "config": {},
        "steps": [{"step": 1, "train_loss": 3.0, "lr": 1e-4, "tokens": 1}],
        "evals": [],
    }
    (tmp_path / "old-record.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    lines = [l for l in table.splitlines() if l.startswith("| old-record")]
    assert lines and "| 0 |" in lines[0]


def test_skipped_and_wraps_default_to_zero_when_absent(tmp_path):
    # Older/test records may not carry the "skipped"/"wraps" extras on steps.
    record = {
        "run_id": "old-record-2",
        "status": "completed",
        "config": {},
        "steps": [{"step": 1, "train_loss": 3.0, "lr": 1e-4, "tokens": 1}],
        "evals": [],
    }
    (tmp_path / "old-record-2.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    lines = [l for l in table.splitlines() if l.startswith("| old-record-2")]
    assert lines
    # last two columns before the trailing pipe are skipped, wraps
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[-2:] == ["0", "0"]


def test_skipped_and_wraps_report_nonzero_values(tmp_path):
    record = {
        "run_id": "flaky",
        "status": "completed",
        "config": {},
        "steps": [{"step": 5, "train_loss": 3.0, "lr": 1e-4, "tokens": 5, "skipped": 3, "wraps": 2}],
        "evals": [],
    }
    (tmp_path / "flaky.json").write_text(json.dumps(record), encoding="utf-8")
    table = build_table(tmp_path)
    lines = [l for l in table.splitlines() if l.startswith("| flaky")]
    assert lines
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[-2:] == ["3", "2"]


def test_unreadable_run_file_gets_a_flagged_row_not_a_crash(tmp_path):
    (tmp_path / "corrupt.json").write_text("{not valid json", encoding="utf-8")
    write_run(tmp_path, "run-a", 3.9, 4.0)
    table = build_table(tmp_path)
    assert "corrupt" in table
    assert "unreadable" in table
    lines = [l for l in table.splitlines() if l.startswith("| corrupt")]
    assert lines
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[1] == "unreadable"
    assert all(c == "-" for c in cells[2:])


def test_rows_are_sorted_by_run_id(tmp_path):
    write_run(tmp_path, "zeta", 3.9, 4.0)
    write_run(tmp_path, "alpha", 3.5, 3.6)
    write_run(tmp_path, "mu", 3.7, 3.8)
    table = build_table(tmp_path)
    body_lines = [l for l in table.splitlines() if l.startswith("| ")][2:]  # skip header + separator
    run_ids = [l.split("|")[1].strip() for l in body_lines]
    assert run_ids == sorted(run_ids)
