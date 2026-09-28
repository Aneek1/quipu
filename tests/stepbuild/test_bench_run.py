"""The benchmark runner: scores, the reference model, the leakage guard and the CLI.

The summary tests use hand-made records, so the arithmetic is checked without
running anything. Only the CLI test runs a real sandbox (npm-marked).
"""
import json

import pytest

from stepbuild.bench import run as bench_run
from stepbuild.bench.acceptance import list_apps, load_reference
from stepbuild.bench.run import LeakageError, ReferenceModel, main, run_bench, summarise
from stepbuild.harness.model import StepModel
from stepbuild.harness.retrieve import Example, ExampleLibrary


def _step(number, key, attempts, passed=True, model_step=True):
    return {
        "number": number, "key": key, "model_step": model_step,
        "passed": passed, "attempts": attempts,
    }


KEYS = ["model", "routes", "api_tests", "components", "wiring"]


def _all_steps(attempts):
    steps = [_step(i + 1, k, a) for i, (k, a) in enumerate(zip(KEYS, attempts))]
    steps.append(_step(6, "run", 1, model_step=False))
    return steps


RECORDS = [
    {   # every step first try, acceptance passes
        "app": "todo", "model": "m", "status": "passed_steps", "acceptance_passed": True,
        "steps": _all_steps([1, 1, 1, 1, 1]), "seconds": 10.0, "trace": "t/todo.json",
    },
    {   # two retries on routes, then acceptance fails
        "app": "notes", "model": "m", "status": "passed_steps", "acceptance_passed": False,
        "steps": _all_steps([1, 3, 1, 1, 1]), "seconds": 20.0, "trace": "t/notes.json",
    },
    {   # fails at step 2 after four attempts; acceptance never ran
        "app": "habits", "model": "m", "status": "failed_at_step_2", "acceptance_passed": None,
        "steps": [_step(1, "model", 1), _step(2, "routes", 4, passed=False)],
        "seconds": 30.0, "trace": "t/habits.json",
    },
]


def test_summarise_totals():
    text = summarise(RECORDS)
    # 1 of 3 apps passes acceptance
    assert "Apps passing acceptance: 1/3 (33.3%)" in text
    # model steps that ran: 5 + 5 + 2 = 12; first try: 5 + 4 + 1 = 10
    assert "First-try step pass rate (model steps): 10/12 (83.3%)" in text
    # retries: 0 + 2 + 3 = 5 over 12 model steps
    assert "Mean retries per model step: 0.42 (5 retries over 12 model steps)" in text
    assert "Failing steps: acceptance 1, routes 1" in text
    assert "Wall time: 60.0 s total, 20.0 s mean per app" in text


def test_summarise_rows():
    lines = summarise(RECORDS).splitlines()
    row = {line.split("|")[1].strip(): line for line in lines if line.startswith("| ")}
    assert [c.strip() for c in row["todo"].split("|")[1:-1]] == [
        "todo", "passed_steps", "pass", "1 1 1 1 1 1", "0", "10.0",
    ]
    assert [c.strip() for c in row["notes"].split("|")[1:-1]] == [
        "notes", "passed_steps", "fail", "1 3 1 1 1 1", "2", "20.0",
    ]
    assert [c.strip() for c in row["habits"].split("|")[1:-1]] == [
        "habits", "failed_at_step_2", "-", "1 4", "3", "30.0",
    ]


def test_summarise_all_passing_is_100_percent_and_no_failures():
    text = summarise(RECORDS[:1])
    assert "Apps passing acceptance: 1/1 (100.0%)" in text
    assert "First-try step pass rate (model steps): 5/5 (100.0%)" in text
    assert "Failing steps: none" in text


def test_summarise_empty():
    assert "no apps were run" in summarise([])


def test_reference_model_serves_each_apps_replies_in_order():
    base = ReferenceModel()
    assert isinstance(base, StepModel)
    assert base.name == "reference" and base.context_tokens is None
    todo = base.for_app("todo")
    notes = base.for_app("notes")
    assert todo.name == "reference" and todo.context_tokens is None
    msgs = [{"role": "user", "content": "x"}]
    assert [todo.complete(msgs) for _ in range(5)] == load_reference("todo")
    assert notes.complete(msgs) == load_reference("notes")[0]
    with pytest.raises(RuntimeError):
        todo.complete(msgs)  # only five replies exist


def test_reference_model_without_an_app_refuses():
    with pytest.raises(RuntimeError, match="for_app"):
        ReferenceModel().complete([{"role": "user", "content": "x"}])


def test_leakage_guard_refuses_a_reference_reply_in_the_library(tmp_path, monkeypatch):
    # The leaked reply belongs to an app that is NOT being run: any benchmark
    # reference in the library is a leak.
    leaked = load_reference("notes")[2]
    library = ExampleLibrary([Example("fine", "=== FILE: a.py ===\nx = 1\n"),
                              Example("leak", leaked.replace("\n", "\r\n"))])

    def boom(*a, **k):
        raise AssertionError("no sandbox may be created when the library leaks")

    monkeypatch.setattr(bench_run, "create_sandbox", boom)
    with pytest.raises(LeakageError, match="notes"):
        run_bench(ReferenceModel(), ["todo"], tmp_path, library)
    assert not any(tmp_path.iterdir())


def test_leakage_guard_accepts_a_clean_library():
    bench_run.check_no_leakage(ExampleLibrary([Example("s", "=== FILE: a.py ===\nx = 1\n")]))


def test_unknown_app_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown app"):
        run_bench(ReferenceModel(), ["nope"], tmp_path, ExampleLibrary([]))


def test_cli_rejects_unknown_model():
    with pytest.raises(SystemExit):
        main(["--model", "gpt-9"])


@pytest.mark.npm
def test_cli_reference_todo_writes_trace_and_summary(tmp_path, capsys):
    code = main(["--model", "reference", "--apps", "todo", "--out", str(tmp_path)])
    assert code == 0
    trace = json.loads((tmp_path / "reference" / "todo.json").read_text(encoding="utf-8"))
    assert trace["status"] == "passed_steps"
    assert trace["acceptance"]["passed"] is True
    assert len(trace["steps"]) == 6
    summary = (tmp_path / "reference" / "summary.md").read_text(encoding="utf-8")
    assert "Apps passing acceptance: 1/1 (100.0%)" in summary
    assert "First-try step pass rate (model steps): 5/5 (100.0%)" in summary
    assert "| todo | passed_steps | pass |" in summary
    assert "Apps passing acceptance: 1/1 (100.0%)" in capsys.readouterr().out


def _fake_sandboxes(monkeypatch, tmp_path):
    """Replace sandbox creation/removal with fakes that record what happened."""
    from stepbuild.harness.sandbox import Sandbox

    made, removed = [], []

    def create(parent, cache):
        box = Sandbox(tmp_path / f"box{len(made)}")
        made.append(box)
        return box

    monkeypatch.setattr(bench_run, "create_sandbox", create)
    monkeypatch.setattr(bench_run, "remove_sandbox", removed.append)
    return made, removed


def test_sandbox_is_removed_even_when_the_run_raises(tmp_path, monkeypatch):
    made, removed = _fake_sandboxes(monkeypatch, tmp_path)

    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(bench_run, "run_app", explode)
    with pytest.raises(OSError):
        run_bench(ReferenceModel(), ["todo"], tmp_path / "out", ExampleLibrary([]))
    assert removed == made and len(made) == 1


def test_failed_steps_skip_acceptance_and_records_are_written(tmp_path, monkeypatch):
    from stepbuild.harness.checks import CheckResult
    from stepbuild.harness.runner import AppResult, Attempt, StepTrace

    made, removed = _fake_sandboxes(monkeypatch, tmp_path)
    seen_models = []

    def fake_run_app(model, plan, sandbox, library, max_retries=3):
        seen_models.append(model)
        bad = Attempt("r", None, (CheckResult("pyflakes", False, "boom", 0.1),))
        ok = Attempt("r", None, (CheckResult("pyflakes", True, "", 0.1),))
        steps = (StepTrace(1, "model", (ok,), True), StepTrace(2, "routes", (bad, bad), False))
        return AppResult(plan.app, model.name, "failed_at_step_2", steps, 1.5)

    def no_acceptance(*a, **k):
        raise AssertionError("acceptance must not run after a failed step")

    monkeypatch.setattr(bench_run, "run_app", fake_run_app)
    monkeypatch.setattr(bench_run, "run_acceptance", no_acceptance)
    out = tmp_path / "out"
    records = run_bench(ReferenceModel(), ["todo", "notes"], out, ExampleLibrary([]))
    assert [r["app"] for r in records] == ["todo", "notes"]
    assert removed == made and len(made) == 2
    # the per-app hook was used: each app got its own bound reference model
    assert [m.app for m in seen_models] == ["todo", "notes"]
    rec = records[0]
    assert rec["status"] == "failed_at_step_2" and rec["acceptance_passed"] is None
    assert rec["steps"] == [
        {"number": 1, "key": "model", "model_step": True, "passed": True, "attempts": 1},
        {"number": 2, "key": "routes", "model_step": True, "passed": False, "attempts": 2},
    ]
    trace = json.loads((out / "reference" / "todo.json").read_text(encoding="utf-8"))
    assert trace["acceptance"] is None and trace["status"] == "failed_at_step_2"
    assert rec["trace"] == str(out / "reference" / "todo.json")
    assert (out / "reference" / "summary.md").is_file()
