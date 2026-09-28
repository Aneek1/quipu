"""The benchmark runner: scores, provenance, the reference model, the leakage
guard, harness-error handling and the CLI.

The summary tests use hand-made records, so the arithmetic is checked without
running anything; run_bench is exercised with a fake sandbox and runner. Only the
CLI test builds a real app (npm-marked).
"""
import itertools
import json
import logging

import pytest

from stepbuild.bench import run as bench_run
from stepbuild.bench.acceptance import list_apps, load_reference
from stepbuild.bench.run import (
    LeakageError, ReferenceModel, main, model_slug, run_bench, summarise,
)
from stepbuild.harness.blocks import parse_blocks
from stepbuild.harness.checks import CheckResult
from stepbuild.harness.model import StepModel
from stepbuild.harness.retrieve import Example, ExampleLibrary
from stepbuild.harness.runner import AppResult, Attempt, StepTrace
from stepbuild.harness.sandbox import Sandbox

KEYS = ["model", "routes", "api_tests", "components", "wiring"]


def _step(number, key, attempts, passed=True, model_step=True, kinds=None):
    return {
        "number": number, "key": key, "model_step": model_step, "passed": passed,
        "attempts": attempts, "failure_kinds": kinds or {},
    }


def _all_steps(attempts):
    steps = [
        _step(i + 1, k, a, kinds={"pyflakes": a - 1} if a > 1 else None)
        for i, (k, a) in enumerate(zip(KEYS, attempts))
    ]
    steps.append(_step(6, "run", 1, model_step=False))
    return steps


def _rec(app, status, acc, steps, seconds, error=None):
    return {
        "app": app, "model": "m", "status": status, "acceptance_passed": acc,
        "steps": steps, "max_retries": 3, "error": error, "cleanup_error": None,
        "run_seconds": seconds, "total_seconds": seconds, "trace": f"t/{app}.json",
    }


RECORDS = [
    # every step first try, acceptance passes
    _rec("todo", "passed_steps", True, _all_steps([1, 1, 1, 1, 1]), 10.0),
    # two retries on routes, every step passes, acceptance fails
    _rec("notes", "passed_steps", False, _all_steps([1, 3, 1, 1, 1]), 20.0),
    # fails at step 2 after exhausting four attempts; acceptance never ran
    _rec("habits", "failed_at_step_2", None,
         [_step(1, "model", 1), _step(2, "routes", 4, passed=False, kinds={"pytest": 4})],
         30.0),
    # step 1 passes on the second try, step 2 hits a model_error at once
    _rec("recipes", "failed_at_step_2", None,
         [_step(1, "model", 2, kinds={"format": 1}),
          _step(2, "routes", 1, passed=False, kinds={"model_error": 1})],
         5.0),
    # the harness itself broke
    _rec("contacts", "harness_error", None, [], 15.0, error="OSError: disk full"),
]


def test_summarise_totals():
    text = summarise(RECORDS)
    # N = 5 apps, 25 model steps in all
    assert "**Apps passing acceptance: 1/5 (20.0%)**" in text
    assert "Apps completing all steps: 2/5 (40.0%)" in text
    # passed: 5 + 5 + 1 + 1 = 12; first try: 5 + 4 + 1 + 0 = 10; reached 5+5+2+2 = 14
    assert "Steps passed (of all 25): 12/25 (48.0%)" in text
    assert "Steps passed first try (of all 25): 10/25 (40.0%)" in text
    assert "Steps reached (of all 25): 14/25 (56.0%)" in text
    assert "First-try rate among reached steps (conditional): 10/14 (71.4%)" in text
    # retries on passed steps: notes routes 2 + recipes model 1 = 3 over 12
    assert "Mean retries per passed step: 0.25 (3 retries over 12 passed steps)" in text
    assert "Steps that exhausted the retry budget: 1" in text
    # attempts on model steps: 5 + 7 + 5 + 3 = 20; one model_error
    assert ("WARNING: Attempts lost to infrastructure (model_error, prompt_too_long): "
            "1/20 (5.0%)") in text
    assert ("Where non-passing apps stopped: step 2 routes: 2, "
            "acceptance (all steps passed): 1, harness_error: 1") in text
    assert "Harness errors: contacts (OSError: disk full)" in text
    assert ("Wall time (steps + acceptance, incl. sandbox setup): 80.0 s total, "
            "16.0 s mean per app") in text
    assert "only the 5 app record(s) of this run" in text


def _row(text, app):
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells and cells[0] == app:
            return cells
    raise AssertionError(f"no row for {app}")


def test_summarise_rows():
    text = summarise(RECORDS)
    assert _row(text, "todo") == [
        "todo", "passed_steps", "pass", "5", "-", "1 1 1 1 1 1", "0", "10.0"]
    assert _row(text, "notes") == [
        "notes", "passed_steps", "fail", "5", "acceptance (all steps passed)",
        "1 3 1 1 1 1", "2", "20.0"]
    assert _row(text, "habits") == [
        "habits", "failed_at_step_2", "-", "1", "step 2 routes", "1 4", "3", "30.0"]
    assert _row(text, "contacts") == [
        "contacts", "harness_error", "-", "0", "harness_error", "-", "0", "15.0"]


def test_summarise_all_passing_has_no_warnings():
    text = summarise(RECORDS[:1])
    assert "**Apps passing acceptance: 1/1 (100.0%)**" in text
    assert "Steps passed first try (of all 5): 5/5 (100.0%)" in text
    assert "- Attempts lost to infrastructure (model_error, prompt_too_long): 0/5 (0.0%)" in text
    assert "WARNING" not in text
    assert "Where non-passing apps stopped: none" in text
    assert "Harness errors: none" in text


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


# ------------------------------------------------------------------ leakage

CLEAN = "=== FILE: a.py ===\nx = 1\n=== END FILE ===\n"


def _no_sandbox(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no sandbox may be created when the library leaks")
    monkeypatch.setattr(bench_run, "create_sandbox", boom)


def test_leakage_guard_refuses_a_reference_reply_in_the_library(tmp_path, monkeypatch):
    # The leaked reply belongs to an app that is NOT being run: any benchmark
    # reference in the library is a leak.
    leaked = load_reference("notes")[2]
    library = ExampleLibrary([Example("fine", CLEAN),
                              Example("leak", leaked.replace("\n", "\r\n"))])
    _no_sandbox(monkeypatch)
    with pytest.raises(LeakageError, match="notes step 3"):
        run_bench(ReferenceModel(), ["todo"], tmp_path, library)
    assert not any(tmp_path.iterdir())


def test_leakage_guard_ignores_whitespace():
    leaked = "  \n" + " ".join(load_reference("habits")[0].split()) + "\n\n"
    with pytest.raises(LeakageError, match="habits step 1.*whitespace"):
        bench_run.check_no_leakage(ExampleLibrary([Example("s", leaked)]))


def test_leakage_guard_catches_a_reindented_file_block():
    # One reference file, re-indented and wrapped in a reply with other files and
    # chatter: the whole reply differs, but the block is a near-copy.
    ref = parse_blocks(load_reference("expenses")[1])[0]
    reindented = "\n".join("  " + line.replace("    ", "  ") for line in ref.content.split("\n"))
    reply = (
        "Here you go.\n"
        f"=== FILE: {ref.path} ===\n{reindented}\n=== END FILE ===\n"
        "=== FILE: backend/extra.py ===\ny = 2\n=== END FILE ===\n"
    )
    with pytest.raises(LeakageError, match=r"expenses step 2 \(5-shingle Jaccard 1\.000"):
        bench_run.check_no_leakage(ExampleLibrary([Example("s", reply)]))


def test_leakage_threshold_does_not_flag_one_app_against_another():
    """Calibration: no genuinely different reference solutions (two apps' blocks
    for the same path) are as similar as the threshold. Byte-identical blocks
    (todo and todo_auth share their components) are the same solution."""
    blocks = {}
    for app in list_apps():
        for reply in load_reference(app):
            for b in parse_blocks(reply):
                blocks.setdefault(b.path, []).append((app, b.content))
    worst = 0.0
    for items in blocks.values():
        for (a, x), (b, y) in itertools.combinations(items, 2):
            if x != y:
                worst = max(worst, bench_run.jaccard(bench_run._shingles(x),
                                                     bench_run._shingles(y)))
    assert 0.5 < worst < bench_run.LEAK_JACCARD


def test_leakage_guard_accepts_a_clean_library():
    result = bench_run.check_no_leakage(ExampleLibrary([Example("s", CLEAN)]))
    assert result.startswith("passed (1 examples")


def test_unknown_app_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown app"):
        run_bench(ReferenceModel(), ["nope"], tmp_path, ExampleLibrary([]))


# ------------------------------------------------------------------ model slug

@pytest.mark.parametrize("name, slug", [
    ("reference", "reference"),
    ("Qwen/Qwen2.5-Coder-1.5B", "Qwen_Qwen2.5-Coder-1.5B"),
    ("a b:c", "a_b_c"),
])
def test_model_slug(name, slug):
    assert model_slug(name) == slug


@pytest.mark.parametrize("name", ["", ".", ".."])
def test_model_slug_refuses_unsafe_names(name):
    with pytest.raises(ValueError):
        model_slug(name)


# ------------------------------------------------------------------ run_bench with fakes

class NamedModel:
    context_tokens = 32768

    def __init__(self, name):
        self.name = name

    def complete(self, messages):
        raise AssertionError("the fake runner never calls the model")


def _fake_sandboxes(monkeypatch, tmp_path):
    made, removed = [], []

    def create(parent, cache):
        box = Sandbox(tmp_path / f"box{len(made)}")
        made.append(box)
        return box

    monkeypatch.setattr(bench_run, "create_sandbox", create)
    monkeypatch.setattr(bench_run, "remove_sandbox", removed.append)
    return made, removed


def _passing_run_app(model, plan, sandbox, library, max_retries=3):
    ok = Attempt("r", None, (CheckResult("pyflakes", True, "", 0.1),), (0,))
    steps = tuple(StepTrace(s.number, s.key, (ok,), True) for s in plan.steps)
    return AppResult(plan.app, model.name, "passed_steps", steps, 1.0)


def test_a_harness_error_is_recorded_and_the_bench_continues(tmp_path, monkeypatch):
    made, removed = _fake_sandboxes(monkeypatch, tmp_path)

    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(bench_run, "run_app", explode)
    out = tmp_path / "out"
    records = run_bench(ReferenceModel(), ["todo", "notes"], out, ExampleLibrary([]))
    assert [r["app"] for r in records] == ["todo", "notes"]
    assert removed == made and len(made) == 2
    for r in records:
        assert r["status"] == "harness_error" and r["acceptance_passed"] is None
        assert r["steps"] == [] and r["error"] == "OSError: disk full"
    trace = json.loads((out / "reference" / "todo.json").read_text(encoding="utf-8"))
    assert trace["error"]["type"] == "OSError"
    assert "Traceback" in trace["error"]["traceback"] and "disk full" in trace["error"]["traceback"]
    summary = (out / "reference" / "summary.md").read_text(encoding="utf-8")
    assert "Harness errors: todo (OSError: disk full); notes (OSError: disk full)" in summary


def test_keyboard_interrupt_stops_the_bench_but_removes_the_sandbox(tmp_path, monkeypatch):
    made, removed = _fake_sandboxes(monkeypatch, tmp_path)

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench_run, "run_app", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_bench(ReferenceModel(), ["todo", "notes"], tmp_path / "out", ExampleLibrary([]))
    assert removed == made and len(made) == 1


def test_a_failed_cleanup_is_logged_and_does_not_mask_the_outcome(tmp_path, monkeypatch, caplog):
    _fake_sandboxes(monkeypatch, tmp_path)

    def cannot_remove(box):
        raise PermissionError("locked")

    monkeypatch.setattr(bench_run, "remove_sandbox", cannot_remove)
    monkeypatch.setattr(bench_run, "run_app", _passing_run_app)
    monkeypatch.setattr(bench_run, "run_acceptance",
                        lambda app, root: CheckResult("acceptance", True, "", 0.1))
    with caplog.at_level(logging.WARNING, logger="stepbuild.bench.run"):
        records = run_bench(ReferenceModel(), ["todo", "notes"], tmp_path / "out",
                            ExampleLibrary([]))
    assert [r["status"] for r in records] == ["passed_steps", "passed_steps"]
    assert all(r["acceptance_passed"] is True for r in records)
    assert all(r["cleanup_error"] == "PermissionError: locked" for r in records)
    assert "could not remove sandbox" in caplog.text


def test_failed_steps_skip_acceptance_and_records_are_written(tmp_path, monkeypatch):
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
        {"number": 1, "key": "model", "model_step": True, "passed": True, "attempts": 1,
         "failure_kinds": {}},
        {"number": 2, "key": "routes", "model_step": True, "passed": False, "attempts": 2,
         "failure_kinds": {"pyflakes": 2}},
    ]
    assert rec["run_seconds"] == 1.5 and rec["total_seconds"] >= 0
    trace = json.loads((out / "reference" / "todo.json").read_text(encoding="utf-8"))
    assert trace["acceptance"] is None and trace["status"] == "failed_at_step_2"
    assert "seconds" not in trace and trace["run_seconds"] == 1.5
    assert rec["trace"] == str(out / "reference" / "todo.json")


def test_provenance_in_summary_and_traces(tmp_path, monkeypatch):
    _fake_sandboxes(monkeypatch, tmp_path)
    monkeypatch.setattr(bench_run, "run_app", _passing_run_app)
    monkeypatch.setattr(bench_run, "run_acceptance",
                        lambda app, root: CheckResult("acceptance", True, "", 0.1))
    out = tmp_path / "out"
    model = NamedModel("Qwen/Qwen2.5-Coder-1.5B")
    info = {"split": "train", "shards": [{"path": "data/train-000.jsonl", "sha256": "abc123"}]}
    records = run_bench(model, ["todo", "notes"], out, ExampleLibrary([Example("s", CLEAN)]),
                        max_retries=2, library_info=info, command="python -m x --flag")
    assert len(records) == 2
    model_dir = out / "Qwen_Qwen2.5-Coder-1.5B"
    summary = (model_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Stepbuild benchmark: Qwen/Qwen2.5-Coder-1.5B")
    assert "Model: Qwen/Qwen2.5-Coder-1.5B (backend " in summary
    assert "test_bench_run.NamedModel; context tokens 32768; max new tokens 2048)" in summary
    assert "Apps run: 2 of 10 (todo, notes) -- SUBSET" in summary
    assert "split train; 1 examples; shards: data/train-000.jsonl (sha256 abc123)" in summary
    assert "leakage guard: passed" in summary
    assert "max_retries 2; reply_reserve 2048; retry_reserve 3072" in summary
    assert "feedback_max_tokens 1024; examples per prompt 2" in summary
    assert "Command: `python -m x --flag`" in summary
    assert "ended in progress" not in summary
    trace = json.loads((model_dir / "todo.json").read_text(encoding="utf-8"))
    assert next(iter(trace)) == "run"
    run = trace["run"]
    assert run["model"]["name"] == "Qwen/Qwen2.5-Coder-1.5B"
    assert run["harness"]["git_commit"] and run["environment"]["started"]
    assert run["environment"]["started"][-6] in "+-" or run["environment"]["started"].endswith("Z")
    assert run["environment"]["python"] and run["environment"]["gpu"]
    assert trace["steps"][0]["attempts"][0]["examples"] == [0]


# ------------------------------------------------------------------ CLI

def test_cli_rejects_unknown_model():
    with pytest.raises(SystemExit):
        main(["--model", "gpt-9"])


def test_cli_leakage_exits_2_with_one_line(tmp_path, capsys):
    shard = tmp_path / "train-000.jsonl"
    row = {
        "split": "train",
        "messages": [
            {"role": "user", "content": "STEP: add models\n\nCONTEXT FILES:\n"},
            {"role": "assistant", "content": load_reference("todo")[0]},
        ],
    }
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    code = main(["--model", "reference", "--apps", "todo", "--out", str(tmp_path / "o"),
                 "--library", str(shard)])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith("error: LeakageError:") and "\n" not in err
    assert "Traceback" not in err


def test_cli_harness_error_exits_2(tmp_path, monkeypatch, capsys):
    _fake_sandboxes(monkeypatch, tmp_path)

    def explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(bench_run, "run_app", explode)
    code = main(["--model", "reference", "--apps", "todo", "--out", str(tmp_path / "o")])
    assert code == 2
    assert "harness_error in todo" in capsys.readouterr().err


def test_cli_require_pass_and_max_retries(tmp_path, monkeypatch):
    _fake_sandboxes(monkeypatch, tmp_path)
    monkeypatch.setattr(bench_run, "run_app", _passing_run_app)
    monkeypatch.setattr(bench_run, "run_acceptance",
                        lambda app, root: CheckResult("acceptance", False, "no", 0.1))
    args = ["--model", "reference", "--apps", "todo", "--out", str(tmp_path / "o"),
            "--max-retries", "5"]
    assert main(args) == 0                        # ran cleanly
    assert main(args + ["--require-pass"]) == 1   # but did not pass
    summary = (tmp_path / "o" / "reference" / "summary.md").read_text(encoding="utf-8")
    assert "max_retries 5" in summary
    assert "--max-retries 5 --require-pass`" in summary


@pytest.mark.npm
def test_cli_reference_todo_writes_trace_and_summary(tmp_path, capsys):
    code = main(["--model", "reference", "--apps", "todo", "--out", str(tmp_path),
                 "--require-pass"])
    assert code == 0
    trace = json.loads((tmp_path / "reference" / "todo.json").read_text(encoding="utf-8"))
    assert trace["status"] == "passed_steps"
    assert trace["acceptance"]["passed"] is True
    assert len(trace["steps"]) == 6
    summary = (tmp_path / "reference" / "summary.md").read_text(encoding="utf-8")
    assert "**Apps passing acceptance: 1/1 (100.0%)**" in summary
    assert "Steps passed first try (of all 5): 5/5 (100.0%)" in summary
    assert "| todo | passed_steps | pass | 5 | - |" in summary
    assert "Apps passing acceptance: 1/1 (100.0%)" in capsys.readouterr().out
