"""The runner: plan -> prompt -> model -> parse -> write -> check -> retry, with traces.

Most tests replay the todo reference into a backend-only project (a copy of the
template backend, no frontend, no npm) with a plan whose checks leave out
npm_build, so the whole six-step loop runs in seconds. One npm-marked test
replays the reference into a real sandbox with every check.
"""
import dataclasses
import json
import pathlib
import shutil

import pytest

from stepbuild.bench.acceptance import load_reference, load_spec
from stepbuild.harness import runner
from stepbuild.harness.checks import CheckResult
from stepbuild.harness.model import ScriptedModel
from stepbuild.harness.plan import make_plan
from stepbuild.harness.prompt import TRIMMED, default_count_tokens
from stepbuild.harness.retrieve import Example, ExampleLibrary
from stepbuild.harness.runner import (
    AppResult,
    Attempt,
    StepTrace,
    project_files,
    run_app,
    write_trace,
)
from stepbuild.harness.sandbox import TEMPLATE_DIR, Sandbox, create_sandbox, remove_sandbox

APP = "todo"
REF = load_reference(APP)
EMPTY = ExampleLibrary([])


def _backend_plan():
    """The todo plan with npm_build dropped from every step's checks."""
    plan = make_plan(APP, load_spec(APP))
    steps = tuple(
        dataclasses.replace(s, checks=tuple(c for c in s.checks if c != "npm_build"))
        for s in plan.steps
    )
    return dataclasses.replace(plan, steps=steps)


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(
        TEMPLATE_DIR / "backend", root / "backend",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    box = Sandbox(root)
    yield box
    remove_sandbox(box)


def _break_python(reply):
    """The reply with a syntax error at the top of its (only) Python file."""
    head, rest = reply.split(" ===\n", 1)
    return head + " ===\ndef broken(:\n" + rest


def _last_user(messages):
    return messages[-1]["content"]


def test_backend_replay_passes_every_step_first_try(sandbox):
    model = ScriptedModel(REF)
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert isinstance(result, AppResult)
    assert (result.app, result.model, result.status) == (APP, "scripted", "passed_steps")
    assert [s.number for s in result.steps] == [1, 2, 3, 4, 5, 6]
    assert all(s.passed and len(s.attempts) == 1 for s in result.steps)
    assert len(model.calls) == 5
    assert result.seconds > 0
    run = result.steps[5]
    assert run.key == "run"
    assert run.attempts[0].reply == "" and run.attempts[0].parse_error is None
    assert [c.name for c in run.attempts[0].checks] == ["pyflakes", "pytest"]


def test_syntax_error_every_time_fails_at_step_two(sandbox):
    bad = _break_python(REF[1])
    model = ScriptedModel([REF[0], bad, bad, bad, bad])
    result = run_app(model, _backend_plan(), sandbox, EMPTY, max_retries=3)
    assert result.status == "failed_at_step_2"
    assert len(result.steps) == 2
    step2 = result.steps[1]
    assert not step2.passed and len(step2.attempts) == 4
    for attempt in step2.attempts:
        assert attempt.reply == bad and attempt.parse_error is None
        pyflakes = next(c for c in attempt.checks if c.name == "pyflakes")
        assert not pyflakes.passed and "app.py" in pyflakes.output
    assert len(model.calls) == 5  # 1 for step 1, 4 for step 2, never step 3


def test_retry_prompt_is_base_plus_the_bad_reply_and_its_feedback(sandbox):
    bad = _break_python(REF[1])
    model = ScriptedModel([REF[0], bad, *REF[1:]])
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert result.status == "passed_steps"
    step2 = result.steps[1]
    assert step2.passed and len(step2.attempts) == 2
    base, retry = model.calls[1], model.calls[2]
    assert len(retry) == len(base) + 2
    assert retry[: len(base)] == base
    assert retry[-2] == {"role": "assistant", "content": bad}
    feedback = _last_user(retry)
    assert feedback.startswith("The checks failed:")
    assert "--- pyflakes ---" in feedback
    pyflakes = next(c for c in step2.attempts[0].checks if c.name == "pyflakes")
    assert pyflakes.output.rstrip() in feedback


def test_retries_do_not_grow_the_prompt(sandbox):
    bad = _break_python(REF[1])
    model = ScriptedModel([REF[0], bad, bad, bad, REF[1], *REF[2:]])
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert result.steps[1].passed and len(result.steps[1].attempts) == 4
    base = model.calls[1]
    for retry in model.calls[2:5]:
        assert len(retry) == len(base) + 2 and retry[: len(base)] == base


def test_disallowed_file_is_a_parse_error_fed_back_as_format(sandbox):
    wrong = "=== FILE: backend/app.py ===\nprint('hi')\n=== END FILE ===\n"
    model = ScriptedModel([wrong, *REF])
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert result.status == "passed_steps"
    first = result.steps[0].attempts[0]
    assert first.reply == wrong
    assert first.parse_error and "not allowed" in first.parse_error
    assert first.checks == (CheckResult("format", False, first.parse_error, 0.0),)
    assert len(result.steps[0].attempts) == 2
    feedback = _last_user(model.calls[1])
    assert "--- format ---" in feedback and first.parse_error in feedback
    # The rejected reply wrote nothing.
    assert "print('hi')" not in (sandbox.root / "backend" / "app.py").read_text(encoding="utf-8")


def test_prompt_too_long_fails_the_step_without_calling_the_model(sandbox):
    model = ScriptedModel(REF, context_tokens=5500)
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert result.status == "failed_at_step_1"
    assert model.calls == []
    (step,) = result.steps
    (attempt,) = step.attempts
    assert attempt.reply == "" and attempt.parse_error is None
    (check,) = attempt.checks
    assert check.name == "prompt_too_long" and not check.passed
    # max_tokens = context_tokens - reply_reserve (2048)
    #              - retry_reserve (reply_reserve + FEEDBACK_MAX_TOKENS = 3072)
    assert "the budget is 380" in check.output


def test_reserves_are_adjustable(sandbox):
    model = ScriptedModel(REF, context_tokens=600)
    result = run_app(
        model, _backend_plan(), sandbox, EMPTY, reply_reserve=100, retry_reserve=50
    )
    # 600 - 100 - 50 = 450 is still too small for the todo prompt.
    check = result.steps[0].attempts[0].checks[0]
    assert check.name == "prompt_too_long" and "the budget is 450" in check.output


def test_a_reply_missing_an_allowed_file_is_a_format_failure_and_writes_nothing(
    sandbox, monkeypatch
):
    only_list = REF[3].split("=== FILE: frontend/src/components/Form.jsx ===")[0]
    assert "List.jsx" in only_list and "Form.jsx" not in only_list
    writes = []
    real_write = runner.write_blocks

    def spy_write(box, blocks):
        writes.append(sorted(b.path for b in blocks))
        real_write(box, blocks)

    monkeypatch.setattr(runner, "write_blocks", spy_write)
    model = ScriptedModel([*REF[:3], only_list, *REF[3:]])
    result = run_app(model, _backend_plan(), sandbox, EMPTY)
    assert result.status == "passed_steps"
    # One write per passing reply, each with all of its step's files; the partial
    # reply wrote nothing.
    plan = _backend_plan()
    assert writes == [sorted(s.allowed_files) for s in plan.steps if s.model_step]
    step4 = result.steps[3]
    assert len(step4.attempts) == 2
    first = step4.attempts[0]
    message = "reply is missing FILE blocks for: frontend/src/components/Form.jsx"
    assert first.parse_error == message
    assert first.checks == (CheckResult("format", False, message, 0.0),)
    feedback = _last_user(model.calls[4])
    assert "--- format ---" in feedback and message in feedback
    assert feedback.endswith("Reply again with complete FILE blocks for every allowed file.")


def test_a_do_nothing_model_fails_step_one_on_the_contract(sandbox):
    placeholder = (TEMPLATE_DIR / "backend" / "models.py").read_text(encoding="utf-8")
    stub = f"=== FILE: backend/models.py ===\n{placeholder}=== END FILE ===\n"
    model = ScriptedModel([stub] * 4)
    result = run_app(model, _backend_plan(), sandbox, EMPTY, max_retries=3)
    assert result.status == "failed_at_step_1"
    (step,) = result.steps
    assert len(step.attempts) == 4
    for attempt in step.attempts:
        checks = {c.name: c for c in attempt.checks}
        assert checks["pyflakes"].passed  # the placeholder is clean Python...
        assert not checks["contract"].passed  # ...but not a Store
        assert checks["contract"].output == "backend/models.py does not define `class Store`"
    assert "does not define `class Store`" in _last_user(model.calls[1])


def test_write_refusal_is_a_format_failure_but_os_errors_propagate(sandbox, monkeypatch):
    real_write = runner.write_blocks
    calls = []

    def refuse_once(box, blocks):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("refusing to write 'backend/models.py': it resolves outside the sandbox")
        real_write(box, blocks)

    monkeypatch.setattr(runner, "write_blocks", refuse_once)
    result = run_app(ScriptedModel([REF[0], *REF]), _backend_plan(), sandbox, EMPTY)
    assert result.status == "passed_steps"
    first = result.steps[0].attempts[0]
    assert first.parse_error and "resolves outside the sandbox" in first.parse_error
    assert first.checks[0].name == "format"

    def disk_error(box, blocks):
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_blocks", disk_error)
    with pytest.raises(OSError, match="disk full"):
        run_app(ScriptedModel(REF), _backend_plan(), sandbox, EMPTY)


def test_retry_prompt_fits_the_context_even_with_long_failure_output(sandbox, monkeypatch):
    """base <= context - reply_reserve - retry_reserve, the previous reply <=
    reply_reserve and the feedback <= FEEDBACK_MAX_TOKENS, so the retry prompt
    fits in context - reply_reserve, leaving room for the next reply."""
    context = 16384
    # Enough ~600-token examples that build_messages fills the prompt budget.
    fillers = [
        Example(f"Example {i}", f"=== FILE: e{i}.py ===\n" + f"# pad {i}\n" * 200 + "=== END FILE ===\n")
        for i in range(40)
    ]

    class Filling:
        def top(self, query, k=2, prefer_paths=None):
            return fillers

    # A reply just under reply_reserve (2048 tokens at ceil(chars / 3)).
    padding = "# padding line\n" * ((2048 * 3 - len(REF[0])) // 15 - 1)
    reply = REF[0].replace("=== END FILE ===", padding + "=== END FILE ===", 1)
    assert default_count_tokens(reply) <= 2048

    def huge_failure(box, names, timeout_s=180, *, step_key=None):
        return [CheckResult("pytest", False, "E " + "very long failure line\n" * 20000, 1.0)]

    monkeypatch.setattr(runner, "run_checks", huge_failure)
    model = ScriptedModel([reply, reply], context_tokens=context)
    run_app(model, _backend_plan(), sandbox, Filling(), max_retries=1)
    base, retry = model.calls
    base_size = sum(default_count_tokens(m["content"]) for m in base)
    assert base_size > context - 2048 - 3072 - 1000  # the budget really was filled
    retry_size = sum(default_count_tokens(m["content"]) for m in retry)
    assert TRIMMED in retry[-1]["content"]
    assert retry_size <= context - 2048


class _Crashing:
    name = "crashing"
    context_tokens = None

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        raise ConnectionError("backend down")


def test_model_exception_fails_the_step_without_retrying(sandbox):
    model = _Crashing()
    result = run_app(model, _backend_plan(), sandbox, EMPTY, max_retries=3)
    assert result.status == "failed_at_step_1"
    assert model.calls == 1
    (attempt,) = result.steps[0].attempts
    (check,) = attempt.checks
    assert check.name == "model_error" and not check.passed
    assert "ConnectionError" in check.output and "backend down" in check.output


class _SpyLibrary:
    EXAMPLE = Example(step="Add the widget store", reply="=== FILE: w.py ===\nW = 1\n=== END FILE ===\n")

    def __init__(self):
        self.queries = []

    def top(self, query, k=2, prefer_paths=None):
        self.queries.append((query, k, prefer_paths))
        return [self.EXAMPLE]


def test_retrieval_uses_the_title_only_and_prefers_the_allowed_files(sandbox):
    plan = _backend_plan()
    library = _SpyLibrary()
    model = ScriptedModel(REF)
    result = run_app(model, plan, sandbox, library)
    assert result.status == "passed_steps"
    model_steps = [s for s in plan.steps if s.model_step]
    assert library.queries == [(s.title, 2, s.allowed_files) for s in model_steps]
    for call in model.calls:
        assert "Add the widget store" in _last_user(call)


def test_prompt_sees_the_files_earlier_steps_wrote(sandbox):
    model = ScriptedModel(REF)
    run_app(model, _backend_plan(), sandbox, EMPTY)
    step3_prompt = _last_user(model.calls[2])
    assert "=== FILE: backend/app.py ===" in step3_prompt
    assert "def create_app" in step3_prompt


def test_project_files_reads_only_project_text(tmp_path):
    def put(rel, data):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    put("backend/models.py", b"x = 1\n")
    put("frontend/src/App.jsx", b"export default 1\n")
    put("frontend/node_modules/vite/index.js", b"junk\n")
    put("frontend/dist/index.js", b"built\n")
    put("backend/__pycache__/models.cpython-313.pyc", b"\x00\x01")
    put("backend/.pytest_cache/v/x", b"cache\n")
    put("frontend/package-lock.json", b"{}\n")
    put("frontend/public/logo.png", b"\x89PNG\r\n\x1a\n\xff\xfe")
    assert project_files(tmp_path) == {
        "backend/models.py": "x = 1\n",
        "frontend/src/App.jsx": "export default 1\n",
    }


def _sample_result():
    ok = CheckResult("pyflakes", True, "", 0.25)
    bad = CheckResult("format", False, "no FILE block found", 0.0)
    return AppResult(
        app="todo",
        model="scripted",
        status="failed_at_step_2",
        steps=(
            StepTrace(1, "model", (Attempt("reply one", None, (ok,)),), True),
            StepTrace(
                2, "routes",
                (Attempt("chatter", "no FILE block found", (bad,)),),
                False,
            ),
        ),
        seconds=1.5,
    )


def test_write_trace_round_trips_as_json(tmp_path):
    path = tmp_path / "out" / "todo.json"
    write_trace(_sample_result(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "app": "todo",
        "model": "scripted",
        "status": "failed_at_step_2",
        "seconds": 1.5,
        "steps": [
            {
                "number": 1, "key": "model", "passed": True,
                "attempts": [{
                    "reply": "reply one", "parse_error": None,
                    "checks": [{"name": "pyflakes", "passed": True, "output": "", "seconds": 0.25}],
                }],
            },
            {
                "number": 2, "key": "routes", "passed": False,
                "attempts": [{
                    "reply": "chatter", "parse_error": "no FILE block found",
                    "checks": [{
                        "name": "format", "passed": False,
                        "output": "no FILE block found", "seconds": 0.0,
                    }],
                }],
            },
        ],
    }


def test_write_trace_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "todo.json"
    path.write_text('{"old": true}', encoding="utf-8")
    real_write_text = pathlib.Path.write_text

    def half_then_fail(self, data, *args, **kwargs):
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(pathlib.Path, "write_text", half_then_fail)
    with pytest.raises(OSError, match="disk full"):
        write_trace(_sample_result(), path)
    monkeypatch.undo()
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["todo.json"]


@pytest.mark.npm
def test_reference_replay_in_a_real_sandbox_passes_every_step(tmp_path, npm_cache):
    box = create_sandbox(tmp_path, npm_cache)
    try:
        model = ScriptedModel(REF)
        result = run_app(model, make_plan(APP, load_spec(APP)), box, EMPTY)
        assert result.status == "passed_steps", json.dumps(
            runner.result_to_dict(result), indent=2
        )[-4000:]
        assert all(len(s.attempts) == 1 for s in result.steps)
        assert len(model.calls) == 5
        assert [c.name for c in result.steps[5].attempts[0].checks] == [
            "pyflakes", "pytest", "npm_build",
        ]
        # node_modules never reaches the prompt.
        assert all("node_modules" not in m["content"] for call in model.calls for m in call)
    finally:
        remove_sandbox(box)
