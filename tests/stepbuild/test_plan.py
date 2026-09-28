"""The build plan is the fixed contract every app is built against: the runner, the
template and the reference solutions all assume these six steps, in this order,
with exactly these files and checks. The tests pin it down so an edit to one side
cannot silently drift from the others."""
import dataclasses

import pytest

from stepbuild.harness.blocks import BlockError
from stepbuild.harness.plan import CHECKS, AppPlan, Step, make_plan

SPEC = "A todo list: each todo has a required title and a done flag."

EXPECTED = [
    (1, "model", ("backend/models.py",), ("pyflakes",), True),
    (2, "routes", ("backend/app.py",), ("pyflakes", "pytest"), True),
    (3, "api_tests", ("backend/tests/test_api.py",), ("pyflakes", "pytest"), True),
    (
        4,
        "components",
        ("frontend/src/components/List.jsx", "frontend/src/components/Form.jsx"),
        ("npm_build",),
        True,
    ),
    (5, "wiring", ("frontend/src/api.js", "frontend/src/App.jsx"), ("npm_build",), True),
    (6, "run", (), ("pyflakes", "pytest", "npm_build"), False),
]


def test_six_steps_in_fixed_order_with_exact_files_and_checks():
    plan = make_plan("todo", SPEC)
    got = [(s.number, s.key, s.allowed_files, s.checks, s.model_step) for s in plan.steps]
    assert got == EXPECTED


def test_step_numbers_are_one_to_six():
    assert [s.number for s in make_plan("todo", SPEC).steps] == [1, 2, 3, 4, 5, 6]


def test_run_step_writes_nothing_and_calls_no_model():
    last = make_plan("todo", SPEC).steps[-1]
    assert last.key == "run"
    assert last.allowed_files == ()
    assert last.model_step is False
    assert all(s.model_step for s in make_plan("todo", SPEC).steps[:-1])


def test_every_title_names_the_app():
    for app in ("todo", "reading_list", "todo-auth2"):
        plan = make_plan(app, SPEC)
        assert plan.app == app and plan.spec == SPEC
        for s in plan.steps:
            assert app in s.title, (s.key, s.title)


def test_titles_state_the_contract_the_template_relies_on():
    steps = {s.key: s for s in make_plan("todo", SPEC).steps}
    assert "create_app()" in steps["routes"].title
    assert "client" in steps["api_tests"].title
    assert "conftest.py" in steps["api_tests"].title
    for s in steps.values():
        for f in s.allowed_files:
            assert f in s.title, (s.key, f)


def test_titles_state_the_cross_task_contract():
    steps = {s.key: s.title for s in make_plan("todo", SPEC).steps}
    for word in ("Store", "create", "list_items", "get", "update", "delete", "validate_", "id"):
        assert word in steps["model"], word
    for word in ("new Store()", "from models import Store", "/api/", "JSON array",
                 "201", "400", "404", "401", "409", "secret_key"):
        assert word in steps["routes"], word
    assert "from app import create_app" in steps["api_tests"]
    assert "do not redefine" in steps["api_tests"]
    assert "onDelete(item.id)" in steps["components"]
    assert "item.id as the React key" in steps["components"]
    assert "main entity" in steps["components"]
    assert "/api/" in steps["wiring"]
    assert "import List from './components/List.jsx'" in steps["wiring"]
    assert "import Form from './components/Form.jsx'" in steps["wiring"]


def test_titles_state_the_task4_review_decisions():
    """Decisions D1-D5 from the Task 4 review: things a model otherwise has to guess."""
    steps = {s.key: s.title for s in make_plan("todo", SPEC).steps}
    # D2: what the Store returns for a missing id.
    assert (
        "`get` and `update` return None and `delete` returns False when the id does not "
        "exist; `delete` returns True when it removed the item." in steps["model"]
    )
    # D5: which rules the validate function checks.
    assert "one for each rule the spec states" in steps["model"]
    assert "a missing or empty required field, a field of the wrong type, and any range rule" in steps["model"]
    # D4: defaults are filled in on create.
    assert (
        "When creating, fill in the spec's default values for optional fields that were "
        "left out." in steps["routes"]
    )
    # D1: PUT is a partial update validated after merging.
    assert (
        "For PUT, merge the sent fields into the existing item and validate the merged item "
        "with the same validate function." in steps["routes"]
    )
    # D3: api.js must cover list, create and delete; more is allowed.
    assert (
        "exports async functions to list, create and delete items (it may export others)"
        in steps["wiring"]
    )


def test_app_name_substitution_is_literal():
    # Titles are filled with str.replace, not str.format: braces in a title must
    # never be interpreted, whatever the table grows to contain.
    for s in make_plan("todo", SPEC).steps:
        assert "{app}" not in s.title and "todo" in s.title
    routes = make_plan("todo", SPEC).steps[1].title
    assert '{"error": ...}' in routes  # would raise KeyError under str.format


def test_checks_come_from_the_known_set():
    assert CHECKS == ("pyflakes", "pytest", "npm_build")
    for s in make_plan("todo", SPEC).steps:
        assert set(s.checks) <= set(CHECKS)


@pytest.mark.parametrize("app", ["", "   ", "\t\n"])
def test_empty_app_rejected(app):
    with pytest.raises(ValueError):
        make_plan(app, SPEC)


@pytest.mark.parametrize("spec", ["", "   ", "\n\t"])
def test_empty_spec_rejected(spec):
    with pytest.raises(ValueError):
        make_plan("todo", spec)


@pytest.mark.parametrize(
    "app", ["todo list", "../todo", "todo/x", "todo.app", " todo", "todo\n", "-todo", "_x", "tödo"]
)
def test_app_name_must_be_a_simple_slug(app):
    with pytest.raises(ValueError):
        make_plan(app, SPEC)


def test_equal_inputs_give_equal_plans():
    assert make_plan("todo", SPEC) == make_plan("todo", SPEC)
    assert make_plan("todo", SPEC) != make_plan("notes", SPEC)


def test_plans_and_steps_are_immutable():
    plan = make_plan("todo", SPEC)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.app = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.steps[0].allowed_files = ("x.py",)  # type: ignore[misc]
    assert isinstance(plan.steps, tuple)
    assert all(isinstance(s.allowed_files, tuple) and isinstance(s.checks, tuple) for s in plan.steps)


def _step(**kw):
    base = dict(number=1, key="model", title="t", allowed_files=("a.py",), checks=("pyflakes",), model_step=True)
    base.update(kw)
    return Step(**base)


def test_step_rejects_unsafe_file_paths():
    with pytest.raises(BlockError):
        _step(allowed_files=("../escape.py",))
    with pytest.raises(BlockError):
        _step(allowed_files=(r"backend\app.py",))


def test_step_rejects_unknown_checks_and_bad_shapes():
    with pytest.raises(ValueError):
        _step(checks=("eslint",))
    with pytest.raises(ValueError):
        _step(checks=("pyflakes", "pyflakes"))
    with pytest.raises(ValueError):
        _step(allowed_files=("a.py", "A.py"))
    with pytest.raises(ValueError):
        _step(model_step=True, allowed_files=())
    with pytest.raises(ValueError):
        _step(title="  ")
    with pytest.raises(TypeError):
        _step(allowed_files=["a.py"])


def test_app_plan_rejects_non_contiguous_numbering():
    plan = make_plan("todo", SPEC)
    s = list(plan.steps)
    with pytest.raises(ValueError):
        AppPlan(plan.app, plan.spec, (s[0], s[2], s[1], *s[3:]))
    with pytest.raises(ValueError):
        AppPlan(plan.app, plan.spec, tuple(s[:-1]))
    with pytest.raises(ValueError):
        AppPlan(plan.app, plan.spec, (dataclasses.replace(s[0], number=0), *s[1:]))
