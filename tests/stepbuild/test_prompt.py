"""The prompt is everything the model knows about a step. These tests pin what goes
in (spec, instruction, allowed files, the files it builds on, examples, tree), in
what order, what never goes in (acceptance tests), how it stays within budget
without cutting a file in half, and how check failures are fed back."""
import copy

import pytest

from stepbuild.harness.checks import CheckResult
from stepbuild.harness.prompt import (
    MAX_USER_CHARS,
    SYSTEM_PROMPT,
    append_feedback,
    build_messages,
)
from stepbuild.harness.plan import make_plan
from stepbuild.harness.retrieve import Example

SPEC = "A todo list: each todo has a required title and a done flag."
PLAN = make_plan("todo", SPEC)
HEADINGS = ["APP SPEC:", "STEP ", "ALLOWED FILES:", "CURRENT FILES:", "EXAMPLES:", "PROJECT TREE:"]

TEMPLATE_FILES = {
    "backend/app.py": "# placeholder app\n",
    "backend/models.py": "# placeholder models\n",
    "backend/tests/conftest.py": "import pytest\n",
    "backend/tests/test_smoke.py": "def test_smoke():\n    pass\n",
    "frontend/src/main.jsx": "import App from './App.jsx'\n",
    "frontend/src/App.jsx": "export default function App() { return null }\n",
    "frontend/package.json": "{}\n",
}


def _step(n):
    return PLAN.steps[n - 1]


def _user(messages):
    return messages[1]["content"]


def _block(path, content):
    return f"=== FILE: {path} ===\n{content}=== END FILE ===\n"


def test_system_prompt_is_the_spec_message_verbatim():
    assert SYSTEM_PROMPT == (
        "You build Flask + React apps one small step at a time. Reply with the complete "
        "new contents of each file you change, in the FILE block format."
    )


def test_system_first_then_one_user_message():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_headings_in_order_with_step_number_and_title():
    ex = [Example("Add a model", _block("models.py", "x = 1\n"))]
    user = _user(build_messages(PLAN, _step(2), TEMPLATE_FILES, ex))
    positions = [user.index(h) for h in HEADINGS]
    assert positions == sorted(positions)
    assert user.startswith("APP SPEC:\n" + SPEC)
    assert f"STEP 2 of 5: {_step(2).title}" in user


def test_allowed_files_listed_one_per_line():
    user = _user(build_messages(PLAN, _step(4), TEMPLATE_FILES, []))
    assert (
        "ALLOWED FILES:\nfrontend/src/components/List.jsx\nfrontend/src/components/Form.jsx\n"
        in user
    )


def test_step_one_shows_only_its_own_existing_file():
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))
    assert _block("backend/models.py", "# placeholder models\n") in user
    assert "=== FILE: backend/app.py ===" not in user


def test_step_two_adds_models_and_its_own_file():
    files = {**TEMPLATE_FILES, "backend/models.py": "class Store:\n    pass\n"}
    user = _user(build_messages(PLAN, _step(2), files, []))
    assert _block("backend/models.py", "class Store:\n    pass\n") in user
    assert _block("backend/app.py", "# placeholder app\n") in user
    assert "=== FILE: frontend/src/App.jsx ===" not in user


def test_step_three_shows_models_app_and_conftest():
    user = _user(build_messages(PLAN, _step(3), TEMPLATE_FILES, []))
    for path in ("backend/models.py", "backend/app.py", "backend/tests/conftest.py"):
        assert f"=== FILE: {path} ===" in user
    # test_api.py does not exist yet, so it is listed as allowed but not shown
    assert "=== FILE: backend/tests/test_api.py ===" not in user
    assert "=== FILE: frontend/" not in user


def test_frontend_steps_show_existing_frontend_sources():
    files = {**TEMPLATE_FILES, "frontend/src/components/List.jsx": "export default 1\n"}
    user = _user(build_messages(PLAN, _step(5), files, []))
    for path in (
        "backend/models.py",
        "backend/app.py",
        "frontend/src/main.jsx",
        "frontend/src/App.jsx",
        "frontend/src/components/List.jsx",
    ):
        assert f"=== FILE: {path} ===" in user
    assert "=== FILE: frontend/package.json ===" not in user


def test_run_step_is_rejected():
    with pytest.raises(ValueError):
        build_messages(PLAN, _step(6), TEMPLATE_FILES, [])


def test_acceptance_and_node_modules_never_appear():
    files = {
        **TEMPLATE_FILES,
        "backend/tests/acceptance/test_acceptance.py": "SECRET = 1\n",
        "acceptance/test_acceptance.py": "SECRET = 2\n",
        "frontend/src/acceptance.jsx": "SECRET\n",
        "frontend/node_modules/react/index.js": "x\n",
    }
    for n in range(1, 6):
        user = _user(build_messages(PLAN, _step(n), files, []))
        assert "acceptance" not in user
        assert "SECRET" not in user
        assert "node_modules" not in user


def test_project_tree_is_sorted_relative_paths():
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))
    tree = user.split("PROJECT TREE:\n", 1)[1].strip().split("\n")
    assert tree == sorted(TEMPLATE_FILES)


def test_examples_rendered_and_section_omitted_when_none():
    ex = [Example("Add a ledger", _block("ledger.py", "rows = []\n"))]
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, ex))
    assert "EXAMPLES:\nExample step: Add a ledger\n" + _block("ledger.py", "rows = []\n") in user
    assert "EXAMPLES:" not in _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))


def test_budget_drops_examples_first_last_ranked_first():
    small = Example("Keep me", _block("keep.py", "k = 1\n"))
    big = Example("Drop me", _block("big.py", "y" * 30_000 + "\n"))
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, [small, big]))
    assert len(user) <= MAX_USER_CHARS
    assert "Keep me" in user and "Drop me" not in user
    # both examples too big together with the rest: all dropped, section gone
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, [big, small]))
    assert "Drop me" not in user
    assert len(user) <= MAX_USER_CHARS


def test_budget_trims_tree_after_examples_but_never_cuts_a_file():
    many = {f"docs/page_{i:05d}.md": "x\n" for i in range(3000)}
    models = "\n".join(f"LINE_{i} = {i}" for i in range(1000)) + "\n"  # ~15k chars
    files = {**TEMPLATE_FILES, **many, "backend/models.py": models}
    ex = [Example("An example", _block("e.py", "e = 1\n"))]
    user = _user(build_messages(PLAN, _step(1), files, ex))
    assert len(user) <= MAX_USER_CHARS
    assert "An example" not in user
    assert _block("backend/models.py", models) in user  # whole file, not truncated
    assert "PROJECT TREE:\n" in user
    assert "more files not shown" in user


def test_huge_current_file_is_kept_whole_even_over_budget():
    models = "z" * (MAX_USER_CHARS + 1000) + "\n"
    user = _user(build_messages(PLAN, _step(1), {"backend/models.py": models}, []))
    assert _block("backend/models.py", models) in user


def _check(name, passed, output):
    return CheckResult(name=name, passed=passed, output=output, seconds=0.1)


def test_append_feedback_includes_only_failed_checks_and_does_not_mutate():
    msgs = build_messages(PLAN, _step(2), TEMPLATE_FILES, [])
    before = copy.deepcopy(msgs)
    out = append_feedback(
        msgs,
        "my reply",
        [_check("pyflakes", True, "LINT OK"), _check("pytest", False, "E assert 1 == 2")],
    )
    assert msgs == before
    assert out[:2] == before
    assert out[2] == {"role": "assistant", "content": "my reply"}
    assert out[3]["role"] == "user"
    fb = out[3]["content"]
    assert fb.startswith("The checks failed:\n")
    assert "--- pytest ---\nE assert 1 == 2" in fb
    assert "pyflakes" not in fb and "LINT OK" not in fb
    assert fb.endswith(
        "\nFix the files and reply again with complete FILE blocks for the files you change."
    )


def test_append_feedback_renders_the_format_pseudo_check():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    out = append_feedback(msgs, "chatter", [_check("format", False, "no FILE block found")])
    assert "--- format ---\nno FILE block found" in out[-1]["content"]


def test_append_feedback_requires_a_failure():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    with pytest.raises(ValueError):
        append_feedback(msgs, "r", [_check("pytest", True, "ok")])
