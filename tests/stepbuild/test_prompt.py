"""The prompt is everything the model knows about a step. These tests pin what goes
in (spec, instruction, allowed files, examples, the files it builds on, tree, reply
format), in what order, what never goes in (acceptance tests), how it stays within
a token budget without cutting a file in half, and how check failures are fed back
without the conversation growing on every retry."""
import copy
import math

import pytest

from stepbuild.harness.checks import CheckResult
from stepbuild.harness.plan import make_plan
from stepbuild.harness.prompt import (
    SYSTEM_PROMPT,
    PromptTooLong,
    append_feedback,
    build_messages,
    default_count_tokens,
    feedback_messages,
)
from stepbuild.harness.retrieve import Example

SPEC = "A todo list: each todo has a required title and a done flag."
PLAN = make_plan("todo", SPEC)
HEADINGS = [
    "APP SPEC:\n",
    "\nSTEP: ",
    "\nALLOWED FILES:\n",
    "\nEXAMPLES:\n",
    "\nCONTEXT FILES:\n",
    "\nPROJECT TREE:\n",
    "\nREPLY WITH:\n",
]
LABEL = "EXAMPLE {n} (from a different project: copy the structure, not the names):"

TEMPLATE_FILES = {
    "backend/app.py": "# placeholder app\n",
    "backend/models.py": "# placeholder models\n",
    "backend/pytest.ini": "[pytest]\n",
    "backend/requirements.txt": "flask\n",
    "backend/tests/conftest.py": "import pytest\n",
    "backend/tests/test_smoke.py": "def test_smoke():\n    pass\n",
    "frontend/src/main.jsx": "import App from './App.jsx'\n",
    "frontend/src/App.jsx": "export default function App() { return null }\n",
    "frontend/src/components/.gitkeep": "",
    "frontend/package.json": "{}\n",
    "frontend/package-lock.json": "{}\n",
}


def _step(n):
    return PLAN.steps[n - 1]


def _user(messages):
    return messages[1]["content"]


def _block(path, content):
    return f"=== FILE: {path} ===\n{content}=== END FILE ===\n"


def _section(user, heading):
    """Text of one section, from its heading to the next heading (or the end)."""
    body = user.split(heading, 1)[1]
    ends = [body.find(h) for h in HEADINGS if h in body]
    return body[: min(ends)] if ends else body


def test_system_prompt_is_the_spec_message_verbatim():
    assert SYSTEM_PROMPT == (
        "You build Flask + React apps one small step at a time. Reply with the complete "
        "new contents of each file you change, in the FILE block format."
    )


def test_system_first_then_one_user_message():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_headings_in_dataset_order_with_step_suffix():
    ex = [Example("Add a model", _block("models.py", "x = 1\n"))]
    user = _user(build_messages(PLAN, _step(2), TEMPLATE_FILES, ex))
    positions = [user.index(h) for h in HEADINGS]
    assert positions == sorted(positions)
    assert user.startswith("APP SPEC:\n" + SPEC)
    assert f"\nSTEP: {_step(2).title} (step 2 of 5)\n" in user


def test_reply_format_is_the_last_section():
    user = _user(build_messages(PLAN, _step(5), TEMPLATE_FILES, []))
    assert user.endswith(
        "\nREPLY WITH:\n"
        "One block per file you write:\n"
        "=== FILE: <path> ===\n"
        "<the complete file>\n"
        "=== END FILE ===\n"
        "Files to write: frontend/src/api.js, frontend/src/App.jsx. "
        "No other text and no ``` fences.\n"
    )


def test_allowed_files_listed_one_per_line():
    user = _user(build_messages(PLAN, _step(4), TEMPLATE_FILES, []))
    assert (
        "ALLOWED FILES:\nfrontend/src/components/List.jsx\nfrontend/src/components/Form.jsx\n"
        in user
    )


def test_step_one_shows_only_its_own_existing_file():
    ctx = _section(_user(build_messages(PLAN, _step(1), TEMPLATE_FILES, [])), "\nCONTEXT FILES:\n")
    assert _block("backend/models.py", "# placeholder models\n") in ctx
    assert "=== FILE: backend/app.py ===" not in ctx


def test_step_two_adds_models_and_its_own_file():
    files = {**TEMPLATE_FILES, "backend/models.py": "class Store:\n    pass\n"}
    ctx = _section(_user(build_messages(PLAN, _step(2), files, [])), "\nCONTEXT FILES:\n")
    assert _block("backend/models.py", "class Store:\n    pass\n") in ctx
    assert _block("backend/app.py", "# placeholder app\n") in ctx
    assert "=== FILE: frontend/" not in ctx


def test_step_three_shows_models_app_and_conftest():
    ctx = _section(_user(build_messages(PLAN, _step(3), TEMPLATE_FILES, [])), "\nCONTEXT FILES:\n")
    for path in ("backend/models.py", "backend/app.py", "backend/tests/conftest.py"):
        assert f"=== FILE: {path} ===" in ctx
    # test_api.py does not exist yet, so it is allowed but not shown
    assert "=== FILE: backend/tests/test_api.py ===" not in ctx
    assert "=== FILE: frontend/" not in ctx


def test_step_four_shows_the_relevant_frontend_files():
    ctx = _section(_user(build_messages(PLAN, _step(4), TEMPLATE_FILES, [])), "\nCONTEXT FILES:\n")
    for path in ("backend/models.py", "backend/app.py", "frontend/src/main.jsx", "frontend/src/App.jsx"):
        assert f"=== FILE: {path} ===" in ctx
    assert "=== FILE: frontend/package.json ===" not in ctx
    assert ".gitkeep" not in ctx


def test_context_files_in_dependency_order():
    files = {
        **TEMPLATE_FILES,
        "frontend/src/api.js": "export const a = 1\n",
        "frontend/src/components/List.jsx": "export default 1\n",
        "frontend/src/components/Form.jsx": "export default 2\n",
    }
    ctx = _section(_user(build_messages(PLAN, _step(5), files, [])), "\nCONTEXT FILES:\n")
    order = [
        "backend/models.py",
        "backend/app.py",
        "frontend/src/api.js",
        "frontend/src/components/Form.jsx",
        "frontend/src/components/List.jsx",
        "frontend/src/App.jsx",
        "frontend/src/main.jsx",
    ]
    positions = [ctx.index(f"=== FILE: {p} ===") for p in order]
    assert positions == sorted(positions)


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


def test_project_tree_is_sorted_and_omits_noise():
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))
    tree = _section(user, "\nPROJECT TREE:\n").strip().split("\n")
    noise = {"frontend/package-lock.json", "frontend/src/components/.gitkeep", "backend/pytest.ini"}
    assert tree == sorted(set(TEMPLATE_FILES) - noise)
    assert "backend/requirements.txt" in tree


def test_examples_labelled_and_section_omitted_when_none():
    ex = [
        Example("Add a ledger", _block("ledger.py", "rows = []\n")),
        Example("Add a list", _block("List.jsx", "export default 1\n")),
    ]
    user = _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, ex))
    section = _section(user, "\nEXAMPLES:\n")
    assert section.startswith(
        LABEL.format(n=1) + "\nStep: Add a ledger\n" + _block("ledger.py", "rows = []\n")
    )
    assert LABEL.format(n=2) + "\nStep: Add a list\n" in section
    assert "EXAMPLES:" not in _user(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))


def test_default_counter_is_ceil_of_chars_over_three():
    assert default_count_tokens("") == 0
    assert default_count_tokens("abcd") == 2
    assert default_count_tokens("x" * 9000) == 3000


def _size(msgs, count=len):
    return sum(count(m["content"]) for m in msgs)


def test_budget_covers_system_and_user_exactly():
    base = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    exact = _size(base)
    ex = [Example("Extra", _block("e.py", "e = 1\n"))]
    fits = build_messages(PLAN, _step(1), TEMPLATE_FILES, ex, max_tokens=exact + 10_000, count_tokens=len)
    assert "Extra" in _user(fits)
    tight = build_messages(PLAN, _step(1), TEMPLATE_FILES, ex, max_tokens=exact, count_tokens=len)
    assert tight == base  # example dropped, nothing else changed
    assert _size(tight) <= exact


def test_default_budget_uses_default_counter():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    assert _size(msgs, default_count_tokens) <= 8000


def test_budget_skips_examples_that_do_not_fit_keeping_smaller_lower_ranked_ones():
    small = Example("Keep me", _block("keep.py", "k = 1\n"))
    big = Example("Drop me", _block("big.py", "y" * 30_000 + "\n"))
    base = _size(build_messages(PLAN, _step(1), TEMPLATE_FILES, []))
    budget = base + 2_000
    for order in ([small, big], [big, small]):
        msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, order, max_tokens=budget, count_tokens=len)
        user = _user(msgs)
        assert "Keep me" in user and "Drop me" not in user
        assert LABEL.format(n=1) in user and LABEL.format(n=2) not in user
        assert _size(msgs) <= budget


def test_budget_trims_tree_after_examples_but_never_cuts_a_file():
    many = {f"docs/page_{i:05d}.md": "x\n" for i in range(3000)}
    models = "\n".join(f"LINE_{i} = {i}" for i in range(1000)) + "\n"  # ~15k chars
    files = {**TEMPLATE_FILES, **many, "backend/models.py": models}
    ex = [Example("An example", _block("e.py", "e = 1\n"))]
    msgs = build_messages(PLAN, _step(1), files, ex, max_tokens=8000)
    user = _user(msgs)
    assert _size(msgs, default_count_tokens) <= 8000
    assert "An example" not in user
    assert _block("backend/models.py", models) in user  # whole file, not truncated
    assert "more files not shown" in _section(user, "\nPROJECT TREE:\n")
    assert user.index("\nPROJECT TREE:\n") < user.index("\nREPLY WITH:\n")


def test_too_long_even_without_examples_or_tree_raises_with_size():
    models = "z" * 30_000 + "\n"
    with pytest.raises(PromptTooLong) as info:
        build_messages(PLAN, _step(1), {"backend/models.py": models}, [], max_tokens=8000)
    err = info.value
    assert isinstance(err, ValueError)
    assert err.max_tokens == 8000
    assert err.tokens > 8000
    assert err.tokens >= math.ceil(30_000 / 3)
    assert str(err.tokens) in str(err)


def _check(name, passed, output):
    return CheckResult(name=name, passed=passed, output=output, seconds=0.1)


FEEDBACK_TAIL = "\nFix the files and reply again with complete FILE blocks for the files you change."


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
    assert fb.endswith(FEEDBACK_TAIL)


def test_append_feedback_renders_the_format_pseudo_check():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    out = append_feedback(msgs, "chatter", [_check("format", False, "no FILE block found")])
    assert "--- format ---\nno FILE block found" in out[-1]["content"]


def test_append_feedback_requires_a_failure():
    msgs = build_messages(PLAN, _step(1), TEMPLATE_FILES, [])
    with pytest.raises(ValueError):
        append_feedback(msgs, "r", [_check("pytest", True, "ok")])


def test_feedback_messages_keep_only_the_original_prompt_and_the_latest_attempt():
    base = build_messages(PLAN, _step(2), TEMPLATE_FILES, [])
    before = copy.deepcopy(base)
    msgs = base
    for i in range(5):
        msgs = feedback_messages(base, f"reply {i}", [_check("pytest", False, f"error {i}")])
        assert len(msgs) == len(base) + 2
    assert base == before
    assert msgs[:2] == before
    assert msgs[2] == {"role": "assistant", "content": "reply 4"}
    assert "--- pytest ---\nerror 4" in msgs[3]["content"]
    assert "error 3" not in msgs[3]["content"]
    assert msgs[3]["content"].endswith(FEEDBACK_TAIL)
    with pytest.raises(ValueError):
        feedback_messages(base, "r", [_check("pytest", True, "ok")])
