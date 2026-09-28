"""Build the messages the model sees for one build step, and the retry turns.

A step prompt is one system message (the dataset's system message, verbatim) and
one user message whose sections follow the dataset's user message ("STEP: ...",
"CONTEXT FILES:", "PROJECT TREE:", spec §3.1), so a model fine-tuned on the step
dataset sees the framing it was trained on, with the benchmark-only sections
around it:

    APP SPEC:        the app's spec.md
    STEP:            the step's title, then "(step n of 5)" (5 = model steps)
    ALLOWED FILES:   the files this step may write, one per line
    EXAMPLES:        retrieved dataset examples, each labelled as coming from a
                     different project (omitted when there are none)
    CONTEXT FILES:   FILE blocks of the files the step builds on (rule below)
    PROJECT TREE:    sorted relative paths of the project's files
    REPLY WITH:      the reply format and the files to write; always last, so it
                     is the freshest thing in the model's context

Examples come before the context files so the project's own code sits nearest
the instruction to write; the label tells the model to copy the examples'
structure, not their names.

CONTEXT FILES rule. The model writes whole files, so it must see the current
version of anything it edits and of anything its code has to agree with:
- the step's allowed files that already exist (template placeholders included);
- backend/models.py from step 2 on (app.py imports Store and the validators);
- backend/app.py from step 3 on (the tests exercise its routes);
- backend/tests/conftest.py in step 3 (the `client` fixture it must not redefine);
- in steps 4 and 5, every existing source file under frontend/src/ (.js, .jsx,
  .css), so components, api.js and App.jsx line up with each other and main.jsx.
The backend files stay in the frontend steps because api.js must call the routes
app.py actually defines. They are shown in dependency order (models.py, app.py,
tests, api.js, components, App.jsx, other frontend files), so each file is read
after the files it uses.

Never shown: any path with a segment containing "acceptance" (the hidden tests
must stay hidden even if a caller passes them in by mistake), node_modules and
__pycache__. The tree also leaves out noise that says nothing about the code
(lockfiles, .gitkeep, pytest.ini).

Budget. The system and user messages together must fit in `max_tokens`, counted
with `count_tokens` (default ceil(chars / 3), deliberately pessimistic for code).
The runner derives max_tokens from the model's context window. When over, examples
are skipped first: each is kept, in rank order, only if it still fits, so a small
lower-ranked example can survive a large higher-ranked one. Then the tree is
trimmed from the end with a note saying how many paths were left out. FILE blocks
are never cut: half a file is worse than no attempt, since the model would rewrite
the missing half from nothing. If the prompt is still over budget, PromptTooLong
is raised with the measured size, and the runner records a failed attempt.

Retries. feedback_messages (what the runner uses) keeps only the original prompt
plus the latest reply and its failures, so a prompt that fit once keeps fitting
however many retries follow. append_feedback, which keeps the whole history, is
kept for callers that want it.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from stepbuild.harness.blocks import FileBlock, render_blocks
from stepbuild.harness.checks import CheckResult
from stepbuild.harness.plan import AppPlan, Step
from stepbuild.harness.retrieve import Example

SYSTEM_PROMPT = (
    "You build Flask + React apps one small step at a time. Reply with the complete "
    "new contents of each file you change, in the FILE block format."
)
DEFAULT_MAX_TOKENS = 8000

_FRONTEND_SRC = "frontend/src/"
_FRONTEND_EXTS = (".js", ".jsx", ".css")
_HIDDEN_SEGMENTS = ("node_modules", "__pycache__")
_TREE_NOISE = (
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock",
    "Pipfile.lock", ".gitkeep", "pytest.ini",
)
_EXAMPLE_LABEL = "EXAMPLE {n} (from a different project: copy the structure, not the names):"
_FEEDBACK_HEAD = "The checks failed:\n"
_FEEDBACK_TAIL = "\nFix the files and reply again with complete FILE blocks for the files you change."


class PromptTooLong(ValueError):
    """The step's prompt does not fit even with no examples and no tree."""

    def __init__(self, tokens: int, max_tokens: int) -> None:
        super().__init__(
            f"prompt needs {tokens} tokens without examples or project tree; "
            f"the budget is {max_tokens}"
        )
        self.tokens = tokens
        self.max_tokens = max_tokens


def default_count_tokens(text: str) -> int:
    return math.ceil(len(text) / 3)


def _visible(path: str) -> bool:
    segments = path.split("/")
    if any(s in _HIDDEN_SEGMENTS for s in segments):
        return False
    return not any("acceptance" in s.casefold() for s in segments)


def _dependency_rank(path: str) -> tuple[int, str]:
    if path == "backend/models.py":
        rank = 0
    elif path == "backend/app.py":
        rank = 1
    elif path.startswith("backend/tests/"):
        rank = 2
    elif path.startswith("backend/"):
        rank = 3
    elif path == "frontend/src/api.js":
        rank = 4
    elif path.startswith("frontend/src/components/"):
        rank = 5
    elif path == "frontend/src/App.jsx":
        rank = 6
    else:
        rank = 7
    return rank, path


def _context_paths(step: Step, files: Mapping[str, str]) -> list[str]:
    wanted = set(step.allowed_files)
    if step.number >= 2:
        wanted.add("backend/models.py")
    if step.number >= 3:
        wanted.add("backend/app.py")
    if step.key == "api_tests":
        wanted.add("backend/tests/conftest.py")
    if step.key in ("components", "wiring"):
        wanted.update(
            p for p in files if p.startswith(_FRONTEND_SRC) and p.endswith(_FRONTEND_EXTS)
        )
    return sorted((p for p in wanted if p in files and _visible(p)), key=_dependency_rank)


def _in_tree(path: str) -> bool:
    return _visible(path) and path.rsplit("/", 1)[-1] not in _TREE_NOISE


def _render_example(n: int, example: Example) -> str:
    reply = example.reply.rstrip("\n")
    return f"{_EXAMPLE_LABEL.format(n=n)}\nStep: {example.step}\n{reply}\n"


def build_messages(
    plan: AppPlan,
    step: Step,
    files: Mapping[str, str],
    examples: Sequence[Example],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> list[dict[str, str]]:
    """System + user messages for `step` of `plan`, given the project's current
    `files` (relative path -> contents) and retrieved `examples` (best first).
    Raises PromptTooLong if even the bare prompt exceeds `max_tokens`."""
    if step not in plan.steps:
        raise ValueError(f"step {step.key!r} is not part of the plan for {plan.app!r}")
    if not step.model_step:
        raise ValueError(f"step {step.key!r} runs checks only; there is no prompt to build")
    count = count_tokens or default_count_tokens
    n_model = sum(s.model_step for s in plan.steps)
    paths = _context_paths(step, files)
    context = render_blocks([FileBlock(p, files[p]) for p in paths]) if paths else "(none yet)\n"
    head = [
        f"APP SPEC:\n{plan.spec.strip()}\n",
        f"STEP: {step.title} (step {step.number} of {n_model})\n",
        "ALLOWED FILES:\n" + "".join(p + "\n" for p in step.allowed_files),
    ]
    reply_with = (
        "REPLY WITH:\n"
        "One block per file you write:\n"
        "=== FILE: <path> ===\n"
        "<the complete file>\n"
        "=== END FILE ===\n"
        f"Files to write: {', '.join(step.allowed_files)}. No other text and no ``` fences.\n"
    )
    tree = sorted(p for p in files if _in_tree(p))
    system_tokens = count(SYSTEM_PROMPT)

    def assemble(kept: Sequence[Example], shown: int) -> str:
        parts = list(head)
        if kept:
            parts.append(
                "EXAMPLES:\n" + "\n".join(_render_example(i, e) for i, e in enumerate(kept, 1))
            )
        parts.append(f"CONTEXT FILES:\n{context}")
        lines = tree[:shown]
        if shown < len(tree):
            lines = lines + [f"... ({len(tree) - shown} more files not shown)"]
        parts.append("PROJECT TREE:\n" + "".join(line + "\n" for line in lines))
        parts.append(reply_with)
        return "\n".join(parts)

    def size(user: str) -> int:
        return system_tokens + count(user)

    kept: list[Example] = []
    for example in examples:  # skip, don't stop: a smaller later example may fit
        if size(assemble(kept + [example], len(tree))) <= max_tokens:
            kept.append(example)
    user = assemble(kept, len(tree))
    if size(user) > max_tokens:
        # kept is empty here (anything kept fit with the full tree). Largest tree
        # prefix that fits, by binary search; the result is checked exactly.
        lo, hi = 0, len(tree)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if size(assemble(kept, mid)) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        user = assemble(kept, lo)
        if size(user) > max_tokens:
            raise PromptTooLong(size(assemble(kept, 0)), max_tokens)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _feedback(reply: str, failures: Sequence[CheckResult]) -> list[dict[str, str]]:
    failed = [c for c in failures if not c.passed]
    if not failed:
        raise ValueError("feedback needs at least one failed check")
    body = "\n".join(f"--- {c.name} ---\n{c.output.rstrip()}" for c in failed)
    return [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": _FEEDBACK_HEAD + body + _FEEDBACK_TAIL},
    ]


def feedback_messages(
    base: Sequence[Mapping[str, str]], reply: str, failures: Sequence[CheckResult]
) -> list[dict[str, str]]:
    """The retry prompt the runner uses: the ORIGINAL prompt `base`, the latest
    `reply`, and a user turn listing each failed check's name and output. Earlier
    attempts are dropped, so the prompt never grows beyond base + 2 messages.

    Passed checks are left out (they would only distract). A reply that could not
    be parsed arrives as a failed pseudo-check named "format" whose output is the
    BlockError message, and renders the same way. `base` is not changed."""
    return [dict(m) for m in base] + _feedback(reply, failures)


def append_feedback(
    messages: Sequence[Mapping[str, str]], reply: str, failures: Sequence[CheckResult]
) -> list[dict[str, str]]:
    """Like feedback_messages but keeps the whole history: `messages` (which may
    already hold earlier attempts), the reply and its feedback. Kept for
    compatibility; the runner uses feedback_messages so retries stay bounded."""
    return [dict(m) for m in messages] + _feedback(reply, failures)
