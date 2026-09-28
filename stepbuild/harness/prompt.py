"""Build the messages the model sees for one build step, and the retry turns.

A step prompt is one system message (the dataset's system message, verbatim, so
a model fine-tuned on the step dataset sees the same framing at build time) and
one user message with these sections, in this order:

    APP SPEC:        the app's spec.md
    STEP n of 5:     the step's title (the instruction; 5 = number of model steps)
    ALLOWED FILES:   the files this step may write, one per line
    CURRENT FILES:   FILE blocks of the files the step builds on (rule below)
    EXAMPLES:        retrieved dataset examples (omitted when there are none)
    PROJECT TREE:    sorted relative paths of every project file

CURRENT FILES rule. The model writes whole files, so it must see the current
version of anything it edits and of anything its code has to agree with:
- the step's allowed files that already exist (template placeholders included);
- backend/models.py from step 2 on (app.py imports Store and the validators);
- backend/app.py from step 3 on (the tests exercise its routes);
- backend/tests/conftest.py in step 3 (the `client` fixture it must not redefine);
- in steps 4 and 5, every existing source file under frontend/src/ (.js, .jsx,
  .css), so components, api.js and App.jsx line up with each other and main.jsx.
The backend files stay in the frontend steps because api.js must call the routes
app.py actually defines.

Never shown: any path with a segment containing "acceptance" (the hidden tests
must stay hidden even if a caller passes them in by mistake), node_modules and
__pycache__ (noise, and the tree would be enormous).

Budget. The user message is capped at MAX_USER_CHARS (about 6,000 tokens at 4
chars per token, the same cap the dataset uses). When over, examples are dropped
first, lowest-ranked first, because they are only hints; then the tree is trimmed
from the end with a note saying how many paths were left out. FILE blocks are
never cut: half a file would be worse than an over-budget prompt, since the model
would rewrite the missing half from nothing. So a very large current file can
still leave the message over the cap.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from stepbuild.harness.blocks import FileBlock, render_blocks
from stepbuild.harness.checks import CheckResult
from stepbuild.harness.plan import AppPlan, Step
from stepbuild.harness.retrieve import Example

SYSTEM_PROMPT = (
    "You build Flask + React apps one small step at a time. Reply with the complete "
    "new contents of each file you change, in the FILE block format."
)
MAX_USER_CHARS = 24_000

_FRONTEND_SRC = "frontend/src/"
_FRONTEND_EXTS = (".js", ".jsx", ".css")
_HIDDEN_SEGMENTS = ("node_modules", "__pycache__")
_FEEDBACK_HEAD = "The checks failed:\n"
_FEEDBACK_TAIL = "\nFix the files and reply again with complete FILE blocks for the files you change."


def _visible(path: str) -> bool:
    segments = path.split("/")
    if any(s in _HIDDEN_SEGMENTS for s in segments):
        return False
    return not any("acceptance" in s.casefold() for s in segments)


def _current_paths(step: Step, files: Mapping[str, str]) -> list[str]:
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
    return sorted(p for p in wanted if p in files and _visible(p))


def _render_example(example: Example) -> str:
    return f"Example step: {example.step}\n{example.reply.rstrip(chr(10))}\n"


def _assemble(
    plan: AppPlan,
    header: str,
    allowed: str,
    current: str,
    examples: Sequence[Example],
    tree: Sequence[str],
    omitted: int,
) -> str:
    parts = [
        f"APP SPEC:\n{plan.spec.strip()}\n",
        header,
        f"ALLOWED FILES:\n{allowed}\n",
        f"CURRENT FILES:\n{current}",
    ]
    if examples:
        parts.append("EXAMPLES:\n" + "\n".join(_render_example(e) for e in examples))
    lines = list(tree)
    if omitted:
        lines.append(f"... ({omitted} more files not shown)")
    parts.append("PROJECT TREE:\n" + "\n".join(lines) + "\n")
    return "\n".join(parts)


def build_messages(
    plan: AppPlan, step: Step, files: Mapping[str, str], examples: Sequence[Example]
) -> list[dict[str, str]]:
    """System + user messages for `step` of `plan`, given the project's current
    `files` (relative path -> contents) and retrieved `examples` (best first)."""
    if step not in plan.steps:
        raise ValueError(f"step {step.key!r} is not part of the plan for {plan.app!r}")
    if not step.model_step:
        raise ValueError(f"step {step.key!r} runs checks only; there is no prompt to build")
    n_model = sum(s.model_step for s in plan.steps)
    header = f"STEP {step.number} of {n_model}: {step.title}\n"
    allowed = "\n".join(step.allowed_files)
    paths = _current_paths(step, files)
    current = render_blocks([FileBlock(p, files[p]) for p in paths]) if paths else "(none yet)\n"
    tree = sorted(p for p in files if _visible(p))

    kept = list(examples)
    user = _assemble(plan, header, allowed, current, kept, tree, 0)
    while len(user) > MAX_USER_CHARS and kept:
        kept.pop()  # lowest-ranked example goes first
        user = _assemble(plan, header, allowed, current, kept, tree, 0)
    if len(user) > MAX_USER_CHARS:
        # Largest tree prefix that fits (binary search; the note's length barely
        # varies, so the fit is monotone in practice and checked exactly anyway).
        lo, hi = 0, len(tree)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            text = _assemble(plan, header, allowed, current, kept, tree[:mid], len(tree) - mid)
            if len(text) <= MAX_USER_CHARS:
                lo = mid
            else:
                hi = mid - 1
        user = _assemble(plan, header, allowed, current, kept, tree[:lo], len(tree) - lo)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def append_feedback(
    messages: Sequence[Mapping[str, str]], reply: str, failures: Sequence[CheckResult]
) -> list[dict[str, str]]:
    """A new message list: `messages`, the model's `reply`, and a user turn listing
    each failed check's name and output. Passed checks are left out (they would
    only distract). A reply that could not be parsed arrives as a failed pseudo-check
    named "format" whose output is the BlockError message, and renders the same way.
    The input list is not changed, so the runner can keep the original prompt."""
    failed = [c for c in failures if not c.passed]
    if not failed:
        raise ValueError("append_feedback needs at least one failed check")
    body = "\n".join(f"--- {c.name} ---\n{c.output.rstrip()}" for c in failed)
    return [dict(m) for m in messages] + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": _FEEDBACK_HEAD + body + _FEEDBACK_TAIL},
    ]
