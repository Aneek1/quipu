"""Drive a model through an app's build plan: prompt, reply, parse, write, check, retry.

For each model step (1-5):

- Retrieval is queried with the step TITLE only and the step's allowed files as
  `prefer_paths` (see retrieve.py for why never the spec).
- The prompt budget comes from the model: max_tokens = context_tokens -
  reply_reserve - retry_reserve, so the prompt, the reply and one round of
  feedback all fit; a model with no stated window gets DEFAULT_MAX_TOKENS.
- The prompt is built from the files currently in the sandbox (project_files).
  If it does not fit (PromptTooLong), the step fails at once with a check named
  "prompt_too_long": retrying would send the same prompt again.
- Each attempt: model.complete -> parse_blocks(allowed=step.allowed_files). A reply
  that does not parse is a failed pseudo-check "format" (nothing is written); a
  reply that parses is written and the step's checks run. All checks pass -> the
  step passes. Otherwise the failures are fed back and the model tries again, up
  to 1 + max_retries attempts. Retries use feedback_messages: the original prompt
  plus only the latest reply and its failures, so a retry never outgrows a prompt
  that fit.
- Anything model.complete raises is a failed check "model_error" and fails the
  step without a retry: a crashing backend is not a model mistake to feed back.

Step 6 makes no model call; its checks run once and are recorded as one attempt
with an empty reply. The run stops at the first failed step: status is
"passed_steps" when all six pass, else f"failed_at_step_{n}".

Traces are JSON (write_trace), written atomically with quipu.fsio.write_text_atomic.
It writes text mode, so on Windows the indented JSON has CRLF line endings; JSON
parsers do not care, and every string inside (replies, check output) keeps its
LF because json escapes newlines in strings as \\n.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

from quipu.fsio import write_text_atomic
from stepbuild.harness.blocks import BlockError, parse_blocks
from stepbuild.harness.checks import CheckResult, run_checks
from stepbuild.harness.model import StepModel
from stepbuild.harness.plan import AppPlan, Step
from stepbuild.harness.prompt import (
    DEFAULT_MAX_TOKENS,
    PromptTooLong,
    build_messages,
    feedback_messages,
)
from stepbuild.harness.retrieve import ExampleLibrary
from stepbuild.harness.sandbox import Sandbox, write_blocks

REPLY_RESERVE = 2048
RETRY_RESERVE = 1536
N_EXAMPLES = 2

# Never part of the prompt: installed or generated trees, and lockfiles (large,
# machine-written, and never something a step edits).
_SKIP_DIRS = frozenset({"node_modules", "dist", "__pycache__", ".pytest_cache"})
_SKIP_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock",
    "Pipfile.lock",
})


@dataclasses.dataclass(frozen=True)
class Attempt:
    reply: str                       # "" for step 6 and when the model was never called
    parse_error: str | None          # the BlockError message when the reply did not parse
    checks: tuple[CheckResult, ...]  # the step's checks, or one failed pseudo-check


@dataclasses.dataclass(frozen=True)
class StepTrace:
    number: int
    key: str
    attempts: tuple[Attempt, ...]
    passed: bool


@dataclasses.dataclass(frozen=True)
class AppResult:
    app: str
    model: str
    status: str                      # "passed_steps" | f"failed_at_step_{n}"
    steps: tuple[StepTrace, ...]     # up to and including the first failed step
    seconds: float


def project_files(root: Path) -> dict[str, str]:
    """The project's text files as {relative POSIX path: contents}.

    Skipped directories are pruned before descending, so frontend/node_modules (a
    junction into the shared npm cache) is never walked. Files that are not UTF-8
    text (images, bytecode) are left out: the prompt can only carry text.
    """
    root = Path(root)
    files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name in _SKIP_FILES:
                continue
            path = Path(dirpath) / name
            try:
                text = path.read_bytes().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "\x00" in text:
                continue
            files[path.relative_to(root).as_posix()] = text.replace("\r\n", "\n")
    return files


def _failed(name: str, output: str) -> tuple[CheckResult, ...]:
    return (CheckResult(name, False, output, 0.0),)


def _run_model_step(
    model: StepModel,
    plan: AppPlan,
    step: Step,
    sandbox: Sandbox,
    library: ExampleLibrary,
    max_retries: int,
    max_tokens: int,
) -> StepTrace:
    examples = library.top(step.title, k=N_EXAMPLES, prefer_paths=step.allowed_files)
    files = project_files(sandbox.root)
    try:
        base = build_messages(plan, step, files, examples, max_tokens=max_tokens)
    except PromptTooLong as e:
        attempt = Attempt("", None, _failed("prompt_too_long", str(e)))
        return StepTrace(step.number, step.key, (attempt,), False)

    attempts: list[Attempt] = []
    messages = base
    for _ in range(1 + max_retries):
        try:
            reply = model.complete(messages)
        except Exception as e:  # any backend failure; KeyboardInterrupt still stops the run
            attempts.append(Attempt("", None, _failed("model_error", f"{type(e).__name__}: {e}")))
            return StepTrace(step.number, step.key, tuple(attempts), False)
        try:
            blocks = parse_blocks(reply, allowed=step.allowed_files)
        except BlockError as e:
            checks = _failed("format", str(e))
            attempts.append(Attempt(reply, str(e), checks))
            messages = feedback_messages(base, reply, checks)
            continue
        write_blocks(sandbox, blocks)
        checks = tuple(run_checks(sandbox, step.checks))
        attempts.append(Attempt(reply, None, checks))
        if all(c.passed for c in checks):
            return StepTrace(step.number, step.key, tuple(attempts), True)
        messages = feedback_messages(base, reply, [c for c in checks if not c.passed])
    return StepTrace(step.number, step.key, tuple(attempts), False)


def _run_check_step(step: Step, sandbox: Sandbox) -> StepTrace:
    checks = tuple(run_checks(sandbox, step.checks))
    passed = all(c.passed for c in checks)
    return StepTrace(step.number, step.key, (Attempt("", None, checks),), passed)


def run_app(
    model: StepModel,
    plan: AppPlan,
    sandbox: Sandbox,
    library: ExampleLibrary,
    max_retries: int = 3,
    *,
    reply_reserve: int = REPLY_RESERVE,
    retry_reserve: int = RETRY_RESERVE,
) -> AppResult:
    """Build `plan` in `sandbox` with `model`, stopping at the first failed step.
    The caller owns the sandbox (create it, then remove_sandbox it)."""
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    context = getattr(model, "context_tokens", None)
    max_tokens = (
        DEFAULT_MAX_TOKENS if context is None else context - reply_reserve - retry_reserve
    )
    start = time.monotonic()
    traces: list[StepTrace] = []
    status = "passed_steps"
    for step in plan.steps:
        if step.model_step:
            trace = _run_model_step(
                model, plan, step, sandbox, library, max_retries, max_tokens
            )
        else:
            trace = _run_check_step(step, sandbox)
        traces.append(trace)
        if not trace.passed:
            status = f"failed_at_step_{step.number}"
            break
    return AppResult(plan.app, model.name, status, tuple(traces), time.monotonic() - start)


def result_to_dict(result: AppResult) -> dict[str, Any]:
    """The trace as plain JSON-ready data, in a fixed key order."""
    return {
        "app": result.app,
        "model": result.model,
        "status": result.status,
        "seconds": result.seconds,
        "steps": [
            {
                "number": s.number,
                "key": s.key,
                "passed": s.passed,
                "attempts": [
                    {
                        "reply": a.reply,
                        "parse_error": a.parse_error,
                        "checks": [dataclasses.asdict(c) for c in a.checks],
                    }
                    for a in s.attempts
                ],
            }
            for s in result.steps
        ],
    }


def write_trace(result: AppResult, path: Path) -> None:
    """Write the trace as indented JSON, atomically (a reader or a crash never sees
    half a file; an existing trace stays intact if the write fails)."""
    payload = json.dumps(result_to_dict(result), indent=2, ensure_ascii=False) + "\n"
    write_text_atomic(path, payload)
