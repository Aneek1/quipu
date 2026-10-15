"""Verified multi-turn code chains.

Spec: docs/superpowers/specs/2026-10-15-multiturn-verified-chat-design.md, section A.

A chain starts from a solved distill v2 problem (turn 1: the problem and the teacher's
answer that passed its tests) and adds follow-up turns. For each follow-up the teacher,
in three separate calls: writes a change request of a named, backward-compatible kind;
writes asserts for that change seeing only the request and the current code (never an
answer); then answers with the whole conversation in view. A turn is kept only if its
code passes the new asserts and every earlier turn's asserts.

The instruction strings below describe each task to the teacher. They are not training
text: training data is the requests and answers the teacher writes."""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Callable

from quipu.sandbox import extract_code, run_tests

CHANGE_KINDS = (
    "add an optional parameter whose default keeps the current behaviour",
    "handle an input or edge case the current code does not handle yet, without "
    "changing the results for inputs it already handles",
    "return extra information, but only when a new optional flag asks for it",
    "validate the input and raise ValueError with a clear message for invalid input, "
    "without changing the results for valid input",
)

REQUEST_SYSTEM = (
    "You write the next message a user sends after receiving the Python code below. "
    "The message asks for exactly one change of this kind: {kind}. Every call that "
    "worked before must still give the same result. Write it as the user, in one to "
    "three plain sentences. No code, no tests, and don't mention tests or backward "
    "compatibility."
)

TESTS_SYSTEM = (
    "You write unit tests for a requested change to a Python function named {entry}. "
    "Reply with one ```python block containing only 3 to 6 assert statements, one per "
    "line, each calling {entry} with literal arguments and comparing with a literal "
    "expected value. Test only the new behaviour the request describes. To check that "
    "a call raises an error, write: assert _raises(ValueError, {entry}, arg1, arg2)"
)

FOLLOWUP_SYSTEM = (
    "When the user asks for a change to code from earlier in the conversation, say in "
    "one or two sentences what changes, then give the complete updated function in one "
    "```python block, keeping its name. Be direct, don't repeat yourself, no filler."
)

# Appended to the code under test so `assert _raises(...)` lines can run.
PRELUDE = (
    "\n\ndef _raises(exc, fn, *args, **kwargs):\n"
    "    try:\n"
    "        fn(*args, **kwargs)\n"
    "    except exc:\n"
    "        return True\n"
    "    except Exception:\n"
    "        return False\n"
    "    return False\n"
)

FENCE_ANY = re.compile(r"```[^\n]*\n(.*?)```", re.S)
# A request that opens like a chat reply, or talks about tests, isn't a user message.
REQUEST_CHATTER = re.compile(r"^(sure|here is|here's|user:|the user)|\btests?\b", re.I)
QUOTES = "\"\u201c\u201d\u2018\u2019"



class SandboxError(RuntimeError):
    """The sandbox couldn't run code (timeout, crash, code that won't load). That says
    nothing about the code's tests, so it must never be read as a pass or a fail."""


# chat(messages, max_tokens) -> (text, finish_reason, completion_tokens)
Chat = Callable[[list[dict], int], "tuple[str, str, int]"]


def pick_kinds(seed_id: str, n: int) -> list[str]:
    """n distinct change kinds for a seed, the same on every run (no global RNG)."""
    if not 0 <= n <= len(CHANGE_KINDS):
        raise ValueError(f"need 0 to {len(CHANGE_KINDS)} change kinds, asked for {n}")
    start = hashlib.sha256(seed_id.encode()).digest()[0] % len(CHANGE_KINDS)
    return [CHANGE_KINDS[(start + i) % len(CHANGE_KINDS)] for i in range(n)]


def parse_request(text: str) -> str | None:
    """The teacher's follow-up request, or None if it isn't plain words of a sane length."""
    t = (text or "").strip().strip(QUOTES).strip()
    if "```" in t or not 15 <= len(t) <= 600 or REQUEST_CHATTER.search(t):
        return None
    return t


def _is_call_of(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _is_entry_call(node: ast.AST, entry_point: str) -> bool:
    """The call `entry_point(...)` itself, or an index or attribute of it (`f(...)[0]`,
    `f(...).x`). Any other wrapper (bool(), str(), isinstance()) hides the real value."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return _is_call_of(node, entry_point)


# A != against one of these is true for almost any result, so it tests nothing.
_TRIVIAL = (0, 0.0, "", [], (), {}, set())


def _literal(node: ast.AST):
    """(True, value) when the node is a literal, else (False, None). Anything the
    evaluator rejects, including odd literals like {[1]: 2}, just isn't a literal."""
    try:
        return True, ast.literal_eval(ast.unparse(node))
    except Exception:
        return False, None


def _checks_entry(test: ast.AST, entry_point: str) -> bool:
    """True for `entry(...) == literal` (either side; `!=` only against a non-trivial
    literal) and for `_raises(ExcClass, entry, ...)`: asserts that pin down a result.
    A literal that is True, False or None is too weak to pin anything down."""
    if isinstance(test, ast.Compare):
        if len(test.ops) != 1 or not isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
            return False
        for call, other in ((test.left, test.comparators[0]), (test.comparators[0], test.left)):
            if not _is_entry_call(call, entry_point):
                continue
            ok, value = _literal(other)
            if not ok or value is None or isinstance(value, bool):
                return False
            return isinstance(test.ops[0], ast.Eq) or value not in _TRIVIAL
        return False
    if _is_call_of(test, "_raises"):
        a = test.args
        return (len(a) >= 2 and isinstance(a[0], ast.Name)
                and isinstance(a[1], ast.Name) and a[1].id == entry_point)
    return False


def is_raises_check(assert_line: str) -> bool:
    """True when the assert is a `_raises(...)` check."""
    try:
        body = ast.parse(assert_line.strip()).body
    except SyntaxError:
        return False
    return len(body) == 1 and isinstance(body[0], ast.Assert) and _is_call_of(body[0].test, "_raises")


def parse_asserts(text: str, entry_point: str, max_len: int = 300) -> list[str]:
    """Single-line asserts that compare a call of `entry_point` with a literal (or check
    that it raises), in order, without repeats. Reads fenced blocks when there are any,
    else the whole reply."""
    blocks = FENCE_ANY.findall(text or "") or [text or ""]
    out: list[str] = []
    for line in "\n".join(blocks).splitlines():
        s = line.strip()
        if not s.startswith("assert ") or len(s) > max_len or s in out:
            continue
        try:
            body = ast.parse(s).body
        except SyntaxError:
            continue
        if len(body) != 1 or not isinstance(body[0], ast.Assert):
            continue
        if not _checks_entry(body[0].test, entry_point):
            continue
        out.append(s)
    return out


def _tests_the_change(assert_line: str, err: str, kind: str) -> bool:
    """Whether the old code failing this assert with `err` means it tests the change.
    "" (passes) and NameError (typo'd name) never do. A wrong result (AssertionError)
    does only where the change is meant to alter results: the edge-case kind, and
    `_raises` checks of the validation kind. A hang does only for the edge-case kind.
    For the other kinds old calls must keep working, so a wrong result or a hang on the
    old code means the teacher's assert contradicts it."""
    if not err or err == "NameError":
        return False
    if err == "timeout":
        return kind == CHANGE_KINDS[1]
    if err == "AssertionError":
        return kind == CHANGE_KINDS[1] or (kind == CHANGE_KINDS[3] and is_raises_check(assert_line))
    return True


def new_asserts(code: str, asserts: list[str], kind: str, timeout: float = 10.0) -> list[str]:
    """The asserts that test the requested change: those the previous turn's code (known
    good) fails for a reason `_tests_the_change` accepts. One sandbox run; if it times
    out, each assert is run alone, since old code that hangs on a new case is a legit
    target of the change. Raises SandboxError if a run failed in a way that proves
    nothing about the old code (process crash, code that won't load)."""
    if not asserts:
        return []
    res = run_tests(code + PRELUDE, asserts, timeout=timeout)
    if res.error == "timeout":
        errors = []
        for a in asserts:
            one = run_tests(code + PRELUDE, [a], timeout=timeout)
            if one.infra and one.error != "timeout":
                raise SandboxError(one.error)
            errors.append(one.errors[0])
    elif res.infra:
        raise SandboxError(res.error)
    else:
        errors = res.errors
    return [a for a, err in zip(asserts, errors) if _tests_the_change(a, err, kind)]


def _normal(code: str) -> list[str]:
    return [line.rstrip() for line in code.splitlines() if line.strip()]


def same_code(a: str, b: str) -> bool:
    """True when two pieces of code differ only in blank lines and trailing spaces."""
    return _normal(a) == _normal(b)


def check_turn(code: str, new_tests: list[str], old_tests: list[str],
               timeout: float = 10.0) -> str:
    """"ok" | "fail_regression" (an earlier turn's test breaks) | "fail_new" |
    "fail_crash" (the answer won't load or hangs) | "sandbox_error" (the process
    itself crashed, so there is no verdict). One sandbox run over the earlier tests
    followed by the new ones. The earlier code already ran fine, so a load failure or
    timeout here is the answer's fault."""
    res = run_tests(code + PRELUDE, old_tests + new_tests, timeout=timeout)
    if res.infra:
        return "sandbox_error" if res.error.startswith("crash:") else "fail_crash"
    errors = res.errors
    if any(errors[:len(old_tests)]):
        return "fail_regression"
    if any(errors[len(old_tests):]):
        return "fail_new"
    return "ok"


def _fence(code: str) -> str:
    return "```python\n" + code.rstrip("\n") + "\n```"


def request_input(turns: list[dict], code: str) -> str:
    """What the request writer sees: what the user asked so far and the current code."""
    asked = "\n".join(f"- {t['user']}" for t in turns)
    return f"The user has asked, in order:\n{asked}\n\nThe current code:\n{_fence(code)}"


def tests_input(request: str, code: str) -> str:
    """What the test writer sees: the current code and the request, never an answer."""
    return f"Current code:\n{_fence(code)}\n\nRequested change:\n{request}"


def conversation(turns: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in turns:
        out += [{"role": "user", "content": t["user"]},
                {"role": "assistant", "content": t["assistant"]}]
    return out


def run_chain(seed: dict, chat: Chat, *, max_turns: int = 3, max_new_tokens: int = 1024,
              timeout: float = 10.0, min_new_tests: int = 2) -> dict:
    """Grow one seed into a chain. The record keeps every turn tried; the turns before
    the first non-ok one are the verified chain, and `stop` says why it ended ("" when
    it reached max_turns). Any exception from the teacher or the sandbox ends the chain
    with stop "error:<ExceptionName>"; turns already verified stay in the record."""
    entry = seed["entry_point"]
    turns = [{"user": seed["prompt"], "assistant": seed["response"], "kind": None,
              "tests": list(seed["tests"]), "status": "ok"}]
    rec = {"id": "ch-" + seed["id"], "seed_id": seed["id"], "split": seed["split"],
           "entry_point": entry, "turns": turns, "stop": ""}
    kinds = pick_kinds(seed["id"], max_turns - 1)   # a bad max_turns raises here, once
    code = extract_code(seed["response"], entry)
    if code is None:
        rec["stop"] = "seed_no_code"
    try:
        for kind in kinds if code is not None else []:
            text, finish, _ = chat([{"role": "system", "content": REQUEST_SYSTEM.format(kind=kind)},
                                    {"role": "user", "content": request_input(turns, code)}], 200)
            request = parse_request(text) if finish != "length" else None
            if request is None:
                rec["stop"] = "request_bad"
                rec["rejected"] = {"request": text}
                break
            text, _, _ = chat([{"role": "system", "content": TESTS_SYSTEM.format(entry=entry)},
                               {"role": "user", "content": tests_input(request, code)}], 600)
            parsed = parse_asserts(text, entry)
            new_tests = new_asserts(code, parsed, kind, timeout)
            if len(new_tests) < min_new_tests:
                rec["stop"] = "tests_too_few"
                rec["rejected"] = {"request": request, "asserts": parsed}
                break
            text, finish, _ = chat([{"role": "system", "content": FOLLOWUP_SYSTEM}]
                                   + conversation(turns) + [{"role": "user", "content": request}],
                                   max_new_tokens)
            new_code = extract_code(text, entry) if finish != "length" else None
            if finish == "length":
                status = "answer_length"
            elif new_code is None:
                status = "no_code"
            elif same_code(new_code, code):
                status = "copy"
            else:
                status = check_turn(new_code, new_tests, [a for t in turns for a in t["tests"]],
                                    timeout)
            turns.append({"user": request, "assistant": text, "kind": kind,
                          "tests": new_tests, "status": status})
            if status != "ok":
                rec["stop"] = status
                break
            code = new_code
    except Exception as e:  # a dead teacher or sandbox ends this chain, not the whole run
        rec["stop"] = "error:" + type(e).__name__
    rec["kept_turns"] = next((i for i, t in enumerate(turns) if t["status"] != "ok"), len(turns))
    return rec
