# Verified Python chains (plan 1 of the multi-turn chat spec) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn solved distill v2 problems into 2-3 turn conversations ("write X" → "now change it") where every turn's code passes its new tests and every earlier turn's tests, and prove it with a 200-chain laptop smoke run.

**Architecture:** `quipu/chains.py` holds the pure chain logic (change kinds, teacher instructions, parsing, assert filtering, turn checks, `run_chain`), tested against a scripted fake teacher and the real sandbox. `scripts/chain_generate.py` drives it over distill v2 parts against an OpenAI-compatible server (llama-server on the laptop, vLLM on Kaggle later), resumable with a PAUSE file. `scripts/chain_verify.py` re-checks every kept turn from scratch and writes chat-ready conversations plus a summary.

**Tech Stack:** Python 3.13, `quipu.sandbox` (subprocess test runner), stdlib `urllib` for the teacher endpoint, pytest, llama.cpp `llama-server` with the Q4_K_M teacher already in `models/`.

**Spec:** `docs/superpowers/specs/2026-10-15-multiturn-verified-chat-design.md` (section "A. Verified Python chains"). Later plans, not this one: B (web pages), C (stepbuild chaining), chat data + eval, the Kaggle fp16 fine-tune.

**Rules that apply to every task:** no Claude-written training text (the instruction strings below describe tasks to the teacher and never reach training data; requests, asserts and answers come from the teacher); stage files by explicit path; commit messages with no trailers; never push.

---

## File structure

| File | Responsibility |
|---|---|
| `quipu/chains.py` (create) | Change kinds, teacher instructions, `parse_request`, `parse_asserts`, `failing_on`, `same_code`, `check_turn`, `run_chain`. No network, no file I/O. |
| `scripts/chain_generate.py` (create) | Seeds from `data/distill-v2/{kept,prompts}`, `make_messages_chat` (OpenAI-compatible client), per-part loop with resume, PAUSE, `--limit`, `--split`. Writes `data/chains/gen/part-NNNN.<tag>.jsonl`. |
| `scripts/chain_verify.py` (create) | Re-verifies every ok turn, writes `data/chains/kept/part-NNNN.jsonl` (chat messages) and `data/chains/verify_summary.json`. |
| `tests/test_chains.py` (create) | All tests for the three files above. |

Run tests with: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q` from the repo root (Git Bash) — `tests/conftest.py` makes `scripts.*` importable, as `tests/test_distill_v2.py` already relies on.

---

### Task 1: Change kinds and parsing

**Files:**
- Create: `quipu/chains.py`
- Test: `tests/test_chains.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Verified multi-turn code chains (quipu.chains, scripts/chain_generate.py,
scripts/chain_verify.py). The teacher is a scripted fake; the sandbox is real."""
import json
from pathlib import Path

from quipu import chains as ch


def test_pick_kinds_is_deterministic_and_distinct():
    a = ch.pick_kinds("dv2-0000001", 2)
    assert a == ch.pick_kinds("dv2-0000001", 2)
    assert len(a) == 2 and a[0] != a[1]
    assert all(k in ch.CHANGE_KINDS for k in a)
    seen = {ch.pick_kinds(f"dv2-{i:07d}", 1)[0] for i in range(200)}
    assert seen == set(ch.CHANGE_KINDS)          # every kind gets used


def test_parse_request_accepts_plain_words_only():
    assert ch.parse_request('  "Let add take an optional third number c."  ') == \
        "Let add take an optional third number c."
    assert ch.parse_request("Sure:\n```python\ndef f(): pass\n```") is None   # code
    assert ch.parse_request("ok") is None                                      # too short
    assert ch.parse_request("x" * 601) is None                                 # too long


def test_parse_asserts_keeps_single_line_asserts_about_the_function():
    text = (
        "Here are the tests:\n```python\n"
        "assert add(1, 2, 3) == 6\n"
        "assert add(1, 2, 3) == 6\n"                  # duplicate
        "assert len([1]) == 1\n"                      # does not use add
        "x = add(1, 2)\n"                             # not an assert
        "assert add(1,\n"                             # broken line
        "assert _raises(ValueError, add, 'a', 1)\n"   # raise check, add passed by name
        "```"
    )
    assert ch.parse_asserts(text, "add") == [
        "assert add(1, 2, 3) == 6",
        "assert _raises(ValueError, add, 'a', 1)",
    ]


def test_parse_asserts_reads_unfenced_replies():
    assert ch.parse_asserts("assert add(2, 2) == 4", "add") == ["assert add(2, 2) == 4"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: FAIL with `ImportError: cannot import name 'chains' from 'quipu'`

- [ ] **Step 3: Write the implementation**

```python
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

# chat(messages, max_tokens) -> (text, finish_reason, completion_tokens)
Chat = Callable[[list[dict], int], "tuple[str, str, int]"]


def pick_kinds(seed_id: str, n: int) -> list[str]:
    """n distinct change kinds for a seed, the same on every run (no global RNG)."""
    start = hashlib.sha256(seed_id.encode()).digest()[0] % len(CHANGE_KINDS)
    return [CHANGE_KINDS[(start + i) % len(CHANGE_KINDS)] for i in range(n)]


def parse_request(text: str) -> str | None:
    """The teacher's follow-up request, or None if it isn't plain words of a sane length."""
    t = (text or "").strip().strip('"').strip()
    if "```" in t or not 15 <= len(t) <= 600:
        return None
    return t


def parse_asserts(text: str, entry_point: str, max_len: int = 300) -> list[str]:
    """Single-line assert statements that use `entry_point`, in order, without repeats.
    Reads fenced blocks when there are any, else the whole reply."""
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
        if not any(isinstance(n, ast.Name) and n.id == entry_point for n in ast.walk(body[0])):
            continue
        out.append(s)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add quipu/chains.py tests/test_chains.py
git commit -m "chains: change kinds, teacher instructions, request and assert parsing"
```

---

### Task 2: Assert filtering and turn checks

**Files:**
- Modify: `quipu/chains.py` (append)
- Test: `tests/test_chains.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
OLD = "def add(a, b):\n    return a + b\n"
NEW = "def add(a, b, c=0):\n    return a + b + c\n"


def test_failing_on_keeps_only_asserts_the_old_code_fails():
    asserts = ["assert add(1, 2, 3) == 6", "assert add(1, 2) == 3", "assert add(0, 0, 5) == 5"]
    assert ch.failing_on(OLD, asserts) == ["assert add(1, 2, 3) == 6", "assert add(0, 0, 5) == 5"]


def test_failing_on_supports_raises_checks():
    old = "def half(x):\n    return x / 2\n"
    assert ch.failing_on(old, ["assert _raises(ValueError, half, 'a')"]) == \
        ["assert _raises(ValueError, half, 'a')"]       # old code raises TypeError, not ValueError


def test_same_code_ignores_blank_lines_and_trailing_spaces():
    assert ch.same_code("def f():\n\n    return 1   \n", "def f():\n    return 1")
    assert not ch.same_code(OLD, NEW)


def test_check_turn_tells_regressions_from_new_failures():
    new_tests = ["assert add(1, 2, 3) == 6"]
    old_tests = ["assert add(1, 2) == 3"]
    assert ch.check_turn(NEW, new_tests, old_tests) == "ok"
    assert ch.check_turn(OLD, new_tests, old_tests) == "fail_new"
    broken = "def add(a, b, c=0):\n    return a + b + c if c else 0\n"
    assert ch.check_turn(broken, new_tests, old_tests) == "fail_regression"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: FAIL with `AttributeError: module 'quipu.chains' has no attribute 'failing_on'`

- [ ] **Step 3: Write the implementation (append to `quipu/chains.py`)**

```python
def failing_on(code: str, asserts: list[str], timeout: float = 10.0) -> list[str]:
    """The asserts that fail on `code`, one sandbox run each: an assert the previous
    turn's code already passes doesn't test the requested change."""
    return [a for a in asserts if run_tests(code + PRELUDE, [a], timeout=timeout).passed == 0]


def _normal(code: str) -> list[str]:
    return [line.rstrip() for line in code.splitlines() if line.strip()]


def same_code(a: str, b: str) -> bool:
    """True when two pieces of code differ only in blank lines and trailing spaces."""
    return _normal(a) == _normal(b)


def check_turn(code: str, new_tests: list[str], old_tests: list[str],
               timeout: float = 10.0) -> str:
    """"ok" | "fail_regression" (an earlier turn's test breaks) | "fail_new"."""
    if old_tests and not run_tests(code + PRELUDE, old_tests, timeout=timeout).all_passed:
        return "fail_regression"
    if not run_tests(code + PRELUDE, new_tests, timeout=timeout).all_passed:
        return "fail_new"
    return "ok"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add quipu/chains.py tests/test_chains.py
git commit -m "chains: keep only asserts the old code fails; tell regressions from new failures"
```

---

### Task 3: `run_chain` against a scripted teacher

**Files:**
- Modify: `quipu/chains.py` (append)
- Test: `tests/test_chains.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
SEED = {
    "id": "dv2-0000007", "split": "train", "entry_point": "add",
    "prompt": "Write a Python function add(a, b) that returns the sum.",
    "response": "Adds two numbers.\n\n```python\n" + OLD + "```",
    "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
}


class FakeTeacher:
    """Answers by which instruction it is given; records every call."""

    def __init__(self, request="Please let add take an optional third number c, default 0.",
                 tests="```python\nassert add(1, 2, 3) == 6\nassert add(0, 0, 5) == 5\n"
                       "assert add(1, 2) == 3\n```",
                 answer="Now add takes c.\n\n```python\n" + NEW + "```", finish="stop"):
        self.request, self.tests, self.answer, self.finish = request, tests, answer, finish
        self.calls: list[list[dict]] = []

    def __call__(self, messages, max_tokens):
        self.calls.append(messages)
        system = messages[0]["content"]
        if system.startswith("You write the next message"):
            return self.request, "stop", 20
        if system.startswith("You write unit tests"):
            return self.tests, "stop", 40
        return self.answer, self.finish, 60


def test_run_chain_keeps_a_good_second_turn():
    t = FakeTeacher()
    rec = ch.run_chain(SEED, t, max_turns=2)
    assert rec["id"] == "ch-dv2-0000007" and rec["stop"] == "" and rec["kept_turns"] == 2
    t2 = rec["turns"][1]
    assert t2["status"] == "ok" and t2["user"] == t.request
    assert t2["tests"] == ["assert add(1, 2, 3) == 6", "assert add(0, 0, 5) == 5"]
    answer_call = t.calls[2]
    assert [m["role"] for m in answer_call] == ["system", "user", "assistant", "user"]
    assert answer_call[2]["content"] == SEED["response"]          # earlier answer in view
    tests_call = t.calls[1]
    assert SEED["response"] not in tests_call[1]["content"]       # tests never see an answer


def test_run_chain_stops_on_a_copied_answer():
    rec = ch.run_chain(SEED, FakeTeacher(answer="Same.\n\n```python\n" + OLD + "```"), max_turns=2)
    assert rec["stop"] == "copy" and rec["kept_turns"] == 1


def test_run_chain_stops_on_a_regression():
    broken = "def add(a, b, c=0):\n    return a + b + c if c else 0\n"
    rec = ch.run_chain(SEED, FakeTeacher(answer="```python\n" + broken + "```"), max_turns=2)
    assert rec["stop"] == "fail_regression" and rec["turns"][1]["status"] == "fail_regression"


def test_run_chain_stops_when_tests_do_not_test_the_change():
    rec = ch.run_chain(SEED, FakeTeacher(tests="```python\nassert add(1, 2) == 3\n```"), max_turns=2)
    assert rec["stop"] == "tests_too_few" and len(rec["turns"]) == 1


def test_run_chain_stops_on_a_cut_off_answer():
    rec = ch.run_chain(SEED, FakeTeacher(finish="length"), max_turns=2)
    assert rec["stop"] == "answer_length"


def test_run_chain_third_turn_needs_tests_the_current_code_fails():
    t = FakeTeacher(tests="```python\nassert add(1, 2, 3) == 6\nassert add(0, 0, 5) == 5\n```")
    rec = ch.run_chain(SEED, t, max_turns=3)
    # Turn 3's tests are the same lines, which NEW already passes, so turn 3 stops at
    # tests_too_few: the chain keeps its two good turns.
    assert rec["kept_turns"] == 2 and rec["stop"] == "tests_too_few"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: FAIL with `AttributeError: module 'quipu.chains' has no attribute 'run_chain'`

- [ ] **Step 3: Write the implementation (append to `quipu/chains.py`)**

```python
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
    it reached max_turns)."""
    entry = seed["entry_point"]
    turns = [{"user": seed["prompt"], "assistant": seed["response"], "kind": None,
              "tests": list(seed["tests"]), "status": "ok"}]
    rec = {"id": "ch-" + seed["id"], "seed_id": seed["id"], "split": seed["split"],
           "entry_point": entry, "turns": turns, "stop": ""}
    code = extract_code(seed["response"], entry)
    if code is None:
        rec["stop"] = "seed_no_code"
    for kind in pick_kinds(seed["id"], max_turns - 1) if code is not None else []:
        text, finish, _ = chat([{"role": "system", "content": REQUEST_SYSTEM.format(kind=kind)},
                                {"role": "user", "content": request_input(turns, code)}], 200)
        request = parse_request(text) if finish != "length" else None
        if request is None:
            rec["stop"] = "request_bad"
            break
        text, _, _ = chat([{"role": "system", "content": TESTS_SYSTEM.format(entry=entry)},
                           {"role": "user", "content": tests_input(request, code)}], 600)
        new_tests = failing_on(code, parse_asserts(text, entry), timeout)
        if len(new_tests) < min_new_tests:
            rec["stop"] = "tests_too_few"
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
            status = check_turn(new_code, new_tests, [a for t in turns for a in t["tests"]], timeout)
        turns.append({"user": request, "assistant": text, "kind": kind,
                      "tests": new_tests, "status": status})
        if status != "ok":
            rec["stop"] = status
            break
        code = new_code
    rec["kept_turns"] = next((i for i, t in enumerate(turns) if t["status"] != "ok"), len(turns))
    return rec
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add quipu/chains.py tests/test_chains.py
git commit -m "chains: run_chain grows a seed turn by turn and stops at the first failed turn"
```

---

### Task 4: The generator script

**Files:**
- Create: `scripts/chain_generate.py`
- Test: `tests/test_chains.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from scripts import chain_generate as cg


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _distill_part(root: Path, n: int = 3) -> None:
    kept, prompts = [], []
    for i in range(n):
        sid = f"dv2-{i:07d}"
        kept.append({"id": sid, "split": "val" if i == 0 else "train", "entry_point": "add",
                     "prompt": SEED["prompt"], "response": SEED["response"],
                     "teacher": "t", "revision": "r", "teacher_tag": "q4"})
        prompts.append({"id": sid, "tests": SEED["tests"], "entry_point": "add"})
    prompts.append({"id": "dv2-9999999", "tests": [], "entry_point": "add"})   # not kept
    _write_jsonl(root / "kept" / "part-0000.jsonl", kept)
    _write_jsonl(root / "prompts" / "part-0000.jsonl", prompts)


def test_load_seeds_joins_tests_by_id_and_filters_split(tmp_path):
    _distill_part(tmp_path)
    seeds = cg.load_seeds(tmp_path, 0)
    assert [s["id"] for s in seeds] == ["dv2-0000000", "dv2-0000001", "dv2-0000002"]
    assert seeds[1]["tests"] == SEED["tests"]
    assert [s["id"] for s in cg.load_seeds(tmp_path, 0, split="val")] == ["dv2-0000000"]


def test_run_part_writes_resumes_and_pauses(tmp_path):
    _distill_part(tmp_path / "distill")
    seeds = cg.load_seeds(tmp_path / "distill", 0)
    out = tmp_path / "chains" / "gen" / "part-0000.q4.jsonl"
    meta = {"teacher": "t", "revision": "r", "teacher_tag": "q4"}
    pause = tmp_path / "chains" / "PAUSE"
    st = cg.run_part(seeds[:2], out, FakeTeacher(), concurrency=2, pause=pause, meta=meta, max_turns=2)
    assert st == "done"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == ["ch-dv2-0000000", "ch-dv2-0000001"]
    assert rows[0]["teacher_tag"] == "q4" and rows[0]["kept_turns"] == 2
    t = FakeTeacher()
    assert cg.run_part(seeds, out, t, concurrency=2, pause=pause, meta=meta, max_turns=2) == "done"
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3      # only the new seed ran
    assert len(t.calls) == 3                                           # 3 calls for one chain
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text("")
    fresh = tmp_path / "chains" / "gen" / "part-0001.q4.jsonl"
    assert cg.run_part(seeds, fresh, FakeTeacher(), concurrency=2, pause=pause, meta=meta) == "paused"
    assert not fresh.exists() or fresh.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: FAIL with `ImportError: cannot import name 'chain_generate' from 'scripts'`

- [ ] **Step 3: Write the implementation**

```python
"""Grow distill v2's solved problems into verified multi-turn chains (quipu.chains).

Reads data/distill-v2/kept/part-NNNN.jsonl (problem + teacher answer that passed) and
joins each row's unit tests by id from data/distill-v2/prompts/part-NNNN.jsonl. Writes
one chain per seed to data/chains/gen/part-NNNN.<teacher-tag>.jsonl, appended and
flushed as each chain finishes. Rerun to resume (finished ids are skipped); create
data/chains/PAUSE to stop cleanly after the chains in flight.

Teacher: an OpenAI-compatible server, the same Qwen2.5-Coder-7B-Instruct as distill v2
(llama-server with the Q4_K_M GGUF on the laptop, vLLM on Kaggle). Greedy decoding.

    .venv/Scripts/python.exe scripts/chain_generate.py --parts 0:0 --limit 200 \\
        --endpoint http://127.0.0.1:8080 --teacher-tag q4km-llamacpp \\
        --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --concurrency 2
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.chains import Chat, run_chain                                   # noqa: E402
from scripts.distill_generate_v2 import TEACHER, _drop_torn_tail, done_ids, part_order  # noqa: E402

SHOW_FAILURES = 5


def load_seeds(distill_root: Path, part: int, split: str = "all") -> list[dict]:
    """Kept rows of one distill v2 part, each with its problem's tests; `split` is
    "all", "train" or "val" (val seeds are the held-out eval set)."""
    root = Path(distill_root)
    with open(root / "prompts" / f"part-{part:04d}.jsonl", encoding="utf-8") as f:
        tests = {r["id"]: r["tests"] for r in map(json.loads, filter(str.strip, f))}
    seeds = []
    with open(root / "kept" / f"part-{part:04d}.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, filter(str.strip, f)):
            if r["id"] in tests and split in ("all", r["split"]):
                seeds.append({k: r[k] for k in ("id", "split", "entry_point", "prompt", "response")}
                             | {"tests": tests[r["id"]]})
    return seeds


def run_part(seeds: list[dict], out: Path, chat: Chat, *, concurrency: int, pause: Path,
             meta: dict, **chain_kw) -> str:
    """Run every seed whose chain isn't in `out` yet. Returns "done" | "paused" |
    "incomplete" (some chains raised, e.g. the server dropped requests; rerun)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(out)
    have = done_ids(out)
    todo = iter([s for s in seeds if "ch-" + s["id"] not in have])
    failed, stop, running = 0, "", set()
    with open(out, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(concurrency) as ex:
        while True:
            if not stop and pause.exists():
                stop = "paused"
            while not stop and len(running) < concurrency:
                s = next(todo, None)
                if s is None:
                    break
                running.add(ex.submit(run_chain, s, chat, **chain_kw))
            if not running:
                break
            finished, running = wait(running, timeout=5.0, return_when=FIRST_COMPLETED)
            for fut in finished:
                if fut.exception() is not None:
                    failed += 1
                    if failed <= SHOW_FAILURES:
                        print(f"chain failed: {fut.exception()!r}", flush=True)
                    continue
                f.write(json.dumps(fut.result() | meta, ensure_ascii=False) + "\n")
                f.flush()
    return stop or ("incomplete" if failed else "done")


def make_messages_chat(endpoint: str, model: str = "teacher", timeout: float = 600.0,
                       retries: int = 3, backoff: float = 5.0) -> Chat:
    """chat(messages, max_tokens) for an OpenAI-compatible server, greedy. Retries
    connection trouble, timeouts and HTTP 5xx; a 4xx is raised at once."""
    import http.client
    import time
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def chat(messages: list[dict], max_tokens: int) -> tuple[str, str, int]:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": messages}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                c = d["choices"][0]
                reason = c.get("finish_reason") or "stop"
                if reason not in ("stop", "length"):
                    raise RuntimeError(f"finish_reason={reason}")
                return (c["message"]["content"] or "", reason,
                        int(d.get("usage", {}).get("completion_tokens", 0)))
            except urllib.error.HTTPError as e:
                e.close()
                if e.code < 500 or attempt == retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
            time.sleep(backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    return chat


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--distill-root", default="data/distill-v2")
    ap.add_argument("--out-root", default="data/chains")
    ap.add_argument("--parts", required=True, help="START:END inclusive, e.g. 0:4")
    ap.add_argument("--direction", choices=("up", "down"), default="up")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N seeds per part (smoke runs)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--teacher-tag", required=True, help="e.g. q4km-llamacpp, fp16-vllm-t4")
    ap.add_argument("--revision", required=True, help="teacher weights revision (sha)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=3)
    args = ap.parse_args(argv)

    out_root = ROOT / args.out_root
    chat = make_messages_chat(args.endpoint)
    meta = {"teacher": TEACHER, "revision": args.revision, "teacher_tag": args.teacher_tag}
    for part in part_order(args.parts, args.direction):
        seeds = load_seeds(ROOT / args.distill_root, part, args.split)[:args.limit]
        out = out_root / "gen" / f"part-{part:04d}.{args.teacher_tag}.jsonl"
        print(f"part {part}: {len(seeds)} seeds", flush=True)
        status = run_part(seeds, out, chat, concurrency=args.concurrency,
                          pause=out_root / "PAUSE", meta=meta, max_turns=args.max_turns)
        print(f"part {part}: {status}", flush=True)
        if status == "paused":
            return 0
        if status == "incomplete":
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/chain_generate.py tests/test_chains.py
git commit -m "chains: generator over distill v2 parts with resume and a pause file"
```

---

### Task 5: The verifier and summary

**Files:**
- Create: `scripts/chain_verify.py`
- Test: `tests/test_chains.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from scripts import chain_verify as cv


def test_verify_rechecks_turns_and_writes_chat_messages(tmp_path):
    good = ch.run_chain(SEED, FakeTeacher(), max_turns=2) | {"teacher_tag": "q4"}
    lying = json.loads(json.dumps(good))                    # says ok, but its code is broken
    lying["id"], lying["seed_id"] = "ch-dv2-0000008", "dv2-0000008"
    lying["turns"][1]["assistant"] = "```python\ndef add(a, b, c=0):\n    return 0\n```"
    short = ch.run_chain(SEED | {"id": "dv2-0000009"},
                         FakeTeacher(answer="```python\n" + OLD + "```"), max_turns=2) | {"teacher_tag": "q4"}
    _write_jsonl(tmp_path / "gen" / "part-0000.q4.jsonl", [good, lying, short])

    summary = cv.verify(tmp_path, workers=2)

    kept = [json.loads(l) for l in (tmp_path / "kept" / "part-0000.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [k["id"] for k in kept] == ["ch-dv2-0000007"]
    assert [m["role"] for m in kept[0]["messages"]] == ["user", "assistant", "user", "assistant"]
    assert kept[0]["turns"] == 2
    assert summary["chains"] == 3 and summary["kept"] == 1
    assert summary["recheck_failed"] == 1                   # the lying chain
    assert summary["stop"]["copy"] == 1
    assert json.loads((tmp_path / "verify_summary.json").read_text(encoding="utf-8")) == summary
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: FAIL with `ImportError: cannot import name 'chain_verify' from 'scripts'`

- [ ] **Step 3: Write the implementation**

```python
"""Re-verify generated chains from scratch and write the kept ones as chat messages.

For every chain in data/chains/gen/*.jsonl, each turn the generator marked ok is checked
again in a fresh sandbox (its code against its own asserts and every earlier turn's).
The chain is cut at the first turn that fails; it is kept when 2+ turns survive. Kept
chains go to data/chains/kept/part-NNNN.jsonl as {"messages": [...]} rows; counts go
to data/chains/verify_summary.json. When a chain id appears under several teacher
tags, the first by scripts.distill_verify's tag priority wins.

    .venv/Scripts/python.exe scripts/chain_verify.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.chains import check_turn                       # noqa: E402
from quipu.fsio import write_text_atomic                   # noqa: E402
from quipu.sandbox import extract_code                     # noqa: E402
from scripts.distill_verify import _tag_rank               # noqa: E402


def recheck(chain: dict, timeout: float = 10.0) -> int:
    """How many leading turns pass a fresh check (turn 1 against its own tests)."""
    entry, earlier = chain["entry_point"], []
    for i, t in enumerate(chain["turns"]):
        if t["status"] != "ok":
            return i
        code = extract_code(t["assistant"], entry)
        if code is None or check_turn(code, t["tests"], earlier, timeout) != "ok":
            return i
        earlier += t["tests"]
    return len(chain["turns"])


def _load(gen_dir: Path) -> dict[str, list[dict]]:
    """Chains per part, first by teacher-tag priority for a repeated id."""
    files = sorted(gen_dir.glob("part-*.jsonl"), key=lambda p: (p.name[:9], _tag_rank(p.name[10:-6])))
    seen: set[str] = set()
    parts: dict[str, list[dict]] = {}
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c["id"] not in seen:
                seen.add(c["id"])
                parts.setdefault(p.name[:9], []).append(c)
    return parts


def verify(root: Path, workers: int = 8, timeout: float = 10.0) -> dict:
    root = Path(root)
    summary = {"chains": 0, "kept": 0, "kept_3_turns": 0, "recheck_failed": 0,
               "stop": Counter(), "kept_by_split": Counter()}
    for part, chains in sorted(_load(root / "gen").items()):
        with ThreadPoolExecutor(workers) as ex:
            ok_turns = list(ex.map(lambda c: recheck(c, timeout), chains))
        rows = []
        for c, n in zip(chains, ok_turns):
            summary["chains"] += 1
            summary["stop"][c["stop"] or "max_turns"] += 1
            if n < c["kept_turns"]:
                summary["recheck_failed"] += 1
            if n < 2:
                continue
            msgs = []
            for t in c["turns"][:n]:
                msgs += [{"role": "user", "content": t["user"]},
                         {"role": "assistant", "content": t["assistant"]}]
            rows.append({"id": c["id"], "seed_id": c["seed_id"], "split": c["split"],
                         "teacher": c.get("teacher"), "revision": c.get("revision"),
                         "teacher_tag": c.get("teacher_tag"), "turns": n, "messages": msgs})
            summary["kept"] += 1
            summary["kept_3_turns"] += n >= 3
            summary["kept_by_split"][c["split"]] += 1
        (root / "kept").mkdir(parents=True, exist_ok=True)
        write_text_atomic(root / "kept" / f"{part}.jsonl",
                          "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = json.loads(json.dumps(summary))               # Counters -> plain dicts
    write_text_atomic(root / "verify_summary.json", json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/chains")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    print(json.dumps(verify(ROOT / args.root, args.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chains.py -q`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/chain_verify.py tests/test_chains.py
git commit -m "chains: verifier re-checks every turn and writes kept chats and a summary"
```

---

### Task 6: 200-chain laptop smoke run

No code; this proves the loop on the real teacher before any Kaggle time. Warn the owner first: the teacher holds ~5-6 GB of the laptop GPU for about an hour, and WebView2/Electron apps (WhatsApp, Teams, Outlook, Claude desktop) may freeze if started meanwhile.

- [ ] **Step 1: Start the teacher server** (PowerShell, repo root; 2 slots x 8,192 tokens so a 3-turn conversation plus a 1,024-token answer fits)

```powershell
Start-Process -WindowStyle Minimized -FilePath .\tools\llama.cpp\llama-server.exe -ArgumentList '-m "models\qwen2.5-coder-7b-instruct-q4_k_m.gguf" -ngl 99 -c 16384 -np 2 --host 127.0.0.1 --port 8080 --alias teacher'
```

Wait until `Invoke-RestMethod http://127.0.0.1:8080/health` answers.

- [ ] **Step 2: Generate 200 chains**

```bash
.venv/Scripts/python.exe scripts/chain_generate.py --parts 0:0 --limit 200 \
  --endpoint http://127.0.0.1:8080 --teacher-tag q4km-llamacpp \
  --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --concurrency 2
```

Expected: `part 0: 200 seeds` … `part 0: done`. Note the wall-clock time (chains per hour sets the Kaggle target).

- [ ] **Step 3: Stop the server** (`Stop-Process -Name llama-server` only for the process started in step 1: use the PID from `Get-Process llama-server` and confirm it is the one started here).

- [ ] **Step 4: Verify**

Run: `.venv/Scripts/python.exe scripts/chain_verify.py`
Expected: a summary with `chains: 200`; read `kept`, `kept_3_turns`, `recheck_failed` (should be 0) and the `stop` counts.

- [ ] **Step 5: Read 10 chains by eye** (5 kept, 5 stopped at `fail_new` or `tests_too_few`): are requests sensible and of the named kind, are the asserts about the change, are answers complete functions? Record findings.

- [ ] **Step 6: Write the results into the spec** under a new "Smoke run (date)" section: chains/hour, kept rate, 3-turn rate, stop-reason table, what the 10 chains showed, and any wording change to the instructions. If the kept rate is under ~20% or the eye check finds systematic problems, fix the instructions (Task 1 strings) and rerun steps 1-5 on parts 1:1 before moving on.

```bash
git add docs/superpowers/specs/2026-10-15-multiturn-verified-chat-design.md
git commit -m "spec: multi-turn chat, laptop smoke run results"
```

---

## After this plan

Next plans, each written once this one's smoke results are in: B (web pages + jsdom checker), C (stepbuild consecutive-commit chains), chat data + `chain_eval.py` metrics, and the Kaggle runs (vLLM generation notebook, fp16-vs-bf16 check, fine-tune).
