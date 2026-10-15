"""Verified multi-turn code chains (quipu.chains, scripts/chain_generate.py,
scripts/chain_verify.py). The teacher is a scripted fake; the sandbox is real."""
import json
from pathlib import Path

import pytest

from quipu import chains as ch


def test_pick_kinds_is_deterministic_and_distinct():
    a = ch.pick_kinds("dv2-0000001", 2)
    assert a == ch.pick_kinds("dv2-0000001", 2)
    assert len(a) == 2 and a[0] != a[1]
    assert all(k in ch.CHANGE_KINDS for k in a)
    seen = {ch.pick_kinds(f"dv2-{i:07d}", 1)[0] for i in range(200)}
    assert seen == set(ch.CHANGE_KINDS)          # every kind gets used


def test_pick_kinds_refuses_more_kinds_than_exist():
    with pytest.raises(ValueError):
        ch.pick_kinds("dv2-0000001", len(ch.CHANGE_KINDS) + 1)


def test_parse_request_accepts_plain_words_only():
    assert ch.parse_request('  "Let add take an optional third number c."  ') == \
        "Let add take an optional third number c."
    assert ch.parse_request("Sure:\n```python\ndef f(): pass\n```") is None   # code
    assert ch.parse_request("ok") is None                                      # too short
    assert ch.parse_request("x" * 601) is None                                 # too long


def test_parse_request_strips_curly_quotes_and_rejects_chatter():
    assert ch.parse_request("\u201cLet add take an optional third number c.\u201d") == \
        "Let add take an optional third number c."
    for bad in ("Sure! Let add take an optional third number c.",
                "Here is the message: let add take a third number.",
                "Here's a request: let add take a third number.",
                "User: let add take an optional third number c.",
                "The user wants add to take a third number c.",
                "Please let add take a third number, and write a test for it."):
        assert ch.parse_request(bad) is None, bad
    assert ch.parse_request("Please add a contest flag to add.") is not None   # whole word only


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


def test_parse_asserts_rejects_weak_or_misleading_asserts():
    text = (
        "assert add(1, 2, 3)\n"                          # bare truthy
        "assert isinstance(add(1, 2, 3), int)\n"         # type check only
        "assert True, add(1, 2)\n"                       # name only in the message
        "assert add(1, 2, 3) > 5\n"                      # not Eq/NotEq
        "assert 1 < add(1, 2, 3) < 9\n"                  # chained comparison
        "assert add(1, 2, 3) == add(3, 2, 1)\n"          # no literal side
        "assert add(1, 2, 3) in (6,)\n"
        "assert _raises(add, 'a', 1)\n"                  # exception class missing
        "assert _raises(ValueError, len, 'a')\n"         # not about add
        "assert add(1, 2, 3) != 7\n"                     # accepted: NotEq with literal
        "assert 6 == add(1, 2, 3)\n"                     # accepted: literal on the left
        "assert add(1, 2, 3) == [6, (1, 'a')]\n"         # accepted: literal container
    )
    assert ch.parse_asserts(text, "add") == [
        "assert add(1, 2, 3) != 7",
        "assert 6 == add(1, 2, 3)",
        "assert add(1, 2, 3) == [6, (1, 'a')]",
    ]


def test_parse_asserts_reads_unfenced_replies():
    assert ch.parse_asserts("assert add(2, 2) == 4", "add") == ["assert add(2, 2) == 4"]


OLD = "def add(a, b):\n    return a + b\n"
NEW = "def add(a, b, c=0):\n    return a + b + c\n"


K_PARAM, K_EDGE, K_FLAG, K_VALIDATE = ch.CHANGE_KINDS


def test_new_asserts_keeps_asserts_the_old_code_fails_for_the_right_reason():
    asserts = ["assert add(1, 2, 3) == 6", "assert add(1, 2) == 3", "assert add(0, 0, 5) == 5"]
    assert ch.new_asserts(OLD, asserts, K_PARAM) == \
        ["assert add(1, 2, 3) == 6", "assert add(0, 0, 5) == 5"]    # TypeError: no third parameter


def test_new_asserts_drops_undefined_names():
    assert ch.new_asserts(OLD, ["assert ad(1, 2) == 3", "assert add(1, 2, 3) == 6"], K_EDGE) == \
        ["assert add(1, 2, 3) == 6"]


def test_new_asserts_only_the_edge_kind_may_change_a_result():
    changed = ["assert add(1, 2) == 4"]                 # old code gives 3: an AssertionError
    assert ch.new_asserts(OLD, changed, K_EDGE) == changed
    for kind in (K_PARAM, K_FLAG, K_VALIDATE):
        assert ch.new_asserts(OLD, changed, kind) == []


def test_new_asserts_supports_raises_checks_for_the_validation_kind():
    old = "def half(x):\n    return x / 2\n"
    raises = ["assert _raises(ValueError, half, 'a')"]  # old code raises TypeError, not ValueError
    assert ch.new_asserts(old, raises, K_VALIDATE) == raises
    assert ch.new_asserts(old, raises, K_PARAM) == []
    assert ch.new_asserts(old, ["assert half(4) == 3"], K_VALIDATE) == []


def test_new_asserts_raises_on_infrastructure_trouble():
    slow = "def add(a, b):\n    while True:\n        pass\n"
    with pytest.raises(ch.SandboxError):
        ch.new_asserts(slow, ["assert add(1, 2) == 3"], K_PARAM, timeout=2)
    with pytest.raises(ch.SandboxError):
        ch.new_asserts("def add(a, b):\n    return a +\n", ["assert add(1, 2) == 3"], K_PARAM)


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


def test_check_turn_reports_infrastructure_trouble_as_sandbox_error():
    slow = "def add(a, b, c=0):\n    while True:\n        pass\n"
    assert ch.check_turn(slow, ["assert add(1, 2, 3) == 6"], ["assert add(1, 2) == 3"],
                         timeout=2) == "sandbox_error"


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
    assert rec["rejected"] == {"request": FakeTeacher().request, "asserts": ["assert add(1, 2) == 3"]}


def test_run_chain_stops_on_a_cut_off_answer():
    rec = ch.run_chain(SEED, FakeTeacher(finish="length"), max_turns=2)
    assert rec["stop"] == "answer_length" and rec["kept_turns"] == 1


def test_run_chain_third_turn_needs_tests_the_current_code_fails():
    t = FakeTeacher(tests="```python\nassert add(1, 2, 3) == 6\nassert add(0, 0, 5) == 5\n```")
    rec = ch.run_chain(SEED, t, max_turns=3)
    # Turn 3's tests are the same lines, which NEW already passes, so turn 3 stops at
    # tests_too_few: the chain keeps its two good turns.
    assert rec["kept_turns"] == 2 and rec["stop"] == "tests_too_few"


def test_run_chain_stops_on_a_bad_request_and_keeps_what_was_rejected():
    rec = ch.run_chain(SEED, FakeTeacher(request="Sure! ```x```"), max_turns=2)
    assert rec["stop"] == "request_bad" and rec["kept_turns"] == 1
    assert rec["rejected"] == {"request": "Sure! ```x```"}


def test_run_chain_stops_when_the_answer_has_no_code():
    rec = ch.run_chain(SEED, FakeTeacher(answer="I would change add to take c."), max_turns=2)
    assert rec["stop"] == "no_code" and rec["kept_turns"] == 1


def test_run_chain_stops_when_the_seed_has_no_code():
    rec = ch.run_chain({**SEED, "response": "No code here."}, FakeTeacher(), max_turns=2)
    assert rec["stop"] == "seed_no_code" and rec["kept_turns"] == 1


def test_run_chain_stops_on_an_answer_that_misses_the_new_tests():
    wrong = "def add(a, b, c=0):\n    return a + b\n"        # takes c, ignores it
    rec = ch.run_chain(SEED, FakeTeacher(answer="```python\n" + wrong + "```"), max_turns=2)
    assert rec["stop"] == "fail_new" and rec["turns"][1]["status"] == "fail_new"
    assert rec["kept_turns"] == 1


class _AnswerRaises(FakeTeacher):
    def __call__(self, messages, max_tokens):
        if not messages[0]["content"].startswith(("You write the next", "You write unit")):
            raise ConnectionError("teacher unreachable")
        return super().__call__(messages, max_tokens)


def test_run_chain_ends_with_an_error_stop_when_the_teacher_fails():
    rec = ch.run_chain(SEED, _AnswerRaises(), max_turns=2)
    assert rec["stop"] == "error:ConnectionError" and rec["kept_turns"] == 1


def test_run_chain_ends_with_an_error_stop_when_the_sandbox_cannot_run_the_code():
    slow = {**SEED, "response": "```python\ndef add(a, b):\n    while True:\n        pass\n```"}
    rec = ch.run_chain(slow, FakeTeacher(), max_turns=2, timeout=2)
    assert rec["stop"] == "error:SandboxError" and rec["kept_turns"] == 1
