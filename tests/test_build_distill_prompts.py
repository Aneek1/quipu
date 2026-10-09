"""scripts/build_distill_prompts.py: the per-row rules (no network)."""
import json
import random
from collections import Counter

from scripts.build_distill_prompts import (apps_problem, entry_point, function_problem,
                                           select)

PROBLEM = "Given a list of integers, return the sum of the even ones. " * 4


def oci(**kw):
    row = {"id": "x1", "input": PROBLEM, "average_test_score": 1.0,
           "unit_tests": json.dumps([f"\nassert sum_even([{i}, 2]) == {2 + (i if i % 2 == 0 else 0)}\n"
                                     for i in range(1, 5)])}
    row.update(kw)
    return row


def apps(**kw):
    io = {"fn_name": "add", "inputs": [[1, 2], [3, 4], [5, 6]], "outputs": [[3], [7], [11]]}
    row = {"id": 7, "difficulty": "introductory", "question": PROBLEM,
           "starter_code": "\ndef add(a, b):\n\t", "input_output": json.dumps(io)}
    row.update(kw)
    return row


def test_entry_point_single_function():
    assert entry_point(["assert f(1) == 2", "assert not f(0)"]) == "f"
    assert entry_point(["assert f(1) == 2", "assert g(1) == 2"]) is None
    assert entry_point(["assert len(f(1)) == 2"]) is None  # builtin first
    assert entry_point(["x = f(1)"]) is None


def test_function_problem_kept_and_prompt_names_function():
    p, why = function_problem(oci())
    assert why == "" and p["entry_point"] == "sum_even" and len(p["tests"]) == 4
    assert p["prompt"].startswith(PROBLEM.strip()) and "`sum_even`" in p["prompt"]
    assert "output" not in p  # the dataset's reference answer is never carried


def test_function_problem_drops():
    assert function_problem(oci(average_test_score=0.9))[1] == "reference_failed_tests"
    assert function_problem(oci(unit_tests=json.dumps(["assert f(1)"])))[1] == "test_count"
    assert function_problem(oci(input="short"))[1] == "prompt_length"


def test_apps_problem_asserts_and_drops():
    p, why = apps_problem(apps())
    assert why == "" and p["entry_point"] == "add"
    assert p["tests"] == ["assert add(1, 2) == 3", "assert add(3, 4) == 7",
                          "assert add(5, 6) == 11"]
    assert "def add(a, b):" in p["prompt"] and "`add`" in p["prompt"]
    assert apps_problem(apps(difficulty="interview"))[1] == "difficulty"
    assert apps_problem(apps(starter_code="class Solution:\n def add(self):"))[1] == "class_solution"
    fl = json.dumps({"fn_name": "add", "inputs": [[1], [2], [3]], "outputs": [[1.5], [2], [3]]})
    assert apps_problem(apps(input_output=fl))[1] == "float_output"
    few = json.dumps({"fn_name": "add", "inputs": [[1]], "outputs": [[1]]})
    assert apps_problem(apps(input_output=few))[1] == "case_count"
    stdin = json.dumps({"inputs": ["1\n"] * 3, "outputs": ["1\n"] * 3})
    assert apps_problem(apps(input_output=stdin))[1] == "not_call_based"


class FakeDecontam:
    def find(self, text):
        return ("humaneval", "HumanEval/0", "substring") if "has_close_elements" in text else None


def test_select_dedupes_and_decontaminates():
    a, _ = function_problem(oci())
    b = dict(a)
    c = dict(a, problem="Check if in given list of numbers, are any two closer " * 3,
             prompt="has_close_elements ...")
    drops, hits = Counter(), Counter()
    out = select([a, b, c], 10, FakeDecontam(), set(), drops, hits, random.Random(0))
    assert len(out) == 1
    assert drops["opencodeinstruct:duplicate"] == 1
    assert drops["opencodeinstruct:decontam_humaneval"] == 1 and hits["humaneval"] == 1
