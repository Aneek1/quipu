"""Retrieval picks the dataset examples shown to the model. It must find the example
that shares a rare identifier with the query, return nothing for an empty library,
and rank deterministically so a benchmark run is reproducible."""
import dataclasses
import json

import pytest

from stepbuild.harness.retrieve import Example, ExampleLibrary

BLOCK = "=== FILE: {path} ===\n{body}\n=== END FILE ===\n"


def _ex(step, path, body):
    return Example(step=step, reply=BLOCK.format(path=path, body=body))


LIB = [
    _ex("Add a user model", "models.py", "class User:\n    pass"),
    _ex("Add the invoice ledger", "ledger.py", "def reconcile_quuxfrobnicate(rows):\n    return rows"),
    _ex("Render the list", "src/List.jsx", "export default function List() { return null }"),
]


def test_example_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LIB[0].step = "x"


def test_rare_identifier_ranks_first():
    top = ExampleLibrary(LIB).top("call reconcile_quuxfrobnicate in the model", k=2)
    assert top[0] is LIB[1]
    assert len(top) <= 2


def test_empty_library_returns_empty():
    assert ExampleLibrary([]).top("anything") == []


def test_k_zero_returns_empty():
    assert ExampleLibrary(LIB).top("reconcile_quuxfrobnicate", k=0) == []


def test_ties_go_to_the_earlier_example():
    same = [_ex("Add a thing", "a.py", "x = 1"), _ex("Add a thing", "a.py", "x = 1")]
    lib = ExampleLibrary(same)
    for _ in range(3):
        got = lib.top("Add a thing", k=2)
        assert got[0] is same[0] and got[1] is same[1]


def test_deterministic_across_instances():
    q = "Render the list of users"
    assert ExampleLibrary(LIB).top(q, k=3) == ExampleLibrary(LIB).top(q, k=3)


def _row(split, step, reply, **extra):
    row = {
        "repo": "o/r", "licence": "MIT", "commit": "abc", "split": split,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"STEP: {step}\n\nCONTEXT FILES:\nx\n\nPROJECT TREE:\na.py"},
            {"role": "assistant", "content": reply},
        ],
    }
    row.update(extra)
    return json.dumps(row)


def test_from_jsonl_reads_only_the_requested_split(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        _row("train", "Add login route", "R1") + "\n\n" + _row("validation", "Val step", "R2") + "\n",
        encoding="utf-8",
    )
    b.write_text(_row("train", "Add logout", "R3") + "\n" + _row("test", "Test step", "R4"), encoding="utf-8")
    train = ExampleLibrary.from_jsonl([a, b])
    assert train.examples == (
        Example("Add login route", "R1"),
        Example("Add logout", "R3"),
    )
    val = ExampleLibrary.from_jsonl([a, b], split="validation")
    assert val.examples == (Example("Val step", "R2"),)
    assert ExampleLibrary.from_jsonl([a], split="test").examples == ()


def test_from_jsonl_rejects_malformed_lines_with_location(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(_row("train", "ok", "R") + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        ExampleLibrary.from_jsonl([p])


def test_from_jsonl_rejects_user_message_without_step(tmp_path):
    p = tmp_path / "nostep.jsonl"
    row = json.loads(_row("train", "x", "R"))
    row["messages"][1]["content"] = "no step line here"
    p.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="STEP"):
        ExampleLibrary.from_jsonl([p])
