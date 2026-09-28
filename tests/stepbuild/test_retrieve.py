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


# Realistic dataset rows: commit messages as steps, full-file replies. Each
# distractor shares many words with the step titles it should NOT win for.
REAL = [
    _ex(  # test file
        "Add pytest tests for the todo API endpoints: create returns 201, list returns "
        "a JSON array, delete then 404",
        "backend/tests/test_api.py",
        "def test_create(client):\n    r = client.post('/api/todos', json={'title': 'a'})\n"
        "    assert r.status_code == 201\n",
    ),
    _ex(  # frontend
        "Add TodoList component with a delete button for each todo item",
        "src/components/TodoList.jsx",
        "export default function TodoList({ items, onDelete }) {\n"
        "  return <ul>{items.map(i => <li key={i.id}>{i.title}"
        "<button onClick={() => onDelete(i.id)}>x</button></li>)}</ul>\n}",
    ),
    _ex(  # models (.py, not test)
        "Add Note model with title validation",
        "server/models.py",
        "class Note:\n    def __init__(self, title):\n        self.title = title\n",
    ),
    _ex(  # routes (.py, not test)
        "Add Flask routes to create and delete notes with proper status codes",
        "server/app.py",
        "from flask import Flask, jsonify, request\n\ndef create_app():\n"
        "    app = Flask(__name__)\n    return app\n",
    ),
    _ex(  # frontend form
        "Add a controlled form to submit a new note and clear the inputs",
        "src/components/NoteForm.jsx",
        "import { useState } from 'react'\nexport default function NoteForm({ onSubmit }) {\n"
        "  const [title, setTitle] = useState('')\n  return null\n}",
    ),
]


def test_routes_query_prefers_a_non_test_python_example():
    from stepbuild.harness.plan import make_plan

    step = make_plan("todo", "spec").steps[1]
    got = ExampleLibrary(REAL).top(step.title, k=2, prefer_paths=step.allowed_files)
    assert {e.reply.split("\n", 1)[0] for e in got} == {
        "=== FILE: server/models.py ===",
        "=== FILE: server/app.py ===",
    }


def test_tests_query_prefers_a_test_file_example():
    from stepbuild.harness.plan import make_plan

    step = make_plan("todo", "spec").steps[2]
    got = ExampleLibrary(REAL).top(step.title, k=2, prefer_paths=step.allowed_files)
    assert got[0] is REAL[0]


def test_components_query_prefers_a_jsx_example():
    from stepbuild.harness.plan import make_plan

    step = make_plan("todo", "spec").steps[3]
    got = ExampleLibrary(REAL).top(step.title, k=2, prefer_paths=step.allowed_files)
    assert {id(e) for e in got} == {id(REAL[1]), id(REAL[4])}


def test_preferred_role_first_then_the_rest_in_bm25_order():
    lib = ExampleLibrary(REAL)
    q = "Add TodoList component with a delete button, pytest tests for the notes routes"
    plain = lib.top(q, k=5)
    preferred = lib.top(q, k=5, prefer_paths=["tests/test_x.py"])
    assert preferred[0] is REAL[0]
    assert preferred[1:] == [e for e in plain if e is not REAL[0]]
    # a role nothing writes changes nothing
    assert lib.top(q, k=5, prefer_paths=["README.md"]) == plain


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


def test_from_jsonl_keeps_multi_paragraph_steps_and_normalises_crlf(tmp_path):
    p = tmp_path / "crlf.jsonl"
    step = "Add login route\r\n\r\nUses a session cookie.\r\nCloses #12"
    p.write_text(_row("train", step, "=== FILE: a.py ===\r\nx = 1\r\n=== END FILE ===\r\n"),
                 encoding="utf-8")
    (ex,) = ExampleLibrary.from_jsonl([p]).examples
    assert ex.step == "Add login route\n\nUses a session cookie.\nCloses #12"
    assert ex.reply == "=== FILE: a.py ===\nx = 1\n=== END FILE ===\n"


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
