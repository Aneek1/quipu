"""format_example turns a kept commit into one chat example (spec §3.1). The harness
reads these rows back (ExampleLibrary.from_jsonl) and a fine-tuned model is prompted
with the harness's framing, so the shape is pinned exactly and checked by a round
trip through the harness's own loader and FILE-block parser."""
import json
import math

import pytest

from stepbuild.dataset.filters import Commit, FileChange
from stepbuild.dataset.format import format_example
from stepbuild.dataset.split import assign_split
from stepbuild.harness.blocks import FileBlock, parse_blocks, render_blocks
from stepbuild.harness.prompt import SYSTEM_PROMPT
from stepbuild.harness.retrieve import ExampleLibrary

MODELS_BEFORE = "class Store:\n    pass\n"
MODELS_AFTER = "class Store:\n    def __init__(self):\n        self.items = {}\n"
LIST_AFTER = "export default function List() {\n  return null\n}\n"


def make_commit(message="Add the item store", sha="abc123"):
    return Commit(
        sha=sha,
        message=message,
        parents=1,
        changes=(
            FileChange("backend/models.py", MODELS_BEFORE, MODELS_AFTER, 2, 1),
            FileChange("frontend/src/List.jsx", None, LIST_AFTER, 3, 0),
        ),
    )


CONTEXT = {"backend/models.py": MODELS_BEFORE, "backend/app.py": "from models import Store\n"}
TREE = ["backend/app.py", "backend/models.py", "frontend/src/List.jsx"]


def fmt(**kw):
    args = dict(
        repo="owner/name", licence="MIT", tag="fullstack", commit=make_commit(),
        context=CONTEXT, tree=TREE,
    )
    args.update(kw)
    return format_example(**args)


def test_exact_shape():
    row = fmt()
    assert list(row) == ["repo", "licence", "tag", "commit", "split", "messages"]
    assert row["repo"] == "owner/name"
    assert row["licence"] == "MIT"
    assert row["tag"] == "fullstack"
    assert row["commit"] == "abc123"
    assert row["split"] == assign_split("owner/name")
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert all(list(m) == ["role", "content"] for m in row["messages"])
    assert row["messages"][0]["content"] == SYSTEM_PROMPT


def test_user_message_exact():
    user = fmt()["messages"][1]["content"]
    context = render_blocks(
        [
            FileBlock("backend/models.py", MODELS_BEFORE),
            FileBlock("backend/app.py", "from models import Store\n"),
        ]
    )
    assert user == (
        "STEP: Add the item store\n\n"
        "CONTEXT FILES:\n" + context + "\n"
        "PROJECT TREE:\n"
        "backend/app.py\nbackend/models.py\nfrontend/src/List.jsx\n"
    )


def test_context_keeps_the_given_order():
    ctx = {"backend/app.py": "a = 1\n", "backend/models.py": MODELS_BEFORE}
    user = fmt(context=ctx)["messages"][1]["content"]
    assert user.index("=== FILE: backend/app.py") < user.index("=== FILE: backend/models.py")


def test_assistant_has_post_commit_contents_of_every_changed_file():
    reply = fmt()["messages"][2]["content"]
    assert reply == render_blocks(
        [FileBlock("backend/models.py", MODELS_AFTER), FileBlock("frontend/src/List.jsx", LIST_AFTER)]
    )
    assert parse_blocks(reply) == [
        FileBlock("backend/models.py", MODELS_AFTER),
        FileBlock("frontend/src/List.jsx", LIST_AFTER),
    ]


def test_empty_context_says_so_like_the_harness():
    user = fmt(context={})["messages"][1]["content"]
    assert "CONTEXT FILES:\n(none yet)\n\nPROJECT TREE:\n" in user


def test_multi_paragraph_message_keeps_its_body_stripped_and_lf():
    msg = "  Add delete endpoint\r\n\r\nReturns 404 when the id is unknown.\r\n\n"
    user = fmt(commit=make_commit(message=msg))["messages"][1]["content"]
    assert user.startswith(
        "STEP: Add delete endpoint\n\nReturns 404 when the id is unknown.\n\nCONTEXT FILES:\n"
    )


def test_over_the_cap_returns_none_never_truncates():
    row = fmt()
    user = row["messages"][1]["content"]
    tokens = math.ceil(len(user) / 3)
    assert fmt(max_user_tokens=tokens) == row
    assert fmt(max_user_tokens=tokens - 1) is None


def test_default_cap_is_6000_tokens():
    big = {"backend/big.py": "x" * 18_000}  # alone ~6000 tokens, plus the rest
    assert fmt(context=big) is None
    assert fmt(context={"backend/big.py": "x" * 1000}) is not None


def test_unsafe_context_paths_are_left_out():
    user = fmt(context={"../secret.py": "k = 1\n", "backend/app.py": "a = 1\n"})["messages"][1][
        "content"
    ]
    assert "secret" not in user
    assert "=== FILE: backend/app.py ===" in user


def test_rejects_a_commit_the_filters_would_drop():
    deleted = Commit("abc", "Remove the old models file", 1, (FileChange("a.py", "x\n", None, 0, 1),))
    with pytest.raises(ValueError, match="deleted_file"):
        fmt(commit=deleted)


@pytest.mark.parametrize(
    "kw", [{"repo": ""}, {"repo": "noslash"}, {"licence": ""}, {"tag": "django"}, {"tree": "a.py"}]
)
def test_argument_validation(kw):
    with pytest.raises((ValueError, TypeError)):
        fmt(**kw)


def test_round_trip_through_the_harness_loader(tmp_path):
    messages = [
        "Add the item store",
        "Add delete endpoint\n\nReturns 404 when the id is unknown.",
    ]
    rows = [fmt(commit=make_commit(message=m, sha=f"sha{i}")) for i, m in enumerate(messages)]
    path = tmp_path / "train-00000.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    split = rows[0]["split"]
    lib = ExampleLibrary.from_jsonl([path], split=split)
    assert [e.step for e in lib.examples] == messages
    for e in lib.examples:
        assert parse_blocks(e.reply) == [
            FileBlock("backend/models.py", MODELS_AFTER),
            FileBlock("frontend/src/List.jsx", LIST_AFTER),
        ]


def test_tree_entries_a_file_block_rejects_are_left_out():
    # The harness tree lists files only: directories ("backend/") are not paths.
    tree = ["backend/", "../up.py", "C:/abs.py", "a\\b.py", "backend/app.py", ""]
    user = fmt(tree=tree)["messages"][1]["content"]
    assert user.endswith("PROJECT TREE:\nbackend/app.py\n")


def test_total_cap_counts_system_user_and_reply():
    row = fmt()
    total = sum(math.ceil(len(m["content"]) / 3) for m in row["messages"])
    assert fmt(max_total_tokens=total) == row
    assert fmt(max_total_tokens=total - 1) is None


def test_default_total_cap_is_8000_tokens():
    # 200 lines of ~100 chars pass every filter; the reply alone is ~6,700 tokens,
    # and showing the old version as context pushes the total past 8,000.
    body = "".join(f"v{i:03} = '" + "a" * 92 + "'\n" for i in range(199))
    big = Commit(
        "abc", "Add the constants table", 1,
        (FileChange("backend/c.py", body, body + "z = 1\n", 1, 0),),
    )
    assert fmt(commit=big, context={}, max_user_tokens=10**6) is not None
    assert fmt(commit=big, context={"backend/c.py": body}, max_user_tokens=10**6) is None


def test_filter_limits_pass_through():
    with pytest.raises(ValueError, match="too_many_files"):
        fmt(max_files=1)
    with pytest.raises(ValueError, match="too_many_lines"):
        fmt(max_lines=4)
    with pytest.raises(ValueError, match="too_long_file"):
        fmt(max_file_lines=2)
    wide = Commit(
        "abc", "Add the item store", 1,
        tuple(FileChange(f"backend/m{i}.py", None, f"x = {i}\n", 1, 0) for i in range(4)),
    )
    with pytest.raises(ValueError, match="too_many_files"):
        fmt(commit=wide)
    assert fmt(commit=wide, max_files=4) is not None


@pytest.mark.parametrize(
    "path, text",
    [
        (".env", "DEBUG=1\n"),
        ("backend/.env.local", "DEBUG=1\n"),
        (".envrc", "export X=1\n"),
        ("backend/config.py", "API_KEY = 'AKIAIOSFODNN7EXAMPLE'\n"),
    ],
)
def test_env_files_and_secret_context_are_left_out(path, text):
    user = fmt(context={path: text, "backend/app.py": "a = 1\n"})["messages"][1]["content"]
    assert f"=== FILE: {path} ===" not in user
    assert "=== FILE: backend/app.py ===" in user


def test_a_name_containing_env_later_is_shown():
    user = fmt(context={"backend/settings.env.py": "X = 1\n"})["messages"][1]["content"]
    assert "=== FILE: backend/settings.env.py ===" in user
