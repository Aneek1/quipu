"""Commit filters decide which commits become step examples (spec §3.1). Each drop
reason is a key in the build report, so every key gets a commit that trips it and a
near-identical commit that does not, which proves the rule is the thing dropping it."""
import dataclasses

import pytest

from stepbuild.dataset.filters import REASONS, Commit, FileChange, drop_reason

GOOD_MESSAGE = "Add the todo model with validation"


def change(path="backend/models.py", before="x = 1\n", after="x = 2\n", added=1, removed=1):
    return FileChange(path=path, before=before, after=after, added=added, removed=removed)


def commit(*changes, message=GOOD_MESSAGE, parents=1, sha="abc123"):
    return Commit(sha=sha, message=message, parents=parents, changes=changes or (change(),))


def test_a_plain_commit_is_kept():
    assert drop_reason(commit()) is None


def test_dataclasses_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        change().path = "y.py"
    with pytest.raises(dataclasses.FrozenInstanceError):
        commit().sha = "x"


# (reason, commit that is dropped for it, commit that is kept) -- the pairs differ
# only in the property the rule is about.
LONG = "".join(f"line {i}\n" for i in range(401))
AT_LIMIT = "".join(f"line {i}\n" for i in range(400))

CASES = {
    "merge": (commit(parents=2), commit(parents=1)),
    "root": (commit(parents=0), commit(parents=1)),
    "no_change": (
        commit(change(added=0, removed=0, after="x = 1\n")),
        commit(change(added=1, removed=0)),
    ),
    "too_many_files": (
        commit(*(change(path=f"backend/m{i}.py") for i in range(4))),
        commit(*(change(path=f"backend/m{i}.py") for i in range(3))),
    ),
    "unsafe_path": (
        commit(change(path="../escape.py")),
        commit(change(path="escape.py")),
    ),
    "excluded_path": (
        commit(change(path="frontend/node_modules/react/index.js")),
        commit(change(path="frontend/src/index.js")),
    ),
    "file_type": (
        commit(change(path="README.md")),
        commit(change(path="requirements.txt")),
    ),
    "deleted_file": (
        commit(change(after=None, added=0, removed=1)),
        commit(change(after="", added=0, removed=1)),
    ),
    "too_many_lines": (
        commit(change(added=150, removed=51)),
        commit(change(added=150, removed=50)),
    ),
    "too_long_file": (
        commit(change(after=LONG)),
        commit(change(after=AT_LIMIT)),
    ),
    "marker_in_content": (
        commit(change(after="x = 1\n=== END FILE ===\n")),
        commit(change(after="x = 1  # === END FILE ===\n")),
    ),
    "low_info_message": (
        commit(message="  fix  "),
        commit(message="Fix the empty-title crash"),
    ),
}


def test_every_reason_has_a_case():
    assert set(CASES) == set(REASONS)


@pytest.mark.parametrize("reason", sorted(CASES))
def test_drop_and_keep_pair(reason):
    dropped, kept = CASES[reason]
    assert drop_reason(dropped) == reason
    assert drop_reason(kept) is None


@pytest.mark.parametrize(
    "path",
    [
        "dist/app.js",
        "frontend/build/static/main.js",
        "vendor/lib.py",
        ".venv/lib/site.py",
        "migrations/versions/001_init.py",
        "frontend/src/app.min.js",
        "static/style.min.css",
        "frontend/src/app.js.map",
        "frontend/package-lock.json",
        "yarn.lock",
        "npm-shrinkwrap.json",
        "Node_Modules/x.js",
    ],
)
def test_excluded_paths(path):
    assert drop_reason(commit(change(path=path))) == "excluded_path"


@pytest.mark.parametrize(
    "path",
    [
        "backend/app.py",
        "src/App.jsx",
        "src/api.ts",
        "src/List.tsx",
        "src/index.css",
        "templates/index.html",
        "schema.sql",
        "backend/requirements.txt",
        "frontend/package.json",
        "frontend/src/builder.js",  # 'build' as part of a name is fine
        "migrations/env.py",  # only migrations/versions/ is generated
    ],
)
def test_wanted_source_files_are_kept(path):
    assert drop_reason(commit(change(path=path))) is None


@pytest.mark.parametrize(
    "path", ["setup.cfg", "Dockerfile", "logo.png", "notes.txt", "requirements-dev.txt"]
)
def test_other_file_types_are_dropped(path):
    assert drop_reason(commit(change(path=path))) == "file_type"


@pytest.mark.parametrize(
    "path", ["C:/x.py", "/abs.py", "a\\b.py", "a//b.py", "./a.py", "CON.py", "a.py.", "a:b.py"]
)
def test_paths_a_file_block_rejects_are_unsafe(path):
    assert drop_reason(commit(change(path=path))) == "unsafe_path"


def test_case_insensitive_duplicate_paths_are_unsafe():
    # One file on NTFS; the FILE-block reply could not carry both.
    assert (
        drop_reason(commit(change(path="src/App.jsx"), change(path="src/app.jsx")))
        == "unsafe_path"
    )


def test_a_long_pre_commit_file_is_also_too_long():
    # It would be shown in full as context.
    assert drop_reason(commit(change(before=LONG))) == "too_long_file"


def test_limits_are_parameters():
    c = commit(change(path="a.py"), change(path="b.py"))
    assert drop_reason(c, max_files=1) == "too_many_files"
    assert drop_reason(commit(change(added=10, removed=0)), max_lines=9) == "too_many_lines"
    assert drop_reason(commit(change(after="a\nb\nc\n")), max_file_lines=2) == "too_long_file"


def test_a_new_file_is_kept():
    assert drop_reason(commit(change(before=None, after="x = 1\n", added=1, removed=0))) is None


@pytest.mark.parametrize(
    "message",
    [
        "WIP",
        "Updates",
        ".",
        "short msg",  # 9 chars
        "12345678901",  # 11 chars
        "Minor\n\n",
        "fix\n\nThis body is long enough but the subject says nothing.",
        "Cleanup.",
        'Revert "Add the todo model with validation"',
        "Revert the login changes from yesterday",
        "Merge branch 'main' of github.com:owner/repo",
        "Merge pull request #12 from owner/feature",
        "Update app.py",
        "Create package.json",
        "Add files via upload",
    ],
)
def test_low_information_messages(message):
    assert drop_reason(commit(message=message)) == "low_info_message"


@pytest.mark.parametrize(
    "message",
    [
        "123456789012",  # exactly 12 chars
        "Update the list view to show due dates",
        "Merge sort the results by date",
        "Fix typo in the delete route",
        "Add delete endpoint\n\nReturns 404 when the id is unknown.",
    ],
)
def test_informative_messages_are_kept(message):
    assert drop_reason(commit(message=message)) is None


def test_validation_of_types():
    with pytest.raises(TypeError):
        drop_reason("not a commit")
    with pytest.raises(ValueError):
        FileChange(path="a.py", before=None, after="x", added=-1, removed=0)
    with pytest.raises(ValueError):
        Commit(sha="", message="x", parents=1, changes=())
    with pytest.raises(TypeError):
        Commit(sha="a", message="x", parents=1, changes=[change()])
