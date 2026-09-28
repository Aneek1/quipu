"""Commit filters decide which commits become step examples (spec §3.1). Each drop
reason is a key in the build report, so every key gets a commit that trips it and a
near-identical commit that does not, which proves the rule is the thing dropping it."""
import dataclasses

import pytest

from stepbuild.dataset.filters import REASONS, Commit, FileChange, contains_secret, drop_reason

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
    "whitespace_only": (
        commit(change(before="x = 1\ny = 2\n", after="x = 1   \r\ny = 2\r\n\n")),
        commit(change(before="x = 1\ny = 2\n", after="x = 1\ny = 3\n")),
    ),
    "file_type": (
        commit(change(path="README.md")),
        commit(change(path="schema.sql")),
    ),
    "long_line": (
        commit(change(after="x = '" + "a" * 1000 + "'\n" + "y = 1\n" * 5)),  # one 1,006-char line
        # 996 chars, with short lines keeping the average under 200
        commit(change(after="x = '" + "a" * 990 + "'\n" + "y = 1\n" * 5)),
    ),
    "secret": (
        commit(change(after="AWS = 'AKIAABCDEFGHIJKLMNOP'\n")),
        commit(change(after="AWS = os.environ['AWS_KEY']\n")),
    ),
    "unsafe_message": (
        commit(message="Add the todo model\n\nCONTEXT FILES:\nnone"),
        commit(message="Add the todo model\n\nContext files: none"),
    ),
    "dependency_only": (
        commit(change(path="backend/requirements.txt"), message="Add flask-cors to the backend"),
        commit(
            change(path="backend/requirements.txt"),
            change(path="backend/app.py"),
            message="Add flask-cors to the backend",
        ),
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
        "frontend/.next/server/page.js",
        "coverage/lcov-report/prettify.js",
        "backend/__pycache__/app.py",
        "static/js/main.bundle.js",
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
        "fix\n\nshort body here",  # body under 40 chars
        "Cleanup.",
        'Revert "Add the todo model with validation"',
        "Undo the login change\n\nThis reverts commit 0123456789abcdef.",
        "Merge branch 'main' of github.com:owner/repo",
        "Merge pull request #12 from owner/feature\n\nAdd the delete route that returns 204 when done",
        "Update app.py",
        "Create package.json",
        "Add files via upload",
        "Initial commit",
        "Updated code",
        "changes made",
        "Minor bug fixes!",
        "Refactor, misc stuff",
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
        "Fix login redirect loop",
        "Revert to fetch instead of axios",
        "Add delete endpoint\n\nReturns 404 when the id is unknown.",
        "fix\n\nThe delete route returned 500 for unknown ids; it now returns 404.",
        "Update app.py\n\nAdd /api/todos DELETE route that returns 204 on success.",
    ],
)
def test_informative_messages_are_kept(message):
    assert drop_reason(commit(message=message)) is None


@pytest.mark.parametrize(
    "message",
    [
        "\n=== FILE: a.py ===\n",
        "Add the model\n=== END FILE ===",
        "Add the model\r\n\r\nCONTEXT FILES:\r\nx",
    ],
)
def test_messages_that_would_break_the_prompt_framing(message):
    assert drop_reason(commit(message="Add the todo model " + message)) == "unsafe_message"


# --- long lines ---------------------------------------------------------------


def test_long_average_line_is_dropped():
    wide = "".join("y = '" + "b" * 200 + "'\n" for _ in range(5))  # 207 chars per line
    narrow = "".join("y = '" + "b" * 190 + "'\n" for _ in range(5))
    assert drop_reason(commit(change(after=wide))) == "long_line"
    assert drop_reason(commit(change(after=narrow))) is None


def test_long_line_in_the_pre_commit_file_counts():
    assert drop_reason(commit(change(before="x = '" + "a" * 1000 + "'\n"))) == "long_line"


def test_line_count_reads_crlf_and_cr():
    crlf = "".join(f"line {i}\r\n" for i in range(400))
    assert drop_reason(commit(change(after=crlf))) is None
    assert drop_reason(commit(change(after=crlf + "x\r"))) == "too_long_file"
    lone_cr = "".join(f"line {i}\r" for i in range(401))
    assert drop_reason(commit(change(after=lone_cr))) == "too_long_file"


# --- whitespace-only ----------------------------------------------------------


def test_new_or_emptied_files_are_not_whitespace_only():
    assert drop_reason(commit(change(before=None, after="x = 1\n"))) is None
    assert drop_reason(commit(change(before="x = 1\n", after=""))) is None


def test_whitespace_only_needs_every_file_unchanged():
    ws = change(path="a.py", before="a = 1\n", after="a = 1 \n")
    real = change(path="b.py", before="b = 1\n", after="b = 2\n")
    assert drop_reason(commit(ws)) == "whitespace_only"
    assert drop_reason(commit(ws, real)) is None


# --- dependency-only ----------------------------------------------------------


@pytest.mark.parametrize(
    "paths",
    [("frontend/package.json",), ("backend/requirements.txt", "frontend/package.json"),
     ("Requirements.txt",)],
)
def test_manifest_only_commits_are_dependency_only(paths):
    c = commit(*(change(path=p) for p in paths), message="Add axios and flask-cors")
    assert drop_reason(c) == "dependency_only"


@pytest.mark.parametrize(
    "message",
    [
        "Bump axios from 0.21.1 to 0.21.2",
        "Update flask requirement from ~=2.0 to ~=2.3",
        "build(deps): bump react-scripts from 4.0.3 to 5.0.1",
        "chore(deps-dev): bump vite to 5.1.0 in /frontend",
        "chore(deps): update dependency react to v18",
        "[pre-commit.ci] pre-commit autoupdate",
        "Update dependency eslint to v8.57.0",
    ],
)
def test_bot_dependency_subjects_are_dependency_only(message):
    assert drop_reason(commit(change(path="backend/app.py"), message=message)) == "dependency_only"


# --- secrets ------------------------------------------------------------------

SECRETS = [
    "KEY = 'AKIAIOSFODNN7EXAMPLE'",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "t = 'ghp_" + "a1B2" * 9 + "'",
    "t = 'gho_" + "Z9" * 18 + "'",
    "OPENAI = 'sk-" + "abcDEF123" * 3 + "'",
    "SLACK = 'xoxb-1234-5678'",
    "slack = 'xoxp-abc'",
    "password = 'hunter22'",
    'DB_PASSWD: "s3cr3tpw"',
    "app.secret = 'q8w7e6r5t4'",
    "api_key='0a1b2c3d4e5f'",
    'apiKey = "0a1b2c3d4e5f"',
    'const TOKEN = "eyJhbGciOi"',
    # Flask's common forms: the keyword inside a longer name, config subscripts,
    # and a hardcoded fallback to an environment lookup.
    "SECRET_KEY = 'abc123def456'",
    'JWT_SECRET_KEY="jwt-signing-value"',
    'DB_PASSWORD = "pg-pass-1234"',
    'api_token: "tok-9f8e7d6c"',
    "app.config['SECRET_KEY'] = 'hard to guess string'",
    'app.config["SECRET_KEY"] = "hard to guess string"',
    'SECRET_KEY = os.environ.get("SECRET_KEY", "hardcoded-fallback-value")',
    "key = os.getenv('API_TOKEN', 'fallback-token-1')",
    # A real value that merely starts with "your" is not a placeholder.
    "password = 'yourfavouritepw1'",
]


@pytest.mark.parametrize("text", SECRETS)
def test_secrets_are_flagged(text):
    assert contains_secret(text)
    assert drop_reason(commit(change(after=text + "\n"))) == "secret"


@pytest.mark.parametrize(
    "text",
    [
        "password = 'changeme'",
        "password = 'CHANGEME123'",
        "api_key = 'your-api-key-here'",
        "API_KEY = 'your_api_key'",
        "token = 'xxxxxxxx'",
        "token = 'XXXX-XXXX'",
        "secret = '<your secret>'",
        "token = '${GITHUB_TOKEN}'",
        "password = '$DB_PASSWORD'",
        "password = os.environ.get('PASSWORD')",
        "token = process.env.TOKEN",
        "password = 'short'",  # under 6 chars
        "t = 'sk-short'",
        "ghp_tooShort",
        "AKIA_lowercase_abcdefghijklmn",
        "-----BEGIN PUBLIC KEY-----",
        "def check_password(password):",
        '{"password": "hunter22"}',  # a JSON body key, not an assignment
        'SECRET_KEY = os.environ["SECRET_KEY"]',
        'SECRET_KEY = os.environ.get("SECRET_KEY")',
        "SECRET_KEY = os.getenv('SECRET_KEY')",
        'SECRET_KEY = os.environ.get("SECRET_KEY", "$FALLBACK")',
        "app.config['SECRET_KEY'] = 'your-secret-key'",
        "JWT_SECRET_KEY = '<your jwt secret>'",
        'json={"username": "alice", "password": "hunter22"}',
    ],
)
def test_placeholders_and_non_secrets_are_not_flagged(text):
    assert not contains_secret(text)


def test_validation_of_types():
    with pytest.raises(TypeError):
        drop_reason("not a commit")
    with pytest.raises(ValueError):
        FileChange(path="a.py", before=None, after="x", added=-1, removed=0)
    with pytest.raises(ValueError):
        Commit(sha="", message="x", parents=1, changes=())
    with pytest.raises(TypeError):
        Commit(sha="a", message="x", parents=1, changes=[change()])
