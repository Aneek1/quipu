"""Mining: tiny git repos built in tmp_path (needs git; no network)."""
import subprocess

import pytest

from stepbuild.dataset import mine as mine_mod
from stepbuild.dataset.filters import drop_reason
from stepbuild.dataset.mine import MineError, Miner, clone, mine_repo
from tests.stepbuild.gitrepo import commit, git, init_repo, make_repo

pytestmark = pytest.mark.git

APP_V1 = "from flask import Flask\n\napp = Flask(__name__)\n"
APP_V2 = APP_V1 + "\n\n@app.get('/health')\ndef health():\n    return {'ok': True}\n"


def test_yields_first_parent_history_oldest_first(tmp_path):
    shas = make_repo(tmp_path / "r", [
        ("Initial project skeleton", {"backend/app.py": APP_V1}),
        ("Add a health endpoint returning ok", {"backend/app.py": APP_V2,
                                                "logo.png": b"\x89PNG\x00\x01\x02",
                                                "docs/read me.py": "x = 1\n"}),
        ("Move the app module to server.py", {"backend/app.py": None,
                                              "backend/server.py": APP_V2}),
    ])
    commits = list(mine_repo(tmp_path / "r"))
    assert [c.sha for c in commits] == shas
    root, edit, rename = commits
    assert root.parents == 0 and root.changes == ()
    assert drop_reason(root) == "root"

    assert edit.parents == 1 and edit.message.strip() == "Add a health endpoint returning ok"
    # The binary is skipped; a path with a space (added: its "before" is missing,
    # which cat-file reports by echoing the path back) is read fine.
    assert [c.path for c in edit.changes] == ["backend/app.py", "docs/read me.py"]
    assert edit.changes[1].before is None and edit.changes[1].after == "x = 1\n"
    ch = edit.changes[0]
    assert (ch.before, ch.after, ch.added, ch.removed) == (APP_V1, APP_V2, 5, 0)
    assert drop_reason(edit) is None

    # --no-renames: a rename is a delete plus an add, dropped as deleted_file.
    by_path = {c.path: c for c in rename.changes}
    assert set(by_path) == {"backend/app.py", "backend/server.py"}
    assert by_path["backend/app.py"].after is None
    assert by_path["backend/server.py"].before is None
    assert drop_reason(rename) == "deleted_file"


def test_merge_commits_are_first_parent_merges(tmp_path):
    repo = tmp_path / "r"
    make_repo(repo, [("Initial project skeleton", {"a.py": "x = 1\n"})])
    git(repo, "checkout", "-q", "-b", "feature")
    commit(repo, "Add the feature module", {"b.py": "y = 2\n"}, n=1)
    git(repo, "checkout", "-q", "main")
    commit(repo, "Tweak the main module", {"a.py": "x = 2\n"}, n=2)
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge branch 'feature'", "feature", n=3)
    commits = list(mine_repo(repo))
    # First-parent history: the feature commit itself is not walked.
    assert [c.message.strip() for c in commits] == [
        "Initial project skeleton", "Tweak the main module", "Merge branch 'feature'"]
    assert commits[-1].parents == 2 and commits[-1].changes == ()
    assert drop_reason(commits[-1]) == "merge"


def test_huge_commits_are_recorded_without_contents(tmp_path):
    files = {f"src/f{i}.py": f"v = {i}\n" for i in range(mine_mod.MAX_LOAD_FILES + 1)}
    make_repo(tmp_path / "r", [("Initial project skeleton", {"a.py": "x\n"}),
                               ("Add many generated modules", files)])
    c = list(mine_repo(tmp_path / "r"))[1]
    assert len(c.changes) == len(files)
    assert all(ch.before is None and ch.after is None and ch.added == 1 for ch in c.changes)
    assert drop_reason(c) == "too_many_files"


def test_oversized_blob_reads_as_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(mine_mod, "MAX_BLOB_BYTES", 50)
    make_repo(tmp_path / "r", [("Initial project skeleton", {"a.py": "x = 1\n"}),
                               ("Add a big table of values", {"a.py": "x = 1\n" * 30})])
    c = list(mine_repo(tmp_path / "r"))[1]
    assert drop_reason(c) == "too_long_file"


def test_context_and_tree(tmp_path):
    other = "\n".join(f"line_{i} = {i}" for i in range(10)) + "\n"
    make_repo(tmp_path / "r", [
        ("Initial project skeleton", {
            "backend/app.py": APP_V1,
            "backend/database.py": "def get_session():\n    return 'database session'\n",
            "backend/utils.py": other,
            "backend/secrets.py": "API_KEY = 'hunter2hunter2'\n# database session\n",
            ".env.py": "database session\n",
            "node_modules/react/index.js": "database session\n",
            "frontend/src/components/deep/Deep.jsx": "export default 1\n",
            "frontend/src/App.jsx": "export default function App() {}\n",
            "README.md": "# demo\n",
        }),
        ("Use the database session in the health endpoint", {"backend/app.py": APP_V2}),
    ])
    with Miner(tmp_path / "r") as m:
        commits = list(m.commits())
        kept = commits[1]
        ctx = m.context(kept)
        tree = m.tree(kept)
    # The changed file's old contents first, then BM25 picks from the parent tree;
    # secrets, .env files and node_modules are never offered.
    assert list(ctx)[0] == "backend/app.py" and ctx["backend/app.py"] == APP_V1
    assert "backend/database.py" in ctx
    assert not {"backend/secrets.py", ".env.py", "node_modules/react/index.js"} & set(ctx)
    assert len(ctx) <= 1 + mine_mod.CONTEXT_BM25
    # Files only, depth <= 2 directories, harness-visible, sorted.
    assert tree == sorted(tree)
    assert "backend/app.py" in tree and "frontend/src/App.jsx" in tree and "README.md" in tree
    assert "frontend/src/components/deep/Deep.jsx" not in tree
    assert not any(p.startswith("node_modules/") or p.endswith("/") for p in tree)


def test_clone_skips_a_repo_already_present(tmp_path):
    (tmp_path / "acme__alpha").mkdir()

    def runner(cmd, **kw):
        raise AssertionError("must not clone again")

    assert clone("acme/alpha", tmp_path, runner=runner) == tmp_path / "acme__alpha"


def test_clone_failure_leaves_nothing_behind(tmp_path):
    def runner(cmd, **kw):
        assert kw["timeout"] and kw["env"]["GIT_TERMINAL_PROMPT"] == "0"
        dest = cmd[-1]
        (tmp_path / "acme__gone.partial").mkdir()
        assert dest.endswith("acme__gone.partial")
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: repository not found\n")

    with pytest.raises(MineError, match="repository not found"):
        clone("acme/gone", tmp_path, runner=runner)
    assert list(tmp_path.iterdir()) == []


def test_clone_is_a_full_bare_clone_renamed_into_place(tmp_path):
    src = tmp_path / "src"
    make_repo(src, [("Initial project skeleton", {"a.py": "x\n"}),
                    ("Add the second module file", {"b.py": "y\n"})])

    def runner(cmd, **kw):
        assert cmd[:4] == ["git", "clone", "--bare", "--quiet"]
        assert cmd[4] == "https://github.com/acme/alpha.git"
        return subprocess.run(["git", "clone", "--bare", "--quiet", str(src), cmd[5]],
                              capture_output=True, text=True, timeout=kw["timeout"])

    dest = clone("acme/alpha", tmp_path / "repos", runner=runner)
    assert dest.name == "acme__alpha" and not (tmp_path / "repos" / "acme__alpha.partial").exists()
    assert len(list(mine_repo(dest))) == 2


def test_empty_repo_raises_mine_error(tmp_path):
    init_repo(tmp_path / "r")
    with pytest.raises(MineError):
        list(mine_repo(tmp_path / "r"))
