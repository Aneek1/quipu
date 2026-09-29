"""The dataset build end to end: recorded gh responses, tiny local repos standing
in for clones (needs git; no network)."""
import json
import subprocess

import pytest

from stepbuild.bench.acceptance import load_reference
from stepbuild.dataset import build as build_mod
from stepbuild.dataset.build import assemble, build
from stepbuild.dataset.filters import REASONS
from stepbuild.dataset.split import assign_split
from stepbuild.harness.blocks import parse_blocks
from tests.stepbuild.fakegh import FakeGh, fixture, http_error, ok, page_of, query_of
from tests.stepbuild.gitrepo import make_repo

APP_V1 = "from flask import Flask\n\napp = Flask(__name__)\n"
APP_V2 = APP_V1 + "\n\n@app.get('/health')\ndef health():\n    return {'ok': True}\n"
COUNTER = (
    "import { useState } from 'react';\n\nexport default function Counter() {\n"
    "  const [n, setN] = useState(0);\n"
    "  return <button onClick={() => setN(n + 1)}>{n}</button>;\n}\n"
)
GPL_CODE = "def gpl_only():\n    return 'copyleft'\n"
LICENCES = {
    "acme/alpha": "licence_mit.json", "acme/alpha-mirror": "licence_mit.json",
    "acme/beta": "licence_mit.json", "acme/broken": "licence_mit.json",
    "acme/gamma": "licence_gpl.json", "acme/delta": "licence_noassertion.json",
}
FLASK_SIDE = ["acme/alpha", "acme/alpha-mirror", "acme/gamma", "acme/delta", "acme/broken"]
REACT_SIDE = ["acme/alpha", "acme/beta"]

pytestmark = pytest.mark.git


def _page(repos):
    return {"total_count": len(repos), "incomplete_results": False,
            "items": [{"path": "x", "repository": {"full_name": r, "fork": False}} for r in repos]}


def _handler(args):
    if args[:3] == ["-X", "GET", "search/code"]:
        if page_of(args) > 1:
            return http_error(422, "Cannot access beyond the first 1000 results")
        return ok(_page(REACT_SIDE if "package.json" in query_of(args) else FLASK_SIDE))
    if "/git/trees/" in args[0]:  # the full-stack check: no trees recorded here
        return http_error(404, "Not Found")
    repo = args[0].removeprefix("repos/").removesuffix("/license")
    return ok(fixture(LICENCES[repo]))


def _clone_runner(cmd, **kw):
    # Every repo but "broken" is already in out/repos, so only broken gets here.
    assert "acme/broken" in cmd[4]
    return subprocess.CompletedProcess(cmd, 128, "", "fatal: repository not found\n")


def _leaked_models():
    (block,) = [b for b in parse_blocks(load_reference("todo")[0]) if b.path == "backend/models.py"]
    return block.content


def _alpha_history():
    return [
        ("Initial project skeleton", {"backend/app.py": APP_V1}),
        ("Add a health endpoint returning ok", {"backend/app.py": APP_V2}),
        ("Add the todo store with its validation", {"backend/models.py": _leaked_models()}),
        ("wip", {"backend/app.py": APP_V2 + "# todo\n"}),
        ("Move the app module to server.py", {"backend/app.py": None,
                                              "backend/server.py": APP_V2 + "# todo\n"}),
    ]


@pytest.fixture
def world(tmp_path):
    out = tmp_path / "out"
    _make_world(out / "repos")
    return out


def _make_world(repos):
    make_repo(repos / "acme__alpha", _alpha_history())
    make_repo(repos / "acme__alpha-mirror", _alpha_history())  # same SHAs
    make_repo(repos / "acme__beta", [
        ("Start the api", {"backend/app.py": APP_V1}),
        # The same reply as alpha's health commit, with another message and history.
        ("Health check route for the load balancer", {"backend/app.py": APP_V2}),
        ("Add a React counter component with increment", {"frontend/src/Counter.jsx": COUNTER}),
    ])
    make_repo(repos / "acme__gamma", [
        ("Initial project skeleton", {"a.py": "x = 1\n"}),
        ("Add the copyleft helper function", {"b.py": GPL_CODE}),
    ])
    make_repo(repos / "acme__delta", [
        ("Initial project skeleton", {"a.py": "x = 1\n"}),
        ("Add the unidentified licence helper", {"c.py": "def c():\n    return 3\n"}),
    ])


def _rows(out):
    rows = []
    for split in ("train", "validation", "test"):
        for shard in sorted(out.glob(f"{split}-*.jsonl")):
            for line in shard.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                assert row["split"] == split == assign_split(row["repo"])
                rows.append(row)
    return rows


def test_build_end_to_end(world):
    out = world
    assembled = build(10, out, runner=FakeGh(_handler), sleep=lambda s: None,
                      clone_runner=_clone_runner)

    sources = [json.loads(line) for line in
               (out / "SOURCES.jsonl").read_text(encoding="utf-8").splitlines()]
    names = {s["repo"] for s in sources}
    # GPL and NOASSERTION repos are never mined, so never credited.
    assert "acme/gamma" not in names and "acme/delta" not in names
    assert "acme/beta" in names and names <= {"acme/alpha", "acme/alpha-mirror", "acme/beta"}
    assert all(s["licence"] == "MIT" and len(s["first_sha"]) == 40 for s in sources)
    for split in ("train", "validation", "test"):
        assert (out / f"{split}-000.jsonl").is_file()
    rows = _rows(out)
    replies = [r["messages"][2]["content"] for r in rows]
    # One health-endpoint example survives out of alpha, its mirror and beta, plus
    # beta's counter. The leaked todo models.py, wip, the rename and GPL are gone.
    assert len(rows) == 2
    assert sum("def health" in r for r in replies) == 1
    assert sum("Counter" in r for r in replies) == 1
    assert not any("class Store" in r or "copyleft" in r for r in replies)
    assert assembled["dedupe_sha"] + assembled["dedupe_reply"] == 2

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["repos"]
    assert manifest["acme/broken"]["status"] == "failed"
    assert "repository not found" in manifest["acme/broken"]["error"]
    assert manifest["acme/gamma"] == {"repo": "acme/gamma", "tag": "flask",
                                      "licence": "GPL-3.0", "status": "unlicensed"}
    assert manifest["acme/delta"]["status"] == "unlicensed"
    assert manifest["acme/alpha"]["tag"] == "fullstack"
    assert manifest["acme/alpha"]["leakage_skipped"] == 1

    report = (out / "report.md").read_text(encoding="utf-8")
    for reason in REASONS:
        assert f"| {reason} |" in report
    assert "| root | 3 |" in report and "| deleted_file | 2 |" in report
    assert "| low_info_message | 2 |" in report and "| merge | 0 |" in report
    assert "| total | 6 | 4 | 3 | 1 | 3 |" in report  # found, licensed, mined, failed, with ex.
    assert "skipped by the benchmark leakage guard: 2" in report
    assert "duplicates removed: 2" in report
    assert "ceil(chars / 3)" in report
    assert "acme/broken: MineError" in report


def test_rerun_skips_mined_repos(world, monkeypatch):
    out = world
    build(10, out, runner=FakeGh(_handler), sleep=lambda s: None, clone_runner=_clone_runner)
    first = _rows(out)

    def no_mining(*a, **kw):
        raise AssertionError("a mined repo was mined again")

    monkeypatch.setattr(build_mod, "mine_one", no_mining)
    runner = FakeGh(_handler)
    build(10, out, runner=runner, sleep=lambda s: None, clone_runner=_clone_runner)
    assert _rows(out) == first
    # Search pages and licences all come from the cache: no gh call at all.
    assert runner.calls == []


def _name(split, prefix):
    return next(f"acme/{prefix}{i}" for i in range(10_000)
                if assign_split(f"acme/{prefix}{i}") == split)


def test_dedupe_keeps_the_train_copy(tmp_path):
    train_repo, test_repo = _name("train", "z"), _name("test", "a")  # the test repo sorts first
    mined = tmp_path / "mined"
    mined.mkdir()

    def row(repo, sha, reply):
        return {"repo": repo, "licence": "MIT", "tag": "flask", "commit": sha,
                "split": assign_split(repo),
                "messages": [{"role": "system", "content": "s"},
                             {"role": "user", "content": "STEP: x\n\nCONTEXT FILES:\n"},
                             {"role": "assistant", "content": reply}]}

    rows = {
        # Same SHA as a train example; and the same reply as another once CRLF and
        # trailing whitespace are normalised.
        test_repo: [row(test_repo, "a" * 40, "=== FILE: a.py ===\nx = 1\n=== END FILE ===\n"),
                    row(test_repo, "c" * 40,
                        "=== FILE: b.py ===\r\ny = 2   \r\n=== END FILE ===\r\n")],
        train_repo: [row(train_repo, "a" * 40, "=== FILE: z.py ===\nz\n=== END FILE ===\n"),
                     row(train_repo, "b" * 40, "=== FILE: b.py ===\ny = 2\n=== END FILE ===\n")],
    }
    entries = []
    for repo, rs in rows.items():
        (mined / f"{repo.replace('/', '__')}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
        entries.append({"repo": repo, "tag": "flask", "licence": "MIT",
                        "first_sha": "f" * 40, "last_sha": "e" * 40})
    got = assemble(tmp_path, entries, mined)
    kept = _rows(tmp_path)
    assert [r["repo"] for r in kept] == [train_repo, train_repo]
    assert got["dedupe_sha"] == 1 and got["dedupe_reply"] == 1
    assert got["examples"] == {"train": 2, "validation": 0, "test": 0}
    sources = (tmp_path / "SOURCES.jsonl").read_text(encoding="utf-8")
    assert train_repo in sources and test_repo not in sources


# ------------------------------------------------------------ the two phases

def _no_clone(cmd, **kw):
    raise AssertionError(f"the discover phase cloned: {cmd}")


def _no_gh(args):
    raise AssertionError(f"the mine phase called gh: {args}")


def test_discover_phase_writes_candidates_and_clones_nothing(tmp_path):
    out = tmp_path / "laptop"
    got = build(10, out, runner=FakeGh(_handler), sleep=lambda s: None,
                clone_runner=_no_clone, phase="discover")
    data = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["limit"] == 10
    recs = {r["repo"]: r for r in data["candidates"]}
    assert set(recs) == set(LICENCES)
    assert recs["acme/alpha"] == {
        "repo": "acme/alpha", "tag": "fullstack", "licence": "MIT",
        "url": "https://github.com/acme/alpha", "stars": None,
        "clone_url": "https://github.com/acme/alpha.git",
    }
    assert recs["acme/gamma"]["licence"] == "GPL-3.0"
    assert recs["acme/delta"]["licence"] == "NOASSERTION"
    assert got == {"candidates": 6, "licensed": 4, "path": out / "candidates.json"}
    # Nothing past discovery: no clones, no manifest, no shards, no report.
    assert sorted(p.name for p in out.iterdir()) == ["cache", "candidates.json"]


def test_mine_phase_matches_the_combined_build_without_gh(world, tmp_path):
    combined = world
    build(10, combined, runner=FakeGh(_handler), sleep=lambda s: None,
          clone_runner=_clone_runner)

    laptop = tmp_path / "laptop"
    build(10, laptop, runner=FakeGh(_handler), sleep=lambda s: None,
          clone_runner=_no_clone, phase="discover")
    cands = (laptop / "candidates.json").read_text(encoding="utf-8")
    assert cands == (combined / "candidates.json").read_text(encoding="utf-8")

    # On the "remote box" the public repos are cloned for real, from local file://
    # mirrors standing in for github.com (same commits, so the same SHAs).
    src = tmp_path / "github"
    _make_world(src)
    data = json.loads(cands)
    for rec in data["candidates"]:
        mirror = src / rec["repo"].replace("/", "__")
        if mirror.is_dir():
            rec["clone_url"] = mirror.as_uri()
    moved = tmp_path / "moved" / "candidates.json"
    moved.parent.mkdir()
    moved.write_text(json.dumps(data), encoding="utf-8")

    cloned = []

    def local_clone(cmd, **kw):
        if "acme/broken" in cmd[4]:
            return _clone_runner(cmd, **kw)
        assert cmd[4].startswith("file://")
        cloned.append(cmd[4])
        return subprocess.run(cmd, **kw)

    remote = tmp_path / "remote"
    gh = FakeGh(_no_gh)
    build(None, remote, runner=gh, sleep=lambda s: None, clone_runner=local_clone,
          phase="mine", candidates_path=moved)
    assert gh.calls == []
    assert len(cloned) == 3  # alpha, its mirror and beta: the licensed ones

    def shard_names(out):
        return sorted(p.name for p in out.glob("*.jsonl"))

    assert shard_names(remote) == shard_names(combined)
    for name in shard_names(combined):
        assert (remote / name).read_bytes() == (combined / name).read_bytes(), name
    assert (remote / "manifest.json").read_bytes() == (combined / "manifest.json").read_bytes()

    def report(out):  # everything but the command, the times and the out path
        lines = (out / "report.md").read_text(encoding="utf-8").splitlines()
        return [ln for ln in lines if not ln.startswith(("- command", "- started", "- limit"))]

    assert report(remote) == report(combined)
    assert "- limit 10;" in (remote / "report.md").read_text(encoding="utf-8")


def test_mine_phase_records_a_failed_licence_lookup(tmp_path):
    cands = tmp_path / "candidates.json"
    cands.write_text(json.dumps({"version": 1, "limit": 2, "candidates": [
        {"repo": "acme/x", "tag": "flask", "licence": None, "url": "https://github.com/acme/x",
         "stars": None, "clone_url": "https://github.com/acme/x.git",
         "licence_error": "GhError: gh api repos/acme/x/license: boom"},
        {"repo": "acme/y", "tag": "react", "licence": "GPL-3.0",
         "url": "https://github.com/acme/y", "stars": None,
         "clone_url": "https://github.com/acme/y.git"},
    ]}), encoding="utf-8")
    out = tmp_path / "out"
    gh = FakeGh(_no_gh)
    build(None, out, runner=gh, clone_runner=_no_clone, guard=object(), phase="mine",
          candidates_path=cands)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["repos"]
    assert manifest["acme/x"] == {"repo": "acme/x", "tag": "flask", "status": "failed",
                                  "error": "GhError: gh api repos/acme/x/license: boom"}
    assert manifest["acme/y"]["status"] == "unlicensed"
    assert gh.calls == []
    assert "- acme/x: GhError" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("url", [
    "git@github.com:acme/x.git", "http://github.com/acme/x.git",
    "https://user:token@github.com/acme/x.git", "ssh://github.com/acme/x.git", None,
])
def test_read_candidates_refuses_other_clone_urls(tmp_path, url):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"version": 1, "limit": 1, "candidates": [
        {"repo": "acme/x", "tag": "flask", "licence": "MIT", "clone_url": url}]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="clone_url"):
        build_mod.read_candidates(path)


def test_cli_phase_arguments(tmp_path):
    with pytest.raises(SystemExit):
        build_mod.main(["--out", str(tmp_path)])  # --limit needed without --phase mine
    with pytest.raises(SystemExit):
        build_mod.main(["--limit", "3", "--candidates", str(tmp_path / "c.json")])
