"""The per-repo cap: applied after dedupe, deterministic, spread over history."""
import json

import pytest

from stepbuild.dataset import build as build_mod
from stepbuild.dataset.build import MAX_PER_REPO, assemble, cap_indices, render_report
from stepbuild.dataset.discover import Candidate
from stepbuild.dataset.split import assign_split


def _repo(split, prefix):
    return next(f"acme/{prefix}{i}" for i in range(10_000)
                if assign_split(f"acme/{prefix}{i}") == split)


BIG, SMALL = _repo("train", "big"), _repo("train", "small")


def _row(repo, n):
    return {"repo": repo, "licence": "MIT", "tag": "flask", "commit": f"{repo}-{n:04d}",
            "split": assign_split(repo),
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "STEP: x\n\nCONTEXT FILES:\n"},
                         {"role": "assistant",
                          "content": f"=== FILE: a.py ===\n# {repo} {n}\n=== END FILE ===\n"}]}


def _world(tmp_path, counts):
    mined = tmp_path / "mined"
    mined.mkdir(parents=True)
    entries = []
    for repo, k in counts.items():
        (mined / f"{repo.replace('/', '__')}.jsonl").write_text(
            "".join(json.dumps(_row(repo, n)) + "\n" for n in range(k)), encoding="utf-8")
        entries.append({"repo": repo, "tag": "flask", "licence": "MIT",
                        "first_sha": "f" * 40, "last_sha": "e" * 40})
    return mined, entries


def _kept(out, repo):
    rows = [json.loads(line) for p in sorted(out.glob("train-*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines()]
    return [int(r["commit"].rsplit("-", 1)[1]) for r in rows if r["repo"] == repo]


def test_cap_indices_rule():
    assert cap_indices(120, 50) == sorted({round(i * 120 / 50) for i in range(50)})
    assert len(cap_indices(120, 50)) == 50
    assert len(cap_indices(51, 50)) == 50 and cap_indices(51, 50)[-1] == 50
    assert cap_indices(30, 50) == list(range(30))
    assert cap_indices(120, 0) == list(range(120))
    for k in range(51, 400):
        idx = cap_indices(k, 50)
        assert len(idx) == 50 and idx[0] == 0 and idx[-1] < k


def test_big_repo_keeps_50_spread_and_small_repo_keeps_all(tmp_path):
    assert MAX_PER_REPO == 50
    runs = []
    for run in ("a", "b"):
        out = tmp_path / run
        mined, entries = _world(out, {BIG: 120, SMALL: 30})
        got = assemble(out, entries, mined, max_per_repo=MAX_PER_REPO)
        runs.append((_kept(out, BIG), _kept(out, SMALL), got))
    (big, small, got), (big2, small2, _) = runs
    assert big == big2 and small == small2  # same input, same selection
    assert len(big) == 50 and big == sorted(big)
    assert big[0] == 0 and big[-1] >= 117  # from the first example to the end of history
    assert max(b - a for a, b in zip(big, big[1:])) <= 3  # evenly spread, no gaps
    assert small == list(range(30))
    assert got["capped"] == 70
    assert got["per_repo"] == {BIG: (120, 50), SMALL: (30, 30)}
    assert got["examples"]["train"] == 80
    sources = {json.loads(line)["repo"]: json.loads(line)["examples"] for line in
               (tmp_path / "b" / "SOURCES.jsonl").read_text(encoding="utf-8").splitlines()}
    assert sources == {BIG: 50, SMALL: 30}


def test_zero_means_no_cap(tmp_path):
    mined, entries = _world(tmp_path, {BIG: 120})
    got = assemble(tmp_path, entries, mined, max_per_repo=0)
    assert _kept(tmp_path, BIG) == list(range(120))
    assert got["capped"] == 0


def test_cap_runs_after_dedupe(tmp_path):
    mined, entries = _world(tmp_path, {BIG: 60})
    path = mined / f"{BIG.replace('/', '__')}.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows + rows[:15]) + "\n", encoding="utf-8")  # 15 repeated SHAs
    got = assemble(tmp_path, entries, mined, max_per_repo=50)
    assert got["dedupe_sha"] == 15
    assert got["per_repo"][BIG] == (60, 50) and got["capped"] == 10


def test_report_shows_the_cap(tmp_path):
    mined, entries = _world(tmp_path, {BIG: 120, SMALL: 30})
    got = assemble(tmp_path, entries, mined, max_per_repo=50)
    manifest = {"repos": {e["repo"]: {**e, "status": "mined", "examples": 0} for e in entries}}
    meta = {"command": "x", "started": "s", "finished": "f", "elapsed_s": 1.0,
            "limit": 2, "out": "out"}
    report = render_report([Candidate(BIG, "flask"), Candidate(SMALL, "flask")],
                           manifest, got, meta)
    assert "- per-repo cap: 50 examples per repo" in report
    assert "- examples removed by the per-repo cap: 70" in report
    assert f"| {BIG} | 120 | 50 |" in report and f"| {SMALL} | 30 | 30 |" in report
    assert report.index(f"| {BIG} |") < report.index(f"| {SMALL} |")  # largest first
    got0 = assemble(tmp_path, entries, mined, max_per_repo=0)
    assert "- per-repo cap: 0 (no cap)" in render_report([], {"repos": {}}, got0, meta)


def test_cli_max_per_repo(tmp_path, monkeypatch):
    seen = {}

    def fake_build(limit, out, **kw):
        seen.update(kw)
        (out / "report.md").write_text("r", encoding="utf-8")
        return {}

    monkeypatch.setattr(build_mod, "build", fake_build)
    build_mod.main(["--limit", "1", "--out", str(tmp_path)])
    assert seen["max_per_repo"] == 50
    build_mod.main(["--limit", "1", "--out", str(tmp_path), "--max-per-repo", "0"])
    assert seen["max_per_repo"] == 0
    with pytest.raises(SystemExit):
        build_mod.main(["--limit", "1", "--out", str(tmp_path), "--max-per-repo", "-1"])
