"""Discovery: parse recorded code-search JSON, tag repos, back off on rate limits,
pace live requests and cache every response (no network)."""
import subprocess

import pytest

from stepbuild.dataset import discover as disc
from stepbuild.dataset.discover import Candidate, GhError, discover, gh_json, valid_repo
from tests.stepbuild.fakegh import FakeGh, fixture, http_error, ok, search_handler

FLASK_Q, REACT_Q = "flask filename:requirements.txt", '"react" filename:package.json'


def _pages():
    return {FLASK_Q: {1: fixture("search_flask.json")}, REACT_Q: {1: fixture("search_react.json")}}


def _discover(tmp_path, runner, limit=10, sleep=None, clock=None, max_checks=0, **kw):
    return discover(
        limit, tmp_path / "cache", runner=runner, sleep=sleep or (lambda s: None),
        clock=clock or (lambda: 0.0), flask_queries=(FLASK_Q,), react_queries=(REACT_Q,),
        max_checks=max_checks, **kw,
    )


def test_parses_and_tags_fullstack_first(tmp_path):
    runner = FakeGh(search_handler(_pages()))
    got = _discover(tmp_path, runner)
    # beta is in both searches; alpha (listed twice) flask only; gamma react only;
    # the fork is skipped.
    assert got == [
        Candidate("acme/beta", "fullstack"),
        Candidate("acme/alpha", "flask"),
        Candidate("acme/gamma", "react"),
    ]
    assert all(c.stars is None for c in got)
    q = runner.calls[0]
    assert q[:3] == ["-X", "GET", "search/code"] and f"q={FLASK_Q}" in q and "per_page=100" in q


def test_limit_prefers_fullstack(tmp_path):
    got = _discover(tmp_path, FakeGh(search_handler(_pages())), limit=1)
    assert got == [Candidate("acme/beta", "fullstack")]


def test_retries_a_429_then_succeeds(tmp_path):
    delays = []
    runner = FakeGh(search_handler(_pages()),
                    script=[http_error(429, "API rate limit exceeded"),
                            http_error(403, "You have exceeded a secondary rate limit")])
    got = _discover(tmp_path, runner, sleep=delays.append)
    assert [c.repo for c in got] == ["acme/beta", "acme/alpha", "acme/gamma"]
    assert delays[:2] == [disc.BASE_BACKOFF, disc.BASE_BACKOFF * 2]
    assert len(runner.calls) == 4  # two failures, then both queries


def test_backoff_is_capped_and_gives_up_after_max_tries():
    delays = []
    runner = FakeGh(lambda a: http_error(429, "too many"))
    with pytest.raises(GhError, match="gave up after 5 tries"):
        gh_json(["search/code"], runner=runner, sleep=delays.append)
    assert len(runner.calls) == disc.MAX_TRIES
    assert len(delays) == disc.MAX_TRIES - 1
    assert max(delays) <= disc.MAX_BACKOFF and delays == sorted(delays)


def test_other_http_errors_are_not_retried():
    runner = FakeGh(lambda a: http_error(401, "Bad credentials"))
    with pytest.raises(GhError) as e:
        gh_json(["search/code"], runner=runner, sleep=lambda s: None)
    assert e.value.status == 401 and len(runner.calls) == 1


def test_timeout_is_retried():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return ok({"items": []})

    assert gh_json(["search/code"], runner=runner, sleep=lambda s: None) == {"items": []}


def test_second_call_uses_the_cache(tmp_path):
    first = _discover(tmp_path, FakeGh(search_handler(_pages())))

    def no_network(cmd, **kw):
        raise AssertionError("cache miss: a live request was made")

    assert _discover(tmp_path, no_network) == first
    assert list((tmp_path / "cache" / "search").glob("*.json"))


def test_live_requests_are_paced(tmp_path):
    now = [100.0]
    delays = []

    def sleep(s):
        delays.append(s)
        now[0] += s

    _discover(tmp_path, FakeGh(search_handler(_pages())), sleep=sleep, clock=lambda: now[0])
    assert delays == [pytest.approx(disc.SEARCH_INTERVAL)]  # two live requests, one gap


def test_pages_until_short_page_and_stops_at_the_cap(tmp_path):
    full = {"total_count": 5000, "incomplete_results": False,
            "items": [{"repository": {"full_name": f"o/r{i}", "fork": False}}
                      for i in range(disc.PER_PAGE)]}
    pages = {FLASK_Q: {1: full, 2: fixture("search_flask.json")}, REACT_Q: {}}
    runner = FakeGh(search_handler(pages))
    got = _discover(tmp_path, runner, limit=1000)
    flask_pages = sorted(int(a[5:]) for c in runner.calls for a in c
                         if a.startswith("page=") and f"q={FLASK_Q}" in c)
    assert flask_pages == [1, 2]  # page 2 is short: no page 3
    assert len(got) == disc.PER_PAGE + 2  # o/r0..99, alpha, beta (react page 1 is 422)


def test_valid_repo():
    assert valid_repo("acme/alpha") and valid_repo("a.b/c-d_e")
    for bad in ("acme", "a/b/c", "../x", "a/..", "a b/c", None, "", "/x"):
        assert not valid_repo(bad), bad


# ------------------------------------------------------------ full-stack check

def _b64(text):
    import base64
    return {"type": "file", "encoding": "base64",
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}


def _check_handler(pages, trees, files):
    """Search pages as before; trees[repo] and files[(repo, path)] for the check;
    anything else is a 404."""
    search = search_handler(pages)

    def handler(args):
        if args[:3] == ["-X", "GET", "search/code"]:
            return search(args)
        path = args[0]
        for repo, tree in trees.items():
            if path == f"repos/{repo}/git/trees/HEAD?recursive=1":
                return ok(tree)
        for (repo, fpath), body in files.items():
            if path == f"repos/{repo}/contents/{fpath}":
                return ok(body)
        return http_error(404, "Not Found")
    return handler


def test_intersection_ignores_case(tmp_path):
    flask = {"total_count": 1, "items": [{"repository": {"full_name": "Acme/Beta"}}]}
    react = {"total_count": 1, "items": [{"repository": {"full_name": "acme/beta"}}]}
    got = _discover(tmp_path, FakeGh(search_handler({FLASK_Q: {1: flask}, REACT_Q: {1: react}})))
    assert got == [Candidate("Acme/Beta", "fullstack")]


def test_tree_check_confirms_fullstack_repos(tmp_path):
    # Recorded from a real repo: backend/requirements.txt plus a React
    # frontend/package.json that the react search did not return.
    tree = fixture("tree_flask_react.json")
    trees = {"acme/alpha": tree, "acme/gamma": {
        "sha": "x", "truncated": False,
        "tree": [{"path": "package.json", "type": "blob"},
                 {"path": "api/requirements.txt", "type": "blob"},
                 {"path": "node_modules/x/requirements.txt", "type": "blob"}]}}
    files = {
        ("acme/alpha", "frontend/package.json"): fixture("contents_package_json_react.json"),
        ("acme/gamma", "api/requirements.txt"): _b64("gunicorn==22\nFlask==3.0.3\n"),
    }
    runner = FakeGh(_check_handler(_pages(), trees, files))
    got = _discover(tmp_path, runner, max_checks=10)
    # beta: both searches. alpha (flask) and gamma (react): confirmed by the check.
    assert got == [Candidate("acme/beta", "fullstack"), Candidate("acme/alpha", "fullstack"),
                   Candidate("acme/gamma", "fullstack")]
    assert not any("node_modules" in c[0] for c in runner.calls)

    def no_network(cmd, **kw):
        raise AssertionError("cache miss: a live request was made")

    assert _discover(tmp_path, no_network, max_checks=10) == got  # checks are cached too


def test_tree_check_leaves_single_side_repos_alone(tmp_path):
    trees = {"acme/alpha": {"tree": [{"path": "package.json", "type": "blob"}]}}
    files = {("acme/alpha", "package.json"): _b64('{"dependencies": {"vue": "^3"}}')}
    # gamma's tree is a 404 (deleted repo): it just stays react.
    got = _discover(tmp_path, FakeGh(_check_handler(_pages(), trees, files)), max_checks=10)
    assert [(c.repo, c.tag) for c in got] == [
        ("acme/beta", "fullstack"), ("acme/alpha", "flask"), ("acme/gamma", "react")]


def test_tree_check_respects_max_checks_and_survives_api_errors(tmp_path):
    runner = FakeGh(lambda a: http_error(500, "Server Error")
                    if "trees" in a[0] else search_handler(_pages())(a))
    got = _discover(tmp_path, runner, max_checks=1)
    assert [c.tag for c in got] == ["fullstack", "flask", "react"]
    assert sum("trees" in c[0] for c in runner.calls) == 1


def test_manifest_readers():
    assert disc.react_in_package_json('{"devDependencies": {"react": "18"}}')
    assert disc.react_in_package_json('{"peerDependencies": {"react": "*"}}')
    assert not disc.react_in_package_json('{"dependencies": {"react-icons": "1"}}')
    assert not disc.react_in_package_json('{"description": "react"}')
    assert not disc.react_in_package_json("not json")
    req = disc.flask_in_manifest
    assert req("requirements.txt", "flask\n") and req("requirements.txt", "Flask>=2.0 ; x\n")
    assert req("requirements.txt", "a==1\n  flask[async]==3\n")
    assert not req("requirements.txt", "flask-cors==4\n# flask\nflask_sqlalchemy\n")
    assert req("pyproject.toml", 'dependencies = ["flask>=3", "x"]\n')
    assert req("pyproject.toml", '[tool.poetry.dependencies]\nFlask = "^3.0"\n')
    assert not req("pyproject.toml", 'dependencies = ["flask-cors"]\n')
