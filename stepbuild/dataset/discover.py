"""Find candidate repos with GitHub code search (spec §3.1, `discover.py`).

Every call goes through the `gh` CLI (`gh api ...`) in a subprocess: gh is
already logged in on this machine and adds its own credentials, so no token is
read, stored or printed by this code. The subprocess runner is injectable, so the
tests replay recorded JSON instead of touching the network.

Groups. Repos whose requirements.txt / pyproject.toml mention flask (FLASK_QUERIES)
and repos whose package.json mentions "react" (REACT_QUERIES) are collected
separately; a repo found by both is `fullstack`, the rest `flask` or `react`
unless the full-stack check below finds the other half. Names are compared
casefolded (GitHub names are case-insensitive).

Confirming full-stack repos. Two independent searches, each capped at 1,000
results out of hundreds of thousands of matches, barely overlap by chance: the
first real run found 1,380 flask and 736 react repos and not one in both, while
a flask hit on backend/requirements.txt (philfung/perplexed) had a React
frontend/package.json that the react search never returned. So single-side
repos are checked for the other half directly: the repo's file tree (`gh api
repos/{repo}/git/trees/HEAD?recursive=1`) is scanned for package.json (for a
flask repo) or requirements.txt / pyproject.toml (for a react repo) at most
MANIFEST_DEPTH directories deep and outside node_modules/, and up to
MAX_MANIFESTS of them are fetched (`gh api repos/{repo}/contents/{path}`) and
read: a package.json with "react" among its dependencies, devDependencies or
peerDependencies; a requirements line or a pyproject dependency naming flask. A
repo that has both is tagged fullstack. Flask and react repos are checked
alternately, repos found by a path-scoped query (path:backend, path:frontend,
...) first, being the likeliest full stacks; at most `max_checks` repos
(default CHECKS_PER_LIMIT x limit), stopping once `limit` full-stack repos are
known. These are core-API calls (5,000 an hour), not code search; they are
cached like the search pages, and a repo whose check fails (deleted, empty, API
error) keeps its single-side tag.

Selection. Full-stack repos come first (spec: preferred): those both searches
returned, then those the check confirmed; then flask-only and react-only
alternately, each in search order, up to `limit`. Searching stops once `limit`
repos are in both searches, once both sides have CHECKS_PER_LIMIT x limit repos
(enough to check), or when every query has run out of pages (a short page, the
1,000-result cap, or HTTP 422 from the API). The path-scoped variants are there
because a full-stack app usually keeps each half in its own directory.

Rate limits and caching. Code search allows about 10 requests a minute, so live
requests are spaced at least SEARCH_INTERVAL seconds apart (cache hits cost
nothing). HTTP 403 (rate limit, secondary rate limit) and 429 are retried with
exponential backoff capped at MAX_BACKOFF seconds, at most MAX_TRIES attempts in
all; so are failures with no HTTP status (network trouble). Any other HTTP error
raises GhError at once. Every successful response is cached as JSON under
`cache_dir` (search pages keyed by query, page and page size; trees and file
contents by repo and path), so a rerun, or a larger --limit after a smaller one,
repeats no request it already made.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import re
import subprocess
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Sequence

from quipu.fsio import write_text_atomic

log = logging.getLogger(__name__)

FLASK_QUERIES = (
    "flask filename:requirements.txt",
    "flask filename:pyproject.toml",
    "flask filename:requirements.txt path:backend",
    "flask filename:requirements.txt path:server",
    "flask filename:requirements.txt path:api",
)
REACT_QUERIES = (
    '"react" filename:package.json',
    '"react" filename:package.json path:frontend',
    '"react" filename:package.json path:client',
)
PER_PAGE = 100
MAX_PAGES = 10          # code search never returns more than 1,000 results
SEARCH_INTERVAL = 6.5   # seconds between live code-search requests (~9 a minute)
MAX_TRIES = 5
BASE_BACKOFF = 4.0
MAX_BACKOFF = 60.0
GH_TIMEOUT = 60         # seconds per gh call
RETRY_STATUSES = frozenset({403, 429})
CHECKS_PER_LIMIT = 3
MANIFEST_DEPTH = 2      # directories above a manifest the full-stack check reads
MAX_MANIFESTS = 3       # manifests fetched per repo

_STATUS = re.compile(r"\(HTTP (\d{3})\)")
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_FLASK_REQ = re.compile(r"(?im)^\s*flask(?![\w-])")
_FLASK_TOML = re.compile(r"""(?i)["']flask(?![\w-])|^\s*flask\s*=""", re.MULTILINE)

Runner = Callable[..., Any]


class GhError(RuntimeError):
    """A gh call that failed for good; `status` is the HTTP status, if gh gave one."""

    def __init__(self, message: str, status: int | None) -> None:
        super().__init__(message)
        self.status = status


@dataclasses.dataclass(frozen=True)
class Candidate:
    repo: str               # "owner/name"
    tag: str                # "fullstack", "flask" or "react"
    stars: int | None = None  # code search results carry no star count; None then


def valid_repo(repo: object) -> bool:
    """True for a plain "owner/name" (no "..", no extra slashes). Checked before a
    name is put in an API path or a directory name."""
    return (
        isinstance(repo, str)
        and bool(_REPO.fullmatch(repo))
        and all(part not in (".", "..") for part in repo.split("/"))
    )


def _status_of(stderr: str) -> int | None:
    m = _STATUS.search(stderr or "")
    return int(m.group(1)) if m else None


def gh_json(
    args: Sequence[str],
    runner: Runner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    not_found_ok: bool = False,
) -> Any:
    """Run `gh api <args>` and return the parsed JSON body. With `not_found_ok`, an
    HTTP 404 returns None instead of raising. Retries as the module docstring says."""
    cmd = ["gh", "api", *args]
    last = ""
    status = None
    for attempt in range(MAX_TRIES):
        try:
            proc = runner(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=GH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            status, last = None, f"gh api timed out after {GH_TIMEOUT} s"
        else:
            if proc.returncode == 0:
                try:
                    return json.loads(proc.stdout)
                except json.JSONDecodeError as e:
                    raise GhError(f"gh api {args[0]}: response is not JSON ({e.msg})", None)
            status = _status_of(proc.stderr)
            lines = (proc.stderr or "").strip().splitlines()
            last = lines[-1] if lines else f"exit {proc.returncode}"
            if status == 404 and not_found_ok:
                return None
            if status is not None and status not in RETRY_STATUSES:
                raise GhError(f"gh api {args[0]}: {last}", status)
        if attempt == MAX_TRIES - 1:
            break
        delay = min(MAX_BACKOFF, BASE_BACKOFF * 2**attempt)
        log.warning("gh api %s: %s; retrying in %.0f s", args[0], last, delay)
        sleep(delay)
    raise GhError(f"gh api {args[0]}: gave up after {MAX_TRIES} tries ({last})", status)


class _Pacer:
    """Keeps live code-search requests SEARCH_INTERVAL seconds apart."""

    def __init__(self, sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
        self.sleep, self.clock, self.last = sleep, clock, None

    def wait(self) -> None:
        if self.last is not None:
            gap = SEARCH_INTERVAL - (self.clock() - self.last)
            if gap > 0:
                self.sleep(gap)
        self.last = self.clock()


def _cache_path(cache_dir: Path, query: str, page: int) -> Path:
    key = hashlib.sha256(f"{query}\n{page}\n{PER_PAGE}".encode("utf-8")).hexdigest()[:32]
    return Path(cache_dir) / "search" / f"{key}.json"


def search_page(
    query: str,
    page: int,
    cache_dir: Path,
    runner: Runner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    pacer: _Pacer | None = None,
) -> dict | None:
    """One page of code-search results (the API's JSON), from the cache if there;
    None past the 1,000-result cap (HTTP 422)."""
    path = _cache_path(cache_dir, query, page)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["response"]
    if pacer is not None:
        pacer.wait()
    args = ["-X", "GET", "search/code", "-f", f"q={query}",
            "-f", f"per_page={PER_PAGE}", "-f", f"page={page}"]
    try:
        response = gh_json(args, runner=runner, sleep=sleep)
    except GhError as e:
        if e.status == 422:
            return None
        raise
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        raise GhError(f"code search for {query!r} page {page}: unexpected response shape", None)
    write_text_atomic(
        path, json.dumps({"query": query, "page": page, "response": response}) + "\n"
    )
    return response


def _repos_in(response: dict) -> list[str]:
    repos = []
    for item in response["items"]:
        repo = item.get("repository", {}) if isinstance(item, dict) else {}
        name = repo.get("full_name")
        if valid_repo(name) and not repo.get("fork"):
            repos.append(name)
    return repos


# ------------------------------------------------------------ full-stack check

def _cached_json(cache_dir: Path, kind: str, key: str, fetch: Callable[[], Any]) -> Any:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    path = Path(cache_dir) / kind / f"{digest}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    response = fetch()
    write_text_atomic(path, json.dumps({"key": key, "response": response}) + "\n")
    return response


def _tree(repo: str, cache_dir: Path, runner: Runner, sleep) -> list[str]:
    """Blob paths on the default branch; [] for a missing or empty repo."""
    def fetch():
        try:
            return gh_json([f"repos/{repo}/git/trees/HEAD?recursive=1"], runner=runner,
                           sleep=sleep, not_found_ok=True)
        except GhError as e:
            if e.status == 409:  # "Git Repository is empty."
                return None
            raise

    response = _cached_json(cache_dir, "tree", repo.casefold(), fetch)
    if not isinstance(response, dict) or not isinstance(response.get("tree"), list):
        return []
    return [
        e["path"] for e in response["tree"]
        if isinstance(e, dict) and e.get("type") == "blob" and isinstance(e.get("path"), str)
    ]


def _file_text(repo: str, path: str, cache_dir: Path, runner: Runner, sleep) -> str | None:
    quoted = urllib.parse.quote(path, safe="/")
    response = _cached_json(
        cache_dir, "contents", f"{repo.casefold()}:{path}",
        lambda: gh_json([f"repos/{repo}/contents/{quoted}"], runner=runner, sleep=sleep,
                        not_found_ok=True),
    )
    if not isinstance(response, dict) or response.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(response.get("content") or "").decode("utf-8", "replace")
    except ValueError:
        return None


def react_in_package_json(text: str) -> bool:
    """True when "react" is a dependency, devDependency or peerDependency."""
    try:
        data = json.loads(text)
    except ValueError:
        return False
    return isinstance(data, dict) and any(
        isinstance(data.get(k), dict) and "react" in data[k]
        for k in ("dependencies", "devDependencies", "peerDependencies")
    )


def flask_in_manifest(path: str, text: str) -> bool:
    """True when a requirements.txt line, or a pyproject.toml dependency, names
    flask itself (not flask-cors alone, not a comment)."""
    if path.rsplit("/", 1)[-1].lower() == "pyproject.toml":
        return bool(_FLASK_TOML.search(text))
    return bool(_FLASK_REQ.search(text))


def _manifests(paths: Sequence[str], names: Sequence[str]) -> list[str]:
    found = [
        p for p in paths
        if p.rsplit("/", 1)[-1].lower() in names
        and p.count("/") <= MANIFEST_DEPTH
        and "node_modules" not in p.split("/")
    ]
    return sorted(found, key=lambda p: (p.count("/"), p))[:MAX_MANIFESTS]


def has_other_side(
    repo: str,
    tag: str,
    cache_dir: Path,
    runner: Runner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """For a `flask` repo: does it also have a React package.json? For a `react`
    repo: a requirements.txt / pyproject.toml naming flask? (module docstring)"""
    if not valid_repo(repo):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    if tag == "flask":
        names = ("package.json",)

        def test(path, text):
            return react_in_package_json(text)
    elif tag == "react":
        names, test = ("requirements.txt", "pyproject.toml"), flask_in_manifest
    else:
        raise ValueError(f"tag must be 'flask' or 'react', got {tag!r}")
    for path in _manifests(_tree(repo, cache_dir, runner, sleep), names):
        text = _file_text(repo, path, cache_dir, runner, sleep)
        if text is not None and test(path, text):
            return True
    return False


# ------------------------------------------------------------ selection

def _select(
    flask: dict[str, bool], react: dict[str, bool], limit: int, confirmed: Sequence[str] = ()
) -> list[Candidate]:
    flask_keys = {r.casefold() for r in flask}
    react_keys = {r.casefold() for r in react}
    full = [r for r in flask if r.casefold() in react_keys]
    full += [r for r in confirmed if r.casefold() not in {f.casefold() for f in full}]
    full_keys = {r.casefold() for r in full}
    chosen = [Candidate(r, "fullstack") for r in full[:limit]]
    only_f = [Candidate(r, "flask") for r in flask if r.casefold() not in full_keys | react_keys]
    only_r = [Candidate(r, "react") for r in react if r.casefold() not in full_keys | flask_keys]
    i = j = 0
    while len(chosen) < limit and (i < len(only_f) or j < len(only_r)):
        if i < len(only_f):
            chosen.append(only_f[i])
            i += 1
        if len(chosen) < limit and j < len(only_r):
            chosen.append(only_r[j])
            j += 1
    return chosen


def _check_order(side: dict[str, bool], other_keys: set[str]) -> list[str]:
    """The side's single-side repos, path-scoped hits first, each in search order."""
    single = [r for r in side if r.casefold() not in other_keys]
    return [r for r in single if side[r]] + [r for r in single if not side[r]]


def _interleave(a: list, b: list) -> list:
    out = [x for pair in zip(a, b) for x in pair]
    return out + a[len(b):] + b[len(a):]


def discover(
    limit: int,
    cache_dir: Path,
    runner: Runner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    flask_queries: Sequence[str] = FLASK_QUERIES,
    react_queries: Sequence[str] = REACT_QUERIES,
    max_pages: int = MAX_PAGES,
    max_checks: int | None = None,
) -> list[Candidate]:
    """Up to `limit` tagged candidates, full-stack first (module docstring)."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    if max_checks is None:
        max_checks = CHECKS_PER_LIMIT * limit
    pacer = _Pacer(sleep, clock)
    # side -> {repo: found by a path-scoped query}, insertion-ordered (search order)
    found: dict[str, dict[str, bool]] = {"flask": {}, "react": {}}
    live = [(side, q) for side, qs in (("flask", flask_queries), ("react", react_queries))
            for q in qs]
    for page in range(1, max_pages + 1):
        still = []
        for side, query in live:
            response = search_page(query, page, cache_dir, runner, sleep, pacer)
            if response is None:
                continue
            for repo in _repos_in(response):
                found[side][repo] = found[side].get(repo, False) or "path:" in query
            total = min(int(response.get("total_count", 0)), MAX_PAGES * PER_PAGE)
            if len(response["items"]) == PER_PAGE and page * PER_PAGE < total:
                still.append((side, query))
        live = still
        both = len({r.casefold() for r in found["flask"]} & {r.casefold() for r in found["react"]})
        log.info("search page %d: %d flask, %d react repos so far, %d in both searches",
                 page, len(found["flask"]), len(found["react"]), both)
        if (both >= limit or not live
                or min(len(found["flask"]), len(found["react"])) >= CHECKS_PER_LIMIT * limit):
            break

    flask_keys = {r.casefold() for r in found["flask"]}
    react_keys = {r.casefold() for r in found["react"]}
    n_both = len(flask_keys & react_keys)
    order = _interleave(
        [(r, "flask") for r in _check_order(found["flask"], react_keys)],
        [(r, "react") for r in _check_order(found["react"], flask_keys)],
    )
    confirmed: list[str] = []
    checked = 0
    for repo, tag in order:
        if n_both + len(confirmed) >= limit or checked >= max_checks:
            break
        checked += 1
        try:
            if has_other_side(repo, tag, cache_dir, runner, sleep):
                confirmed.append(repo)
        except GhError as e:
            log.warning("full-stack check of %s failed: %s", repo, e)
        if checked % 25 == 0:
            log.info("full-stack check: %d checked, %d confirmed", checked, len(confirmed))
    log.info("full-stack check: %d repos checked, %d confirmed full-stack", checked,
             len(confirmed))
    return _select(found["flask"], found["react"], limit, confirmed)
