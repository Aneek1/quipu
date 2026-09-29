"""Build the step dataset (spec §3.1).

    python -m stepbuild.dataset.build --limit N --out data/stepbuild
    python -m stepbuild.dataset.build --phase discover --limit N --out data/stepbuild
    python -m stepbuild.dataset.build --phase mine --candidates PATH --out data/stepbuild

discover (gh code search) -> licence filter -> clone -> mine -> filter
(drop_reason) -> format (format_example) -> leakage guard -> dedupe -> split ->
shards, SOURCES.jsonl, manifest.json and report.md under --out.

Phases. Everything that talks to the GitHub API (code search, the full-stack
tree checks, the licence lookups; all through the logged-in `gh`, all cached
under out/cache/) is the discover phase; everything else is the mine phase, which
needs no GitHub login at all: it clones public repos over https with plain git.
So the gh login can stay on the owner's laptop while the heavy mining runs on a
rented machine.
- `--phase discover` writes out/candidates.json (atomically) and stops: an object
  {"version", "limit", "candidates"}, one record per candidate in selection order
  with repo, tag, licence (SPDX id or null), url, stars and clone_url, plus
  licence_error when the lookup failed. Nothing is cloned.
- `--phase mine --candidates PATH` reads such a file (default out/candidates.json)
  and does the rest, making no gh call. A candidate whose licence lookup failed
  is recorded as failed with that error (rerun discover to retry it). --limit is
  optional here and keeps only the first N candidates.
- With no --phase the build runs both, as one command, and also writes
  out/candidates.json; its shards, manifest and report are those of the two
  phases run one after the other.

Per repo. A licensed repo is cloned into out/repos/, mined commit by commit, and
its surviving examples written atomically to out/mined/<owner>__<name>.jsonl;
then its entry in out/manifest.json (status, licence, commit range, counts per
drop reason) is written, also atomically. A crash therefore loses at most the
repo in progress, and a rerun skips every repo whose manifest entry says "mined"
with the same filter settings (a change of settings re-mines it). Any exception
while handling one repo (licence lookup, clone, mining) is logged, recorded as
status "failed" with its message, and the build moves on (spec §4). A failed repo
is retried on the next run.

Per commit. drop_reason and format_example get the same limits (max_files,
max_lines, max_file_lines), so a commit the filters keep is never refused by the
formatter. Context and tree are computed only for commits the filters keep. A
formatted example is then skipped when:
- it is over a token cap: user message over MAX_USER_TOKENS, or system + user +
  reply over MAX_TOTAL_TOKENS (counted separately in the report);
- the benchmark's leakage guard (stepbuild.bench.run.LeakageGuard, the same check
  the bench applies to its example library) says the reply copies a benchmark
  reference solution: such an example would teach, and later show, the answer.

Assembly (every run, over the current candidates' mined repos). Examples are
deduplicated across the whole build by commit SHA (mirrors and copied repos share
history) and by a hash of the normalised reply (LF line endings, trailing
whitespace stripped): the same files written twice teach nothing new and, across
splits, would leak. Repos are visited train first, then validation, then test
(each sorted by name), and the first example with a given key is kept, so a
collision keeps the train copy. The split is format_example's (assign_split, by
repo). Shards are `<split>-NNN.jsonl`, SHARD_ROWS rows each, written atomically
(every split gets at least an empty -000 shard; stale higher-numbered shards from
an earlier, larger build are removed). SOURCES.jsonl lists the repos that
contributed at least one kept example. report.md gives repos found / licensed /
mined per tag, examples per split, drops for every filters.REASONS key (zeros
included), the token caps, leakage skips, dedupe counts and the token counter.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from quipu.fsio import write_text_atomic
from stepbuild.dataset.discover import Candidate, discover, valid_repo
from stepbuild.dataset.filters import REASONS, drop_reason
from stepbuild.dataset.format import format_example
from stepbuild.dataset.licence import is_allowed, licence_of, source_record
from stepbuild.dataset.mine import MAX_LOAD_FILES, Miner, clone, repo_dir_name
from stepbuild.dataset.split import SPLITS

log = logging.getLogger(__name__)

MAX_FILES = 3
MAX_LINES = 200
MAX_FILE_LINES = 400
MAX_USER_TOKENS = 6000
MAX_TOTAL_TOKENS = 8000
SHARD_ROWS = 2000
TAG_ORDER = ("fullstack", "flask", "react")
PARAMS = {
    "max_files": MAX_FILES, "max_lines": MAX_LINES, "max_file_lines": MAX_FILE_LINES,
    "max_user_tokens": MAX_USER_TOKENS, "max_total_tokens": MAX_TOTAL_TOKENS,
    "max_load_files": MAX_LOAD_FILES, "format": 1,
}
TOKEN_COUNTER = (
    "ceil(chars / 3), the harness's default_count_tokens (stepbuild.harness.prompt), "
    "not the plan's chars / 4: pessimistic for code, so an example that fits these "
    "caps also fits the harness's prompt budget"
)

CANDIDATES_FILE = "candidates.json"
CANDIDATES_VERSION = 1
PHASES = ("discover", "mine")
CLONE_SCHEMES = ("https://", "file://")  # file:// is for tests and local mirrors

assert MAX_FILES < MAX_LOAD_FILES  # mine leaves contents out only above MAX_LOAD_FILES


# ------------------------------------------------------------ candidates.json

def candidate_record(cand: Candidate, licence: str | None, error: str | None = None) -> dict:
    rec = {
        "repo": cand.repo, "tag": cand.tag, "licence": licence,
        "url": f"https://github.com/{cand.repo}", "stars": cand.stars,
        "clone_url": f"https://github.com/{cand.repo}.git",
    }
    if error is not None:
        rec["licence_error"] = error
    return rec


def discover_candidates(
    limit: int,
    cache: Path,
    runner: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """The discover phase: candidates and their licences (all gh work, cached)."""
    candidates = discover(limit, cache, runner=runner, sleep=sleep)
    log.info("discovered %d candidates (%s)", len(candidates),
             ", ".join(f"{t} {sum(c.tag == t for c in candidates)}" for t in TAG_ORDER))
    records = []
    for cand in candidates:
        try:
            licence = licence_of(cand.repo, runner=runner, cache_dir=cache, sleep=sleep)
        except Exception as e:  # recorded; the mine phase reports it as a failed repo
            error = f"{type(e).__name__}: {e}"
            log.warning("%s: licence lookup failed: %s", cand.repo, error)
            records.append(candidate_record(cand, None, error))
            continue
        records.append(candidate_record(cand, licence))
    return records


def write_candidates(path: Path, limit: int, records: Sequence[dict]) -> None:
    body = {"version": CANDIDATES_VERSION, "limit": limit, "candidates": list(records)}
    write_text_atomic(Path(path), json.dumps(body, indent=1) + "\n")


def _check_clone_url(url: object) -> bool:
    if not isinstance(url, str) or not url.startswith(CLONE_SCHEMES):
        return False
    authority = url.split("://", 1)[1].split("/", 1)[0]
    return "@" not in authority  # no credentials embedded in the URL


def read_candidates(path: Path) -> tuple[int, list[dict]]:
    """(limit, records) from a candidates.json; ValueError when it is malformed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != CANDIDATES_VERSION:
        raise ValueError(f"{path}: not a version {CANDIDATES_VERSION} candidates file")
    limit, records = data.get("limit"), data.get("candidates")
    if not isinstance(limit, int) or not isinstance(records, list):
        raise ValueError(f"{path}: needs an int 'limit' and a 'candidates' list")
    for i, rec in enumerate(records):
        ok = (
            isinstance(rec, dict) and valid_repo(rec.get("repo"))
            and rec.get("tag") in TAG_ORDER
            and (rec.get("licence") is None or isinstance(rec.get("licence"), str))
            and _check_clone_url(rec.get("clone_url"))
        )
        if not ok:
            raise ValueError(f"{path}: candidate {i} is malformed or has a disallowed "
                             f"clone_url (https:// or file://, no credentials): {rec!r}")
    return limit, records


def _reply_key(row: dict) -> str:
    reply = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    text = reply.replace("\r\n", "\n").replace("\r", "\n")
    norm = "\n".join(line.rstrip() for line in text.split("\n")).rstrip("\n")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("repos"), dict):
            return data
    return {"repos": {}}


def _save_manifest(path: Path, manifest: dict) -> None:
    write_text_atomic(path, json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def mine_one(
    cand: Candidate,
    licence: str,
    repo_dir: Path,
    guard,
) -> tuple[dict, list[dict]]:
    """Mine one cloned repo: (its manifest entry, its formatted examples)."""
    drops: Counter = Counter({r: 0 for r in REASONS})
    over_user = over_total = leaked = kept_by_filters = 0
    rows: list[dict] = []
    with Miner(repo_dir, max_load_files=MAX_LOAD_FILES, max_file_lines=MAX_FILE_LINES) as m:
        for commit in m.commits():
            reason = drop_reason(
                commit, max_files=MAX_FILES, max_lines=MAX_LINES, max_file_lines=MAX_FILE_LINES
            )
            if reason is not None:
                drops[reason] += 1
                continue
            kept_by_filters += 1
            context, tree = m.context(commit), m.tree(commit)
            limits = dict(max_files=MAX_FILES, max_lines=MAX_LINES, max_file_lines=MAX_FILE_LINES)
            row = format_example(
                cand.repo, licence, cand.tag, commit, context, tree,
                max_user_tokens=MAX_USER_TOKENS, max_total_tokens=MAX_TOTAL_TOKENS, **limits,
            )
            if row is None:
                loose = format_example(
                    cand.repo, licence, cand.tag, commit, context, tree,
                    max_user_tokens=MAX_USER_TOKENS, max_total_tokens=10**12, **limits,
                )
                if loose is None:
                    over_user += 1
                else:
                    over_total += 1
                continue
            reply = row["messages"][2]["content"]
            copied = guard.find(reply) if guard is not None else None
            if copied is not None:
                leaked += 1
                log.info("%s %s: skipped, the reply%s", cand.repo, commit.sha[:10], copied)
                continue
            rows.append(row)
        entry = {
            "repo": cand.repo, "tag": cand.tag, "licence": licence, "status": "mined",
            "params": PARAMS, "first_sha": m.first_sha, "last_sha": m.last_sha,
            "commits": m.walked, "kept_by_filters": kept_by_filters,
            "drops": dict(drops), "over_user_tokens": over_user,
            "over_total_tokens": over_total, "leakage_skipped": leaked,
            "examples": len(rows),
        }
    return entry, rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def assemble(out: Path, entries: Sequence[dict], mined_dir: Path) -> dict:
    """Dedupe, shard, and write SOURCES.jsonl. Returns the assembly counts."""
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    split_of_repo: dict[str, str] = {}
    for e in entries:
        path = mined_dir / f"{repo_dir_name(e['repo'])}.jsonl"
        first = next(_read_jsonl(path), None) if path.is_file() else None
        if first is None:
            continue
        split_of_repo[e["repo"]] = first["split"]
    ordered = [
        e for s in SPLITS
        for e in sorted(entries, key=lambda e: e["repo"].casefold())
        if split_of_repo.get(e["repo"]) == s
    ]
    seen_sha: set[str] = set()
    seen_reply: set[str] = set()
    dup_sha = dup_reply = 0
    kept_per_repo: Counter = Counter()
    rows_out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    shards: dict[str, list[str]] = {s: [] for s in SPLITS}

    def flush(split: str, final: bool = False) -> None:
        rows = rows_out[split]
        if not rows and not (final and not shards[split]):
            return
        name = f"{split}-{len(shards[split]):03d}.jsonl"
        _write_jsonl(out / name, rows)
        shards[split].append(name)
        rows_out[split] = []

    for e in ordered:
        split = split_of_repo[e["repo"]]
        for row in _read_jsonl(mined_dir / f"{repo_dir_name(e['repo'])}.jsonl"):
            key = _reply_key(row)
            dup = "sha" if row["commit"] in seen_sha else "reply" if key in seen_reply else None
            seen_sha.add(row["commit"])
            seen_reply.add(key)
            if dup == "sha":
                dup_sha += 1
                continue
            if dup == "reply":
                dup_reply += 1
                continue
            rows_out[split].append(row)
            kept_per_repo[e["repo"]] += 1
            by_split[split].append(e["repo"])
            if len(rows_out[split]) >= SHARD_ROWS:
                flush(split)
    for split in SPLITS:
        flush(split, final=True)
        n = len(shards[split])
        for stale in out.glob(f"{split}-*.jsonl"):
            idx = stale.stem.rsplit("-", 1)[-1]
            if idx.isdigit() and int(idx) >= n:
                stale.unlink()

    sources = [
        source_record(e["repo"], e["licence"], e["first_sha"], e["last_sha"], e["tag"])
        | {"examples": kept_per_repo[e["repo"]]}
        for e in sorted(entries, key=lambda e: e["repo"].casefold())
        if kept_per_repo[e["repo"]] > 0
    ]
    _write_jsonl(out / "SOURCES.jsonl", sources)
    return {
        "examples": {s: len(by_split[s]) for s in SPLITS},
        "repos_per_split": {s: len(set(by_split[s])) for s in SPLITS},
        "dedupe_sha": dup_sha,
        "dedupe_reply": dup_reply,
        "shards": shards,
        "sources": len(sources),
    }


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def render_report(
    candidates: Sequence[Candidate], manifest: dict, assembled: dict, meta: dict
) -> str:
    repos = manifest["repos"]
    entries = [repos.get(c.repo, {}) for c in candidates]
    lines = [
        "# Step dataset build report",
        "",
        f"- command: `{meta['command']}`",
        f"- started {meta['started']}, finished {meta['finished']} "
        f"({meta['elapsed_s']:.0f} s)",
        f"- limit {meta['limit']}; out `{meta['out']}`",
        f"- filters: max_files={MAX_FILES}, max_lines={MAX_LINES}, "
        f"max_file_lines={MAX_FILE_LINES} (the same values go to drop_reason and "
        "format_example)",
        f"- caps: user message <= {MAX_USER_TOKENS} tokens, system + user + reply <= "
        f"{MAX_TOTAL_TOKENS} tokens",
        f"- token counter: {TOKEN_COUNTER}",
        "",
        "## Repos",
        "",
    ]
    rows = []
    for tag in TAG_ORDER + ("total",):
        sel = [
            (c, e) for c, e in zip(candidates, entries) if tag == "total" or c.tag == tag
        ]
        rows.append([
            tag,
            len(sel),
            sum(1 for _, e in sel if is_allowed(e.get("licence"))),
            sum(1 for _, e in sel if e.get("status") == "mined"),
            sum(1 for _, e in sel if e.get("status") == "failed"),
            sum(1 for _, e in sel if e.get("status") == "mined" and e.get("examples", 0) > 0),
        ])
    lines += _table(["tag", "found", "licensed", "mined", "failed", "with examples (before dedupe)"], rows)
    lic = Counter(str(e.get("licence")) for e in entries if "licence" in e)
    lines += ["", "Licences seen: " + (", ".join(
        f"{k} {v}" for k, v in sorted(lic.items(), key=lambda kv: (-kv[1], kv[0]))) or "none")]
    mined = [e for e in entries if e.get("status") == "mined"]
    drops: Counter = Counter({r: 0 for r in REASONS})
    for e in mined:
        drops.update(e.get("drops", {}))
    walked = sum(e.get("commits", 0) for e in mined)
    kept = sum(e.get("kept_by_filters", 0) for e in mined)
    over_user = sum(e.get("over_user_tokens", 0) for e in mined)
    over_total = sum(e.get("over_total_tokens", 0) for e in mined)
    leaked = sum(e.get("leakage_skipped", 0) for e in mined)
    formatted = sum(e.get("examples", 0) for e in mined)
    ex = assembled["examples"]
    lines += [
        "",
        "## Examples",
        "",
        *_table(
            ["split", "examples", "repos", "shards"],
            [[s, ex[s], assembled["repos_per_split"][s], ", ".join(assembled["shards"][s])]
             for s in SPLITS]
            + [["total", sum(ex.values()), sum(assembled["repos_per_split"].values()), ""]],
        ),
        "",
        "## Commits",
        "",
        f"- first-parent commits walked: {walked}",
        f"- kept by the filters: {kept}",
        f"- over the user-message cap ({MAX_USER_TOKENS} tokens): {over_user}",
        f"- over the whole-example cap ({MAX_TOTAL_TOKENS} tokens): {over_total}",
        f"- skipped by the benchmark leakage guard: {leaked}",
        f"- examples after the caps and the guard: {formatted}",
        f"- duplicates removed: {assembled['dedupe_sha'] + assembled['dedupe_reply']} "
        f"({assembled['dedupe_sha']} by commit SHA, {assembled['dedupe_reply']} by reply)",
        f"- kept in the shards: {sum(ex.values())}",
        "",
        "### Drops per filter reason",
        "",
        *_table(["reason", "commits"], [[r, drops[r]] for r in REASONS]
                + [["total", sum(drops[r] for r in REASONS)]]),
    ]
    failed = [(c.repo, e.get("error", "")) for c, e in zip(candidates, entries)
              if e.get("status") == "failed"]
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- {repo}: {err}" for repo, err in failed]
    return "\n".join(lines) + "\n"


def build(
    limit: int | None,
    out: Path,
    runner: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    clone_runner: Callable[..., Any] | None = None,
    guard=None,
    command: str = "",
    phase: str | None = None,
    candidates_path: Path | None = None,
) -> dict:
    """Run the build, or one phase of it (module docstring). Returns the assembly
    counts, or for the discover phase {"candidates", "licensed", "path"}. `runner`
    runs gh (discover phase only); `clone_runner` runs git clone (mine phase)."""
    if phase not in (None, *PHASES):
        raise ValueError(f"phase must be one of {PHASES} or None, got {phase!r}")
    started = datetime.datetime.now().astimezone()
    t0 = time.monotonic()
    out = Path(out)
    cache, repos_dir, mined_dir = out / "cache", out / "repos", out / "mined"
    manifest_path = out / "manifest.json"
    if phase == "mine":
        path = Path(candidates_path) if candidates_path else out / CANDIDATES_FILE
        file_limit, records = read_candidates(path)
        if limit is not None:
            records = records[:limit]
        limit = file_limit if limit is None else limit
        log.info("read %d candidates from %s", len(records), path)
    else:
        if limit is None:
            raise ValueError("limit is required unless phase is 'mine'")
        records = discover_candidates(limit, cache, runner=runner, sleep=sleep)
        path = out / CANDIDATES_FILE
        write_candidates(path, limit, records)
        if phase == "discover":
            licensed = sum(1 for r in records if is_allowed(r["licence"]))
            log.info("wrote %d candidates (%d licensed) to %s", len(records), licensed, path)
            return {"candidates": len(records), "licensed": licensed, "path": path}
    if guard is None:
        from stepbuild.bench.run import LeakageGuard
        guard = LeakageGuard()
    candidates = [Candidate(r["repo"], r["tag"], r.get("stars")) for r in records]
    manifest = _load_manifest(manifest_path)
    for n, (cand, rec) in enumerate(zip(candidates, records), start=1):
        prev = manifest["repos"].get(cand.repo, {})
        mined_file = mined_dir / f"{repo_dir_name(cand.repo)}.jsonl"
        if prev.get("status") == "mined" and prev.get("params") == PARAMS and mined_file.is_file():
            log.info("[%d/%d] %s: already mined, skipped", n, len(candidates), cand.repo)
            continue
        entry: dict = {"repo": cand.repo, "tag": cand.tag}
        try:
            licence = rec["licence"]
            if rec.get("licence_error") is not None:  # the discover phase's lookup failed
                entry["status"] = "failed"
                entry["error"] = rec["licence_error"]
                log.warning("[%d/%d] %s failed: %s", n, len(candidates), cand.repo,
                            entry["error"])
            elif not is_allowed(licence):
                entry["licence"] = licence
                entry["status"] = "unlicensed"
                log.info("[%d/%d] %s: licence %s, skipped", n, len(candidates), cand.repo, licence)
            else:
                entry["licence"] = licence
                t = time.monotonic()
                repo_dir = clone(cand.repo, repos_dir, runner=clone_runner or subprocess.run,
                                 url=rec["clone_url"])
                entry, rows = mine_one(cand, licence, repo_dir, guard)
                _write_jsonl(mined_file, rows)
                log.info("[%d/%d] %s (%s, %s): %d commits, %d examples, %.0f s",
                         n, len(candidates), cand.repo, cand.tag, licence,
                         entry["commits"], len(rows), time.monotonic() - t)
        except Exception as e:  # one bad repo never stops the build (spec §4)
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            log.warning("[%d/%d] %s failed: %s\n%s", n, len(candidates), cand.repo,
                        entry["error"], traceback.format_exc(limit=3))
        manifest["repos"][cand.repo] = entry
        _save_manifest(manifest_path, manifest)
    entries = [
        manifest["repos"][c.repo] for c in candidates
        if manifest["repos"].get(c.repo, {}).get("status") == "mined"
    ]
    assembled = assemble(out, entries, mined_dir)
    finished = datetime.datetime.now().astimezone()
    meta = {
        "command": command or f"build(limit={limit}, out={out})",
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "elapsed_s": time.monotonic() - t0,
        "limit": limit,
        "out": out.as_posix(),
    }
    write_text_atomic(out / "report.md", render_report(candidates, manifest, assembled, meta))
    return assembled


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="number of repos to use (required unless --phase mine)")
    parser.add_argument("--out", type=Path, default=Path("data") / "stepbuild")
    parser.add_argument("--phase", choices=PHASES, default=None,
                        help="discover: gh work only, writes candidates.json; mine: no gh, "
                             "reads candidates.json; default: both")
    parser.add_argument("--candidates", type=Path, default=None,
                        help=f"candidates file for --phase mine (default OUT/{CANDIDATES_FILE})")
    args = parser.parse_args(argv)
    if args.limit is None and args.phase != "mine":
        parser.error("--limit is required unless --phase mine")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.candidates is not None and args.phase != "mine":
        parser.error("--candidates only goes with --phase mine")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    argv_list = sys.argv[1:] if argv is None else list(argv)
    result = build(args.limit, args.out, phase=args.phase, candidates_path=args.candidates,
                   command="python -m stepbuild.dataset.build " + " ".join(argv_list))
    if args.phase == "discover":
        print(f"wrote {result['candidates']} candidates ({result['licensed']} licensed) "
              f"to {result['path']}")
    else:
        print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
