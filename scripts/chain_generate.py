"""Grow distill v2's solved problems into verified multi-turn chains (quipu.chains).

Reads data/distill-v2/kept/part-NNNN.jsonl (problem + teacher answer that passed) and
joins each row's unit tests by id from data/distill-v2/prompts/part-NNNN.jsonl. Writes
one chain per seed to data/chains/gen/part-NNNN.<teacher-tag>.jsonl, appended and
flushed as each chain finishes. Rerun to resume (finished ids are skipped); create
data/chains/PAUSE to stop cleanly after the chains in flight (do that, not Ctrl+C).
Chains that fail on a dropped connection or a sandbox crash are not written; they are
logged in <out>.failures.jsonl and retried on a rerun, up to --max-attempts times.

Teacher: an OpenAI-compatible server, the same Qwen2.5-Coder-7B-Instruct as distill v2
(llama-server with the Q4_K_M GGUF on the laptop, vLLM on Kaggle). Greedy decoding.

    .venv/Scripts/python.exe scripts/chain_generate.py --parts 0:0 --limit 200 \\
        --endpoint http://127.0.0.1:8080 --teacher-tag q4km-llamacpp \\
        --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --concurrency 2
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu import sandbox                                                  # noqa: E402
from quipu.chains import PROMPT_VERSION, Chat, run_chain                                   # noqa: E402
from scripts.distill_generate_v2 import TEACHER, _drop_torn_tail, done_ids, part_order  # noqa: E402

SHOW_FAILURES = 5


def load_seeds(distill_root: Path, part: int, split: str = "all") -> list[dict]:
    """Kept rows of one distill v2 part, each with its problem's tests; `split` is
    "all", "train" or "val" (val seeds are the held-out eval set)."""
    root = Path(distill_root)
    with open(root / "prompts" / f"part-{part:04d}.jsonl", encoding="utf-8") as f:
        tests = {r["id"]: r["tests"] for r in map(json.loads, filter(str.strip, f))}
    seeds = []
    with open(root / "kept" / f"part-{part:04d}.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, filter(str.strip, f)):
            if r["id"] in tests and split in ("all", r["split"]):
                seeds.append({k: r[k] for k in ("id", "split", "entry_point", "prompt", "response")}
                             | {"tests": tests[r["id"]]})
    return seeds


class PermanentRequestError(RuntimeError):
    """The server rejected this one request because it is too long (HTTP 400, context
    overflow). It would fail the same way every time, so the chain is recorded."""


class FatalRequestError(RuntimeError):
    """The server rejected the request for a reason that affects every request (wrong
    endpoint or model, bad credentials: 401/403/404, or any other non-retryable 4xx).
    The whole run stops; nothing is written."""


OVERFLOW_WORDS = ("context", "exceed", "maximum context length", "too long")

# Stops caused by the connection, not the seed: never written, never counted against
# --max-attempts; only the failure streak ends a run that keeps hitting them.
CONNECTION_STOPS = frozenset("error:" + n for n in (
    "URLError", "HTTPError", "TimeoutError", "ConnectionError", "RemoteDisconnected",
    "ConnectionResetError", "ConnectionRefusedError", "ConnectionAbortedError", "OSError",
    "IncompleteRead", "BadStatusLine", "JSONDecodeError"))


SETTINGS = ("max_turns", "min_new_tests", "max_new_tokens", "timeout")
SETTING_DEFAULTS = {k: v.default for k, v in inspect.signature(run_chain).parameters.items()
                    if k in SETTINGS}


def _transient(rec: dict) -> bool:
    """A chain that ended on a teacher/sandbox exception says nothing about the seed,
    so it is not recorded: a rerun retries it. A too-long request (PermanentRequestError)
    would fail the same way every time, so that record is kept."""
    stop = rec.get("stop", "")
    if stop == "error:PermanentRequestError":
        return False
    return stop.startswith("error:") or stop == "sandbox_error"


def _counts_as_attempt(rec: dict) -> bool:
    """Transient failures that repeat for the same seed (a sandbox crash) use up
    --max-attempts; connection trouble does not."""
    return rec["stop"] not in CONNECTION_STOPS


def _attempts(ledger: Path) -> dict[str, int]:
    """Failed attempts so far per chain id, from the retry ledger."""
    counts: dict[str, int] = {}
    if ledger.exists():
        with open(ledger, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    cid = json.loads(line)["id"]
                except (ValueError, KeyError, TypeError):
                    continue
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def _check_settings(out: Path, settings: dict) -> None:
    """Refuse to extend a file written with other chain settings: one file, one setting."""
    if not out.exists():
        return
    with open(out, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                first = json.loads(line)
            except ValueError:
                continue
            for k, v in settings.items():
                if first.get(k) != v:
                    raise ValueError(f"{out} was written with {k}={first.get(k)!r}, now {v!r}; "
                                     "use a new --teacher-tag or output file")
            return


def run_part(seeds: list[dict], out: Path, chat: Chat, *, concurrency: int, pause: Path,
             meta: dict, max_attempts: int = 3, **chain_kw) -> str:
    """Run every seed whose chain isn't in `out` yet. Returns "done" | "paused" |
    "error" (a streak of failures: the server stopped answering) | "incomplete" (some
    chains raised or ended on a transient error; those are not written but logged in
    `<out>.failures.jsonl`, rerun to retry them). A chain that already failed
    `max_attempts` times is written as it ended instead, so it stops being retried."""
    out = Path(out)
    settings = (SETTING_DEFAULTS | {k: v for k, v in chain_kw.items() if k in SETTINGS}
                | {"prompt_version": PROMPT_VERSION})
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(out)
    _check_settings(out, settings)
    ledger = Path(str(out) + ".failures.jsonl")
    attempts = _attempts(ledger)
    have = done_ids(out)
    pending = [s for s in seeds if "ch-" + s["id"] not in have]
    todo = iter(pending)
    failed = written = streak = 0
    max_streak = max(20, 2 * concurrency)
    stop, running, fatal = "", set(), False
    t0 = time.monotonic()
    with open(out, "a", encoding="utf-8", newline="\n") as f, \
            open(ledger, "a", encoding="utf-8", newline="\n") as lf, \
            ThreadPoolExecutor(concurrency) as ex:
        while True:
            if not stop and pause.exists():
                stop = "paused"
            if not stop and streak >= max_streak:
                stop = "error"
            while not stop and len(running) < concurrency:
                s = next(todo, None)
                if s is None:
                    break
                running.add(ex.submit(run_chain, s, chat, **chain_kw))
            if not running:
                break
            finished, running = wait(running, timeout=5.0, return_when=FIRST_COMPLETED)
            for fut in finished:
                if fut.exception() is not None:
                    failed += 1
                    streak += 1
                    if failed <= SHOW_FAILURES:
                        print(f"chain failed: {fut.exception()!r}", flush=True)
                    continue
                rec = fut.result()
                if rec["stop"] == "error:FatalRequestError":
                    print(f"fatal: {rec['stop']} on {rec['id']}; check --endpoint and the server",
                          flush=True)
                    fatal = True
                    stop = "error"
                    continue
                if _transient(rec) and (not _counts_as_attempt(rec)
                                        or attempts.get(rec["id"], 0) < max_attempts):
                    failed += 1
                    streak += 1
                    if _counts_as_attempt(rec):
                        attempts[rec["id"]] = attempts.get(rec["id"], 0) + 1
                        lf.write(json.dumps({"id": rec["id"], "stop": rec["stop"]}) + "\n")
                        lf.flush()
                    if failed <= SHOW_FAILURES:
                        print(f"chain {rec['id']} not written: {rec['stop']}", flush=True)
                    continue
                if not rec["stop"].startswith("error:"):
                    streak = 0              # only clean records show the server is healthy
                f.write(json.dumps(rec | meta | settings, ensure_ascii=True) + "\n")
                f.flush()
                written += 1
                if written % 100 == 0:
                    rate = written / max(time.monotonic() - t0, 1e-9) * 3600
                    print(f"  {written}/{len(pending)} written, {rate:.0f} chains/hour", flush=True)
    print(f"  {written} written, {failed} failed", flush=True)
    return stop or ("incomplete" if failed else "done")


def make_messages_chat(endpoint: str, model: str = "teacher", timeout: float = 600.0,
                       retries: int = 3, backoff: float = 5.0) -> Chat:
    """chat(messages, max_tokens) for an OpenAI-compatible server, greedy. Retries
    connection trouble, timeouts and HTTP 5xx; a 4xx is raised at once."""
    import http.client
    import time
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def chat(messages: list[dict], max_tokens: int) -> tuple[str, str, int]:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": messages}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                c = d["choices"][0]
                reason = c.get("finish_reason") or "stop"
                if reason not in ("stop", "length"):
                    raise RuntimeError(f"finish_reason={reason}")
                return (c["message"]["content"] or "", reason,
                        int(d.get("usage", {}).get("completion_tokens", 0)))
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read(1000).decode("utf-8", "replace")
                except Exception:
                    detail = ""
                e.close()
                if e.code < 500 and e.code not in (408, 429):
                    msg = f"HTTP {e.code}: {detail}"
                    if e.code == 400 and any(w in detail.lower() for w in OVERFLOW_WORDS):
                        raise PermanentRequestError(msg) from e
                    raise FatalRequestError(msg) from e
                if attempt == retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
            time.sleep(backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    return chat


def _positive(text: str) -> int:
    n = int(text)
    if n < 1:
        raise ValueError("must be >= 1")
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--distill-root", default="data/distill-v2")
    ap.add_argument("--out-root", default="data/chains")
    ap.add_argument("--parts", required=True, help="START:END inclusive, e.g. 0:4")
    ap.add_argument("--direction", choices=("up", "down"), default="up")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--limit", type=_positive, default=None, help="first N seeds per part (smoke runs)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--teacher-tag", required=True, help="e.g. q4km-llamacpp, fp16-vllm-t4")
    ap.add_argument("--revision", required=True, help="teacher weights revision (sha)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="failed attempts per chain before it is written as it ended")
    ap.add_argument("--sandbox-workers", type=_positive, default=os.cpu_count() or 1,
                    help="most sandbox subprocesses at once (separate from --concurrency)")
    args = ap.parse_args(argv)

    sandbox.set_max_parallel(args.sandbox_workers)
    out_root = ROOT / args.out_root
    chat = make_messages_chat(args.endpoint)
    meta = {"teacher": TEACHER, "revision": args.revision, "teacher_tag": args.teacher_tag}
    incomplete = []
    for part in part_order(args.parts, args.direction):
        seeds = load_seeds(ROOT / args.distill_root, part, args.split)[:args.limit]
        out = out_root / "gen" / f"part-{part:04d}.{args.teacher_tag}.jsonl"
        print(f"part {part}: {len(seeds)} seeds", flush=True)
        status = run_part(seeds, out, chat, concurrency=args.concurrency,
                          pause=out_root / "PAUSE", meta=meta, max_turns=args.max_turns,
                          max_attempts=args.max_attempts)
        print(f"part {part}: {status}", flush=True)
        if status == "paused":
            return 0
        if status == "error":
            return 4
        if status == "incomplete":
            incomplete.append(part)
    if incomplete:
        print(f"incomplete parts (rerun to retry): {incomplete}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
