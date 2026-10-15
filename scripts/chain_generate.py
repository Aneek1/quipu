"""Grow distill v2's solved problems into verified multi-turn chains (quipu.chains).

Reads data/distill-v2/kept/part-NNNN.jsonl (problem + teacher answer that passed) and
joins each row's unit tests by id from data/distill-v2/prompts/part-NNNN.jsonl. Writes
one chain per seed to data/chains/gen/part-NNNN.<teacher-tag>.jsonl, appended and
flushed as each chain finishes. Rerun to resume (finished ids are skipped); create
data/chains/PAUSE to stop cleanly after the chains in flight.

Teacher: an OpenAI-compatible server, the same Qwen2.5-Coder-7B-Instruct as distill v2
(llama-server with the Q4_K_M GGUF on the laptop, vLLM on Kaggle). Greedy decoding.

    .venv/Scripts/python.exe scripts/chain_generate.py --parts 0:0 --limit 200 \\
        --endpoint http://127.0.0.1:8080 --teacher-tag q4km-llamacpp \\
        --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --concurrency 2
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.chains import Chat, run_chain                                   # noqa: E402
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


def _transient(rec: dict) -> bool:
    """A chain that ended on a teacher/sandbox exception says nothing about the seed,
    so it is not recorded: a rerun retries it."""
    stop = rec.get("stop", "")
    return stop.startswith("error:") or stop == "sandbox_error"


def run_part(seeds: list[dict], out: Path, chat: Chat, *, concurrency: int, pause: Path,
             meta: dict, **chain_kw) -> str:
    """Run every seed whose chain isn't in `out` yet. Returns "done" | "paused" |
    "incomplete" (some chains raised or ended on a transient error, e.g. the server
    dropped requests; they are not written, rerun to retry them)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(out)
    have = done_ids(out)
    todo = iter([s for s in seeds if "ch-" + s["id"] not in have])
    failed, stop, running = 0, "", set()
    with open(out, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(concurrency) as ex:
        while True:
            if not stop and pause.exists():
                stop = "paused"
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
                    if failed <= SHOW_FAILURES:
                        print(f"chain failed: {fut.exception()!r}", flush=True)
                    continue
                rec = fut.result()
                if _transient(rec):
                    failed += 1
                    if failed <= SHOW_FAILURES:
                        print(f"chain {rec['id']} not written: {rec['stop']}", flush=True)
                    continue
                f.write(json.dumps(rec | meta, ensure_ascii=False) + "\n")
                f.flush()
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
                e.close()
                if e.code < 500 or attempt == retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
            time.sleep(backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    return chat


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--distill-root", default="data/distill-v2")
    ap.add_argument("--out-root", default="data/chains")
    ap.add_argument("--parts", required=True, help="START:END inclusive, e.g. 0:4")
    ap.add_argument("--direction", choices=("up", "down"), default="up")
    ap.add_argument("--split", choices=("all", "train", "val"), default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N seeds per part (smoke runs)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--teacher-tag", required=True, help="e.g. q4km-llamacpp, fp16-vllm-t4")
    ap.add_argument("--revision", required=True, help="teacher weights revision (sha)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=3)
    args = ap.parse_args(argv)

    out_root = ROOT / args.out_root
    chat = make_messages_chat(args.endpoint)
    meta = {"teacher": TEACHER, "revision": args.revision, "teacher_tag": args.teacher_tag}
    for part in part_order(args.parts, args.direction):
        seeds = load_seeds(ROOT / args.distill_root, part, args.split)[:args.limit]
        out = out_root / "gen" / f"part-{part:04d}.{args.teacher_tag}.jsonl"
        print(f"part {part}: {len(seeds)} seeds", flush=True)
        status = run_part(seeds, out, chat, concurrency=args.concurrency,
                          pause=out_root / "PAUSE", meta=meta, max_turns=args.max_turns)
        print(f"part {part}: {status}", flush=True)
        if status == "paused":
            return 0
        if status == "incomplete":
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
