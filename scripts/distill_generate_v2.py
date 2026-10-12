"""Distillation v2, step 1: the teacher answers a range of prompt parts.

Talks to any OpenAI-compatible /v1/chat/completions server: vLLM on the rented box
(bf16), llama.cpp llama-server on the laptop (4-bit GGUF). Teacher:
Qwen/Qwen2.5-Coder-7B-Instruct (Apache-2.0). Greedy decoding, the fixed
SYSTEM_PROMPT below (a description of the answer style only; no example answers),
the problem as the user turn.

Each worker gets an explicit part range (--parts START:END, --direction up|down;
box 0:4 up, laptop 5:19 down), skips parts that have a .done sidecar and works the
first unfinished one. Every answer is appended and flushed as it arrives, so
stopping at any point loses only in-flight requests; rerunning resumes. Creating
data/distill-v2/PAUSE makes it finish in-flight requests and exit. On the box,
--budget-usd stops it when quipu.spend reports the cap reached (or when the spend
check itself fails: the box is billing, so it fails closed). If the server stops
answering (max(20, 2 * concurrency) failures in a row) it stops with "error".

Output: generations/part-NNNN.<tag>.jsonl rows: id, part, teacher, revision,
teacher_tag, response, finish ("eos" | "length"), new_tokens; and, when every
prompt id of the part is answered, part-NNNN.<tag>.done: {"rows", "sha256"}.
"rows" is the part's prompt count, not the line count of the jsonl: a resumed file
can hold a duplicate row for an id (a request in flight at a hard stop whose row
landed anyway), so the verifier dedups by id.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEACHER = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "Start with one or two sentences restating what the problem asks and the approach "
    "you'll take. Then give a short numbered plan (2-4 steps). Then the complete function "
    "in one ```python block, with clear names and a brief docstring, and comments only "
    "where the logic isn't obvious. End with a short \"Edge cases\" line covering empty "
    "input, boundaries, and time complexity. Be direct and friendly, don't repeat "
    "yourself, and no filler like \"Great question!\"."
)

WAIT_TIMEOUT = 5.0   # s; PAUSE and budget are re-checked at least this often
BUDGET_EVERY = 60.0  # s between over_budget() calls
SHOW_FAILURES = 5    # failure messages printed per part; the rest are counted

Chat = Callable[[str, str, int], "tuple[str, str, int]"]  # -> (text, finish_reason, tokens)


def part_order(spec: str, direction: str) -> list[int]:
    start, end = (int(x) for x in spec.split(":"))
    parts = list(range(start, end + 1))
    return parts[::-1] if direction == "down" else parts


def gen_paths(root: Path, part: int, tag: str) -> tuple[Path, Path]:
    g = Path(root) / "generations"
    return g / f"part-{part:04d}.{tag}.jsonl", g / f"part-{part:04d}.{tag}.done"


def is_done(root: Path, part: int) -> bool:
    return any((Path(root) / "generations").glob(f"part-{part:04d}.*.done"))


def done_ids(path: Path) -> set[str]:
    out: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    out.add(json.loads(line)["id"])
                except (ValueError, KeyError, TypeError):
                    continue  # defensive: a damaged line never blocks a resume
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _drop_torn_tail(path: Path) -> None:
    """An interrupted run can leave a half-written last line. Cut it off, so the
    next appended row starts on a fresh line and a finished file holds only whole
    JSON rows (that prompt is simply asked again)."""
    if not path.exists() or not path.stat().st_size:
        return
    with open(path, "r+b") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return
        size = f.seek(0, 2)
        pos, cut = size, 0
        while pos > 0:  # scan back to the last newline
            step = min(1 << 16, pos)
            pos -= step
            f.seek(pos)
            nl = f.read(step).rfind(b"\n")
            if nl != -1:
                cut = pos + nl + 1
                break
        f.seek(cut)
        try:  # a whole row that only lacks its newline is kept
            whole = "id" in json.loads(f.read().decode("utf-8"))
        except (ValueError, TypeError):
            whole = False
        if whole:
            f.seek(0, 2)
            f.write(b"\n")
        else:
            f.truncate(cut)


def run_part(root: Path, part: int, tag: str, chat: Chat, *, concurrency: int,
             max_new_tokens: int, teacher: str, revision: str,
             over_budget: Callable[[], bool] = lambda: False) -> str:
    """Answer every unanswered prompt of one part.
    Returns "done" | "paused" | "budget" | "error" (server stopped answering) |
    "incomplete" (some requests failed; a rerun retries them)."""
    root = Path(root)
    with open(root / "prompts" / f"part-{part:04d}.jsonl", encoding="utf-8") as pf:
        prompts = [json.loads(l) for l in pf if l.strip()]
    out, done_file = gen_paths(root, part, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    _drop_torn_tail(out)
    have = done_ids(out)
    todo = [p for p in prompts if p["id"] not in have]
    pause = root / "PAUSE"
    lock = threading.Lock()
    failed = streak = 0
    max_streak = max(20, 2 * concurrency)
    stop = ""
    t0, n_tok, n_new = time.monotonic(), 0, 0

    # Exit order: the executor (listed last) shuts down first and waits for every
    # in-flight request, which writes its row; only then is the file closed.
    with open(out, "a", encoding="utf-8", newline="\n") as f, \
            ThreadPoolExecutor(concurrency) as ex:

        def one(p: dict) -> None:
            nonlocal n_tok, n_new
            text, reason, tokens = chat(SYSTEM_PROMPT, p["prompt"], max_new_tokens)
            rec = {"id": p["id"], "part": part, "teacher": teacher, "revision": revision,
                   "teacher_tag": tag, "response": text,
                   "finish": "length" if reason == "length" else "eos",
                   "new_tokens": tokens}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_tok += tokens
                n_new += 1
                if n_new % 200 == 0:
                    el = max(time.monotonic() - t0, 1e-6)
                    print(f"part {part}: {len(have) + n_new}/{len(prompts)}, "
                          f"{n_tok / el:.0f} tok/s", flush=True)

        it = iter(todo)
        running: set = set()
        last_budget_check = None
        while True:
            if not stop and pause.exists():
                stop = "paused"
            if not stop and (last_budget_check is None
                             or time.monotonic() - last_budget_check >= BUDGET_EVERY):
                last_budget_check = time.monotonic()
                try:
                    over = over_budget()
                except Exception as e:  # fail closed: the box is billing
                    print(f"budget check failed, stopping: {e!r}", flush=True)
                    over = True
                if over:
                    stop = "budget"
            while not stop and len(running) < concurrency:
                p = next(it, None)
                if p is None:
                    break
                running.add(ex.submit(one, p))
            if not running:
                break
            finished, running = wait(running, timeout=WAIT_TIMEOUT,
                                     return_when=FIRST_COMPLETED)
            for fut in finished:
                err = fut.exception()
                if err is None:
                    streak = 0
                    continue
                failed += 1
                streak += 1
                if failed <= SHOW_FAILURES:
                    print(f"request failed: {err!r}", flush=True)
            if not stop and streak >= max_streak:
                print(f"part {part}: {streak} requests failed in a row; "
                      f"server looks down, stopping", flush=True)
                stop = "error"

    if failed:
        print(f"part {part}: {failed} request(s) failed in total", flush=True)
    if stop:
        return stop
    missing = {p["id"] for p in prompts} - done_ids(out)
    if failed or missing:
        return "incomplete"
    done_file.write_text(json.dumps({"rows": len(prompts), "sha256": _sha256(out)}) + "\n",
                         encoding="utf-8")
    return "done"


def make_chat(endpoint: str, model: str = "teacher", timeout: float = 600.0,
              retries: int = 3, backoff: float = 5.0) -> Chat:
    """A chat() for an OpenAI-compatible server (vLLM, llama-server).
    Retries connection trouble, timeouts and HTTP 5xx; a 4xx is raised at once."""
    import http.client
    import urllib.error
    import urllib.request

    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def chat(system: str, user: str, max_tokens: int) -> tuple[str, str, int]:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                c = d["choices"][0]
                return (c["message"]["content"] or "", c.get("finish_reason") or "stop",
                        int(d.get("usage", {}).get("completion_tokens", 0)))
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.RemoteDisconnected):
                if attempt == retries - 1:
                    raise
            time.sleep(backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    return chat


def spend_over(budget: float) -> Callable[[], bool]:
    """True when quipu.spend says the cap is reached; fails closed."""
    import subprocess

    def check() -> bool:
        r = subprocess.run([sys.executable, "-m", "quipu.spend", "show", "--usd-only"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print("spend ledger unreadable: stopping", flush=True)
            return True
        return float(r.stdout.strip()) >= budget

    return check


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/distill-v2")
    ap.add_argument("--parts", required=True, help="START:END inclusive, e.g. 0:4")
    ap.add_argument("--direction", choices=("up", "down"), default="up")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--teacher-tag", required=True, help="e.g. bf16-vllm, q4km-llamacpp")
    ap.add_argument("--revision", required=True, help="teacher weights revision (sha)")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--budget-usd", type=float, default=None, help="box only")
    args = ap.parse_args(argv)

    root = ROOT / args.root
    chat = make_chat(args.endpoint)
    over = spend_over(args.budget_usd) if args.budget_usd is not None else (lambda: False)
    for part in part_order(args.parts, args.direction):
        if is_done(root, part):
            continue
        if not (root / "prompts" / f"part-{part:04d}.jsonl").exists():
            continue
        print(f"part {part}: start", flush=True)
        status = run_part(root, part, args.teacher_tag, chat, concurrency=args.concurrency,
                          max_new_tokens=args.max_new_tokens, teacher=TEACHER,
                          revision=args.revision, over_budget=over)
        print(f"part {part}: {status}", flush=True)
        if status in ("paused", "budget"):
            return 0
        if status == "incomplete":
            return 3  # rerun to retry the failed requests
        if status == "error":
            return 4  # server looks down; do not move on to the next part
    print("all parts in range done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
