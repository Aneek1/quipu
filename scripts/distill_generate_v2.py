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
--budget-usd stops it when quipu.spend reports the cap reached.

Output: generations/part-NNNN.<tag>.jsonl rows: id, part, teacher, revision,
teacher_tag, response, finish ("eos" | "length"), new_tokens; and, when every
prompt of the part is answered, part-NNNN.<tag>.done: {"rows", "sha256"}.
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
        with open(path, encoding="utf-8") as f:
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
        pos = size
        while pos > 0:  # scan back to the last newline
            step = min(1 << 16, pos)
            pos -= step
            f.seek(pos)
            nl = f.read(step).rfind(b"\n")
            if nl != -1:
                f.truncate(pos + nl + 1)
                return
        f.truncate(0)  # the only line was torn


def run_part(root: Path, part: int, tag: str, chat: Chat, *, concurrency: int,
             max_new_tokens: int, teacher: str, revision: str,
             over_budget: Callable[[], bool] = lambda: False) -> str:
    """Answer every unanswered prompt of one part.
    Returns "done" | "paused" | "budget" | "incomplete" (some requests failed)."""
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
    failed = 0
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
                             or time.monotonic() - last_budget_check > 60):
                last_budget_check = time.monotonic()
                if over_budget():
                    stop = "budget"
            while not stop and len(running) < concurrency:
                p = next(it, None)
                if p is None:
                    break
                running.add(ex.submit(one, p))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                if fut.exception() is not None:
                    failed += 1
                    print(f"request failed: {fut.exception()!r}", flush=True)

    if stop:
        return stop
    if failed or len(done_ids(out)) < len(prompts):
        return "incomplete"
    done_file.write_text(json.dumps({"rows": len(prompts), "sha256": _sha256(out)}) + "\n",
                         encoding="utf-8")
    return "done"
