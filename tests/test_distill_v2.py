"""Distillation v2: prompt parts, generator, verifier (no network, no GPU)."""
import json
from pathlib import Path

from quipu.sandbox import extract_code

STYLED = (
    "We need the sum of the even numbers in a list; one pass with a filter does it.\n\n"
    "1. Walk the list.\n2. Keep values divisible by 2.\n3. Add them up.\n\n"
    "```python\ndef sum_even(xs):\n    \"\"\"Sum of the even values in xs.\"\"\"\n"
    "    return sum(x for x in xs if x % 2 == 0)\n```\n\n"
    "Example:\n```python\nprint(sum_even([1, 2, 4]))\n```\n\n"
    "Edge cases: an empty list gives 0; negatives work; O(n) time, O(1) space."
)


def test_extract_code_explain_first_reply():
    code = extract_code(STYLED, "sum_even")
    assert code.startswith("def sum_even(xs):")
    assert "print(" not in code  # the usage block is not taken


from scripts.build_distill_prompts_v2 import PartWriter, take_unique


def test_part_writer_rolls_parts_and_ids(tmp_path):
    w = PartWriter(tmp_path, part_size=3)
    for i in range(7):
        w.add({"source": "opencodeinstruct", "source_id": f"s{i}", "licence": "CC-BY-4.0",
               "kind": "function", "prompt": f"p{i}", "entry_point": "f", "tests": ["assert f()"]})
    parts = w.close()
    assert [p["rows"] for p in parts] == [3, 3, 1]
    rows = [json.loads(l) for l in open(tmp_path / "part-0001.jsonl", encoding="utf-8")]
    assert [r["id"] for r in rows] == ["dv2-0000003", "dv2-0000004", "dv2-0000005"]
    assert all(r["part"] == 1 for r in rows)
    assert rows[0]["split"] == "train"
    first = json.loads(open(tmp_path / "part-0000.jsonl", encoding="utf-8").readline())
    assert first["split"] == "val"  # every 20th prompt (index % 20 == 0) is val
    assert len(parts[0]["sha256"]) == 64


def test_take_unique_drops_seen_and_contaminated():
    from collections import Counter

    class Fake:
        def find(self, text):
            return ("mbpp", "1", "substring") if "LEAK" in text else None

    seen, drops, hits = set(), Counter(), Counter()
    probs = [{"source": "opencodeinstruct", "problem": t, "prompt": t, "tests": ["assert f()"]}
             for t in ("a  b", "a b", "LEAK", "c")]
    kept = list(take_unique(probs, Fake(), seen, drops, hits))
    assert [p["problem"] for p in kept] == ["a  b", "c"]
    assert drops["opencodeinstruct:duplicate"] == 1 and hits["mbpp"] == 1
    assert len(seen) == 2 and all(isinstance(k, bytes) for k in seen)  # hashes, not text


def test_publish_replaces_old_out_wholesale(tmp_path):
    from scripts.build_distill_prompts_v2 import publish
    out, tmp = tmp_path / "prompts", tmp_path / "prompts.tmp"
    out.mkdir()
    (out / "part-0009.jsonl").write_text("old")
    (out / "part-0000.jsonl").write_text("old")
    (out / "manifest.json").write_text("old")
    tmp.mkdir()
    (tmp / "part-0000.jsonl").write_text("new")
    (tmp / "manifest.json").write_text("new")
    publish(tmp, out)
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "part-0000.jsonl"]
    assert (out / "part-0000.jsonl").read_text() == "new"
    assert not tmp.exists()


from scripts.distill_generate_v2 import SYSTEM_PROMPT, part_order, run_part


def _prompts(tmp_path, n=5, part=0):
    d = tmp_path / "prompts"
    d.mkdir(exist_ok=True)
    with open(d / f"part-{part:04d}.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"dv2-{i:07d}", "part": part, "prompt": f"p{i}"}) + "\n")
    return d


def fake_chat(system, user, max_tokens):
    assert system == SYSTEM_PROMPT
    return f"answer to {user}", "stop", 7


def test_part_order_ranges():
    assert part_order("0:4", "up") == [0, 1, 2, 3, 4]
    assert part_order("5:8", "down") == [8, 7, 6, 5]


def test_run_part_writes_all_then_done(tmp_path):
    _prompts(tmp_path)
    gen = tmp_path / "generations"
    status = run_part(tmp_path, 0, "bf16-vllm", fake_chat, concurrency=2,
                      max_new_tokens=64, teacher="T", revision="R")
    assert status == "done"
    rows = [json.loads(l) for l in open(gen / "part-0000.bf16-vllm.jsonl", encoding="utf-8")]
    assert sorted(r["id"] for r in rows) == [f"dv2-{i:07d}" for i in range(5)]
    assert rows[0]["finish"] == "eos" and rows[0]["new_tokens"] == 7
    assert rows[0]["teacher_tag"] == "bf16-vllm" and rows[0]["teacher"] == "T"
    done = json.loads((gen / "part-0000.bf16-vllm.done").read_text())
    assert done["rows"] == 5 and len(done["sha256"]) == 64


def test_run_part_resumes_and_skips_done_ids(tmp_path):
    _prompts(tmp_path)
    gen = tmp_path / "generations"
    gen.mkdir()
    (gen / "part-0000.q4.jsonl").write_text(
        json.dumps({"id": "dv2-0000000", "response": "old"}) + "\n" + '{"id": "dv2-00', encoding="utf-8")
    asked = []

    def chat(system, user, max_tokens):
        asked.append(user)
        return "x", "length", 64

    assert run_part(tmp_path, 0, "q4", chat, concurrency=1, max_new_tokens=64,
                    teacher="T", revision="R") == "done"
    assert "p0" not in asked and len(asked) == 4
    rows = [json.loads(l) for l in open(gen / "part-0000.q4.jsonl", encoding="utf-8")
            if l.strip().endswith("}")]
    assert len({r["id"] for r in rows}) == 5
    assert any(r.get("finish") == "length" for r in rows)


def test_pause_file_stops_cleanly(tmp_path):
    _prompts(tmp_path, n=50)
    calls = []

    def chat(system, user, max_tokens):
        calls.append(user)
        if len(calls) == 3:
            (tmp_path / "PAUSE").write_text("")
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", chat, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "paused"
    assert not (tmp_path / "generations" / "part-0000.q4.done").exists()
    assert len(calls) < 50


def test_failed_request_is_retried_next_run(tmp_path):
    _prompts(tmp_path, n=3)

    def flaky(system, user, max_tokens):
        if user == "p1":
            raise OSError("server down")
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", flaky, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "incomplete"
    assert run_part(tmp_path, 0, "q4", fake_chat_any, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "done"


def fake_chat_any(system, user, max_tokens):
    return "x", "stop", 1


def test_budget_stop(tmp_path):
    _prompts(tmp_path, n=20)
    assert run_part(tmp_path, 0, "bf16-vllm", fake_chat_any, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R", over_budget=lambda: True) == "budget"


def test_pause_writes_in_flight_answers(tmp_path):
    import threading
    _prompts(tmp_path, n=40)
    calls = []
    lock = threading.Lock()
    release = threading.Event()

    def chat(system, user, max_tokens):
        with lock:
            calls.append(user)
            if len(calls) == 1:
                (tmp_path / "PAUSE").write_text("")
            if len(calls) == 4:
                release.set()
        release.wait(5)  # hold every request until all four are in flight
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", chat, concurrency=4, max_new_tokens=8,
                    teacher="T", revision="R") == "paused"
    rows = [json.loads(l) for l in
            open(tmp_path / "generations" / "part-0000.q4.jsonl", encoding="utf-8")]
    assert len(calls) == 4 and len(rows) == 4
