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


def test_dead_server_stops_with_error(tmp_path):
    import threading
    _prompts(tmp_path, n=100)
    calls = []
    lock = threading.Lock()

    def dead(system, user, max_tokens):
        with lock:
            calls.append(user)
        raise OSError("connection refused")

    assert run_part(tmp_path, 0, "q4", dead, concurrency=2, max_new_tokens=8,
                    teacher="T", revision="R") == "error"
    assert len(calls) < 40
    assert not (tmp_path / "generations" / "part-0000.q4.done").exists()


def test_failures_interleaved_with_successes_do_not_trip_error(tmp_path):
    _prompts(tmp_path, n=60)

    def half(system, user, max_tokens):
        if int(user[1:]) % 2:
            raise OSError("flaky")
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", half, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "incomplete"


def test_budget_checked_while_requests_hang(tmp_path, monkeypatch):
    import threading
    import time as _time
    import scripts.distill_generate_v2 as g
    monkeypatch.setattr(g, "WAIT_TIMEOUT", 0.01)
    monkeypatch.setattr(g, "BUDGET_EVERY", 0.0)
    _prompts(tmp_path, n=5)
    release = threading.Event()
    checks = []

    def hang(system, user, max_tokens):
        release.wait(3)
        return "x", "stop", 1

    def over():
        checks.append(1)
        if len(checks) >= 3:
            release.set()
            return True
        return False

    t = _time.monotonic()
    assert run_part(tmp_path, 0, "q4", hang, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R", over_budget=over) == "budget"
    assert _time.monotonic() - t < 1.5  # checked during the hang, not after it


def test_budget_callback_error_fails_closed(tmp_path):
    _prompts(tmp_path, n=5)

    def broken():
        raise RuntimeError("spend api down")

    assert run_part(tmp_path, 0, "q4", fake_chat_any, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R", over_budget=broken) == "budget"


def test_complete_unterminated_last_row_is_kept(tmp_path):
    _prompts(tmp_path, n=3)
    gen = tmp_path / "generations"
    gen.mkdir()
    (gen / "part-0000.q4.jsonl").write_text(
        json.dumps({"id": "dv2-0000000", "response": "old"}), encoding="utf-8")
    asked = []

    def chat(system, user, max_tokens):
        asked.append(user)
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", chat, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "done"
    assert sorted(asked) == ["p1", "p2"]
    rows = [json.loads(l) for l in open(gen / "part-0000.q4.jsonl", encoding="utf-8")]
    assert [r["id"] for r in rows][0] == "dv2-0000000" and len(rows) == 3


def test_resume_survives_bad_utf8_mid_file(tmp_path):
    _prompts(tmp_path, n=3)
    gen = tmp_path / "generations"
    gen.mkdir()
    (gen / "part-0000.q4.jsonl").write_bytes(
        b'{"id": "dv2-0000000", "response": "a"}\n\xff\xfe junk\n'
        b'{"id": "dv2-0000001", "response": "b"}\n')
    asked = []

    def chat(system, user, max_tokens):
        asked.append(user)
        return "x", "stop", 1

    assert run_part(tmp_path, 0, "q4", chat, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "done"
    assert asked == ["p2"]


def test_completeness_is_by_id_not_count(tmp_path):
    # a duplicated prompt id must not leave the part "incomplete" forever
    d = tmp_path / "prompts"
    d.mkdir()
    rows = [{"id": "dv2-0000000", "prompt": "p0"}, {"id": "dv2-0000001", "prompt": "p1"},
            {"id": "dv2-0000001", "prompt": "p1"}]
    (d / "part-0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                       encoding="utf-8")
    assert run_part(tmp_path, 0, "q4", fake_chat_any, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "done"
    assert (tmp_path / "generations" / "part-0000.q4.done").exists()
    assert run_part(tmp_path, 0, "q4", fake_chat_any, concurrency=1, max_new_tokens=8,
                    teacher="T", revision="R") == "done"


import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scripts.distill_generate_v2 import make_chat


def _serve(statuses, finish="length"):
    """Fake OpenAI server: answers with each status in turn (200 = a real reply)."""
    seen = {"calls": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            seen["body"] = json.loads(self.rfile.read(n))
            code = statuses[min(seen["calls"], len(statuses) - 1)]
            seen["calls"] += 1
            body = json.dumps({"choices": [{"message": {"content": "hi"},
                                            "finish_reason": finish}],
                               "usage": {"completion_tokens": 9}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_make_chat_against_fake_openai_server():
    srv, seen = _serve([200])
    chat = make_chat(f"http://127.0.0.1:{srv.server_port}")
    assert chat("SYS", "USER", 32) == ("hi", "length", 9)
    b = seen["body"]
    assert b["temperature"] == 0 and b["max_tokens"] == 32 and b["model"] == "teacher"
    assert b["messages"] == [{"role": "system", "content": "SYS"},
                             {"role": "user", "content": "USER"}]
    srv.shutdown()


def test_make_chat_retries_503_then_succeeds():
    srv, seen = _serve([503, 200])
    chat = make_chat(f"http://127.0.0.1:{srv.server_port}", backoff=0)
    assert chat("S", "U", 8) == ("hi", "length", 9)
    assert seen["calls"] == 2
    srv.shutdown()


def test_make_chat_does_not_retry_4xx():
    import urllib.error
    srv, seen = _serve([400, 200])
    chat = make_chat(f"http://127.0.0.1:{srv.server_port}", backoff=0)
    with pytest.raises(urllib.error.HTTPError):
        chat("S", "U", 8)
    assert seen["calls"] == 1
    srv.shutdown()


def test_make_chat_abort_finish_reason_raises_without_retry():
    srv, seen = _serve([200], finish="abort")
    chat = make_chat(f"http://127.0.0.1:{srv.server_port}", backoff=0)
    with pytest.raises(RuntimeError, match="finish_reason=abort"):
        chat("S", "U", 8)
    assert seen["calls"] == 1
    srv.shutdown()


def test_main_empty_range_returns_2(tmp_path, capsys):
    from scripts.distill_generate_v2 import main
    rc = main(["--root", str(tmp_path), "--parts", "0:2", "--teacher-tag", "t",
               "--revision", "r"])
    assert rc == 2
    assert "no prompt file, skipping" in capsys.readouterr().out


from scripts.distill_verify import TAG_PRIORITY, verify_part, verify_parts


def _setup_part(tmp_path, gens_by_tag):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "generations").mkdir()
    with open(tmp_path / "prompts" / "part-0000.jsonl", "w", encoding="utf-8") as f:
        for i in range(2):
            f.write(json.dumps({"id": f"dv2-{i:07d}", "part": 0, "split": "train",
                                "source": "opencodeinstruct", "prompt": f"add two numbers {i}",
                                "entry_point": "add",
                                "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}) + "\n")
    for tag, responses in gens_by_tag.items():
        p = tmp_path / "generations" / f"part-0000.{tag}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i, resp in enumerate(responses):
                f.write(json.dumps({"id": f"dv2-{i:07d}", "teacher_tag": tag, "teacher": "T",
                                    "response": resp, "finish": "eos"}) + "\n")
        (tmp_path / "generations" / f"part-0000.{tag}.done").write_text(
            json.dumps({"rows": 2, "sha256": "x"}))


GOOD = "Plan.\n```python\ndef add(a, b):\n    return a + b\n```\nEdge cases: none."
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_verify_part_keeps_passing_answers(tmp_path):
    _setup_part(tmp_path, {"bf16-vllm": [GOOD, BAD]})
    s = verify_part(tmp_path, 0, workers=2, timeout=10)
    assert s == {"answers": 2, "kept": 1}
    kept = [json.loads(l) for l in open(tmp_path / "kept" / "part-0000.jsonl", encoding="utf-8")]
    assert [k["id"] for k in kept] == ["dv2-0000000"]
    assert kept[0]["response"] == GOOD and kept[0]["prompt"] == "add two numbers 0"
    assert (tmp_path / "verified" / "part-0000.meta.json").exists()


def test_verify_part_prefers_bf16_over_laptop(tmp_path):
    assert TAG_PRIORITY[0] == "bf16-vllm"
    _setup_part(tmp_path, {"q4km-llamacpp": [BAD, GOOD], "bf16-vllm": [GOOD, BAD]})
    verify_part(tmp_path, 0, workers=2, timeout=10)
    rows = [json.loads(l) for l in open(tmp_path / "verified" / "part-0000.jsonl", encoding="utf-8")]
    assert [r["teacher_tag"] for r in rows] == ["bf16-vllm", "bf16-vllm"]


def test_verify_part_skips_when_up_to_date(tmp_path):
    _setup_part(tmp_path, {"bf16-vllm": [GOOD, GOOD]})
    assert verify_part(tmp_path, 0, workers=1, timeout=10) == {"answers": 2, "kept": 2}
    assert verify_part(tmp_path, 0, workers=1, timeout=10) is None  # nothing changed


def test_verify_part_ignores_parts_without_done(tmp_path):
    _setup_part(tmp_path, {"bf16-vllm": [GOOD, GOOD]})
    (tmp_path / "generations" / "part-0000.bf16-vllm.done").unlink()
    assert verify_part(tmp_path, 0, workers=1, timeout=10) is None


def test_verify_part_duplicate_ids_first_wins(tmp_path):
    _setup_part(tmp_path, {"bf16-vllm": [GOOD, GOOD]})
    with open(tmp_path / "generations" / "part-0000.bf16-vllm.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "dv2-0000000", "teacher_tag": "bf16-vllm", "teacher": "T",
                            "response": BAD, "finish": "eos"}) + "\n")
    assert verify_part(tmp_path, 0, workers=1, timeout=10) == {"answers": 2, "kept": 2}


def test_verify_parts_without_done_parts_does_not_crash(tmp_path):
    _setup_part(tmp_path, {"bf16-vllm": [GOOD, GOOD]})
    (tmp_path / "generations" / "part-0000.bf16-vllm.done").unlink()
    s = verify_parts(tmp_path, workers=1, timeout=10)
    assert s["answers"] == 0 and s["kept"] == 0
    assert (tmp_path / "verify_summary.json").exists()
