"""M10: scripts/code_eval.py (HumanEval / MBPP harness). Fixtures only, no network."""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import pytest

from tests.moe_fixtures import fresh_model, tiny_moe

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("code_eval", ROOT / "scripts" / "code_eval.py")
code_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["code_eval"] = code_eval
_SPEC.loader.exec_module(code_eval)

HE_ROW = {
    "task_id": "Fixture/0",
    "prompt": 'from typing import List\n\n\ndef add_all(xs: List[int]) -> int:\n'
              '    """Return the sum of xs.\n    >>> add_all([1, 2])\n    3\n    """\n',
    "canonical_solution": "    return sum(xs)\n",
    "test": "def check(candidate):\n    assert candidate([1, 2]) == 3\n"
            "    assert candidate([]) == 0\n    assert candidate([5, -5, 7]) == 7\n",
    "entry_point": "add_all",
}
MBPP_ROW = {
    "task_id": 900, "prompt": "Write a function to square a number.",
    "code": "def square(x):\n    return x * x\n",
    "test_imports": ["import math"],
    "test_list": ["assert square(3) == 9", "assert square(-2) == 4", "assert math.isclose(square(0.5), 0.25)"],
}


# ---- pass@k -----------------------------------------------------------------------------

@pytest.mark.parametrize("n,c,k,want", [
    (20, 0, 1, 0.0), (20, 20, 1, 1.0), (20, 1, 1, 0.05), (20, 5, 1, 0.25),
    (20, 1, 10, 0.5),                               # 1 - C(19,10)/C(20,10) = 1 - 10/20
    (20, 2, 10, 1 - (10 * 9) / (20 * 19)),          # 1 - C(18,10)/C(20,10)
    (5, 2, 4, 1.0),                                 # n - c < k
    (10, 3, 2, 1 - (7 * 6) / (10 * 9)),
])
def test_pass_at_k_matches_hand_values(n, c, k, want):
    assert code_eval.pass_at_k(n, c, k) == pytest.approx(want, abs=1e-12)
    assert code_eval.pass_at_k(n, c, k) == pytest.approx(
        1 - math.comb(n - c, k) / math.comb(n, k), abs=1e-12)


def test_pass_at_k_rejects_bad_counts():
    with pytest.raises(ValueError):
        code_eval.pass_at_k(5, 6, 1)
    with pytest.raises(ValueError):
        code_eval.pass_at_k(5, 1, 6)


def test_summary_averages_the_estimator_over_problems():
    per = [{"greedy": "passed", "n": 20, "c": 20}, {"greedy": "failed", "n": 20, "c": 1}]
    s = code_eval.summarise(per, 20)
    assert s["greedy_pass@1"] == 50.0
    assert s["pass@1"] == pytest.approx(100 * (1.0 + 0.05) / 2)
    assert s["pass@10"] == pytest.approx(100 * (1.0 + 0.5) / 2)
    assert code_eval.summarise(per, 0)["pass@10"] is None


# ---- programs and execution ---------------------------------------------------------------

def test_known_correct_completion_passes_and_wrong_one_fails():
    [task] = code_eval.humaneval_tasks([HE_ROW])
    ok, _ = code_eval.run_program(code_eval.build_program(task, "    return sum(xs)\n", False))
    bad, err = code_eval.run_program(code_eval.build_program(task, "    return len(xs)\n", False))
    assert ok == "passed"
    assert bad == "failed" and "AssertionError" in err


def test_mbpp_known_correct_and_wrong_with_test_imports():
    [task] = code_eval.mbpp_tasks([MBPP_ROW])
    assert task.prompt == ('"""\nWrite a function to square a number.\n'
                           'assert square(3) == 9\n"""\n')
    assert code_eval.run_program(code_eval.build_program(task, MBPP_ROW["code"], False))[0] == "passed"
    assert code_eval.run_program(
        code_eval.build_program(task, "def square(x):\n    return 2 * x\n", False))[0] == "failed"
    # chat: code in a fence, imports added by the harness
    reply = "Here you go:\n```python\ndef square(x):\n    return x ** 2\n```\nDone."
    assert code_eval.run_program(code_eval.build_program(task, reply, True))[0] == "passed"


def test_chat_reply_with_the_whole_function_or_only_the_body():
    [task] = code_eval.humaneval_tasks([HE_ROW])
    whole = "```python\ndef add_all(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n```"
    body = "    return sum(xs)\n"
    assert code_eval.run_program(code_eval.build_program(task, whole, True))[0] == "passed"
    assert code_eval.run_program(code_eval.build_program(task, body, True))[0] == "passed"


def test_timeout_kills_an_infinite_loop():
    t0 = time.perf_counter()
    status, detail = code_eval.run_program("while True:\n    pass\n", timeout=2.0)
    assert status == "timeout" and "killed" in detail
    assert time.perf_counter() - t0 < 15


def test_network_is_disabled_in_the_candidate():
    prog = ("import socket\ntry:\n    socket.create_connection(('example.com', 80), timeout=1)\n"
            "except OSError as e:\n    assert 'disabled' in str(e)\nelse:\n    raise SystemExit(3)\n")
    assert code_eval.run_program(prog)[0] == "passed"


def test_candidate_runs_in_a_temp_dir_with_proxies_unreachable():
    prog = ("import os, sys\nassert 'quipu_code_eval_' in os.getcwd(), os.getcwd()\n"
            f"assert os.environ['HTTPS_PROXY'] == {code_eval.UNREACHABLE_PROXY!r}\n")
    assert code_eval.run_program(prog)[0] == "passed"


# ---- stop sequences ------------------------------------------------------------------------

@pytest.mark.parametrize("text,stops,want", [
    ("    return x\n\ndef other():\n    pass", code_eval.STOP_HUMANEVAL, "    return x\n"),
    ("    return x\nclass A:\n", code_eval.STOP_HUMANEVAL, "    return x"),
    ("    y = 1\n    return y\n\n\nif __name__ == '__main__':\n", code_eval.STOP_HUMANEVAL,
     "    y = 1\n    return y\n\n"),
    ("    return x\nprint(f(1))", code_eval.STOP_HUMANEVAL, "    return x"),
    ("    return x  # fine\n# a trailing comment", code_eval.STOP_HUMANEVAL, "    return x  # fine"),
    ("    return x\n", code_eval.STOP_HUMANEVAL, "    return x\n"),
    ("def f(x):\n    return g(x)\n\ndef g(x):\n    return x\nassert f(1) == 1",
     code_eval.STOP_MBPP, "def f(x):\n    return g(x)\n\ndef g(x):\n    return x"),
])
def test_stop_sequences_cut_correctly(text, stops, want):
    assert code_eval.cut_at_stop(text, stops) == want


def test_extract_code_prefers_the_first_fence():
    assert code_eval.extract_code("a\n```python\nx = 1\n```\n```python\ny\n```") == "x = 1\n"
    assert code_eval.extract_code("x = 2") == "x = 2"
    assert code_eval.extract_code("```\nz = 3\n") == "z = 3\n"


def test_load_tasks_needs_a_pinned_revision_and_the_full_problem_count(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with pytest.raises(ValueError, match="pinned"):
        code_eval.load_tasks("humaneval", "main")
    path = tmp_path / "he.parquet"
    pq.write_table(pa.Table.from_pylist([HE_ROW]), path)
    with pytest.raises(ValueError, match="expected 164"):
        code_eval.load_tasks("humaneval", "a" * 40, download=lambda *a, **k: str(path))


# ---- generation on a tiny model ------------------------------------------------------------

def test_generation_is_greedy_reproducible_and_stops(tmp_path):
    cfg, tok = tiny_moe(tmp_path, moe_dispatch="padded")
    model = fresh_model(cfg, 4)
    ids = tok.encode("def area(width, height):\n")
    a = code_eval.generate_batch(model, ids, 1, temperature=0.0, max_new_tokens=12, decode=tok.decode)
    b = code_eval.generate_batch(model, ids, 1, temperature=0.0, max_new_tokens=12, decode=tok.decode)
    assert a == b
    s1 = code_eval.generate_batch(model, ids, 3, temperature=0.8, max_new_tokens=12,
                                  decode=tok.decode, seed=5)
    s2 = code_eval.generate_batch(model, ids, 3, temperature=0.8, max_new_tokens=12,
                                  decode=tok.decode, seed=5)
    assert s1 == s2 and len(s1) == 3
    # every token id is a stop id: nothing is generated
    stop_all = code_eval.generate_batch(model, ids, 2, temperature=0.8, max_new_tokens=12,
                                        decode=tok.decode, stop_ids=range(cfg.model.vocab_size))
    assert stop_all == ["", ""]
    assert all(b.moe.dispatch == "padded" for b in model.blocks)


def test_evaluate_end_to_end_on_a_tiny_model(tmp_path):
    cfg, tok = tiny_moe(tmp_path)
    model = fresh_model(cfg, 5)
    tasks = code_eval.humaneval_tasks([HE_ROW]) + code_eval.mbpp_tasks([MBPP_ROW])
    res = code_eval.evaluate(model, tok, tasks, samples=2, max_new_tokens=6, workers=2,
                             log=lambda _m: None)
    assert res["problems"] == 2 and [p["n"] for p in res["per_problem"]] == [2, 2]
    assert res["greedy_pass@1"] == 0.0          # an untrained model writes nonsense
    assert res["pass@1"] == 0.0 and res["pass@10"] is None      # n = 2 < 10
    chat = code_eval.evaluate(model, tok, tasks[:1], chat=True, samples=0, max_new_tokens=4,
                              log=lambda _m: None)
    assert chat["pass@1"] is None and chat["problems"] == 1


def test_report_cites_published_numbers_as_not_re_run():
    table = code_eval.results_table([{"model": "quipu-moe-1B-A149M", "params": "1.0B total / 149M active",
                                      "results": {"humaneval": {"greedy_pass@1": 1.2, "pass@10": 3.4,
                                                                "samples": 20}}}])
    assert "**quipu-moe-1B-A149M**" in table and "| 1.2 | 3.4 |" in table
    for name in ["CodeGen-350M-mono", "SantaCoder-1.1B", "SmolLM-135M", "SmolLM-360M",
                 "Qwen2.5-Coder-1.5B"]:
        row = next(l for l in table.splitlines() if name in l)
        assert "published, not re-run" in row
    assert "arXiv:2203.13474" in table and "arXiv:2409.12186" in table
