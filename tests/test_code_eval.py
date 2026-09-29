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

def _ideal(row: dict, task) -> str:
    """What a perfect base model generates after the stripped prompt: the whitespace
    rstrip() removed, then the canonical solution."""
    return row["prompt"][len(task.prompt):] + row["canonical_solution"]


def test_humaneval_base_prompt_is_rstripped_bigcode_style():
    [task] = code_eval.humaneval_tasks([HE_ROW])
    assert task.prompt == HE_ROW["prompt"].rstrip()
    assert task.prompt.endswith('"""') and _ideal(HE_ROW, task).startswith("\n    return")
    # the chat instruction still shows the problem as published
    assert HE_ROW["prompt"] in task.instruction


def test_known_correct_completion_passes_and_wrong_one_fails():
    [task] = code_eval.humaneval_tasks([HE_ROW])
    ok, _ = code_eval.run_program(code_eval.build_program(task, _ideal(HE_ROW, task), False))
    bad, err = code_eval.run_program(code_eval.build_program(task, "\n    return len(xs)\n", False))
    assert ok == "passed"
    assert bad == "failed" and "AssertionError" in err


# ---- C1: the prompt's tokens must be a prefix of prompt + solution tokens -----------------

REAL_TOKENIZER = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
HE_REV = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
MBPP_REV = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"


@pytest.fixture(scope="module")
def real_tok():
    if not REAL_TOKENIZER.is_file():
        pytest.skip("artifacts/tokenizer/tokenizer.json is not here")
    from quipu.bpe import BPETokenizer
    return BPETokenizer(REAL_TOKENIZER)


def _cached_rows(benchmark: str, revision: str) -> list[dict]:
    """The benchmark's rows from the local Hugging Face cache, or skip (no network)."""
    import pyarrow.parquet as pq
    try:
        from huggingface_hub import hf_hub_download
        dataset, file, _lic, _n = code_eval.FILES[benchmark]
        path = hf_hub_download(dataset, file, repo_type="dataset", revision=revision,
                               local_files_only=True)
    except Exception as exc:  # not cached (or hub not installed): nothing to check
        pytest.skip(f"{benchmark} at {revision[:12]} is not in the HF cache ({type(exc).__name__})")
    return pq.read_table(path).to_pylist()


def _prefix_failures(tok, tasks, rows, solution_key: str, stripped_of_prompt: bool) -> list[str]:
    bad = []
    for task, row in zip(tasks, rows):
        tail = row["prompt"][len(task.prompt):] if stripped_of_prompt else ""
        p, full = tok.encode(task.prompt), tok.encode(task.prompt + tail + row[solution_key])
        if full[: len(p)] != p:
            bad.append(task.task_id)
    return bad


def test_raw_humaneval_prompt_is_not_a_token_prefix_but_the_stripped_one_is(real_tok):
    # The byte-level pre-tokeniser merges "\n" + the next line's indentation into one
    # token, so a prompt ending in "\n" ends on a token the solution never produces.
    raw = real_tok.encode(HE_ROW["prompt"])
    assert real_tok.encode(HE_ROW["prompt"] + HE_ROW["canonical_solution"])[: len(raw)] != raw
    tasks = code_eval.humaneval_tasks([HE_ROW])
    assert _prefix_failures(real_tok, tasks, [HE_ROW], "canonical_solution", True) == []


def test_every_humaneval_problem_is_a_token_prefix_and_passes_stripped(real_tok):
    from concurrent.futures import ThreadPoolExecutor

    rows = _cached_rows("humaneval", HE_REV)
    assert len(rows) == 164
    tasks = code_eval.humaneval_tasks(rows)
    assert _prefix_failures(real_tok, tasks, rows, "canonical_solution", True) == []
    # The raw (unstripped) prompt broke the prefix property on every problem; that is
    # what the fix is for.
    raw_bad = sum(real_tok.encode(r["prompt"] + r["canonical_solution"])[
        : len(real_tok.encode(r["prompt"]))] != real_tok.encode(r["prompt"]) for r in rows)
    assert raw_bad == 164
    progs = [code_eval.build_program(t, _ideal(r, t), False) for t, r in zip(tasks, rows)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda p: code_eval.run_program(p, 20.0), progs))
    failed = [(t.task_id, r) for t, r in zip(tasks, results) if r[0] != "passed"]
    assert failed == []


def test_mbpp_prompt_is_a_token_prefix(real_tok):
    tasks = code_eval.mbpp_tasks([MBPP_ROW])
    assert _prefix_failures(real_tok, tasks, [MBPP_ROW], "code", False) == []
    rows = _cached_rows("mbpp", MBPP_REV)
    assert len(rows) == 257
    assert _prefix_failures(real_tok, code_eval.mbpp_tasks(rows), rows, "code", False) == []


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


# ---- I5: the reliability guard ------------------------------------------------------------

def test_candidate_cannot_delete_files(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    for call in ("os.remove", "os.unlink", "shutil.rmtree", "os.rename"):
        args = f"({str(victim)!r}, 'y')" if call == "os.rename" else f"({str(victim)!r})"
        prog = f"import os, shutil\n{call}{args}\n"
        assert code_eval.run_program(prog)[0] == "failed", call
    assert victim.read_text() == "x"


def test_candidate_cannot_write_outside_its_temp_dir(tmp_path):
    target = tmp_path / "written.txt"
    for prog in (f"open({str(target)!r}, 'w').write('x')\n",
                 f"import pathlib\npathlib.Path({str(target)!r}).write_text('x')\n",
                 f"import os\nos.open({str(target)!r}, os.O_WRONLY | os.O_CREAT)\n"):
        assert code_eval.run_program(prog)[0] == "failed", prog
    assert not target.exists()
    # inside the temp dir, and reading anywhere, still work
    ok = ("open('scratch.txt', 'w').write('hi')\nassert open('scratch.txt').read() == 'hi'\n"
          f"import os\nassert os.path.isfile({str(ROOT / 'pyproject.toml')!r})\n")
    assert code_eval.run_program(ok)[0] == "passed"


def test_candidate_cannot_start_processes_or_exit_cleanly():
    assert code_eval.run_program("import subprocess\nsubprocess.run(['python', '-V'])\n")[0] == "failed"
    assert code_eval.run_program("import os\nos.system('echo hi')\n")[0] == "failed"
    # exit()/quit() are disabled: calling them is an error, not a pass
    assert code_eval.run_program("exit(0)\nraise SystemExit(1)\n")[0] == "failed"


def test_network_libraries_import_but_connect_raises():
    prog = ("import ssl, asyncio, urllib.request, http.client, socket\n"
            "s = socket.socket()\n"
            "for f, a in ((s.connect, (('127.0.0.1', 9),)), (s.connect_ex, (('127.0.0.1', 9),)),\n"
            "             (s.sendto, (b'x', ('127.0.0.1', 9)))):\n"
            "    try:\n        f(*a)\n    except OSError as e:\n"
            "        assert 'disabled' in str(e), e\n"
            "    else:\n        raise SystemExit(3)\n"
            "assert isinstance(s, socket.socket)\n")
    status, err = code_eval.run_program(prog)
    assert status == "passed", err


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


def test_chat_prompt_is_the_user_turn_as_plain_text(tmp_path, monkeypatch):
    cfg, tok = tiny_moe(tmp_path)
    task = code_eval.humaneval_tasks([HE_ROW])[0]
    task = code_eval.dataclasses.replace(
        task, instruction=task.instruction + "\n=== FILE: a.py ===\nx = 1\n=== END FILE ===\n")
    seen: list[list[int]] = []

    def fake(model, ids, n, **kw):
        seen.append(list(ids))
        return [""] * n

    monkeypatch.setattr(code_eval, "generate_batch", fake)
    code_eval.evaluate(None, tok, [task], chat=True, samples=0, log=lambda _m: None)
    sid = tok.special_id
    assert seen == [[sid("<|user|>")] + tok.encode(task.instruction)
                    + [sid("<|end|>"), sid("<|assistant|>")]]
    assert not {sid("=== FILE: "), sid("=== END FILE ===")} & set(seen[0])


def _row(table: str, *needles: str) -> list[str]:
    line = next(l for l in table.splitlines() if all(n in l for n in needles))
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_report_cites_published_numbers_as_not_re_run_with_their_settings():
    table = code_eval.results_table([{"model": "quipu-moe-1B-A149M", "params": "1.0B total / 149M active",
                                      "results": {"humaneval": {"greedy_pass@1": 1.2, "pass@1": 0.9,
                                                                "pass@10": 3.4, "samples": 20,
                                                                "temperature": 0.8, "top_p": 0.95}}}])
    header = [c.strip() for c in table.splitlines()[0].strip().strip("|").split("|")]
    assert header[:3] == ["model", "params", "metric / setting"]
    greedy = _row(table, "**quipu-moe-1B-A149M**", "greedy")
    sampled = _row(table, "**quipu-moe-1B-A149M**", "sampled")
    assert greedy[3:5] == ["1.2", "-"]              # greedy: pass@1 only
    assert sampled[3:5] == ["0.9", "3.4"]           # sampled pass@1 shown next to greedy
    assert "n=20" in sampled[2] and "T=0.8" in sampled[2] and "HumanEval 164" in sampled[2]
    assert "MBPP sanitized test (257)" in greedy[2]
    for name in ["CodeGen-350M-mono", "SantaCoder-1.1B", "SmolLM-135M", "SmolLM-360M",
                 "Qwen2.5-Coder-1.5B"]:
        row = _row(table, name)
        assert "published, not re-run" in row[-1] and row[2], name
    santa = _row(table, "SantaCoder")[2]
    assert "T=0.2" in santa and "T=0.8" in santa and "n=200" in santa and "MultiPL-E" in santa
    assert "unverified setting" in _row(table, "Qwen2.5-Coder")[2]
    assert "0.2, 0.6, 0.8" in _row(table, "CodeGen-350M")[2]
    assert "arXiv:2203.13474" in table and "arXiv:2409.12186" in table


def test_report_md_states_prompt_formats_stops_and_sandbox_limits():
    payload = {"chat": False, "checkpoint": "c.pt", "benchmarks": {},
               "results": {}, "protocol": code_eval.protocol(chat=False, samples=20)}
    md = code_eval.report_md("m", payload)
    for stop in code_eval.STOP_HUMANEVAL + code_eval.STOP_MBPP:
        assert repr(stop) in md, stop
    assert "prompt.rstrip()" in md and "first assert" in md and "docstring" in md
    assert "best effort, not a security boundary" in md


def test_revision_cross_check_against_the_manifest(tmp_path):
    import json
    man = tmp_path / "manifest.json"
    revs = {"humaneval": "a" * 40, "mbpp": "b" * 40}
    man.write_text(json.dumps({"decontamination": {"benchmarks": {
        "humaneval": {"revision": "a" * 40}, "mbpp": {"revision": "c" * 40}}}}))
    with pytest.raises(ValueError, match="mbpp"):
        code_eval.check_revisions(revs, man)
    man.write_text(json.dumps({"decontamination": {"benchmarks": {
        "humaneval": {"revision": "a" * 40}, "mbpp": {"revision": "b" * 40}}}}))
    code_eval.check_revisions(revs, man)                  # agrees: no error
    code_eval.check_revisions(revs, tmp_path / "missing.json")   # no manifest: nothing to check


def test_default_model_name_is_the_release_name(tmp_path):
    cfg, _ = tiny_moe(tmp_path)
    from quipu import model_card
    want = model_card.release_name(*model_card.count_params(cfg.model))
    assert code_eval.default_model_name(cfg, chat=False) == want
    assert code_eval.default_model_name(cfg, chat=True) == want + "-chat"
