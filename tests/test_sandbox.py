"""quipu.sandbox: code extraction and the isolated test runner (runs real subprocesses)."""
import os

from quipu.sandbox import child_env, extract_code, run_tests

ADD = "def add(a, b):\n    return a + b\n"


def test_extract_prefers_block_defining_entry_point():
    reply = "Here:\n```python\nimport math\n```\nand\n```python\n" + ADD + "```\nDone."
    assert extract_code(reply, "add") == ADD
    assert extract_code("```\n" + ADD + "```", "add") == ADD
    assert extract_code(ADD, "add") == ADD          # bare code, no fence
    assert extract_code("no code here", "add") is None
    assert extract_code("```python\ndef other(): pass\n```", "add") is None
    assert extract_code("```python\ndef add_all(): pass\n```", "add") is None  # whole name only
    cls = "class Calc:\n    def add(self, a, b):\n        return a + b\n"
    assert extract_code("```python\n" + cls + "```", "Calc") == cls


def test_run_tests_all_pass_and_partial():
    r = run_tests(ADD, ["assert add(1, 2) == 3", "assert add(2, 2) == 4"])
    assert r.all_passed and r.passed == 2 and r.error == ""
    r = run_tests(ADD, ["assert add(1, 2) == 3", "assert add(2, 2) == 5"])
    assert not r.all_passed and r.passed == 1 and r.error == "AssertionError"


def test_run_tests_solution_error_and_timeout():
    r = run_tests("def add(a, b):\n    return a +\n", ["assert add(1, 2) == 3"])
    assert r.passed == 0 and r.error == "solution:SyntaxError"
    import time
    t0 = time.time()
    r = run_tests("while True:\n    pass\n", ["assert True"], timeout=2)
    assert r.error == "timeout"
    assert time.time() - t0 < 15  # the loop is killed, not waited on
    r = run_tests("import subprocess, sys\n"
                  "subprocess.Popen([sys.executable, '-c', 'while True: pass'])\n"
                  "while True:\n    pass\n", ["assert True"], timeout=2)
    assert r.error == "timeout" and time.time() - t0 < 30  # a grandchild cannot hold it open


def test_child_sees_no_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert "ANTHROPIC_API_KEY" not in child_env()
    code = "import os\nLEAK = os.environ.get('ANTHROPIC_API_KEY')\n"
    r = run_tests(code, ["assert LEAK is None"])
    assert r.all_passed
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-real"


def test_solution_cannot_overwrite_the_harness_counters():
    # Answers that define globals named like the harness's own (ok, first, TESTS, json)
    # must not change the count: these once reported 10/10 with error [..] or 2.
    code = ADD + "ok = 99\nfirst = ['apple']\nTESTS = []\njson = None\n"
    r = run_tests(code, ["assert add(1, 2) == 3", "assert add(2, 2) == 5"])
    assert r.passed == 1 and r.total == 2 and r.error == "AssertionError"
    r = run_tests(ADD + "first = [0, 0, 2]\n", ["assert add(1, 2) == 3"])
    assert r.all_passed and r.error == ""
