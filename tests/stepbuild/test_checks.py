"""Checks are what turn "the model wrote something" into "the step passed", so each
one is tested on both sides: the untouched template passes, and a specific break
fails that check with output the model can act on."""
import shutil
import subprocess
import sys

import pytest

from stepbuild.harness import checks
from stepbuild.harness import sandbox as sb
from stepbuild.harness.blocks import FileBlock
from stepbuild.harness.checks import CheckResult, run_checks
from stepbuild.harness.plan import CHECKS
from stepbuild.harness.sandbox import Sandbox, create_sandbox, remove_sandbox, write_blocks

LIST = "frontend/src/components/List.jsx"

# A backend test that starts a detached child sleeping for a minute, records its
# pid in child.pid and passes at once.
SPAWN_DETACHED = """import subprocess
import sys


def test_spawn():
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with open("child.pid", "w") as f:
        f.write(str(child.pid))
"""


@pytest.fixture(scope="module")
def box(npm_cache, tmp_path_factory):
    """One full sandbox for the npm tests in this module; each test that adds a
    component removes it again, so test order does not matter."""
    box = create_sandbox(tmp_path_factory.mktemp("checks"), npm_cache)
    yield box
    remove_sandbox(box)


@pytest.fixture
def py_box(tmp_path):
    """A fresh template copy without node_modules, for the Python-only checks."""
    root = tmp_path / "py"
    shutil.copytree(sb.TEMPLATE_DIR, root, ignore=sb._TEMPLATE_IGNORE)
    return Sandbox(root)


def _write(box, path, content):
    write_blocks(box, [FileBlock(path, content)])


def test_template_passes_python_checks(py_box):
    results = run_checks(py_box, ["pyflakes", "pytest"])
    assert [r.name for r in results] == ["pyflakes", "pytest"]
    for r in results:
        assert isinstance(r, CheckResult)
        assert r.passed, r.output
        assert r.seconds >= 0


def test_pytest_ignores_config_in_a_parent_directory(tmp_path):
    # A pyproject.toml above the sandbox (there is a stray one in %TEMP% on the dev
    # machine) must not change how the backend tests run.
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-k no_such_test"\n', encoding="utf-8"
    )
    root = tmp_path / "py"
    shutil.copytree(sb.TEMPLATE_DIR, root, ignore=sb._TEMPLATE_IGNORE)
    [r] = run_checks(Sandbox(root), ["pytest"])
    assert r.passed, r.output
    assert "1 passed" in r.output


def test_check_names_come_from_the_plan():
    assert checks.CHECK_NAMES is CHECKS


def test_unknown_check_is_rejected(py_box):
    with pytest.raises(ValueError, match="unknown check"):
        run_checks(py_box, ["mypy"])


def test_syntax_error_fails_pyflakes_with_file_name(py_box):
    _write(py_box, "backend/models.py", "def broken(:\n    pass\n")
    [r] = run_checks(py_box, ["pyflakes"])
    assert not r.passed
    assert "models.py" in r.output


def test_undefined_name_fails_pyflakes(py_box):
    _write(py_box, "backend/models.py", "def f():\n    return missing\n")
    [r] = run_checks(py_box, ["pyflakes"])
    assert not r.passed
    assert "missing" in r.output


def test_failing_test_fails_pytest(py_box):
    _write(py_box, "backend/tests/test_api.py", "def test_x():\n    assert 1 == 2\n")
    [r] = run_checks(py_box, ["pytest"])
    assert not r.passed
    assert "test_x" in r.output


def test_broken_create_app_fails_pytest_via_smoke_test(py_box):
    _write(py_box, "backend/app.py", "def create_app():\n    return None\n")
    [r] = run_checks(py_box, ["pytest"])
    assert not r.passed
    assert "test_create_app_returns_a_flask_app" in r.output


def test_client_fixture_and_models_import_work(py_box):
    _write(py_box, "backend/models.py", "class Store:\n    pass\n")
    _write(
        py_box,
        "backend/tests/test_api.py",
        "from models import Store\n\n\n"
        "def test_404(client):\n"
        "    assert Store is not None\n"
        "    assert client.get('/api/nothing').status_code == 404\n",
    )
    [r] = run_checks(py_box, ["pytest"])
    assert r.passed, r.output


def test_output_is_last_60_lines(py_box):
    body = "".join(f"def test_{i}():\n    print('x')\n    assert False\n\n\n" for i in range(40))
    _write(py_box, "backend/tests/test_api.py", body)
    [r] = run_checks(py_box, ["pytest"])
    assert not r.passed
    assert 0 < len(r.output.splitlines()) <= 60
    assert "40 failed" in r.output  # the summary at the end is what survives


def test_timeout_returns_failed_with_message(py_box):
    _write(
        py_box,
        "backend/tests/test_api.py",
        "import time\n\n\ndef test_slow():\n    time.sleep(60)\n",
    )
    [r] = run_checks(py_box, ["pytest"], timeout_s=1)
    assert not r.passed
    assert r.output == "timed out after 1s"
    assert r.seconds < 20


def test_checks_see_an_unreachable_proxy(py_box):
    _write(
        py_box,
        "backend/tests/test_api.py",
        "import os\n\n\ndef test_proxy():\n"
        "    assert os.environ['HTTP_PROXY'] == 'http://127.0.0.1:9'\n"
        "    assert os.environ['HTTPS_PROXY'] == 'http://127.0.0.1:9'\n"
        "    assert os.environ['NO_PROXY'] == ''\n",
    )
    [r] = run_checks(py_box, ["pytest"])
    assert r.passed, r.output


def test_check_env_is_public_and_strips_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "somewhere")
    env = checks.check_env()
    assert "PYTHONPATH" not in env
    assert env["HTTP_PROXY"] == checks.UNREACHABLE_PROXY
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_run_command_check_passes_fails_and_trims_output(tmp_path):
    env = checks.check_env()
    ok = checks.run_command_check(
        "demo", [sys.executable, "-c", "print('\\n'.join(str(i) for i in range(100)))"],
        tmp_path, env, 30,
    )
    assert isinstance(ok, CheckResult)
    assert ok.name == "demo" and ok.passed
    assert ok.output.split("\n") == [str(i) for i in range(40, 100)]
    bad = checks.run_command_check(
        "demo", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], tmp_path, env, 30
    )
    assert not bad.passed and bad.output == "boom"


def test_run_command_check_timeout_message(tmp_path):
    r = checks.run_command_check(
        "demo", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
        checks.check_env(), 1,
    )
    assert not r.passed
    assert r.output == "timed out after 1s"


def _process_exited(pid: int, wait_ms: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    synchronize = 0x00100000
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True  # no such process any more
    try:
        return kernel32.WaitForSingleObject(handle, wait_ms) == 0  # WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_detached_grandchild_does_not_outlive_the_check(py_box):
    # The test starts a detached child that would sleep for a minute, records its
    # pid and passes at once. The check's Job Object must kill the child when the
    # check ends, although taskkill /T never runs (nothing timed out).
    _write(py_box, "backend/tests/test_api.py", SPAWN_DETACHED)
    [r] = run_checks(py_box, ["pytest"])
    assert r.passed, r.output
    pid = int((py_box.root / "backend" / "child.pid").read_text())
    try:
        assert _process_exited(pid, 5000), "detached child survived the check"
    finally:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


@pytest.mark.npm
def test_template_passes_all_three_checks(box):
    results = run_checks(box, ["pyflakes", "pytest", "npm_build"])
    assert [r.name for r in results] == ["pyflakes", "pytest", "npm_build"]
    for r in results:
        assert r.passed, f"{r.name}:\n{r.output}"


@pytest.mark.npm
def test_jsx_syntax_error_in_unimported_component_fails_npm_build(box):
    # App.jsx does not import List.jsx; only main.jsx's eager glob pulls it in.
    _write(box, LIST, "export default function List() {\n  return <ul><li></ul>\n}\n")
    try:
        [r] = run_checks(box, ["npm_build"])
    finally:
        (box.root / LIST).unlink()
    assert not r.passed
    assert "List.jsx" in r.output


@pytest.mark.npm
def test_valid_component_builds(box):
    _write(
        box,
        LIST,
        "export default function List({ items }) {\n"
        "  return <ul>{items.map((i) => <li key={i.id}>{i.id}</li>)}</ul>\n}\n",
    )
    try:
        [r] = run_checks(box, ["npm_build"])
    finally:
        (box.root / LIST).unlink()
    assert r.passed, r.output
