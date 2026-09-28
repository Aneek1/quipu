"""Benchmark apps on disk, and the hidden acceptance run against a finished project.

Each app lives in `apps/<name>/`:

    spec.md                        one paragraph, the only description the model gets
    acceptance/test_acceptance.py  hidden REST-contract tests, never shown to the model
    reference/step_1.txt .. step_5.txt
                                   the FILE-block reply a perfect model would give for
                                   each model step (rendered with render_blocks)

`run_acceptance` runs an app's acceptance tests against a project root. The tests
are copied into a fresh temp dir rather than run in place or inside the sandbox,
so the model's files can never shadow or edit them and the sandbox's own conftest
(the template `client` fixture) plays no part: the acceptance tests define their
own client from `create_app()`, which is the only thing they import.

Two details keep the run honest:

- pytest's config is pinned with a `pytest.ini` written into the temp dir and
  passed with `-c`, and the rootdir is pinned too. Without that, pytest walks up
  from the temp dir looking for config, and a stray `pyproject.toml` or
  `pytest.ini` in %TEMP% (there is one on the dev machine) would change the run.
- `PYTHONPATH` is set to the project's `backend/`, so `from app import
  create_app` (and the app's own `from models import ...`) resolve there. The
  checks' environment strips PYTHONPATH on purpose, so it is added back here after
  building the rest of the environment with the same helper the checks use (no
  network proxies, no bytecode written into the sandbox, UTF-8 output).

The run goes through the checks' `run_command_check`, so it reports exactly
like the other checks: the last 60 lines of output, and a hung app (a server
started at import time, an infinite loop) fails with "timed out after Ns" and
its whole process tree is killed.
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

from stepbuild.harness.checks import CheckResult, check_env, run_command_check

APPS_DIR = Path(__file__).resolve().parent / "apps"
N_REFERENCE_STEPS = 5
CHECK_NAME = "acceptance"

_APP_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_PYTEST_INI = (
    "# Written by stepbuild.bench.acceptance: pins pytest's config to this directory\n"
    "# so no config file in a parent directory can change the acceptance run.\n"
    "[pytest]\n"
)


def list_apps() -> list[str]:
    """Names of every benchmark app (a directory under apps/ with a spec.md), sorted."""
    if not APPS_DIR.is_dir():
        return []
    return sorted(p.name for p in APPS_DIR.iterdir() if (p / "spec.md").is_file())


def _app_dir(app: str) -> Path:
    if not isinstance(app, str) or not _APP_NAME.fullmatch(app) or app not in list_apps():
        raise ValueError(f"unknown app {app!r}; known apps: {', '.join(list_apps())}")
    return APPS_DIR / app


def load_spec(app: str) -> str:
    """The app's spec.md text: the plain-English description the model builds from."""
    return (_app_dir(app) / "spec.md").read_text(encoding="utf-8")


def load_reference(app: str) -> list[str]:
    """The reference replies for model steps 1..5, in order."""
    ref = _app_dir(app) / "reference"
    replies = []
    for n in range(1, N_REFERENCE_STEPS + 1):
        path = ref / f"step_{n}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"app {app!r} has no reference reply {path.name}")
        # Universal-newline reading turns CRLF into LF: git autocrlf checks these
        # files out with CRLF on Windows, and the replies must equal what
        # render_blocks wrote (LF) wherever the repo was cloned.
        replies.append(path.read_text(encoding="utf-8"))
    return replies


def run_acceptance(app: str, project_root: Path, timeout_s: int = 180) -> CheckResult:
    """Run the app's hidden acceptance tests against `project_root`/backend."""
    source = _app_dir(app) / "acceptance"
    backend = Path(project_root).resolve() / "backend"
    if not (backend / "app.py").is_file():
        return CheckResult(CHECK_NAME, False, f"no backend/app.py under {project_root}", 0.0)
    with tempfile.TemporaryDirectory(prefix=f"accept-{app}-", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp) / "acceptance"
        shutil.copytree(
            source, workdir, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
        )
        ini = workdir / "pytest.ini"
        ini.write_text(_PYTEST_INI, encoding="utf-8")
        env = check_env()
        env["PYTHONPATH"] = str(backend)
        cmd = [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-c", str(ini), "--rootdir", str(workdir), str(workdir),
        ]
        return run_command_check(CHECK_NAME, cmd, workdir, env, timeout_s)
