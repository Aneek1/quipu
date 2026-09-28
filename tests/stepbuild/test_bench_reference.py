"""The benchmark gate: every app's reference solution builds cleanly step by step
and passes its own hidden acceptance tests, and those tests fail on the bare
template, so they really test the app rather than the scaffolding.

The fast tests (no npm) check the reference files themselves and run the
acceptance tests against a copy of the backend only; the npm-marked gate replays
all five reference steps into a real sandbox, with every step's checks, the
final step-6 checks and the acceptance run.
"""
import re
import shutil
from pathlib import Path

import pytest

from stepbuild.bench import acceptance
from stepbuild.bench.acceptance import list_apps, load_reference, load_spec, run_acceptance
from stepbuild.harness.blocks import parse_blocks, render_blocks
from stepbuild.harness.checks import CheckResult, run_checks
from stepbuild.harness.plan import make_plan
from stepbuild.harness.sandbox import TEMPLATE_DIR, create_sandbox, remove_sandbox, write_blocks

APPS = list_apps()


def _plan(app):
    return make_plan(app, load_spec(app))


def _model_steps(app):
    return [s for s in _plan(app).steps if s.model_step]


def _backend_only(tmp_path):
    """A project root holding only a copy of the template backend: enough for the
    acceptance tests, no npm, and no node_modules link to worry about deleting."""
    root = tmp_path / "project"
    shutil.copytree(
        TEMPLATE_DIR / "backend", root / "backend",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    return root


def _write_backend_steps(app, root):
    for step, reply in zip(_model_steps(app), load_reference(app)):
        blocks = parse_blocks(reply, allowed=step.allowed_files)
        backend = [b for b in blocks if b.path.startswith("backend/")]
        for b in backend:
            target = root / b.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b.content.encode("utf-8"))


def test_the_first_two_apps_are_present():
    assert {"todo", "notes"} <= set(APPS)
    assert APPS == sorted(APPS)


@pytest.mark.parametrize("app", APPS)
def test_spec_is_one_paragraph_naming_api_endpoints(app):
    spec = load_spec(app).strip()
    assert "\n\n" not in spec
    assert "/api/" in spec
    for method in ("GET", "POST", "PUT", "DELETE"):
        assert method in spec
    assert "400" in spec and "404" in spec and "error" in spec
    assert "`id`" in spec and "JSON array" in spec
    assert "ids are never reused, even after a delete" in spec
    assert "every new app instance starts with no items" in spec.lower()
    # D1: the partial-update rule, word for word.
    assert (
        "PUT accepts any subset of the fields; fields left out keep their values; a field "
        "that is sent follows the same rules as on create (so an empty `title` is 400)" in spec
    )


@pytest.mark.parametrize("app", APPS)
def test_reference_has_one_reply_per_model_step(app):
    assert len(load_reference(app)) == len(_model_steps(app)) == 5


@pytest.mark.parametrize("app", APPS)
def test_each_reference_step_writes_exactly_its_allowed_files(app):
    for step, reply in zip(_model_steps(app), load_reference(app)):
        blocks = parse_blocks(reply, allowed=step.allowed_files)
        assert sorted(b.path for b in blocks) == sorted(step.allowed_files), step.key
        # The file is exactly what render_blocks produces: no chatter, canonical form.
        assert render_blocks(blocks) == reply, step.key


def test_reference_reads_as_lf_even_from_a_crlf_checkout(tmp_path, monkeypatch):
    """git autocrlf on Windows checks the step files out with CRLF; the replies must
    still come back exactly as render_blocks wrote them."""
    lf = load_reference("todo")
    apps = tmp_path / "apps"
    shutil.copytree(acceptance.APPS_DIR / "todo", apps / "todo")
    for path in (apps / "todo" / "reference").glob("step_*.txt"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(acceptance, "APPS_DIR", apps)
    crlf = load_reference("todo")
    assert all("\r" not in reply for reply in crlf)
    assert crlf == lf


def test_unknown_app_is_rejected():
    with pytest.raises(ValueError, match="unknown app"):
        load_reference("no_such_app")
    with pytest.raises(ValueError, match="unknown app"):
        run_acceptance("../todo", Path("."))


@pytest.mark.parametrize("app", APPS)
def test_acceptance_fails_on_the_bare_template(app, tmp_path):
    result = run_acceptance(app, _backend_only(tmp_path))
    assert isinstance(result, CheckResult)
    assert result.name == "acceptance"
    assert not result.passed
    assert "failed" in result.output


@pytest.mark.parametrize("app", APPS)
def test_acceptance_passes_on_the_reference_backend(app, tmp_path):
    root = _backend_only(tmp_path)
    _write_backend_steps(app, root)
    result = run_acceptance(app, root)
    assert result.passed, result.output
    assert "passed" in result.output


# Plausible model mistakes the acceptance tests must catch: (file, pattern, replacement).
BREAKAGES = {
    # PUT replaces the whole item, so a partial PUT loses (or fails on) the other fields.
    "full_replace_put": (
        "backend/app.py",
        r"changes = \{key: data\[key\] for key in FIELDS if key in data\}",
        "changes = {key: data.get(key) for key in FIELDS}",
    ),
    # PUT skips validation.
    "unvalidated_put": (
        "backend/app.py",
        r"errors = validate_\w+\(\{\*\*\w+, \*\*changes\}\)",
        "errors = []",
    ),
    # ids from len + 1, so a delete lets a later create reuse an id.
    "reused_ids": (
        "backend/models.py",
        r'item\["id"\] = self\._next_id',
        'item["id"] = len(self._items) + 1',
    ),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
@pytest.mark.parametrize("app", APPS)
def test_acceptance_catches_plausible_breakages(app, breakage, tmp_path):
    root = _backend_only(tmp_path)
    _write_backend_steps(app, root)
    rel, pattern, replacement = BREAKAGES[breakage]
    path = root / rel
    text = path.read_text(encoding="utf-8")
    broken, count = re.subn(pattern, lambda _: replacement, text)
    assert count == 1, f"{breakage}: pattern not found once in the {app} reference {rel}"
    path.write_text(broken, encoding="utf-8")
    result = run_acceptance(app, root)
    assert not result.passed, f"{breakage} went unnoticed by the {app} acceptance tests"


def test_acceptance_ignores_a_stray_pytest_config_above_its_temp_dir(tmp_path, monkeypatch):
    """%TEMP% on the dev machine holds a pyproject.toml; a config in a parent
    directory must not change the acceptance run."""
    root = _backend_only(tmp_path)
    _write_backend_steps("todo", root)
    hostile = tmp_path / "hostile-temp"
    hostile.mkdir()
    (hostile / "pytest.ini").write_text("[pytest]\naddopts = --no-such-option\n", encoding="utf-8")
    (hostile / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "--no-such-option"\n', encoding="utf-8"
    )
    monkeypatch.setattr(acceptance.tempfile, "tempdir", str(hostile))
    result = run_acceptance("todo", root)
    assert result.passed, result.output


def test_acceptance_without_a_backend_fails_clearly(tmp_path):
    result = run_acceptance("todo", tmp_path / "missing")
    assert not result.passed
    assert "backend" in result.output


def test_acceptance_timeout_is_a_failed_result(tmp_path):
    root = _backend_only(tmp_path)
    (root / "backend" / "app.py").write_text(
        "import time\n\ntime.sleep(60)\n\n\ndef create_app():\n    pass\n", encoding="utf-8"
    )
    result = run_acceptance("todo", root, timeout_s=2)
    assert not result.passed
    assert result.output == "timed out after 2s"


@pytest.mark.npm
@pytest.mark.parametrize("app", APPS)
def test_reference_replays_through_every_step_and_passes_acceptance(app, npm_cache, tmp_path):
    plan = _plan(app)
    sandbox = create_sandbox(tmp_path, npm_cache)
    try:
        replies = iter(load_reference(app))
        for step in plan.steps:
            if step.model_step:
                write_blocks(sandbox, parse_blocks(next(replies), allowed=step.allowed_files))
            results = run_checks(sandbox, step.checks)
            failed = [r for r in results if not r.passed]
            assert not failed, f"step {step.number} ({step.key}): " + "\n".join(
                f"{r.name}:\n{r.output}" for r in failed
            )
        result = run_acceptance(app, sandbox.root)
        assert result.passed, result.output
    finally:
        remove_sandbox(sandbox)
