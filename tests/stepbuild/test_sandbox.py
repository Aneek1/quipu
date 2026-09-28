"""The sandbox is where model output lands, so write_blocks is tested as a second
safety boundary (after FileBlock validation), and create_sandbox is tested for the
things every later step relies on: the template is copied intact, node_modules is
reachable from frontend/ without a per-sandbox install, and deleting a sandbox
never reaches through its node_modules link into the shared cache."""
import os
import shutil
from pathlib import Path

import pytest

from stepbuild.harness import sandbox as sb
from stepbuild.harness.blocks import FileBlock
from stepbuild.harness.sandbox import (
    Sandbox,
    create_sandbox,
    default_cache_dir,
    remove_sandbox,
    write_blocks,
)

TEMPLATE_FILES = [
    "backend/app.py",
    "backend/models.py",
    "backend/requirements.txt",
    "backend/pytest.ini",
    "backend/tests/conftest.py",
    "backend/tests/test_smoke.py",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/vite.config.js",
    "frontend/index.html",
    "frontend/src/main.jsx",
    "frontend/src/App.jsx",
    "frontend/src/components/.gitkeep",
]


def test_template_has_every_file():
    for rel in TEMPLATE_FILES:
        assert (sb.TEMPLATE_DIR / rel).is_file(), rel


def test_main_jsx_eagerly_globs_components():
    main = (sb.TEMPLATE_DIR / "frontend/src/main.jsx").read_text(encoding="utf-8")
    assert "import.meta.glob('./components/*.jsx', { eager: true })" in main


def test_template_has_no_build_output():
    assert not (sb.TEMPLATE_DIR / "frontend/node_modules").exists()
    assert not (sb.TEMPLATE_DIR / "frontend/dist").exists()


def test_write_blocks_creates_nested_files(tmp_path):
    box = Sandbox(tmp_path)
    write_blocks(box, [
        FileBlock("backend/models.py", "x = 1\n"),
        FileBlock("frontend/src/components/deep/List.jsx", "export default 1\n"),
    ])
    assert (tmp_path / "backend/models.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "frontend/src/components/deep/List.jsx").is_file()


def test_write_blocks_overwrites_and_leaves_no_temp_file(tmp_path):
    box = Sandbox(tmp_path)
    write_blocks(box, [FileBlock("a.py", "x = 1\n")])
    write_blocks(box, [FileBlock("a.py", "x = 2\n")])
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.py"]


def test_write_blocks_writes_lf_bytes(tmp_path):
    write_blocks(Sandbox(tmp_path), [FileBlock("a.py", "x = 1\ny = 2\n")])
    assert (tmp_path / "a.py").read_bytes() == b"x = 1\ny = 2\n"


def _unvalidated_block(path):
    # FileBlock already rejects '..'; build one that bypasses its validation to
    # prove write_blocks does not trust its input blindly.
    evil = object.__new__(FileBlock)
    object.__setattr__(evil, "path", path)
    object.__setattr__(evil, "content", "x = 1\n")
    return evil


def test_write_blocks_refuses_paths_escaping_root(tmp_path):
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError, match="outside the sandbox"):
        write_blocks(Sandbox(root), [_unvalidated_block("../outside.py")])
    assert not (tmp_path / "outside.py").exists()


def test_write_blocks_checks_every_path_before_writing_any(tmp_path):
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError):
        write_blocks(Sandbox(root), [FileBlock("ok.py", "x = 1\n"), _unvalidated_block("../o.py")])
    assert not (root / "ok.py").exists()


def test_write_blocks_refuses_writing_through_a_link_out_of_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "box"
    root.mkdir()
    try:
        sb._make_link(outside, root / "linked")
    except OSError:
        pytest.skip("cannot create a directory link here")
    with pytest.raises(ValueError, match="outside the sandbox"):
        write_blocks(Sandbox(root), [FileBlock("linked/x.py", "x = 1\n")])
    assert not (outside / "x.py").exists()


def test_default_cache_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STEPBUILD_CACHE", str(tmp_path / "c"))
    assert default_cache_dir() == tmp_path / "c"
    monkeypatch.delenv("STEPBUILD_CACHE")
    assert default_cache_dir().name == "stepbuild-npm-cache"


def _frontend(dir_: Path, package: str, lock: str) -> Path:
    dir_.mkdir(parents=True)
    (dir_ / "package.json").write_bytes(package.encode())
    (dir_ / "package-lock.json").write_bytes(lock.encode())
    return dir_


def test_cache_key_follows_package_json_and_lockfile(tmp_path):
    base = sb._cache_key(_frontend(tmp_path / "a", '{"x": 1}\n', '{"l": 1}\n'))
    assert base == sb._cache_key(_frontend(tmp_path / "b", '{"x": 1}\n', '{"l": 1}\n'))
    assert base != sb._cache_key(_frontend(tmp_path / "c", '{"x": 2}\n', '{"l": 1}\n'))
    assert base != sb._cache_key(_frontend(tmp_path / "d", '{"x": 1}\n', '{"l": 2}\n'))
    # A CRLF checkout (git autocrlf) of the same files shares the cache.
    assert base == sb._cache_key(_frontend(tmp_path / "e", '{"x": 1}\r\n', '{"l": 1}\r\n'))


def test_link_dir_falls_back_to_copy(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "index.js").write_text("1", encoding="utf-8")

    def refuse(*a, **k):
        raise OSError("no links here")

    monkeypatch.setattr(sb, "_make_link", refuse)
    sb._link_dir(src, tmp_path / "dst")
    assert (tmp_path / "dst" / "pkg" / "index.js").read_text(encoding="utf-8") == "1"


def _key_dir(cache: Path) -> Path:
    return cache / sb._cache_key(sb.TEMPLATE_DIR / "frontend")


def _fake_install(calls):
    """Stands in for npm ci: records the call and creates a minimal node_modules."""

    def install(workdir, timeout_s):
        calls.append(workdir)
        assert (workdir / "package.json").is_file()
        assert (workdir / "package-lock.json").is_file()
        vite = workdir / "node_modules" / "vite"
        vite.mkdir(parents=True)
        (vite / "package.json").write_text("{}", encoding="utf-8")

    return install


def test_failed_install_leaves_no_half_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise sb.SandboxError("npm ci failed")

    monkeypatch.setattr(sb, "_npm_install", boom)
    cache = tmp_path / "cache"
    with pytest.raises(sb.SandboxError):
        sb.ensure_node_modules(cache)
    assert not (_key_dir(cache) / "node_modules").exists()
    assert list(cache.glob("*.staging-*")) == []


def test_install_runs_once_then_reuses_the_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sb, "_npm_install", _fake_install(calls))
    cache = tmp_path / "cache"
    first = sb.ensure_node_modules(cache)
    second = sb.ensure_node_modules(cache)
    assert first == second == _key_dir(cache) / "node_modules"
    assert len(calls) == 1
    assert list(cache.glob("*.staging-*")) == []


def test_cache_marked_complete_but_missing_vite_is_rebuilt(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    key_dir = _key_dir(cache)
    (key_dir / "node_modules").mkdir(parents=True)
    (key_dir / ".complete").write_text("ok\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(sb, "_npm_install", _fake_install(calls))
    nm = sb.ensure_node_modules(cache)
    assert len(calls) == 1
    assert (nm / "vite" / "package.json").is_file()


def test_losing_a_concurrent_first_install_returns_the_winner(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(sb, "_npm_install", _fake_install([]))

    def other_process_wins(src, dst):
        # Another process renames its complete cache into place just before us.
        vite = Path(dst) / "node_modules" / "vite"
        vite.mkdir(parents=True)
        (vite / "package.json").write_text('{"winner": true}', encoding="utf-8")
        (Path(dst) / ".complete").write_text("ok\n", encoding="utf-8")
        raise PermissionError("destination exists")

    monkeypatch.setattr(sb, "replace_with_retry", other_process_wins)
    nm = sb.ensure_node_modules(cache)
    assert (nm / "vite" / "package.json").read_text(encoding="utf-8") == '{"winner": true}'
    assert list(cache.glob("*.staging-*")) == []


def test_failed_final_rename_without_a_winner_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "_npm_install", _fake_install([]))

    def refuse(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(sb, "replace_with_retry", refuse)
    with pytest.raises(PermissionError):
        sb.ensure_node_modules(tmp_path / "cache")
    assert list((tmp_path / "cache").glob("*.staging-*")) == []


def test_npm_install_uses_npm_ci(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, cwd, env, timeout_s):
        seen.append(list(cmd))
        return 0, ""

    monkeypatch.setattr(sb, "run_with_timeout", fake_run)
    monkeypatch.setattr(sb.shutil, "which", lambda name: "npm")
    sb._npm_install(tmp_path, 10)
    assert seen[0][:2] == ["npm", "ci"]


@pytest.fixture
def fake_cache_box(tmp_path, monkeypatch):
    """A sandbox linked to a fake cache holding a sentinel file, so deleting the
    sandbox can be checked against the cache without npm."""
    nm = tmp_path / "cache" / "node_modules"
    nm.mkdir(parents=True)
    sentinel = nm / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(sb, "ensure_node_modules", lambda cache_dir: nm)
    box = create_sandbox(tmp_path / "boxes", tmp_path / "cache")
    assert (box.root / "frontend" / "node_modules" / "sentinel.txt").is_file()
    return box, sentinel


def test_remove_sandbox_keeps_the_cache(fake_cache_box):
    box, sentinel = fake_cache_box
    remove_sandbox(box)
    assert not box.root.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_plain_rmtree_of_a_sandbox_keeps_the_cache(fake_cache_box):
    box, sentinel = fake_cache_box
    shutil.rmtree(box.root)
    assert not box.root.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_remove_sandbox_handles_a_copied_node_modules(tmp_path, monkeypatch):
    nm = tmp_path / "cache" / "node_modules"
    nm.mkdir(parents=True)
    (nm / "sentinel.txt").write_text("keep me", encoding="utf-8")

    def refuse(*a, **k):
        raise OSError("no links here")

    monkeypatch.setattr(sb, "ensure_node_modules", lambda cache_dir: nm)
    monkeypatch.setattr(sb, "_make_link", refuse)
    box = create_sandbox(tmp_path / "boxes", tmp_path / "cache")
    remove_sandbox(box)
    assert not box.root.exists()
    assert (nm / "sentinel.txt").is_file()


@pytest.mark.npm
def test_create_sandbox_copies_template_and_links_node_modules(npm_cache, tmp_path):
    box = create_sandbox(tmp_path, npm_cache)
    other = create_sandbox(tmp_path, npm_cache)
    try:
        assert box.root.parent == tmp_path
        for rel in TEMPLATE_FILES:
            assert (box.root / rel).is_file(), rel
        nm = box.root / "frontend" / "node_modules"
        assert (nm / "vite" / "package.json").is_file()
        # Two sandboxes share one install instead of installing twice.
        assert other.root != box.root
        assert os.path.realpath(nm) == os.path.realpath(other.root / "frontend" / "node_modules")
    finally:
        remove_sandbox(box)
        remove_sandbox(other)
    assert (sb.ensure_node_modules(npm_cache) / "vite" / "package.json").is_file()
