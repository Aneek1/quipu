"""The sandbox is where model output lands, so write_blocks is tested as a second
safety boundary (after FileBlock validation), and create_sandbox is tested for the
two things every later step relies on: the template is copied intact and
node_modules is reachable from frontend/ without a per-sandbox npm install."""
import os

import pytest

from stepbuild.harness import sandbox as sb
from stepbuild.harness.blocks import FileBlock
from stepbuild.harness.sandbox import Sandbox, create_sandbox, default_cache_dir, write_blocks

TEMPLATE_FILES = [
    "backend/app.py",
    "backend/models.py",
    "backend/requirements.txt",
    "backend/pytest.ini",
    "backend/tests/conftest.py",
    "backend/tests/test_smoke.py",
    "frontend/package.json",
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


def test_write_blocks_refuses_paths_escaping_root(tmp_path):
    # FileBlock already rejects '..'; build one that bypasses its validation to
    # prove write_blocks does not trust its input blindly.
    evil = object.__new__(FileBlock)
    object.__setattr__(evil, "path", "../outside.py")
    object.__setattr__(evil, "content", "x = 1\n")
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError, match="outside the sandbox"):
        write_blocks(Sandbox(root), [evil])
    assert not (tmp_path / "outside.py").exists()


def test_write_blocks_checks_every_path_before_writing_any(tmp_path):
    evil = object.__new__(FileBlock)
    object.__setattr__(evil, "path", "../outside.py")
    object.__setattr__(evil, "content", "x = 1\n")
    root = tmp_path / "box"
    root.mkdir()
    with pytest.raises(ValueError):
        write_blocks(Sandbox(root), [FileBlock("ok.py", "x = 1\n"), evil])
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


def test_cache_key_follows_package_json(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    b.write_text('{"x": 2}', encoding="utf-8")
    assert sb._cache_key(a) != sb._cache_key(b)
    assert sb._cache_key(a) == sb._cache_key(a)


def test_link_dir_falls_back_to_copy(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "index.js").write_text("1", encoding="utf-8")

    def refuse(*a, **k):
        raise OSError("no links here")

    monkeypatch.setattr(sb, "_make_link", refuse)
    sb._link_dir(src, tmp_path / "dst")
    assert (tmp_path / "dst" / "pkg" / "index.js").read_text(encoding="utf-8") == "1"


def test_failed_install_leaves_no_half_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise sb.SandboxError("npm install failed")

    monkeypatch.setattr(sb, "_npm_install", boom)
    with pytest.raises(sb.SandboxError):
        sb.ensure_node_modules(tmp_path / "cache")
    key_dir = tmp_path / "cache" / sb._cache_key(sb.TEMPLATE_DIR / "frontend" / "package.json")
    assert not (key_dir / "node_modules").exists()


@pytest.mark.npm
def test_create_sandbox_copies_template_and_links_node_modules(npm_cache, tmp_path):
    box = create_sandbox(tmp_path, npm_cache)
    assert box.root.parent == tmp_path
    for rel in TEMPLATE_FILES:
        assert (box.root / rel).is_file(), rel
    nm = box.root / "frontend" / "node_modules"
    assert (nm / "vite" / "package.json").is_file()
    # Two sandboxes share one install instead of installing twice.
    other = create_sandbox(tmp_path, npm_cache)
    assert other.root != box.root
    assert os.path.realpath(nm) == os.path.realpath(other.root / "frontend" / "node_modules")
