"""scripts/remote/modal_pool.py without Modal: a stub `modal` module records how the app
is declared (CPU, Volume, one container per source) and runs the remote functions in
this process on a temporary pool root; and the --local-smoke path, which needs no Modal
at all. Nothing is launched."""
import contextlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "remote" / "modal_pool.py"
TOKENIZER = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
needs_tokenizer = pytest.mark.skipif(not TOKENIZER.exists(), reason="no trained tokenizer")

from tests.test_build_pool import NoDecontam, local_pool, write_jsonl  # noqa: E402,F401


def fake_modal(pool_dir):
    m = types.ModuleType("modal")
    m.declared = {}
    m.commits = []

    class Fn:
        def __init__(self, fn, kw):
            self.fn, self.kw, self.calls = fn, kw, []

        def remote(self, *a, **k):
            self.calls.append(a)
            return self.fn(*a, **k)

        def starmap(self, items, return_exceptions=False):
            out = []
            for a in items:
                self.calls.append(tuple(a))
                try:
                    out.append(self.fn(*a))
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    out.append(exc)
            return out

    class App:
        def __init__(self, name, **kw):
            self.name = name

        def function(self, **kw):
            def deco(fn):
                f = Fn(fn, kw)
                m.declared[fn.__name__] = f
                return f
            return deco

        def local_entrypoint(self):
            return lambda fn: fn

    class Image:
        files = []

        @classmethod
        def debian_slim(cls, **kw):
            return cls()

        def pip_install(self, *a, **k):
            self.pip = a
            return self

        def env(self, d):
            return self

        def add_local_file(self, src, dst, **k):
            assert Path(src).exists(), src
            Image.files.append(dst)
            return self

    class Vol:
        def __init__(self, name):
            self.name = name

        def commit(self):
            m.commits.append(self.name)

        def reload(self):
            pass

        def listdir(self, path, recursive=False):
            p = Path(pool_dir) / path.strip("/")
            if not p.exists():
                raise FileNotFoundError(path)
            return [types.SimpleNamespace(path=str(x.relative_to(pool_dir)).replace("\\", "/"))
                    for x in p.iterdir()]

        @contextlib.contextmanager
        def batch_upload(self, force=False):
            class Up:
                def put_file(self_inner, local, remote):
                    dst = Path(pool_dir) / remote.strip("/")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, dst)
            yield Up()

    class Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            m.volume_name = name
            return Vol(name)

    class Secret:
        @staticmethod
        def from_name(name, required_keys=None):
            return ("secret", name)

    m.App, m.Image, m.Volume, m.Secret = App, Image, Volume, Secret
    m.is_local = lambda: True
    return m


def load(monkeypatch, modal_module):
    if modal_module is None:
        monkeypatch.setitem(sys.modules, "modal", None)  # import modal -> ImportError
    else:
        monkeypatch.setitem(sys.modules, "modal", modal_module)
    spec = importlib.util.spec_from_file_location("modal_pool_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_app_is_one_cpu_container_per_source_on_the_quipu_pool_volume(monkeypatch, tmp_path):
    fm = fake_modal(tmp_path)
    mod = load(monkeypatch, fm)
    assert fm.volume_name == "quipu-pool"
    b = fm.declared["build"].kw
    assert b["cpu"] == 16.0 and "gpu" not in b
    assert b["volumes"] == {"/pool": mod.vol}
    assert ("secret", "huggingface") in b["secrets"]
    assert fm.declared["merge"].kw["volumes"] == {"/pool": mod.vol}
    for f in ("quipu/pool.py", "quipu/pool_sources.py", "scripts/build_pool.py",
              "configs/data-v3.toml", "artifacts/tokenizer/tokenizer.json"):
        assert f"/src/{f}" in fm.Image.files


@needs_tokenizer
def test_the_entrypoint_builds_each_source_in_its_own_call_then_merges(
        monkeypatch, tmp_path, local_pool):
    pool_dir = tmp_path / "vol"
    fm = fake_modal(pool_dir)
    mod = load(monkeypatch, fm)
    monkeypatch.setattr(mod, "POOL_ROOT", pool_dir)
    monkeypatch.setattr(mod, "CONFIG", local_pool)
    build_pool = mod._bp()  # the module the remote functions use
    monkeypatch.setattr(build_pool, "load_decontam", lambda cfg: (NoDecontam(), {"fake": 1}))
    mod.main(chunk_tokens=6000, workers=0, lid_filter=False)
    calls = fm.declared["build"].calls
    assert sorted(c[0] for c in calls) == ["a", "b"]          # one call (container) per source
    # each build wrote under its own sources/<name>/ only
    assert {p.name for p in (pool_dir / "sources").iterdir()} == {"a", "b"}
    m = json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["chunks"][0]["tokens"] >= 6000
    assert len(fm.declared["merge"].calls) == 1


@needs_tokenizer
def test_local_smoke_runs_the_same_per_source_function_without_modal(
        monkeypatch, tmp_path, local_pool, capsys):
    mod = load(monkeypatch, None)
    assert mod.modal is None
    monkeypatch.setattr(mod, "CONFIG", local_pool)
    build_pool = mod._bp()
    monkeypatch.setattr(build_pool, "load_decontam", lambda cfg: (NoDecontam(), {"fake": 1}))
    out = tmp_path / "smoke"
    rep = mod.local_smoke(docs=30, out=str(out), lid_filter=False)
    assert set(rep["sources"]) == {"a", "b"}
    assert all(r["documents"] == 30 for r in rep["sources"].values())
    assert rep["chunk"]["tokens"] > 0 and (out / "mixed").exists()
    assert "local smoke" in capsys.readouterr().out
