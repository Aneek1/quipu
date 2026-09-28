import pytest

import quipu.fsio as fsio_module
from quipu.fsio import replace_with_retry


def test_succeeds_after_transient_permission_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(fsio_module.time, "sleep", lambda *_: None)

    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"
    src.write_text("new content", encoding="utf-8")

    real_replace = fsio_module.os.replace
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("transiently locked")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(fsio_module.os, "replace", flaky)

    replace_with_retry(src, dst)

    assert calls["n"] == 3
    assert dst.read_text(encoding="utf-8") == "new content"


def test_gives_up_after_all_attempts_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(fsio_module.time, "sleep", lambda *_: None)

    def always_fails(*args, **kwargs):
        raise PermissionError("still locked")

    monkeypatch.setattr(fsio_module.os, "replace", always_fails)

    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"
    src.write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        replace_with_retry(src, dst, attempts=4, base_delay=0.01)


def test_non_permission_error_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(fsio_module.time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def not_found(*args, **kwargs):
        calls["n"] += 1
        raise FileNotFoundError("gone")

    monkeypatch.setattr(fsio_module.os, "replace", not_found)

    src = tmp_path / "src.tmp"
    dst = tmp_path / "dst.txt"

    with pytest.raises(FileNotFoundError):
        replace_with_retry(src, dst)

    assert calls["n"] == 1


def test_write_text_atomic(tmp_path):
    from quipu.fsio import write_text_atomic
    target = tmp_path / "sub" / "out.md"
    write_text_atomic(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    write_text_atomic(target, "again\n")
    assert target.read_text(encoding="utf-8") == "again\n"
    assert not list(target.parent.glob("*.tmp"))
