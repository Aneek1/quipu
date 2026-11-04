"""scripts/ocr/fetch_fonts.py with the network replaced by a fake _get."""
import hashlib
import http.client
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "ocr" / "fetch_fonts.py"
spec = importlib.util.spec_from_file_location("fetch_fonts", SCRIPT)
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)

TTF = b"fake-ttf-bytes"
REF = "abc123"


def http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


class FakeNet:
    """A tiny google/fonts: dir -> {name: bytes} (or a str for a plain file)."""
    def __init__(self, tree, api_error=None, fail_raw=(), truncate=0):
        self.tree, self.api_error, self.fail_raw = tree, api_error or {}, set(fail_raw)
        self.truncate = truncate            # first N raw downloads raise IncompleteRead
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if "/commits/main" in url:
            return json.dumps({"sha": REF}).encode()
        if "/contents/" in url:
            d = url.split("/contents/")[1].split("?")[0]
            if d in self.api_error:
                raise http_error(self.api_error[d])
            if d not in self.tree:
                raise http_error(404)
            v = self.tree[d]
            if isinstance(v, str):                       # a file, not a folder
                return json.dumps({"type": "file", "name": d}).encode()
            return json.dumps([{"name": n, "type": "file"} for n in v]).encode()
        d, name = url.split(f"/{REF}/")[1].rsplit("/", 1)
        if self.truncate > 0:
            self.truncate -= 1
            raise http.client.IncompleteRead(b"par", 10)
        if name in self.fail_raw:
            raise http_error(500)
        return self.tree[d][name]


def family():
    return {"OFL.txt": b"licence", "A-Regular.ttf": TTF, "A-Italic.ttf": b"ital"}


CFG = 'min_per_style = 1\nprinted = ["ofl/a"]\nhand = ["ofl/b"]\n'
TREE = {"ofl/a": family(), "ofl/b": family()}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(ff.time, "sleep", lambda s: None)
    out = tmp_path / "out"

    def go(tree, config=CFG, **net_kw):
        net = FakeNet(tree, **net_kw)
        monkeypatch.setattr(ff, "_get", net)
        cfg = tmp_path / "fonts.toml"
        cfg.write_text(config, encoding="utf-8")
        return ff.main(["--out", str(out), "--config", str(cfg)]), out, net
    return go


def test_manifest_contents_and_sha(run):
    rc, out, _ = run(TREE)
    assert rc == 0
    m = json.loads((out / "fonts.json").read_text())
    assert m["ref"] == REF
    assert sorted((e["family"], e["style"]) for e in m["fonts"]) == [("a", "printed"),
                                                                    ("b", "hand")]
    for e in m["fonts"]:
        assert e["path"].endswith("A-Regular.ttf")            # italic skipped
        assert hashlib.sha256((out / e["path"]).read_bytes()).hexdigest() == e["sha256"]
        assert (out / e["licence"]).read_bytes() == b"licence"
    assert not list(out.rglob("*.tmp"))


def test_404_skips_the_family(run):
    cfg = 'min_per_style = 1\nprinted = ["ofl/gone", "ofl/a"]\nhand = ["ofl/b"]\n'
    rc, out, _ = run(TREE, cfg)
    assert rc == 0
    assert len(json.loads((out / "fonts.json").read_text())["fonts"]) == 2


def test_too_few_families_fails(run):
    rc, out, _ = run({"ofl/b": family()})                    # printed 0 of 1
    assert rc == 1
    assert not (out / "fonts.json").exists()


@pytest.mark.parametrize("code", [403, 429, 500])
def test_other_http_errors_abort(run, code, capsys):
    rc, out, _ = run(TREE, api_error={"ofl/a": code})
    assert rc == 1
    assert not (out / "fonts.json").exists()
    assert str(code) in capsys.readouterr().err


def test_failed_raw_download_leaves_nothing(run, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "fonts.json").write_text("stale")
    rc, out, net = run(TREE, fail_raw=["A-Regular.ttf"])
    assert rc == 1
    assert not (out / "fonts.json").exists()                 # stale manifest removed too
    assert not (out / "ofl" / "a").exists()
    assert not list(out.rglob("*.tmp"))
    assert sum("A-Regular.ttf" in u for u in net.calls) == 3  # retried 3 times


def test_existing_family_survives_a_failed_refetch(run, tmp_path):
    rc, out, _ = run(TREE)
    assert rc == 0
    rc, out, _ = run(TREE, fail_raw=["A-Regular.ttf"])
    assert rc == 1
    assert not list(out.rglob("*.tmp"))
    assert (out / "ofl" / "a" / "A-Regular.ttf").read_bytes() == TTF   # old copy intact


def test_folder_that_is_a_file_is_skipped(run):
    cfg = 'min_per_style = 1\nprinted = ["ofl/a", "ofl/b"]\nhand = ["ofl/b"]\n'
    rc, out, _ = run({"ofl/a": "file", "ofl/b": family()}, cfg)
    assert rc == 0


def test_non_plain_file_names_are_rejected(run, tmp_path):
    bad = {"OFL.txt": b"l", "../evil.ttf": TTF}
    rc, out, _ = run({"ofl/a": bad, "ofl/b": family()})
    assert rc == 1                                           # a skipped -> 0 printed
    assert not (out / "ofl" / "evil.ttf").exists()
    assert not (out.parent / "evil.ttf").exists()


def test_token_is_sent_when_set(monkeypatch):
    seen = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"x"

    def fake_urlopen(req, timeout):
        seen.update(req.header_items())
        return R()
    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    ff._get("http://x")
    assert seen["Authorization"] == "Bearer tok"
    seen.clear()
    monkeypatch.delenv("GITHUB_TOKEN")
    ff._get("http://x")
    assert "Authorization" not in seen


def test_truncated_download_is_retried(run):
    rc, out, net = run(TREE, truncate=2)
    assert rc == 0
    assert (out / "ofl" / "a" / "A-Regular.ttf").read_bytes() == TTF
    rc, out, net = run(TREE, truncate=3)                     # all three tries truncated
    assert rc == 1


def test_manifest_records_missing_ocr_characters(run):
    from tests.ocr_fixtures import DEJAVU
    real = {"OFL.txt": b"licence", "A-Regular.ttf": DEJAVU.read_bytes()}
    rc, out, _ = run({"ofl/a": real, "ofl/b": family()})
    assert rc == 0
    m = {e["family"]: e for e in json.loads((out / "fonts.json").read_text())["fonts"]}
    assert m["a"]["missing"] == ""                    # DejaVu has every OCR character
    assert m["b"]["missing"] is None                  # fake bytes: no character map
