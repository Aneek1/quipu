"""scripts/ocr/eval_ocr.py: fixed eval sets and the scored run."""
import importlib.util
import random
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from quipu import ocr_ctc as oc
from quipu.ocr_data import PATCH, preprocess
from tests.ocr_fixtures import make_font_dir

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_ocr", ROOT / "scripts" / "ocr" / "eval_ocr.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _make(tmp_path, seed=0, n=12, force=False):
    fonts = tmp_path / "fonts"
    if not fonts.exists():
        make_font_dir(fonts)
    held = tmp_path / "held.txt"
    held.write_text("\n".join(f"held line number {i}" for i in range(50)) + "\n", encoding="utf-8")
    out = tmp_path / f"eval{seed}"
    ev.make_sets(held, fonts, out, n=n, seed=seed, force=force)
    return out


def test_make_sets_writes_three_sets_with_labels_styles_and_sums(tmp_path):
    out = _make(tmp_path)
    for name in ("printed", "hand", "nonsense"):
        rows = ev.read_set(out / name)
        assert len(rows) == 12
        for png, text, style in rows:
            assert png.is_file() and text
            assert style == name if name != "nonsense" else style in ("printed", "hand")
    assert {s for *_, s in ev.read_set(out / "nonsense")} == {"printed", "hand"}
    assert (out / "SHA256SUMS").is_file()
    ev.verify_sums(out)                       # does not raise


def test_make_sets_keeps_only_scorable_lines(tmp_path):
    out = _make(tmp_path)
    for name in ("printed", "hand", "nonsense"):
        for png, text, _ in ev.read_set(out / name):
            t = preprocess(Image.open(png))   # would raise LineTooWide
            assert oc.ctc_steps_needed(text) <= (t.shape[-1] // PATCH) * oc.STEPS_PER_COL


def test_make_sets_is_deterministic(tmp_path):
    a, b = _make(tmp_path, 0), _make(tmp_path / "again", 0)
    assert (a / "SHA256SUMS").read_text() == (b / "SHA256SUMS").read_text()


def test_make_refuses_a_non_empty_dir_without_force(tmp_path):
    out = _make(tmp_path)
    with pytest.raises(FileExistsError, match="--force"):
        _make(tmp_path)


def test_remake_with_force_and_smaller_n_leaves_no_stale_files(tmp_path):
    out = _make(tmp_path, n=12)
    (out / "notes.txt").write_text("mine")                 # not ours: kept, not hashed
    _make(tmp_path, n=5, force=True)
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 3 * (5 + 1)                        # 5 PNGs + labels.tsv per set
    assert not any("notes.txt" in s for s in sums)
    for name in ("printed", "hand", "nonsense"):
        assert len(list((out / name).glob("*.png"))) == 5
    assert (out / "notes.txt").read_text() == "mine"
    ev.verify_sums(out)


def test_verify_sums_catches_a_changed_file(tmp_path):
    out = _make(tmp_path)
    png, *_ = ev.read_set(out / "printed")[0]
    png.write_bytes(png.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        ev.verify_sums(out)


def test_make_sets_gives_up_when_nothing_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "render_fitting", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="1000"):
        _make(tmp_path)


def test_nonsense_words():
    for seed in range(200):
        w = ev.nonsense_line(random.Random(seed))
        assert w and all(part.isalpha() for part in w.split())
        assert w == w.strip() and len(w) <= ev.MAX_CHARS      # render_line's strip rule


def _tiny_ckpt(tmp_path, **extra):
    import torch
    from quipu.ocr_ctc import LineCTC
    from quipu.vision import ViTConfig
    vit = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)
    ck = tmp_path / "best.pt"
    torch.save({"model": LineCTC(vit).state_dict(), "vit": vars(vit), "step": 1,
                **oc.ckpt_compat(), **extra}, ck)
    return ck


def _run(tmp_path, out, baselines, **ck_extra):
    table = tmp_path / "results.md"
    ev.main(["run", "--ckpt", str(_tiny_ckpt(tmp_path, **ck_extra)), "--sets", str(out),
             "--out", str(table), "--baselines", baselines])
    return table.read_text(encoding="utf-8")


def test_run_scores_a_ctc_checkpoint_and_writes_a_table(tmp_path):
    text = _run(tmp_path, _make(tmp_path), "")
    assert "| CER (any case) |" in text
    for name in ("printed", "hand", "nonsense"):
        assert f"| quipu-ocr-ctc | {name} | 12 |" in text


def test_run_refuses_a_checkpoint_with_another_charset(tmp_path):
    with pytest.raises(ValueError, match="charset"):
        _run(tmp_path, _make(tmp_path), "", charset_sha256="0" * 64)


def test_run_skips_a_missing_tesseract_and_still_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    text = _run(tmp_path, _make(tmp_path), "tesseract")
    assert capsys.readouterr().err.count("tesseract not on PATH: skipped") == 1
    assert "| quipu-ocr-ctc | printed |" in text and "| tesseract |" not in text


def _fake_tesseract(monkeypatch, returncode, stdout="", stderr=""):
    calls = []
    real = subprocess.run

    def fake(cmd, *a, **k):
        if cmd[0] != "fake-tesseract":
            return real(cmd, *a, **k)
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr("shutil.which", lambda name: "fake-tesseract")
    monkeypatch.setattr(subprocess, "run", fake)
    return calls


def test_run_scores_tesseract_in_english(tmp_path, monkeypatch):
    calls = _fake_tesseract(monkeypatch, 0, stdout="held line number 1\n")
    text = _run(tmp_path, _make(tmp_path), "tesseract")
    assert "| tesseract | printed | 12 |" in text
    assert all(c[c.index("-l") + 1] == "eng" for c in calls)


def test_a_broken_tesseract_raises_instead_of_scoring(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch, 1, stderr="Error opening data file eng.traineddata")
    with pytest.raises(RuntimeError, match="eng.traineddata"):
        _run(tmp_path, _make(tmp_path), "tesseract")


def test_trocr_is_routed_per_line_by_style(tmp_path, monkeypatch):
    seen = {}

    def fake_read(name, paths):
        seen.setdefault(name, []).extend(paths)
        return ["HELD LINE" if "printed" in name else "held line" for _ in paths]

    monkeypatch.setattr(ev, "_trocr_read", fake_read)
    out = _make(tmp_path)
    text = _run(tmp_path, out, "trocr")
    styles = {p: s for name in ev.SETS for p, _, s in ev.read_set(out / name)}
    hw, pr = "microsoft/trocr-small-handwritten", "microsoft/trocr-small-printed"
    assert seen[hw] and all(styles[p] == "hand" for p in seen[hw])
    assert seen[pr] and all(styles[p] == "printed" for p in seen[pr])
    assert len(seen[hw]) + len(seen[pr]) == 36
    assert "| trocr-small | nonsense | 12 |" in text and "receipts" in text


def test_run_refuses_a_changed_set(tmp_path):
    out = _make(tmp_path)
    png, *_ = ev.read_set(out / "hand")[0]
    png.write_bytes(png.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        _run(tmp_path, out, "")
