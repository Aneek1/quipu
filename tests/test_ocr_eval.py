"""scripts/ocr/eval_ocr.py: fixed eval sets."""
import importlib.util
import random
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


def _make(tmp_path, seed=0):
    fonts = make_font_dir(tmp_path / "fonts")
    held = tmp_path / "held.txt"
    held.write_text("\n".join(f"held line number {i}" for i in range(50)) + "\n", encoding="utf-8")
    out = tmp_path / f"eval{seed}"
    ev.make_sets(held, fonts, out, n=12, seed=seed)
    return out


def test_make_sets_writes_three_sets_with_labels_and_sums(tmp_path):
    out = _make(tmp_path)
    for name in ("printed", "hand", "nonsense"):
        rows = ev.read_set(out / name)
        assert len(rows) == 12
        for png, text in rows:
            assert png.is_file() and text
    assert (out / "SHA256SUMS").is_file()
    ev.verify_sums(out)                       # does not raise


def test_make_sets_keeps_only_scorable_lines(tmp_path):
    out = _make(tmp_path)
    for name in ("printed", "hand", "nonsense"):
        for png, text in ev.read_set(out / name):
            t = preprocess(Image.open(png))   # would raise LineTooWide
            assert oc.ctc_steps_needed(text) <= (t.shape[-1] // PATCH) * oc.STEPS_PER_COL


def test_make_sets_is_deterministic(tmp_path):
    a, b = _make(tmp_path, 0), _make(tmp_path / "again", 0)
    assert (a / "SHA256SUMS").read_text() == (b / "SHA256SUMS").read_text()


def test_verify_sums_catches_a_changed_file(tmp_path):
    out = _make(tmp_path)
    png, _ = ev.read_set(out / "printed")[0]
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


def _tiny_ckpt(tmp_path):
    import torch
    from quipu.ocr_ctc import LineCTC
    from quipu.vision import ViTConfig
    vit = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)
    ck = tmp_path / "best.pt"
    torch.save({"model": LineCTC(vit).state_dict(), "vit": vars(vit), "step": 1}, ck)
    return ck


def test_run_scores_a_ctc_checkpoint_and_writes_a_table(tmp_path):
    out = _make(tmp_path)
    ck = _tiny_ckpt(tmp_path)
    table = tmp_path / "results.md"
    ev.main(["run", "--ckpt", str(ck), "--sets", str(out), "--out", str(table),
             "--baselines", ""])
    text = table.read_text(encoding="utf-8")
    assert "| quipu-ocr-ctc" in text
    for name in ("printed", "hand", "nonsense"):
        assert f"| quipu-ocr-ctc | {name} | 12 |" in text


def test_run_skips_a_missing_tesseract_and_still_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    out = _make(tmp_path)
    table = tmp_path / "results.md"
    ev.main(["run", "--ckpt", str(_tiny_ckpt(tmp_path)), "--sets", str(out),
             "--out", str(table), "--baselines", "tesseract"])
    assert "tesseract not on PATH: skipped" in capsys.readouterr().err
    text = table.read_text(encoding="utf-8")
    assert "| quipu-ocr-ctc | printed |" in text and "| tesseract |" not in text


def test_run_refuses_a_changed_set(tmp_path):
    out = _make(tmp_path)
    png, _ = ev.read_set(out / "hand")[0]
    png.write_bytes(png.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        ev.main(["run", "--ckpt", str(_tiny_ckpt(tmp_path)), "--sets", str(out),
                 "--out", str(tmp_path / "r.md"), "--baselines", ""])
