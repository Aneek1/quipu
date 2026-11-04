"""quipu/ocr_data.py."""
import itertools

import pytest
import torch
from PIL import Image

from quipu import ocr_ctc as oc
from quipu import ocr_data
from quipu import ocr_render as r
from quipu.ocr_data import LineStream, LineTooWide, collate, preprocess
from tests.ocr_fixtures import make_font_dir


def test_preprocess_resizes_to_64_and_pads_to_16():
    t = preprocess(Image.new("L", (300, 50), 255))
    assert t.shape == (1, 64, 384)               # 300*64/50 = 384
    t = preprocess(Image.new("L", (100, 64), 255))
    assert t.shape == (1, 64, 112)               # 100 padded to 112
    assert float(t.max()) == 0.0                 # white paper is 0, ink is 1


def test_preprocess_rejects_too_wide_lines():
    with pytest.raises(LineTooWide):
        preprocess(Image.new("L", (5000, 64), 0))
    assert issubclass(LineTooWide, ValueError)


def test_preprocess_accepts_exactly_max_width():
    assert preprocess(Image.new("L", (1024, 64), 0)).shape == (1, 64, 1024)


def test_collate_pads_and_masks():
    a = (torch.ones(1, 64, 32), "ab")
    b = (torch.ones(1, 64, 64), "abc")
    images, mask, texts = collate([a, b])
    assert images.shape == (2, 1, 64, 64)
    assert mask.tolist() == [[True, True, False, False], [True] * 4]
    assert float(images[0, :, :, 32:].abs().sum()) == 0.0
    assert texts == ["ab", "abc"]


def test_line_stream_yields_fitting_lines(tmp_path):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    lines = ["hello world", "x" * 80, "naïve", " padded ", "Quipu reads 42 lines"]
    stream = LineStream(lines, fonts, hand_frac=0.5, max_chars=36, seed=3)
    items = list(itertools.islice(iter(stream), 6))
    for img, text in items:
        assert text in ("hello world", "Quipu reads 42 lines")   # long, non-charset, padded dropped
        assert img.shape[1] == 64
        assert oc.ctc_steps_needed(text) <= img.shape[-1] // 16


def test_line_stream_hand_frac_needs_hand_fonts(tmp_path):
    fonts = [f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == "printed"]
    with pytest.raises(ValueError):
        LineStream(["hello"], fonts, hand_frac=0.5, max_chars=36, seed=0)


def test_line_stream_gives_up_when_every_line_is_too_wide(tmp_path, monkeypatch):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    monkeypatch.setattr(ocr_data, "render_line", lambda text, font, rng: Image.new("L", (5000, 64), 255))
    stream = LineStream(["hello world"], fonts, hand_frac=0.0, max_chars=36, seed=0)
    with pytest.raises(RuntimeError, match="max_chars"):
        next(iter(stream))
