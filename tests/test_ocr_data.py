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
        assert oc.ctc_steps_needed(text) <= (img.shape[-1] // 16) * oc.STEPS_PER_COL


def test_line_stream_hand_frac_needs_hand_fonts(tmp_path):
    fonts = [f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == "printed"]
    with pytest.raises(ValueError):
        LineStream(["hello"], fonts, hand_frac=0.5, max_chars=36, seed=0)


def test_line_stream_gives_up_when_every_line_is_too_wide(tmp_path, monkeypatch):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    monkeypatch.setattr(ocr_data, "render_line", lambda text, font, rng: Image.new("L", (5000, 64), 255))
    stream = LineStream(["hello world"], fonts, hand_frac=0.0, max_chars=36, seed=0)
    with pytest.raises(RuntimeError, match=r"lower max_chars"):
        next(iter(stream))


def test_preprocess_subtracts_the_paper_level():
    img = Image.new("L", (160, 64), 200)
    img.paste(20, (40, 20, 120, 40))
    t = preprocess(img)
    assert float(t[0, 2, 2]) == 0.0                     # paper
    assert float(t[0, 30, 80]) > 0.6                    # ink
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def _stream(tmp_path, lines=("hello world",), **kw):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    return LineStream(list(lines), fonts, **{"hand_frac": 0.5, "max_chars": 36, "seed": 0, **kw})


def test_workers_get_different_streams(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_data, "render_line",
                        lambda text, font, rng: Image.new("L", (400, 64), 255))
    lines = [f"line number {i}" for i in range(200)]
    stream = _stream(tmp_path, lines)

    def first_texts(worker_id):
        class Info:
            id = worker_id
        monkeypatch.setattr(ocr_data, "get_worker_info", lambda: Info)
        return [t for _, t in itertools.islice(iter(stream), 10)]

    assert first_texts(0) != first_texts(1)
    assert first_texts(0) == first_texts(0)


def test_ctc_miss_retries_same_text_and_style_with_another_font(tmp_path, monkeypatch):
    calls, picks = [], []
    orig_pick = ocr_data.pick_font

    def pick(fonts, style, rng):
        picks.append(style)
        return orig_pick(fonts, style, rng)

    def render(text, font, rng):
        calls.append(text)
        width = 16 if len(calls) <= 3 else 400          # first three renders are too narrow
        return Image.new("L", (width, 64), 255)

    monkeypatch.setattr(ocr_data, "pick_font", pick)
    monkeypatch.setattr(ocr_data, "render_line", render)
    stream = _stream(tmp_path, ["hello world", "another line here"])
    _, text = next(iter(stream))
    assert len(calls) == 4 and len(set(calls)) == 1 and calls[0] == text
    assert len(picks) == 4 and len(set(picks)) == 1


def test_ctc_miss_gives_up_on_a_text_after_five_renders(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ocr_data, "render_line",
                        lambda text, font, rng: (calls.append(text), Image.new("L", (16, 64), 255))[1])
    stream = _stream(tmp_path)
    with pytest.raises(RuntimeError, match="too narrow for the CTC step") as e:
        next(iter(stream))
    assert "1000 CTC misses" in str(e.value) and "0 too wide" in str(e.value)
    assert len(calls) == 5 * 1000
