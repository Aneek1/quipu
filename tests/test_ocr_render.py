"""quipu/ocr_render.py: licence refusal, font split, rendering."""
import random
import shutil

import numpy as np
import pytest

from quipu import ocr_render as r
from tests.ocr_fixtures import DEJAVU, make_font_dir


def test_font_without_licence_is_refused(tmp_path):
    shutil.copy(DEJAVU, tmp_path / "x.ttf")
    with pytest.raises(r.FontLicenceError, match="no licence file"):
        r.find_licence(tmp_path / "x.ttf")
    font = r.Font("x", "printed", tmp_path / "x.ttf")
    with pytest.raises(r.FontLicenceError):
        r.render_line("hello", font, random.Random(0))


def test_load_fonts_reads_manifest(tmp_path):
    fonts = r.load_fonts(make_font_dir(tmp_path))
    assert sorted((f.style, f.family) for f in fonts) == [
        ("hand", "delta"), ("hand", "gamma"), ("printed", "alpha"), ("printed", "beta")]


def test_split_fonts_keeps_every_style_on_both_sides(tmp_path):
    train, held = r.split_fonts(r.load_fonts(make_font_dir(tmp_path)))
    for style in ("printed", "hand"):
        assert any(f.style == style for f in train)
        assert any(f.style == style for f in held)
    assert not {f.family for f in train} & {f.family for f in held}


@pytest.mark.parametrize("style", ["printed", "hand"])
def test_render_line_is_grayscale_with_ink(tmp_path, style):
    font = next(f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == style)
    img = r.render_line("Hello, World 42", font, random.Random(1))
    assert img.mode == "L"
    arr = np.asarray(img)
    assert arr.min() < 128 < arr.max()          # dark ink on light paper
    assert img.width > img.height


def test_render_line_is_deterministic_per_seed(tmp_path):
    font = r.load_fonts(make_font_dir(tmp_path))[0]
    a = r.render_line("same seed", font, random.Random(7)).tobytes()
    b = r.render_line("same seed", font, random.Random(7)).tobytes()
    c = r.render_line("same seed", font, random.Random(8)).tobytes()
    assert a == b
    assert a != c


@pytest.mark.parametrize("style", ["printed", "hand"])
def test_crop_margins_leave_edge_columns_free_of_ink(tmp_path, style):
    font = next(f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == style)
    bg, ink = 230, 20
    for seed in range(10):
        text = "Hello, World 42 gjpqy Tall Hill"
        img = r._draw(text, font, 40, bg, ink, random.Random(seed), hand=style == "hand")
        arr = np.asarray(img)
        thr = (bg + ink) / 2
        assert (arr < thr).any()
        assert not (arr[:, :2] < thr).any(), f"ink in left 2 columns, seed {seed}"
        assert not (arr[:, -2:] < thr).any(), f"ink in right 2 columns, seed {seed}"
