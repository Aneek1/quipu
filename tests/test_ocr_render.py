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


def test_pick_axis_values_varies_only_the_weight_axis():
    axes = [{"name": b"Weight", "minimum": 400, "default": 400, "maximum": 700},
            {"name": "Width", "minimum": 75, "default": 100, "maximum": 125}]
    seen = set()
    for seed in range(30):
        vals = r.pick_axis_values(axes, random.Random(seed))
        assert 400 <= vals[0] <= 700
        assert vals[1] == 100
        seen.add(vals[0])
    assert len(seen) > 5
    assert r.pick_axis_values(axes, random.Random(3)) == r.pick_axis_values(axes, random.Random(3))


def test_pick_axis_values_without_weight_axis_keeps_defaults():
    axes = [{"name": b"Slant", "minimum": -10, "default": 0, "maximum": 0}]
    assert r.pick_axis_values(axes, random.Random(0)) == [0]


def test_static_font_has_no_axes_and_still_renders(tmp_path):
    font = r.load_fonts(make_font_dir(tmp_path))[0]
    assert r._axes(font.path) is None
    assert r.render_line("static", font, random.Random(2)).mode == "L"


def test_licence_names_are_ofl_and_apache_only():
    import importlib.util
    from pathlib import Path
    assert "UFL.txt" not in r.LICENCE_NAMES
    spec = importlib.util.spec_from_file_location(
        "fetch_fonts", Path(__file__).parents[1] / "scripts" / "ocr" / "fetch_fonts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "UFL.txt" not in m.LICENCE_NAMES
