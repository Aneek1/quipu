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


# ---- code-review fixes -------------------------------------------------------------

import json  # noqa: E402

from PIL import ImageFont  # noqa: E402


def _fake_variable(monkeypatch, lo=100, hi=900):
    """Pretend every font is variable with a weight axis, recording the weights set."""
    seen = []
    monkeypatch.setattr(r, "_axes", lambda path: [
        {"name": b"Weight", "minimum": lo, "default": 400, "maximum": hi}])
    monkeypatch.setattr(ImageFont.FreeTypeFont, "set_variation_by_axes",
                        lambda self, v: seen.append(v[0]))
    r._font.cache_clear()
    return seen


def test_font_cache_is_bounded_and_weights_are_few(tmp_path, monkeypatch):
    seen = _fake_variable(monkeypatch)
    fonts = r.load_fonts(make_font_dir(tmp_path))
    rng = random.Random(0)
    for i in range(500):
        r.render_line("Hello world", fonts[i % len(fonts)], rng)
    assert r._font.cache_info().currsize <= r._font.cache_info().maxsize == 4096
    assert len(set(seen)) <= 11
    assert min(seen) >= 300 and max(seen) <= 800
    assert all(w % 50 == 0 for w in seen)


def test_pick_axis_values_is_quantised_and_clamped():
    axes = [{"name": b"Weight", "minimum": 100, "default": 400, "maximum": 900}]
    vals = {r.pick_axis_values(axes, random.Random(s))[0] for s in range(300)}
    assert vals <= set(range(300, 801, 50))
    assert len(vals) >= 8


def test_height_does_not_depend_on_text(tmp_path):
    for style in ("printed", "hand"):
        font = next(f for f in r.load_fonts(make_font_dir(tmp_path / style)) if f.style == style)
        hs = {t: r._draw(t, font, 40, 230, 20, random.Random(4), hand=style == "hand").height
              for t in ("acemnorsuvwxz", "bdfhklt", ".", "Hello")}
        assert len(set(hs.values())) == 1, hs


def test_rotation_does_not_blow_up_long_lines(tmp_path):
    font = r.load_fonts(make_font_dir(tmp_path))[0]
    text = "The quick brown fox jumps over the lazy dog. 0123456789 abc"[:60]
    for seed in range(20):
        pre = r._draw(text, font, 40, 230, 20, random.Random(seed), hand=False)
        post = r._augment(pre, 230, random.Random(seed))
        assert post.height <= 1.6 * pre.height, (seed, pre.size, post.size)


@pytest.mark.parametrize("bad", ["", "   ", " lead", "trail ", "\tx", "x\n"])
def test_render_line_rejects_bad_text(tmp_path, bad):
    font = r.load_fonts(make_font_dir(tmp_path))[0]
    with pytest.raises(ValueError):
        r.render_line(bad, font, random.Random(0))


def test_long_hand_line_keeps_its_last_glyph(tmp_path):
    font = next(f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == "hand")
    text = ("Mw" * 30)[:60]
    for seed in range(20):
        arr = np.asarray(r._draw(text, font, 48, 230, 20, random.Random(seed), hand=True))
        thr = (230 + 20) / 2
        assert not (arr[:, -2:] < thr).any(), seed
        assert not (arr[:, :2] < thr).any(), seed


def test_printed_uses_one_draw_call_when_spacing_is_small(tmp_path, monkeypatch):
    from PIL import ImageDraw
    calls = []
    real = ImageDraw.ImageDraw.text
    monkeypatch.setattr(ImageDraw.ImageDraw, "text",
                        lambda self, xy, text, *a, **k: (calls.append(text), real(self, xy, text, *a, **k))[1])
    font = next(f for f in r.load_fonts(make_font_dir(tmp_path)) if f.style == "printed")
    modes = set()
    for seed in range(40):
        calls.clear()
        r._draw("AVAVAV", font, 40, 230, 20, random.Random(seed), hand=False)
        modes.add(len(calls) == 1)
    assert modes == {True, False}      # whole-string kerning path and per-char spacing path


def test_pick_font_picks_family_first(tmp_path):
    d = tmp_path
    fonts = r.load_fonts(make_font_dir(d, families=(("printed", "big"), ("printed", "small"),
                                                    ("hand", "h1"), ("hand", "h2"))))
    big = [r.Font("big", "printed", fonts[0].path.with_name(f"big{i}.ttf")) for i in range(4)]
    pool = big + [f for f in fonts if f.family == "small"] + [f for f in fonts if f.style == "hand"]
    rng = random.Random(0)
    picks = [r.pick_font(pool, "printed", rng) for _ in range(2000)]
    assert all(f.style == "printed" for f in picks)
    n_big = sum(f.family == "big" for f in picks)
    assert 850 < n_big < 1150
    assert {f.path.name for f in picks if f.family == "big"} == {f"big{i}.ttf" for i in range(4)}


def _edit_manifest(root, fn):
    m = json.loads((root / "fonts.json").read_text())
    fn(m)
    (root / "fonts.json").write_text(json.dumps(m))


def test_load_fonts_rejects_sha_mismatch(tmp_path):
    make_font_dir(tmp_path)
    (tmp_path / "ofl" / "alpha" / "alpha.ttf").write_bytes(b"tampered")
    with pytest.raises(r.FontManifestError, match="sha256"):
        r.load_fonts(tmp_path)


@pytest.mark.parametrize("bad", ["../outside.ttf", "ofl/../../outside.ttf", "/abs/x.ttf",
                                 "C:/abs/x.ttf"])
def test_load_fonts_rejects_paths_outside_dir(tmp_path, bad):
    root = tmp_path / "fonts"
    root.mkdir()
    make_font_dir(root)
    _edit_manifest(root, lambda m: m["fonts"][0].__setitem__("path", bad))
    with pytest.raises(r.FontManifestError):
        r.load_fonts(root)


def test_line_box_comes_from_glyph_bounds_not_declared_metrics(tmp_path, monkeypatch):
    """Fonts with understated ascent/descent must not clip tall glyphs or descenders."""
    real = ImageFont.FreeTypeFont.getmetrics
    monkeypatch.setattr(ImageFont.FreeTypeFont, "getmetrics",
                        lambda self, *a, **k: tuple(int(0.6 * v) for v in real(self, *a, **k)))
    r._font.cache_clear()
    text = "Hgjp|[]{}() Tally"
    bg, ink = 230, 20
    for style in ("printed", "hand"):
        font = next(f for f in r.load_fonts(make_font_dir(tmp_path / style)) if f.style == style)
        for seed in range(50):
            arr = np.asarray(r._draw(text, font, 48, bg, ink, random.Random(seed),
                                     hand=style == "hand"))
            thr = (bg + ink) / 2
            assert not (arr[0] < thr).any(), (style, seed, "ink in top row")
            assert not (arr[-1] < thr).any(), (style, seed, "ink in bottom row")
    r._font.cache_clear()


def test_font_cache_holds_a_realistic_pool():
    assert r._font.cache_info().maxsize >= 4096
