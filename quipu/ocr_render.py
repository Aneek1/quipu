"""Render one line of text as a scan-like grayscale image (spec sections 4.2, 4.3).

Fonts come from the manifest that scripts/ocr/fetch_fonts.py writes; a font without a
licence file beside it is refused here, at render time, so no code path can use one.
"printed" draws characters with even spacing; "hand" adds per-character baseline,
size and spacing jitter and a per-line slant. Both then go through the same scan
augmentation. Everything random comes from the `rng` passed in, so a seed fixes the
image exactly."""
from __future__ import annotations

import functools
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt")
STYLES = ("printed", "hand")


class FontLicenceError(Exception):
    pass


class FontManifestError(Exception):
    pass


@dataclass(frozen=True)
class Font:
    family: str
    style: str
    path: Path


def find_licence(font_path: str | Path) -> Path:
    d = Path(font_path).parent
    for name in LICENCE_NAMES:
        if (d / name).is_file():
            return d / name
    raise FontLicenceError(f"{font_path}: no licence file ({', '.join(LICENCE_NAMES)}) "
                           "beside it; not used")


def load_fonts(font_dir: str | Path) -> list[Font]:
    """Fonts listed in fonts.json. Every path must stay inside font_dir, have a licence
    file beside it, and match the sha256 recorded when it was downloaded."""
    font_dir = Path(font_dir)
    root = font_dir.resolve()
    manifest = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))
    fonts = []
    for e in manifest["fonts"]:
        if e["style"] not in STYLES:
            raise ValueError(f"unknown style {e['style']!r} for {e['path']}")
        rel = Path(e["path"])
        path = font_dir / rel
        if rel.is_absolute() or rel.drive or not path.resolve().is_relative_to(root):
            raise FontManifestError(f"{e['path']}: path is outside the font directory")
        if not path.is_file():
            raise FontManifestError(f"{e['path']}: listed in fonts.json but missing")
        find_licence(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != e.get("sha256"):
            raise FontManifestError(f"{e['path']}: sha256 {digest[:12]}... does not match "
                                    f"the manifest ({str(e.get('sha256'))[:12]}...)")
        if _too_thin(path):
            continue
        fonts.append(Font(e["family"], e["style"], path))
    return fonts


def _too_thin(path: Path) -> bool:
    """A static font file lighter than WEIGHT_MIN (Thin, ExtraLight, Hairline): its
    strokes are near-invisible at 64 px, which makes labels for blank-looking images.
    Variable fonts are kept; pick_axis_values clamps their weight instead."""
    from fontTools.ttLib import TTFont
    with TTFont(str(path), lazy=True) as f:
        return "fvar" not in f and f["OS/2"].usWeightClass < WEIGHT_MIN


def pick_font(fonts: list[Font], style: str, rng: random.Random) -> Font:
    """Family first, then a file within it, so a family with many files (static weights)
    is not sampled more often than a single-file one."""
    pool = [f for f in fonts if f.style == style]
    if not pool:
        raise ValueError(f"no {style} fonts")
    family = rng.choice(sorted({f.family for f in pool}))
    return rng.choice([f for f in pool if f.family == family])


@functools.lru_cache(maxsize=None)
def _cmap(path: Path) -> frozenset[int]:
    """Codepoints the font maps to a glyph, read once per file on first use (not in
    load_fonts, which stays fast)."""
    from fontTools.ttLib import TTFont
    with TTFont(str(path), lazy=True) as f:
        return frozenset(f.getBestCmap() or {})


def missing_chars(path: str | Path, chars: str) -> str:
    """The characters of `chars` (space excepted) the font file has no glyph for."""
    cmap = _cmap(Path(path))
    return "".join(c for c in dict.fromkeys(chars) if c != " " and ord(c) not in cmap)


def font_covers(font: Font, text: str) -> bool:
    """True if the font has a glyph for every character of text (space always counts:
    it draws nothing). Rendering a missing glyph draws a box or nothing, so the image
    would not show its label."""
    cmap = _cmap(font.path)
    return all(c == " " or ord(c) in cmap for c in set(text))


def split_fonts(fonts: list[Font], held_every: int = 5) -> tuple[list[Font], list[Font]]:
    """Held-out fonts by family: sha256(family) % held_every == 0. Each style keeps at
    least one family on each side (the first family by name moves if needed)."""
    def held(fam: str) -> bool:
        return int(hashlib.sha256(fam.encode()).hexdigest(), 16) % held_every == 0

    held_fams = {f.family for f in fonts if held(f.family)}
    for style in STYLES:
        fams = sorted({f.family for f in fonts if f.style == style})
        if len(fams) < 2:
            raise ValueError(f"need at least 2 {style} families to split, have {len(fams)}")
        if not any(f in held_fams for f in fams):
            held_fams.add(fams[0])
        if all(f in held_fams for f in fams):
            held_fams.discard(fams[0])
    return ([f for f in fonts if f.family not in held_fams],
            [f for f in fonts if f.family in held_fams])


_AXES_CACHE: dict[str, list | None] = {}
WEIGHT_MIN, WEIGHT_MAX, WEIGHT_STEP = 300, 800, 50
VPAD = 24                    # vertical canvas padding: more than the largest margin (12)
BOX_REF = "Hbdfhklgjpqy|[]{}()"   # tall, deep and bracket glyphs that set the line box


def _axes(path: Path) -> list | None:
    """Variation axes of a variable font, or None for a static one."""
    key = str(path)
    if key not in _AXES_CACHE:
        try:
            _AXES_CACHE[key] = ImageFont.truetype(key, 20).get_variation_axes()
        except (OSError, AttributeError):
            _AXES_CACHE[key] = None
    return _AXES_CACHE[key]


def pick_axis_values(axes: list, rng: random.Random) -> list[float]:
    """A random weight on the weight axis, every other axis at its default. The weight
    is clamped to [300, 800] (thinner is near-invisible at 64 px) and quantised to steps
    of 50, so at most 11 distinct fonts per size are ever built."""
    vals = []
    for a in axes:
        name = a["name"]
        name = name.decode("latin-1") if isinstance(name, bytes) else str(name)
        if name.lower() == "weight":
            lo, hi = max(a["minimum"], WEIGHT_MIN), min(a["maximum"], WEIGHT_MAX)
            if lo > hi:
                lo, hi = a["minimum"], a["maximum"]
            w = round(rng.uniform(lo, hi) / WEIGHT_STEP) * WEIGHT_STEP
            vals.append(min(max(w, lo), hi))
        else:
            vals.append(a["default"])
    return vals


@functools.lru_cache(maxsize=4096)
def _font(path: Path, size: int, axis_values: tuple | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(path), size)
    if axis_values is not None:
        f.set_variation_by_axes(list(axis_values))
    return f


def _draw(text: str, font: Font, size: int, bg: int, ink: int, rng: random.Random,
          hand: bool, axis_values: tuple | None = None) -> Image.Image:
    """Draw on a generous canvas, then crop: horizontally to the ink plus a margin,
    vertically to a fixed box built from the font (declared ascent/descent or the bounds
    of BOX_REF glyphs, whichever is larger, plus jitter room for the hand style), so the
    glyph scale in the final image does not depend on which letters the line contains
    and fonts with understated metrics are not clipped."""
    spacing = rng.uniform(-0.5, 2.5)
    mx, my = rng.randint(6, 16), rng.randint(4, 12)       # drawn first: text-independent
    base_font = _font(font.path, size, axis_values)
    big = 2 * round(size * 1.1 / 2) if hand else size      # largest per-letter size
    ref_font = _font(font.path, big, axis_values)
    asc, desc = ref_font.getmetrics()
    _, ref_top, _, ref_bottom = ref_font.getbbox(BOX_REF, anchor="ls")
    top, bottom = max(asc, -ref_top), max(desc, ref_bottom)
    if hand:                                              # 3 sigma of the baseline jitter + stroke
        top += math.ceil(0.12 * size) + 1
        bottom += math.ceil(0.12 * size) + 1
    pad = size
    baseline = VPAD + top
    whole = not hand and spacing <= 0.5                   # keep the font's kerning
    if whole:
        width = int(base_font.getlength(text)) + 2 * pad + 2
    else:
        sizes = [max(8, 2 * round(size * rng.uniform(0.9, 1.1) / 2)) if hand else size
                 for _ in text]
        dys = [rng.gauss(0, size * 0.04) if hand else 0.0 for _ in text]
        x = float(pad)
        xs = []
        for ch, s in zip(text, sizes):
            xs.append(x)
            x += _font(font.path, s, axis_values).getlength(ch) + spacing + (
                rng.gauss(0, size * 0.05) if hand else 0.0)
        width = int(max(x, xs[-1] + size)) + 2 * pad + 2
    img = Image.new("L", (width, 2 * VPAD + top + bottom), bg)
    d = ImageDraw.Draw(img)
    stroke = rng.choice([0, 0, 1]) if hand else 0
    if whole:
        d.text((pad, baseline), text, font=base_font, fill=ink, anchor="ls")
    else:
        for ch, s, cx, dy in zip(text, sizes, xs, dys):
            d.text((cx, baseline + dy), ch, font=_font(font.path, s, axis_values), fill=ink,
                   anchor="ls", stroke_width=stroke, stroke_fill=ink)
    if hand:
        img = _shear(img, rng.uniform(-0.3, 0.3), bg)
    return _crop_to_ink(img, bg, ink, mx, (VPAD - my, VPAD + top + bottom + my))


def _shear(img: Image.Image, s: float, bg: int) -> Image.Image:
    w, h = img.size
    new_w = w + int(abs(s) * h)
    c = w / 2 - new_w / 2 - s * h / 2
    return img.transform((new_w, h), Image.Transform.AFFINE, (1, s, c, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=bg)


def _crop_to_ink(img: Image.Image, bg: int, ink: int, mx: int, vbox: tuple[int, int]
                 ) -> Image.Image:
    arr = np.asarray(img)
    xs = np.nonzero((arr < (bg + ink) / 2).any(axis=0))[0]
    if len(xs) == 0:
        return img.crop((0, vbox[0], img.width, vbox[1]))
    return img.crop((max(0, xs.min() - mx), vbox[0], min(img.width, xs.max() + 1 + mx),
                     vbox[1]))


PERSPECTIVE = 0.03     # largest corner shift, as a fraction of the image's width / height


def perspective_quad(w: int, h: int, rng: random.Random) -> tuple[float, ...]:
    """Source corners (top-left, bottom-left, bottom-right, top-right) for a QUAD warp.
    Every corner moves outward only, by up to PERSPECTIVE of the size: the warp
    shrinks and skews the line inside the frame but never crops it. (Corners moved
    inward cut the first or last letters off long lines while the label kept them:
    3% of a 1,000 px line is 30 px, the side margin only 6-16 px.)"""
    jx = lambda: rng.uniform(0, PERSPECTIVE) * w                        # noqa: E731
    jy = lambda: rng.uniform(0, PERSPECTIVE) * h                        # noqa: E731
    return (-jx(), -jy(), -jx(), h + jy(), w + jx(), h + jy(), w + jx(), -jy())


def _augment(img: Image.Image, bg: int, rng: random.Random) -> Image.Image:
    nrng = np.random.default_rng(rng.getrandbits(32))
    w0, h0 = img.size
    lim = min(3.0, math.degrees(math.asin(min(1.0, 0.25 * h0 / w0))))   # w*sin(a) <= h/4
    img = img.rotate(rng.uniform(-lim, lim), resample=Image.Resampling.BILINEAR, expand=True,
                     fillcolor=bg)
    if rng.random() < 0.3:
        img = img.transform(img.size, Image.Transform.QUAD, perspective_quad(*img.size, rng),
                            resample=Image.Resampling.BILINEAR, fillcolor=bg)
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.MinFilter(3))                       # ink bleed
    elif rng.random() < 0.15:
        img = img.filter(ImageFilter.MaxFilter(3))                       # ink erosion
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.3:                                               # paper texture
        h, w = arr.shape
        low = nrng.normal(0, 1, (max(2, h // 16), max(2, w // 16))).astype(np.float32)
        tex = np.asarray(Image.fromarray(low).resize((w, h), Image.Resampling.BICUBIC))
        arr = arr + tex * rng.uniform(4, 15)
    if rng.random() < 0.3:                                               # lighting gradient
        arr = arr + np.linspace(0, rng.uniform(-30, 30), arr.shape[1])[None, :]
    if rng.random() < 0.5:
        arr = arr + nrng.normal(0, rng.uniform(2, 12), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(30, 95))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return img


def render_line(text: str, font: Font, rng: random.Random) -> Image.Image:
    find_licence(font.path)
    if not text.strip() or text != text.strip():
        raise ValueError(f"text must be non-empty with no leading or trailing "
                         f"whitespace: {text!r}")
    size = rng.randint(28, 48)
    bg, ink = rng.randint(190, 255), rng.randint(0, 90)
    axes = _axes(font.path)
    axis_values = tuple(pick_axis_values(axes, rng)) if axes else None
    img = _draw(text, font, size, bg, ink, rng, hand=font.style == "hand",
                axis_values=axis_values)
    return _augment(img, bg, rng)
