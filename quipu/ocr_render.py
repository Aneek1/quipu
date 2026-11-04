"""Render one line of text as a scan-like grayscale image (spec sections 4.2, 4.3).

Fonts come from the manifest that scripts/ocr/fetch_fonts.py writes; a font without a
licence file beside it is refused here, at render time, so no code path can use one.
"printed" draws characters with even spacing; "hand" adds per-character baseline,
size and spacing jitter and a per-line slant. Both then go through the same scan
augmentation. Everything random comes from the `rng` passed in, so a seed fixes the
image exactly."""
from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt", "UFL.txt")
STYLES = ("printed", "hand")


class FontLicenceError(Exception):
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
    font_dir = Path(font_dir)
    manifest = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))
    fonts = []
    for e in manifest["fonts"]:
        if e["style"] not in STYLES:
            raise ValueError(f"unknown style {e['style']!r} for {e['path']}")
        path = font_dir / e["path"]
        find_licence(path)
        fonts.append(Font(e["family"], e["style"], path))
    return fonts


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


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    key = (str(path), size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return _FONT_CACHE[key]


def _draw(text: str, font: Font, size: int, bg: int, ink: int, rng: random.Random,
          hand: bool) -> Image.Image:
    spacing = rng.uniform(-0.5, 2.5)
    sizes = [max(8, round(size * rng.uniform(0.9, 1.1))) if hand else size for _ in text]
    advances = [_font(font.path, s).getlength(ch) for ch, s in zip(text, sizes)]
    width = int(sum(advances) + spacing * len(text) + 2 * size) + 16
    height = int(size * 2.2)
    img = Image.new("L", (width, height), bg)
    d = ImageDraw.Draw(img)
    x, base = float(size // 2), size * 0.5
    for ch, s, adv in zip(text, sizes, advances):
        dy = rng.gauss(0, size * 0.04) if hand else 0.0
        d.text((x, base + dy + (size - s) * 0.8), ch, font=_font(font.path, s), fill=ink)
        x += adv + spacing + (rng.gauss(0, size * 0.05) if hand else 0.0)
    if hand:
        img = _shear(img, rng.uniform(-0.3, 0.3), bg)
    return _crop_to_ink(img, bg, ink, rng)


def _shear(img: Image.Image, s: float, bg: int) -> Image.Image:
    w, h = img.size
    new_w = w + int(abs(s) * h)
    c = w / 2 - new_w / 2 - s * h / 2
    return img.transform((new_w, h), Image.Transform.AFFINE, (1, s, c, 0, 1, 0),
                         resample=Image.Resampling.BILINEAR, fillcolor=bg)


def _crop_to_ink(img: Image.Image, bg: int, ink: int, rng: random.Random) -> Image.Image:
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr < (bg + ink) / 2)
    if len(xs) == 0:
        return img
    mx, my = rng.randint(6, 16), rng.randint(4, 12)
    box = (max(0, xs.min() - mx), max(0, ys.min() - my),
           min(img.width, xs.max() + 1 + mx), min(img.height, ys.max() + 1 + my))
    return img.crop(box)


def _augment(img: Image.Image, bg: int, rng: random.Random) -> Image.Image:
    nrng = np.random.default_rng(rng.getrandbits(32))
    img = img.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BILINEAR, expand=True,
                     fillcolor=bg)
    if rng.random() < 0.3:
        w, h = img.size
        j = lambda n: rng.uniform(-0.03, 0.03) * n                      # noqa: E731
        quad = (j(w), j(h), j(w), h + j(h), w + j(w), h + j(h), w + j(w), j(h))
        img = img.transform(img.size, Image.Transform.QUAD, quad,
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
    size = rng.randint(28, 48)
    bg, ink = rng.randint(190, 255), rng.randint(0, 90)
    img = _draw(text, font, size, bg, ink, rng, hand=font.style == "hand")
    return _augment(img, bg, rng)
