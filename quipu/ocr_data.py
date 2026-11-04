"""OCR data: image preprocessing, batching, and the on-the-fly training stream.

Spec sections 3.1 and 4.5. Images become [1, 64, W] float tensors with ink near 1
and paper 0 (the median grey level of the image is subtracted, then clipped to [0, 1]), W a multiple of 16 and at most 1024. A line is skipped (never truncated or
squashed) when it is too long for the encoder stage, has characters outside the set,
has leading or trailing whitespace, renders wider than 1024 px at 64 px high, or
renders narrower than its CTC path needs."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from quipu.ocr_ctc import STEPS_PER_COL, ctc_steps_needed, in_charset
from quipu.ocr_render import Font, pick_font, render_line

HEIGHT, MAX_WIDTH, PATCH = 64, 1024, 16
MAX_SKIPS = 1000          # consecutive skipped texts before the stream gives up
FONT_TRIES = 5            # renders of one text (first try plus 4 retries) on a CTC miss


class LineTooWide(ValueError):
    """The line is wider than MAX_WIDTH once scaled to HEIGHT; squashing it would change
    the glyph aspect, so it is rejected instead."""


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    w = max(PATCH, round(img.width * HEIGHT / img.height))
    if w > MAX_WIDTH:
        raise LineTooWide(f"line is {w} px wide at {HEIGHT} px high; the limit is {MAX_WIDTH}")
    img = img.resize((w, HEIGHT), Image.Resampling.BILINEAR)
    ink = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    ink = np.clip(ink - np.median(ink), 0.0, 1.0)      # paper to 0; no contrast stretch
    t = torch.from_numpy(ink)[None]
    pad = (-w) % PATCH
    return F.pad(t, (0, pad)) if pad else t


def collate(items: list[tuple[torch.Tensor, str]]):
    widths = [t.shape[-1] for t, _ in items]
    W = max(widths)
    images = torch.zeros(len(items), 1, HEIGHT, W)
    for i, (t, _) in enumerate(items):
        images[i, :, :, : t.shape[-1]] = t
    cols = torch.tensor(widths) // PATCH
    mask = torch.arange(W // PATCH)[None, :] < cols[:, None]
    return images, mask, [s for _, s in items]


def render_fitting(text: str, style: str, fonts: list[Font], rng: random.Random,
                   tries: int = FONT_TRIES) -> tuple[Image.Image, torch.Tensor] | None:
    """Render `text` in a random `style` font (pick_font) until the image is wide enough
    for its CTC path, trying up to `tries` fonts. Returns (image, preprocessed tensor), or
    None if every try was too narrow. Raises LineTooWide as soon as one render is wider
    than MAX_WIDTH: another font would not shorten the text much."""
    for _ in range(tries):
        img = render_line(text, pick_font(fonts, style, rng), rng)
        t = preprocess(img)
        if ctc_steps_needed(text) <= (t.shape[-1] // PATCH) * STEPS_PER_COL:
            return img, t
    return None


class LineStream(IterableDataset):
    """Endless (image, text) pairs: a random line, a random style (hand with
    probability hand_frac), a random family then font of that style, rendered and
    preprocessed."""

    def __init__(self, lines: list[str], fonts: list[Font], hand_frac: float,
                 max_chars: int, seed: int) -> None:
        self.lines = [s for s in lines
                      if 0 < len(s) <= max_chars and s == s.strip() and in_charset(s)]
        if not self.lines:
            raise ValueError("no usable lines after filtering")
        self.fonts = fonts
        if hand_frac > 0 and not any(f.style == "hand" for f in fonts):
            raise ValueError("hand_frac > 0 but no hand fonts")
        self.hand_frac, self.seed, self.max_chars = hand_frac, seed, max_chars

    def __iter__(self):
        info = get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        too_wide = ctc_miss = 0
        while True:
            text = rng.choice(self.lines)
            style = "hand" if rng.random() < self.hand_frac else "printed"
            try:                                    # same text and style, up to 5 fonts
                result = render_fitting(text, style, self.fonts, rng)
            except LineTooWide:
                result = None
                too_wide += 1
            else:
                if result is None:
                    ctc_miss += 1
            if result is not None:
                too_wide = ctc_miss = 0
                yield result[1], text
            elif too_wide + ctc_miss >= MAX_SKIPS:
                if ctc_miss >= too_wide:
                    why = "the fonts are too narrow for the CTC step"
                else:
                    why = f"the lines are too long for these fonts; lower max_chars (now {self.max_chars})"
                raise RuntimeError(f"{MAX_SKIPS} candidate lines in a row were skipped "
                                   f"({too_wide} too wide, {ctc_miss} CTC misses): {why}.")
