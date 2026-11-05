"""Render sample lines to PNGs for a visual check of fonts and augmentation.

    .venv/Scripts/python scripts/ocr/render_lines.py --n 40 --out runs/ocr-samples

Each sample goes through the training path (usable_line, render_fitting, preprocess),
styles alternate printed / hand, and the PNG is the 64 px tensor the model sees (ink
dark on white). labels.tsv beside them holds png, text, style, font file."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_ctc import usable_line  # noqa: E402
from quipu.ocr_data import LineTooWide, NoCoveringFont, render_fitting  # noqa: E402
from quipu.ocr_render import STYLES, load_fonts  # noqa: E402

MAX_CHARS = 36      # configs/ocr-ctc.toml data.max_chars


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fonts", default=str(ROOT / "data" / "ocr" / "fonts"))
    ap.add_argument("--text", default=str(ROOT / "data" / "ocr" / "text" / "held.txt"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "ocr-samples"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    fonts = load_fonts(args.fonts)
    lines = [s for s in Path(args.text).read_text(encoding="utf-8").splitlines()
             if usable_line(s, MAX_CHARS)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0
    while i < args.n:
        style = STYLES[i % len(STYLES)]
        text = rng.choice(lines)
        try:
            got = render_fitting(text, style, fonts, rng)
        except (NoCoveringFont, LineTooWide):
            continue
        if got is None:
            continue
        _, t = got
        png = f"{i:03d}_{style}.png"
        Image.fromarray((255 * (1 - t[0].numpy())).astype(np.uint8)).save(out / png)
        rows.append(f"{png}\t{text}\t{style}")
        i += 1
    (out / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.n} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
