"""Render sample lines to PNGs for a visual check of fonts and augmentation.

    .venv/Scripts/python scripts/ocr/render_lines.py --n 40 --out runs/ocr-samples"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_render import load_fonts, render_line  # noqa: E402


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
             if s.strip() and s == s.strip()]          # render_line's rule
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        font = fonts[i % len(fonts)]
        text = rng.choice(lines)
        render_line(text, font, rng).save(out / f"{i:03d}_{font.style}_{font.family}.png")
    print(f"{args.n} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
