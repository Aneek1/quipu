"""Quipu-OCR evaluation (spec section 6).

    .venv/Scripts/python scripts/ocr/eval_ocr.py make [--n 1000]
    .venv/Scripts/python scripts/ocr/eval_ocr.py run --ckpt runs/ocr-ctc/best.pt \
        [--baselines tesseract,trocr]

make: fixed sets from held-out text and held-out fonts (quipu.ocr_render.split_fonts):
printed/, hand/ and nonsense/ (random letters and made-up words, a held-out font of
either style). Lines are at most MAX_CHARS characters, and a line is kept only if its
render is scorable by the CTC model (no wider than 1024 px at 64 px high, wide enough
for its CTC path); otherwise a new text and font are drawn. Each set is PNGs plus
labels.tsv; SHA256SUMS covers every file and `run` refuses a set whose files changed."""
from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_ctc import in_charset  # noqa: E402
from quipu.ocr_data import LineTooWide, render_fitting  # noqa: E402
from quipu.ocr_render import load_fonts, split_fonts  # noqa: E402

SETS = ("printed", "hand", "nonsense")
MAX_CHARS = 36          # the encoder stage's limit (~35 characters fit 64x1024)
MAX_ATTEMPTS = 1000     # draws per eval line before make gives up


def nonsense_line(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(4, 24)))
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiou"
    words = ["".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.randint(1, 4)))
             for _ in range(rng.randint(1, 4))]          # at most 4*8 + 3 = 35 characters
    return " ".join(words)


def make_sets(held_text: Path, font_dir: Path, out: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    _, held_fonts = split_fonts(load_fonts(font_dir))
    lines = [s for s in held_text.read_text(encoding="utf-8").splitlines()
             if 0 < len(s) <= MAX_CHARS and s == s.strip() and in_charset(s)]
    if not lines:
        raise ValueError(f"{held_text}: no usable lines")
    for name in SETS:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            for _ in range(MAX_ATTEMPTS):
                if name == "nonsense":
                    text = nonsense_line(rng)
                    style = "hand" if rng.random() < 0.5 else "printed"
                else:
                    text, style = rng.choice(lines), name
                try:
                    fit = render_fitting(text, style, held_fonts, rng, tries=1)
                except LineTooWide:
                    fit = None
                if fit is not None:
                    break
            else:
                raise RuntimeError(f"{name} line {i}: no scorable render in {MAX_ATTEMPTS} "
                                   "draws of text and font")
            png = f"{i:05d}.png"
            fit[0].save(d / png)
            rows.append(f"{png}\t{text}")
        (d / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def verify_sums(out: Path) -> None:
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if hashlib.sha256((out / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{rel} changed since the eval set was made")


def read_set(d: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in (d / "labels.tsv").read_text(encoding="utf-8").splitlines():
        png, text = line.split("\t", 1)
        rows.append((d / png, text))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--text", default=str(ROOT / "data" / "ocr" / "text" / "held.txt"))
    mk.add_argument("--fonts", default=str(ROOT / "data" / "ocr" / "fonts"))
    mk.add_argument("--out", default=str(ROOT / "data" / "ocr" / "eval"))
    mk.add_argument("--n", type=int, default=1000)
    mk.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make_sets(Path(args.text), Path(args.fonts), Path(args.out), args.n, args.seed)
        print(f"eval sets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
