"""Quipu-OCR evaluation (spec section 6).

    .venv/Scripts/python scripts/ocr/eval_ocr.py make [--n 1000] [--force]
    .venv/Scripts/python scripts/ocr/eval_ocr.py run --ckpt runs/ocr-ctc/best.pt \
        [--baselines tesseract,trocr]

make: fixed sets from held-out text and held-out fonts (quipu.ocr_render.split_fonts):
printed/, hand/ and nonsense/ (random letters and made-up words, a held-out font of
either style). Lines are at most MAX_CHARS characters, and a line is kept only if its
render is scorable by the CTC model (no wider than 1024 px at 64 px high, wide enough
for its CTC path); otherwise a new text and font are drawn. Each set is PNGs plus
labels.tsv (png, text, style). SHA256SUMS covers each labels.tsv and the PNGs it names,
and `run` refuses a set whose files changed. make refuses a non-empty output directory
unless --force, which first removes the three set directories and SHA256SUMS.

run: scores the CTC checkpoint and the baselines on every set. TrOCR reads each line
with trocr-small-handwritten if it was rendered in a hand font, trocr-small-printed
otherwise. The table has CER both as is and ignoring case."""
from __future__ import annotations

import argparse
import functools
import hashlib
import random
import shutil
import string
import subprocess
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
TROCR = {"hand": "microsoft/trocr-small-handwritten",
         "printed": "microsoft/trocr-small-printed"}


def nonsense_line(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(4, 24)))
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiou"
    words = ["".join(rng.choice(cons) + rng.choice(vow) for _ in range(rng.randint(1, 4)))
             for _ in range(rng.randint(1, 4))]          # at most 4*8 + 3 = 35 characters
    return " ".join(words)


def make_sets(held_text: Path, font_dir: Path, out: Path, n: int, seed: int,
              force: bool = False) -> None:
    if out.is_dir() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass --force to replace its eval sets")
        for name in SETS:
            shutil.rmtree(out / name, ignore_errors=True)
        (out / "SHA256SUMS").unlink(missing_ok=True)
    rng = random.Random(seed)
    _, held_fonts = split_fonts(load_fonts(font_dir))
    lines = [s for s in held_text.read_text(encoding="utf-8").splitlines()
             if 0 < len(s) <= MAX_CHARS and s == s.strip() and in_charset(s)]
    if not lines:
        raise ValueError(f"{held_text}: no usable lines")
    written = []
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
            rows.append(f"{png}\t{text}\t{style}")
            written.append(d / png)
        (d / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        written.append(d / "labels.tsv")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}"
            for p in sorted(written)]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def verify_sums(out: Path) -> None:
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        if hashlib.sha256((out / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{rel} changed since the eval set was made")


def read_set(d: Path) -> list[tuple[Path, str, str]]:
    """(png path, text, style) per line; style is "printed" or "hand"."""
    rows = []
    for line in (d / "labels.tsv").read_text(encoding="utf-8").splitlines():
        png, text, style = line.split("\t")
        rows.append((d / png, text, style))
    return rows


class CTCReader:
    """The CTC checkpoint, loaded once. Its "vit" dict rebuilds the config; the charset
    and CTC steps per column must match this code."""

    def __init__(self, ckpt: Path) -> None:
        import torch
        from quipu.ocr_ctc import LineCTC, check_ckpt_compat
        from quipu.vision import ViTConfig

        ck = torch.load(ckpt, weights_only=False, map_location="cpu")
        check_ckpt_compat(ck, ckpt)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LineCTC(ViTConfig(**ck["vit"])).to(self.device).eval()
        self.model.load_state_dict(ck["model"])

    def __call__(self, paths: list[Path]) -> list[str]:
        import torch
        from PIL import Image
        from quipu.ocr_ctc import ctc_greedy_decode
        from quipu.ocr_data import collate, preprocess

        hyps = []
        with torch.no_grad():
            for i in range(0, len(paths), 64):
                items = []
                for p in paths[i:i + 64]:
                    with Image.open(p) as im:
                        items.append((preprocess(im), ""))
                images, mask, _ = collate(items)
                lp, lens = self.model(images.to(self.device), mask.to(self.device))
                hyps += ctc_greedy_decode(lp.cpu(), lens.cpu())
        return hyps


def predict_tesseract(paths: list[Path]) -> list[str] | None:
    exe = shutil.which("tesseract")
    if exe is None:
        print("tesseract not on PATH: skipped", file=sys.stderr)
        return None
    hyps = []
    for p in paths:
        r = subprocess.run([exe, str(p), "stdout", "-l", "eng", "--psm", "7"],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:              # a broken install must not score as CER 1.0
            raise RuntimeError(f"tesseract failed on {p} (exit {r.returncode}): "
                               f"{r.stderr.strip()}")
        hyps.append(r.stdout.strip())
    return hyps


@functools.lru_cache(maxsize=None)
def _trocr_model(name: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (TrOCRProcessor.from_pretrained(name),
            VisionEncoderDecoderModel.from_pretrained(name).to(device).eval(), device)


def _trocr_read(name: str, paths: list[Path]) -> list[str]:
    import torch
    from PIL import Image
    proc, model, device = _trocr_model(name)
    hyps = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            imgs = []
            for p in paths[i:i + 16]:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            pixels = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(pixels, max_new_tokens=64)
            hyps += [s.strip() for s in proc.batch_decode(ids, skip_special_tokens=True)]
    return hyps


def predict_trocr(paths: list[Path], styles: list[str]) -> list[str]:
    """Each line through the TrOCR model for the style it was rendered in."""
    hyps: list[str | None] = [None] * len(paths)
    for style, name in TROCR.items():
        idx = [i for i, s in enumerate(styles) if (s == "hand") == (style == "hand")]
        if idx:
            for i, h in zip(idx, _trocr_read(name, [paths[i] for i in idx])):
                hyps[i] = h
    return hyps


def run(ckpt: Path, sets: Path, out: Path, baselines: list[str]) -> str:
    from quipu.ocr_metrics import cer, exact_match, wer
    verify_sums(sets)
    ctc = CTCReader(ckpt)
    rows = []
    for name in SETS:
        data = read_set(sets / name)
        paths, refs, styles = [p for p, _, _ in data], [t for _, t, _ in data], [s for *_, s in data]
        systems = {"quipu-ocr-ctc": ctc(paths)}
        if "tesseract" in baselines:
            hyp = predict_tesseract(paths)
            if hyp is None:
                baselines = [b for b in baselines if b != "tesseract"]   # say it once
            else:
                systems["tesseract"] = hyp
        if "trocr" in baselines:
            systems["trocr-small"] = predict_trocr(paths, styles)
        for system, hyps in systems.items():
            ci = cer([r.lower() for r in refs], [h.lower() for h in hyps])
            rows.append(f"| {system} | {name} | {len(refs)} | {cer(refs, hyps):.4f} | "
                        f"{ci:.4f} | {wer(refs, hyps):.4f} | {exact_match(refs, hyps):.3f} |")
    table = ("| system | set | lines | CER | CER (any case) | WER | exact |\n"
             "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    note = (f"Checkpoint: `{ckpt}`. Sets: `{sets}` (SHA256SUMS verified). "
            "Greedy decoding, no language-model rescoring.")
    if "trocr" in baselines:
        note += ("\n\ntrocr-small reads hand lines with trocr-small-handwritten and the rest "
                 "with trocr-small-printed. trocr-small-printed was fine-tuned on receipts "
                 "(SROIE) and tends to output uppercase; the any-case CER column shows "
                 "how much of its error is case alone.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(note + "\n\n" + table, encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--text", default=str(ROOT / "data" / "ocr" / "text" / "held.txt"))
    mk.add_argument("--fonts", default=str(ROOT / "data" / "ocr" / "fonts"))
    mk.add_argument("--out", default=str(ROOT / "data" / "ocr" / "eval"))
    mk.add_argument("--n", type=int, default=1000)
    mk.add_argument("--seed", type=int, default=12345)
    mk.add_argument("--force", action="store_true",
                    help="replace the eval sets in a non-empty --out")
    rn = sub.add_parser("run")
    rn.add_argument("--ckpt", required=True)
    rn.add_argument("--sets", default=str(ROOT / "data" / "ocr" / "eval"))
    rn.add_argument("--out", default=str(ROOT / "results" / "ocr-encoder-stage.md"))
    rn.add_argument("--baselines", default="tesseract,trocr")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        try:
            make_sets(Path(args.text), Path(args.fonts), Path(args.out), args.n, args.seed,
                      args.force)
        except FileExistsError as e:
            raise SystemExit(str(e)) from None
        print(f"eval sets -> {args.out}")
    if args.cmd == "run":
        print(run(Path(args.ckpt), Path(args.sets), Path(args.out),
                  [b for b in args.baselines.split(",") if b]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
