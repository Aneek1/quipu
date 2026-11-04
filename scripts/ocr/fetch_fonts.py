"""Download the OCR font families listed in configs/ocr-fonts.toml from
github.com/google/fonts, each with its licence file, and write data/ocr/fonts/fonts.json.

    .venv/Scripts/python scripts/ocr/fetch_fonts.py [--out data/ocr/fonts] [--ref <commit>]

--ref defaults to the current commit of google/fonts main, resolved once and recorded
in fonts.json, so a later run with --ref <that sha> fetches the same files. Uses the
GitHub contents API without a token (60 requests/hour is enough for ~50 families)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.github.com/repos/google/fonts"
RAW = "https://raw.githubusercontent.com/google/fonts"
LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt", "UFL.txt")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quipu-ocr-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def resolve_ref(ref: str | None) -> str:
    if ref:
        return ref
    return json.loads(_get(f"{API}/commits/main"))["sha"]


def fetch_family(family_dir: str, style: str, ref: str, out: Path) -> list[dict]:
    try:
        listing = json.loads(_get(f"{API}/contents/{family_dir}?ref={ref}"))
    except urllib.error.HTTPError as e:
        print(f"skip {family_dir}: HTTP {e.code}", file=sys.stderr)
        return []
    names = [e["name"] for e in listing if e["type"] == "file"]
    licence = next((n for n in LICENCE_NAMES if n in names), None)
    fonts = [n for n in names if n.lower().endswith(".ttf")]
    if licence is None or not fonts:
        print(f"skip {family_dir}: licence={licence} fonts={len(fonts)}", file=sys.stderr)
        return []
    # Upright faces only: italics are a different shape and the augmenter adds slant.
    fonts = [n for n in fonts if "italic" not in n.lower()] or fonts
    dest = out / family_dir
    dest.mkdir(parents=True, exist_ok=True)
    for name in [licence, *fonts]:
        (dest / name).write_bytes(_get(f"{RAW}/{ref}/{family_dir}/{name}"))
    return [{"family": family_dir.split("/")[1], "style": style,
             "path": f"{family_dir}/{n}", "licence": f"{family_dir}/{licence}",
             "sha256": hashlib.sha256((dest / n).read_bytes()).hexdigest()} for n in fonts]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "fonts"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "ocr-fonts.toml"))
    ap.add_argument("--ref", default=None)
    args = ap.parse_args(argv)
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    ref = resolve_ref(args.ref)
    entries: list[dict] = []
    for style in ("printed", "hand"):
        got = 0
        for family_dir in cfg[style]:
            fam = fetch_family(family_dir, style, ref, out)
            got += bool(fam)
            entries.extend(fam)
        print(f"{style}: {got} families")
        if got < cfg["min_per_style"]:
            print(f"only {got} {style} families, need {cfg['min_per_style']}", file=sys.stderr)
            return 1
    (out / "fonts.json").write_text(json.dumps({"repo": "google/fonts", "ref": ref,
                                                "fonts": entries}, indent=1), encoding="utf-8")
    print(f"{len(entries)} font files at {ref}; manifest {out / 'fonts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
