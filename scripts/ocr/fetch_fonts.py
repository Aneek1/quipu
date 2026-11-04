"""Download the OCR font families listed in configs/ocr-fonts.toml from
github.com/google/fonts, each with its licence file, and write data/ocr/fonts/fonts.json.

    .venv/Scripts/python scripts/ocr/fetch_fonts.py [--out data/ocr/fonts] [--ref <commit>]

--ref defaults to the current commit of google/fonts main, resolved once and recorded
in fonts.json, so a later run with --ref <that sha> fetches the same files. Uses the
GitHub contents API; set GITHUB_TOKEN to lift the 60 requests/hour anonymous limit.

A family that does not exist (HTTP 404) is skipped with a warning. Any other HTTP error
(rate limit, server error) aborts the run, so a throttled run cannot pass as a small
font set. A family is downloaded into <dir>.tmp and moved into place only when complete,
and fonts.json is removed at the start and written last, so it never describes a
half-fetched directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.github.com/repos/google/fonts"
RAW = "https://raw.githubusercontent.com/google/fonts"
LICENCE_NAMES = ("OFL.txt", "LICENSE.txt", "LICENCE.txt")
RAW_TRIES = 3


class FetchError(Exception):
    pass


def _get(url: str) -> bytes:
    headers = {"User-Agent": "quipu-ocr-fetch"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _get_raw(url: str) -> bytes:
    """A file download, retried with backoff; raises FetchError when all tries fail."""
    for attempt in range(RAW_TRIES):
        try:
            return _get(url)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == RAW_TRIES - 1:
                raise FetchError(f"download failed after {RAW_TRIES} tries: {url}: {e}") from e
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def resolve_ref(ref: str | None) -> str:
    if ref:
        return ref
    return json.loads(_get(f"{API}/commits/main"))["sha"]


def _plain(name: str) -> bool:
    return name not in ("", ".", "..") and Path(name).name == name and "\\" not in name


def fetch_family(family_dir: str, style: str, ref: str, out: Path) -> list[dict]:
    parts = Path(family_dir).parts
    if len(parts) != 2 or Path(family_dir).is_absolute() or ".." in parts:
        raise FetchError(f"bad family entry {family_dir!r}: expected 'ofl/<name>'")
    try:
        listing = json.loads(_get(f"{API}/contents/{family_dir}?ref={ref}"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"skip {family_dir}: not found at {ref[:10]}", file=sys.stderr)
            return []
        hint = " (rate limited? set GITHUB_TOKEN)" if e.code in (403, 429) else ""
        raise FetchError(f"{family_dir}: HTTP {e.code} from the GitHub API{hint}") from e
    if not isinstance(listing, list):
        print(f"skip {family_dir}: not a folder", file=sys.stderr)
        return []
    names = [e["name"] for e in listing if e.get("type") == "file"]
    bad = [n for n in names if not _plain(n)]
    if bad:
        print(f"skip {family_dir}: unsafe file names {bad}", file=sys.stderr)
        return []
    licence = next((n for n in LICENCE_NAMES if n in names), None)
    fonts = [n for n in names if n.lower().endswith(".ttf")]
    if licence is None or not fonts:
        print(f"skip {family_dir}: licence={licence} fonts={len(fonts)}", file=sys.stderr)
        return []
    # Upright faces only: italics are a different shape and the augmenter adds slant.
    fonts = [n for n in fonts if "italic" not in n.lower()] or fonts
    dest = out / family_dir
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        digests = {}
        for name in [licence, *fonts]:
            data = _get_raw(f"{RAW}/{ref}/{family_dir}/{name}")
            (tmp / name).write_bytes(data)
            digests[name] = hashlib.sha256(data).hexdigest()
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return [{"family": parts[1], "style": style, "path": f"{family_dir}/{n}",
             "licence": f"{family_dir}/{licence}", "sha256": digests[n]} for n in fonts]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "fonts"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "ocr-fonts.toml"))
    ap.add_argument("--ref", default=None)
    args = ap.parse_args(argv)
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "fonts.json"
    manifest.unlink(missing_ok=True)                  # never leave a manifest that lies
    try:
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
                print(f"only {got} {style} families, need {cfg['min_per_style']}",
                      file=sys.stderr)
                return 1
    except (FetchError, urllib.error.URLError) as e:
        print(f"fetch aborted: {e}", file=sys.stderr)
        return 1
    for e in entries:                                 # verify what is on disk
        if hashlib.sha256((out / e["path"]).read_bytes()).hexdigest() != e["sha256"]:
            print(f"fetch aborted: {e['path']} changed on disk after download", file=sys.stderr)
            return 1
    tmp = manifest.with_name("fonts.json.tmp")
    tmp.write_text(json.dumps({"repo": "google/fonts", "ref": ref, "fonts": entries},
                              indent=1), encoding="utf-8")
    os.replace(tmp, manifest)
    print(f"{len(entries)} font files at {ref}; manifest {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
