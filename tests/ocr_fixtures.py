"""A font directory for OCR tests, built from matplotlib's bundled DejaVu Sans so no
download is needed. The licence file is a stand-in; only its presence is checked.
The manifest carries the real sha256 of the font file, which load_fonts verifies."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

DEJAVU = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"


def make_font_dir(root: Path, families=(("printed", "alpha"), ("printed", "beta"),
                                        ("hand", "gamma"), ("hand", "delta"))) -> Path:
    entries = []
    for style, fam in families:
        d = root / "ofl" / fam
        d.mkdir(parents=True)
        shutil.copy(DEJAVU, d / f"{fam}.ttf")
        (d / "OFL.txt").write_text("test licence stand-in\n", encoding="utf-8")
        entries.append({"family": fam, "style": style, "path": f"ofl/{fam}/{fam}.ttf",
                        "licence": f"ofl/{fam}/OFL.txt", "sha256": hashlib.sha256(DEJAVU.read_bytes()).hexdigest()})
    (root / "fonts.json").write_text(json.dumps({"ref": "test", "fonts": entries}),
                                     encoding="utf-8")
    return root


def drop_glyphs(font_dir: Path, family: str, chars: str) -> None:
    """Rewrite one family's font in a make_font_dir tree without the glyphs for `chars`
    (a real, smaller font file) and update its sha256 in fonts.json."""
    from fontTools import subset
    from fontTools.ttLib import TTFont

    path = font_dir / "ofl" / family / f"{family}.ttf"
    font = TTFont(str(path))
    keep = set(font.getBestCmap()) - {ord(c) for c in chars}
    sub = subset.Subsetter(subset.Options())
    sub.populate(unicodes=keep)
    sub.subset(font)
    font.save(str(path))
    m = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))
    for e in m["fonts"]:
        if e["family"] == family:
            e["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (font_dir / "fonts.json").write_text(json.dumps(m), encoding="utf-8")
