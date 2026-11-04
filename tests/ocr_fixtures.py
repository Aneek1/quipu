"""A font directory for OCR tests, built from matplotlib's bundled DejaVu Sans so no
download is needed. The licence file is a stand-in; only its presence is checked."""
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
                        "licence": f"ofl/{fam}/OFL.txt", "sha256": "x"})
    (root / "fonts.json").write_text(json.dumps({"ref": "test", "fonts": entries}),
                                     encoding="utf-8")
    return root
