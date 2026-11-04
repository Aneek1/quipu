"""scripts/ocr/train_ctc.py end to end on CPU with the tiny config."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tests.ocr_fixtures import make_font_dir

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_ctc", ROOT / "scripts" / "ocr" / "train_ctc.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)
_spec2 = importlib.util.spec_from_file_location("eval_ocr", ROOT / "scripts" / "ocr" / "eval_ocr.py")
ev = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ev)

SMOKE = str(ROOT / "configs" / "ocr-ctc-smoke.toml")


def _setup(tmp_path):
    fonts = make_font_dir(tmp_path / "fonts")
    text = tmp_path / "train.txt"
    text.write_text("\n".join(f"line {i} of text" for i in range(100)) + "\n", encoding="utf-8")
    ev.make_sets(text, fonts, tmp_path / "eval", n=4, seed=0)
    return ["--config", SMOKE,
            "--set", f"data.text={text}", "--set", f"data.fonts={fonts}",
            "--set", f"data.eval_dir={tmp_path / 'eval'}",
            "--set", f"train.ckpt_dir={tmp_path / 'run'}"]


def test_smoke_trains_checkpoints_and_resumes(tmp_path):
    args = _setup(tmp_path)
    assert tc.main(args) == 0
    run = tmp_path / "run"
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert ck["step"] == 6
    assert (run / "best.pt").is_file()
    log = [json.loads(s) for s in (run / "log.jsonl").read_text().splitlines()]
    assert any("cer_printed" in e for e in log)
    for e in log:
        assert {"lr", "elapsed_s", "lines_per_s", "cer_hand", "cer_nonsense"} <= e.keys()
    # resume with more steps: starts at 6, ends at 9
    assert tc.main(args + ["--set", "train.steps=9"]) == 0
    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 9


def test_placeholder_paths_fail_fast(tmp_path):
    with pytest.raises(SystemExit, match="PLACEHOLDER"):
        tc.main(["--config", SMOKE])


def test_set_rejects_an_unknown_key():
    with pytest.raises(SystemExit, match="train.stepz"):
        tc.load_config(SMOKE, ["train.stepz=5"])
    with pytest.raises(SystemExit, match="optim.lr"):
        tc.load_config(SMOKE, ["optim.lr=5"])


def test_set_coerces_to_the_config_type(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[a]\nflag = false\nn = 3\nx = 0.5\ns = "hi"\n', encoding="utf-8")
    cfg = tc.load_config(str(p), ["a.flag=true", "a.n=7", "a.x=1e-3", "a.s=there"])
    assert cfg["a"] == {"flag": True, "n": 7, "x": 1e-3, "s": "there"}
    assert tc.load_config(str(p), ["a.flag=False"])["a"]["flag"] is False
    with pytest.raises(SystemExit, match="a.flag"):
        tc.load_config(str(p), ["a.flag=yes"])
    with pytest.raises(SystemExit, match="a.n"):
        tc.load_config(str(p), ["a.n=lots"])
