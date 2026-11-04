"""scripts/ocr/train_ctc.py end to end on CPU with the tiny config."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from quipu import ocr_ctc as oc
from tests.ocr_fixtures import make_font_dir

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_ctc", ROOT / "scripts" / "ocr" / "train_ctc.py")
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)
_spec2 = importlib.util.spec_from_file_location("eval_ocr", ROOT / "scripts" / "ocr" / "eval_ocr.py")
ev = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ev)

SMOKE = str(ROOT / "configs" / "ocr-ctc-smoke.toml")


def _setup(tmp_path, eval_sets=True, run="run"):
    fonts = tmp_path / "fonts"
    if not fonts.exists():
        make_font_dir(fonts)
    text = tmp_path / "train.txt"
    text.write_text("\n".join(f"line {i} of text" for i in range(100)) + "\n", encoding="utf-8")
    if eval_sets and not (tmp_path / "eval").exists():
        ev.make_sets(text, fonts, tmp_path / "eval", n=4, seed=0, n_val=4)
    return ["--config", SMOKE,
            "--set", f"data.text={text}", "--set", f"data.fonts={fonts}",
            "--set", f"data.eval_dir={tmp_path / 'eval'}",
            "--set", f"train.ckpt_dir={tmp_path / run}"]


def _ck(path):
    return torch.load(path, weights_only=False)


def _fake_eval(monkeypatch, cers):
    """evaluate() returns the given val CERs in order (the test sets 0.9)."""
    it = iter(cers)
    monkeypatch.setattr(tc, "evaluate", lambda *a, **k: {
        "cer_val": next(it), "cer_printed": 0.9, "cer_hand": 0.9, "cer_nonsense": 0.9})


def _log(run):
    return [json.loads(s) for s in (run / "log.jsonl").read_text().splitlines()]


def test_smoke_trains_checkpoints_and_resumes(tmp_path):
    args = _setup(tmp_path)
    assert tc.main(args) == 0
    run = tmp_path / "run"
    ck = _ck(run / "latest.pt")
    assert ck["step"] == 6 and ck["precision"] == "fp32" and ck["elapsed_s"] > 0
    assert ck["charset_sha256"] == oc.CHARSET_SHA256 and ck["steps_per_col"] == oc.STEPS_PER_COL
    best = _ck(run / "best.pt")
    assert best["charset_sha256"] == oc.CHARSET_SHA256
    log = _log(run)
    for e in log:
        assert {"loss", "lr", "elapsed_s", "lines_per_s", "cer_val", "cer_printed", "cer_hand",
                "cer_nonsense"} <= e.keys()
        assert "scale" not in e                           # fp32: no grad scaler
    # resume with more steps: starts at 6, ends at 9
    assert tc.main(args + ["--set", "train.steps=9"]) == 0
    ck2 = _ck(run / "latest.pt")
    assert ck2["step"] == 9 and ck2["elapsed_s"] > ck["elapsed_s"]


def test_resume_keeps_optimizer_scaler_best_and_patience(tmp_path, monkeypatch, capsys):
    args = _setup(tmp_path, eval_sets=False) + ["--set", "train.patience=3"]
    run = tmp_path / "run"
    _fake_eval(monkeypatch, [0.5, 0.6])
    assert tc.main(args) == 0
    ck = _ck(run / "latest.pt")
    assert (ck["best_cer"], ck["bad_evals"], ck["scaler"]) == (0.5, 1, {})
    with open(run / "log.jsonl", "a") as f:                 # an entry past the checkpoint
        f.write(json.dumps({"step": 99}) + "\n")

    # fp32 -> fp16: the empty state of a disabled scaler must not be loaded
    _fake_eval(monkeypatch, [0.55])
    assert tc.main(args + ["--set", "train.steps=9", "--set", "train.precision=fp16"]) == 0
    out = capsys.readouterr().out
    assert "precision changed" in out and str(run.resolve()) in out and "resuming" in out
    ck = _ck(run / "latest.pt")
    assert (ck["step"], ck["best_cer"], ck["bad_evals"]) == (9, 0.5, 2)   # 0.55 is not better
    # AdamW kept its count: a fresh optimiser would be at <= 3 (fp16 may skip a step)
    assert ck["opt"]["state"][0]["step"] > 6
    assert ck["scaler"]["scale"] > 0 and ck["precision"] == "fp16"
    steps = [e["step"] for e in _log(run)]
    assert steps == [3, 6, 9]                                # step 99 dropped on resume
    assert "scale" in _log(run)[-1]

    _fake_eval(monkeypatch, [0.7])
    assert tc.main(args + ["--set", "train.steps=12", "--set", "train.precision=fp16"]) == 0
    assert "stopping" in capsys.readouterr().out
    assert _ck(run / "latest.pt")["bad_evals"] == 3

    _fake_eval(monkeypatch, [])                              # must not train or evaluate
    assert tc.main(args + ["--set", "train.steps=15", "--set", "train.precision=fp16"]) == 0
    assert "already stopped early at step 12" in capsys.readouterr().out
    assert _ck(run / "latest.pt")["step"] == 12


def test_resume_tolerates_a_torn_last_log_line(tmp_path, monkeypatch):
    args = _setup(tmp_path, eval_sets=False) + ["--set", "train.patience=5"]
    run = tmp_path / "run"
    _fake_eval(monkeypatch, [0.5, 0.6])
    tc.main(args)
    with open(run / "log.jsonl", "a") as f:                 # killed mid-write
        f.write('{"step": 9, "loss": 1.2')
    _fake_eval(monkeypatch, [0.4])
    assert tc.main(args + ["--set", "train.steps=9"]) == 0
    assert [e["step"] for e in _log(run)] == [3, 6, 9]


def test_resume_takes_the_lower_of_latest_and_best(tmp_path, monkeypatch):
    args = _setup(tmp_path, eval_sets=False) + ["--set", "train.patience=5"]
    run = tmp_path / "run"
    _fake_eval(monkeypatch, [0.5, 0.6])
    tc.main(args)
    best = _ck(run / "best.pt")
    torch.save({**best, "cer_val": 0.1}, run / "best.pt")  # e.g. latest written, crash
    _fake_eval(monkeypatch, [0.2])
    tc.main(args + ["--set", "train.steps=9"])
    assert _ck(run / "latest.pt")["best_cer"] == 0.1 and _ck(run / "best.pt")["cer_val"] == 0.1


def test_resume_from_another_path_copies_latest_and_best(tmp_path, monkeypatch):
    _fake_eval(monkeypatch, [0.5, 0.6])
    tc.main(_setup(tmp_path, eval_sets=False, run="old"))
    old = tmp_path / "old"
    _fake_eval(monkeypatch, [0.4])
    args = _setup(tmp_path, eval_sets=False, run="new")
    assert tc.main(args + ["--resume", str(old / "latest.pt"), "--set", "train.steps=9"]) == 0
    new = tmp_path / "new"
    assert _ck(new / "latest.pt")["step"] == 9 and _ck(new / "best.pt")["cer_val"] == 0.4
    assert _ck(old / "latest.pt")["step"] == 6                # source untouched


def test_fresh_start_over_an_old_log_needs_fresh(tmp_path, monkeypatch):
    args = _setup(tmp_path, eval_sets=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "log.jsonl").write_text('{"step": 3}\n')
    with pytest.raises(SystemExit, match="--fresh"):
        tc.main(args)
    _fake_eval(monkeypatch, [0.5, 0.4])
    assert tc.main(args + ["--fresh"]) == 0
    assert [e["step"] for e in _log(run)] == [3, 6]
    assert (run / "log.jsonl.prev").read_text() == '{"step": 3}\n'


def test_resume_refuses_another_charset(tmp_path, monkeypatch):
    args = _setup(tmp_path, eval_sets=False)
    run = tmp_path / "run"
    _fake_eval(monkeypatch, [0.5, 0.6])
    tc.main(args)
    torch.save({**_ck(run / "latest.pt"), "charset_sha256": "0" * 64}, run / "latest.pt")
    with pytest.raises(SystemExit, match="charset"):
        tc.main(args + ["--set", "train.steps=9"])


def test_placeholder_paths_fail_fast(tmp_path):
    with pytest.raises(SystemExit, match="PLACEHOLDER"):
        tc.main(["--config", SMOKE])


def test_precision_checks(monkeypatch):
    with pytest.raises(SystemExit, match="bf16, fp16, fp32"):
        tc.check_precision("fp8", "cpu")
    tc.check_precision("bf16", "cpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    with pytest.raises(SystemExit, match="7.5"):
        tc.check_precision("bf16", "cuda")
    tc.check_precision("fp16", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    tc.check_precision("bf16", "cuda")


def test_full_config_defaults_suit_a_kaggle_t4():
    cfg = tc.load_config(str(ROOT / "configs" / "ocr-ctc.toml"), [])
    assert cfg["train"]["precision"] == "fp16" and cfg["data"]["workers"] == 4


def test_load_scaler_state_skips_an_empty_state():
    on = torch.amp.GradScaler("cpu", enabled=True)
    tc.load_scaler_state(on, {})                             # does not raise
    off = torch.amp.GradScaler("cpu", enabled=False)
    tc.load_scaler_state(off, on.state_dict())               # nothing to load into


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


def test_val_set_picks_the_best_checkpoint(tmp_path, capsys):
    args = _setup(tmp_path)
    assert tc.main(args) == 0
    run = tmp_path / "run"
    log = _log(run)
    assert all("cer_val" in e for e in log)
    best = _ck(run / "best.pt")
    assert best["cer_val"] == min(e["cer_val"] for e in log)
    assert _ck(run / "latest.pt")["best_cer"] == best["cer_val"]
