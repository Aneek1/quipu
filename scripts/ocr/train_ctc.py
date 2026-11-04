"""Train the Quipu-OCR encoder with a CTC head (spec section 5.1).

    .venv/Scripts/python scripts/ocr/train_ctc.py --config configs/ocr-ctc.toml \
        [--set train.precision=fp16] [--set train.steps=40000]

Single GPU (the model is ~22M parameters; a second T4 is not worth DDP here). Logs one
JSON line per eval to <ckpt_dir>/log.jsonl (step, loss, lr, elapsed_s, lines_per_s and
the CER of each eval set). Checkpoints: latest.pt every eval (resumed automatically),
best.pt whenever the printed-set CER improves. Stops early when that CER has not
improved for `patience` evals in a row. The data stream is seeded by seed + step on
resume, so a resumed run does not replay the same images. Runs are not bit-for-bit
reproducible on GPU: CUDA's CTC backward has no deterministic kernel."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from quipu.ocr_ctc import LineCTC, ctc_greedy_decode, ctc_loss  # noqa: E402
from quipu.ocr_data import LineStream, collate, preprocess  # noqa: E402
from quipu.ocr_metrics import cer  # noqa: E402
from quipu.ocr_render import load_fonts, split_fonts  # noqa: E402
from quipu.vision import ViTConfig  # noqa: E402

EVAL_SETS = ("printed", "hand", "nonsense")


def _coerce(key: str, old, value: str):
    if isinstance(old, bool):                    # before int: bool is an int subclass
        if value.lower() not in ("true", "false"):
            raise SystemExit(f"--set {key}={value}: expected true or false")
        return value.lower() == "true"
    try:
        return type(old)(value)
    except ValueError:
        raise SystemExit(f"--set {key}={value}: expected a {type(old).__name__}") from None


def load_config(path: str, sets: list[str]) -> dict:
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for item in sets:
        key, eq, value = item.partition("=")
        section, dot, name = key.partition(".")
        if not eq or not dot:
            raise SystemExit(f"--set {item!r}: expected section.key=value")
        if section not in cfg or name not in cfg[section]:
            raise SystemExit(f"--set {key}: unknown config key")
        cfg[section][name] = _coerce(key, cfg[section][name], value)
    for section in cfg.values():
        for k, v in section.items():
            if isinstance(v, str) and "PLACEHOLDER" in v:
                raise SystemExit(f"config value {k}={v!r}: PLACEHOLDER not replaced")
    return cfg


def lr_at(step: int, cfg: dict) -> float:
    t = cfg["train"]
    if step < t["warmup"]:
        return t["lr"] * (step + 1) / t["warmup"]
    p = (step - t["warmup"]) / max(1, t["steps"] - t["warmup"])
    return t["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, eval_dir: Path, n: int, device, autocast) -> dict:
    """CER on the first n lines of each eval set. The sets are built to fit (eval_ocr
    make), so a LineTooWide from preprocess is let through: the set and code disagree."""
    sys.path.insert(0, str(ROOT / "scripts" / "ocr"))
    from eval_ocr import read_set
    model.eval()
    out = {}
    for name in EVAL_SETS:
        rows = read_set(eval_dir / name)[:n]
        refs, hyps = [], []
        for i in range(0, len(rows), 64):
            items = []
            for p, t in rows[i:i + 64]:
                with Image.open(p) as im:
                    items.append((preprocess(im), t))
            images, mask, texts = collate(items)
            with autocast():
                lp, lens = model(images.to(device), mask.to(device))
            hyps += ctc_greedy_decode(lp.float().cpu(), lens.cpu())
            refs += texts
        out[f"cer_{name}"] = cer(refs, hyps)
    model.train()
    return out


def save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    t, d = cfg["train"], cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[t["precision"]]
    autocast = (lambda: torch.autocast(device, dtype=dtype)) if dtype else (
        lambda: torch.autocast(device, enabled=False))
    scaler = torch.amp.GradScaler(device, enabled=t["precision"] == "fp16")

    ckpt_dir = Path(t["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vit = ViTConfig(**cfg["model"])
    model = LineCTC(vit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    step, best, bad = 0, float("inf"), 0
    if (ckpt_dir / "latest.pt").is_file():
        ck = torch.load(ckpt_dir / "latest.pt", weights_only=False, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, bad = ck["step"], ck["best_cer"], ck["bad_evals"]
        print(f"resumed at step {step} (best printed CER {best:.4f})")

    train_fonts, _ = split_fonts(load_fonts(d["fonts"]))
    lines = Path(d["text"]).read_text(encoding="utf-8").splitlines()
    stream = LineStream(lines, train_fonts, d["hand_frac"], d["max_chars"], t["seed"] + step)
    del lines                     # the stream keeps its filtered copy; workers get their own
    loader = iter(DataLoader(stream, batch_size=t["batch"], collate_fn=collate,
                             num_workers=d["workers"], persistent_workers=d["workers"] > 0))
    log = open(ckpt_dir / "log.jsonl", "a", encoding="utf-8")
    t0, seen = time.time(), 0
    model.train()
    while step < t["steps"]:
        images, mask, texts = next(loader)
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast():
            lp, lens = model(images.to(device), mask.to(device))
        loss = ctc_loss(lp.float(), lens, texts)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += len(texts)
        if step % t["eval_every"] == 0 or step == t["steps"]:
            metrics = evaluate(model, Path(d["eval_dir"]), t["eval_n"], device, autocast)
            elapsed = time.time() - t0
            entry = {"step": step, "loss": round(loss.item(), 4), "lr": lr,
                     "elapsed_s": round(elapsed, 1),
                     "lines_per_s": round(seen / elapsed, 1), **metrics}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if metrics["cer_printed"] < best:
                best, bad = metrics["cer_printed"], 0
                save(ckpt_dir / "best.pt", {"model": model.state_dict(), "vit": vars(vit),
                                            "step": step, "cer_printed": best})
            else:
                bad += 1
            save(ckpt_dir / "latest.pt", {"model": model.state_dict(), "opt": opt.state_dict(),
                                          "scaler": scaler.state_dict(), "step": step,
                                          "best_cer": best, "bad_evals": bad, "vit": vars(vit)})
            if bad >= t["patience"]:
                print(f"stopping: printed CER has not improved for {bad} evals")
                break
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
