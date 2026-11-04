"""Train the Quipu-OCR encoder with a CTC head (spec section 5.1).

    .venv/Scripts/python scripts/ocr/train_ctc.py --config configs/ocr-ctc.toml \
        [--set train.steps=40000] [--resume /kaggle/input/<prev>/latest.pt] [--fresh]

Single GPU (the model is ~22M parameters; a second T4 is not worth DDP here). The
default precision is fp16 with a grad scaler (a T4 has no bf16). Logs one JSON line per
eval to <ckpt_dir>/log.jsonl: step, mean loss since the last eval, lr, elapsed_s (summed
over sessions), lines_per_s (this session), the grad-scaler scale under fp16, and the CER
of each eval set.

Checkpoints: best.pt when the printed-set CER improves, then latest.pt every eval
(resume takes the lower of the two best CERs, so a crash between them is safe).
<ckpt_dir>/latest.pt is resumed automatically; --resume PATH first copies a latest.pt
(and the best.pt and log.jsonl beside it) from elsewhere, e.g. a read-only Kaggle input.
On resume, log entries after the checkpoint's step are dropped, and a run that already
stopped early stays stopped. A fresh start over an existing log.jsonl needs --fresh,
which moves the old log and best.pt aside to *.prev. Stops early when the printed CER
has not improved for `patience` evals in a row. The data stream is seeded by
seed + step on resume, so a resumed run does not replay the same images. Runs are not
bit-for-bit reproducible on GPU: CUDA's CTC backward has no deterministic kernel."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "ocr"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from eval_ocr import read_set  # noqa: E402
from quipu.ocr_ctc import (LineCTC, check_ckpt_compat, ckpt_compat,  # noqa: E402
                           ctc_greedy_decode, ctc_loss)
from quipu.ocr_data import LineStream, collate, preprocess  # noqa: E402
from quipu.ocr_metrics import cer  # noqa: E402
from quipu.ocr_render import load_fonts, split_fonts  # noqa: E402
from quipu.vision import ViTConfig  # noqa: E402

PRECISIONS = ("bf16", "fp16", "fp32")
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


def check_precision(precision: str, device: str) -> None:
    if precision not in PRECISIONS:
        raise SystemExit(f"train.precision={precision!r}: must be one of {', '.join(PRECISIONS)}")
    if precision == "bf16" and device == "cuda":
        cap = tuple(torch.cuda.get_device_capability())
        if cap < (8, 0):
            raise SystemExit(f"train.precision=bf16 needs a GPU of compute capability 8.0 or "
                             f"newer; this one is {cap[0]}.{cap[1]} (a T4 is 7.5). "
                             "Use --set train.precision=fp16")


def load_scaler_state(scaler, state: dict) -> None:
    """A disabled GradScaler saves {} and an enabled one refuses to load {}: then (and
    when this scaler is disabled) the scaler keeps its fresh state."""
    if state and scaler.is_enabled():
        scaler.load_state_dict(state)


@torch.no_grad()
def evaluate(model, eval_dir: Path, n: int, device, autocast) -> dict:
    """CER on the first n lines of each eval set. The sets are built to fit (eval_ocr
    make), so a LineTooWide from preprocess is let through: the set and code disagree."""
    model.eval()
    try:
        out = {}
        for name in EVAL_SETS:
            rows = read_set(eval_dir / name)[:n]
            refs, hyps = [], []
            for i in range(0, len(rows), 64):
                items = []
                for p, t, *_ in rows[i:i + 64]:
                    with Image.open(p) as im:
                        items.append((preprocess(im), t))
                images, mask, texts = collate(items)
                with autocast():
                    lp, lens = model(images.to(device), mask.to(device))
                hyps += ctc_greedy_decode(lp.float().cpu(), lens.cpu())
                refs += texts
            out[f"cer_{name}"] = cer(refs, hyps)
        return out
    finally:
        model.train()


def save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _prepare_dir(ckpt_dir: Path, resume: str | None, fresh: bool) -> None:
    """Copy a --resume checkpoint in, or guard a fresh start against an old run."""
    latest, log = ckpt_dir / "latest.pt", ckpt_dir / "log.jsonl"
    if resume:
        if fresh:
            raise SystemExit("--resume and --fresh contradict each other")
        src = Path(resume)
        if not src.is_file():
            raise SystemExit(f"--resume {src}: no such file")
        if latest.exists():
            raise SystemExit(f"{latest} exists already; drop --resume to continue it, or "
                             "use another train.ckpt_dir")
        shutil.copyfile(src, latest)
        for name in ("best.pt", "log.jsonl"):
            if (src.parent / name).is_file():
                shutil.copyfile(src.parent / name, ckpt_dir / name)
        print(f"copied {src} (and best.pt, log.jsonl beside it, if any) into {ckpt_dir}")
    elif fresh:
        if latest.exists():
            raise SystemExit(f"--fresh, but {latest} exists: delete it or use another "
                             "train.ckpt_dir")
        for name in ("log.jsonl", "best.pt"):
            if (ckpt_dir / name).exists():
                os.replace(ckpt_dir / name, ckpt_dir / (name + ".prev"))
    elif log.exists() and not latest.exists():
        raise SystemExit(f"{log} exists but {latest} does not: a run was here. Pass --fresh "
                         "to start over (the old log and best.pt move to *.prev) or "
                         "--resume PATH to continue from a checkpoint elsewhere")


def _truncate_log(log: Path, step: int) -> None:
    """Drop entries written after the checkpoint (a session that died between them) and
    any line that is not valid JSON (one killed mid-write leaves a torn last line)."""
    if not log.is_file():
        return
    keep = []
    for s in log.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("step", 0) <= step:
            keep.append(s)
    log.write_text("".join(k + "\n" for k in keep), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="section.key=value")
    ap.add_argument("--resume", help="a latest.pt to copy into ckpt_dir and continue")
    ap.add_argument("--fresh", action="store_true",
                    help="start over in a ckpt_dir that has an old log.jsonl")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    t, d = cfg["train"], cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_precision(t["precision"], device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[t["precision"]]
    autocast = (lambda: torch.autocast(device, dtype=dtype)) if dtype else (
        lambda: torch.autocast(device, enabled=False))
    scaler = torch.amp.GradScaler(device, enabled=t["precision"] == "fp16")

    ckpt_dir = Path(t["ckpt_dir"]).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _prepare_dir(ckpt_dir, args.resume, args.fresh)
    vit = ViTConfig(**cfg["model"])
    model = LineCTC(vit).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    step, best, bad, elapsed0 = 0, float("inf"), 0, 0.0
    if (ckpt_dir / "latest.pt").is_file():
        ck = torch.load(ckpt_dir / "latest.pt", weights_only=False, map_location=device)
        try:
            check_ckpt_compat(ck, ckpt_dir / "latest.pt")
        except ValueError as e:
            raise SystemExit(str(e)) from None
        if ck.get("precision") != t["precision"]:
            print(f"warning: precision changed on resume: {ck.get('precision')} -> "
                  f"{t['precision']}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        load_scaler_state(scaler, ck["scaler"])
        step, best, bad = ck["step"], ck["best_cer"], ck["bad_evals"]
        elapsed0 = ck.get("elapsed_s", 0.0)
        del ck
        if (ckpt_dir / "best.pt").is_file():
            best = min(best, torch.load(ckpt_dir / "best.pt", weights_only=False,
                                        map_location="cpu")["cer_printed"])
        _truncate_log(ckpt_dir / "log.jsonl", step)
        print(f"checkpoints: {ckpt_dir} (resuming at step {step}, best printed CER {best:.4f})")
        if bad >= t["patience"]:
            print(f"already stopped early at step {step} (printed CER not improved for {bad} "
                  "evals; raise train.patience to go on)")
            return 0
    else:
        print(f"checkpoints: {ckpt_dir} (starting fresh)")

    train_fonts, _ = split_fonts(load_fonts(d["fonts"]))
    lines = Path(d["text"]).read_text(encoding="utf-8").splitlines()
    stream = LineStream(lines, train_fonts, d["hand_frac"], d["max_chars"], t["seed"] + step)
    del lines                     # the stream keeps its filtered copy; workers get their own
    cuda = device == "cuda"
    loader = iter(DataLoader(stream, batch_size=t["batch"], collate_fn=collate,
                             num_workers=d["workers"], pin_memory=cuda))
    log = open(ckpt_dir / "log.jsonl", "a", encoding="utf-8")
    t0, seen = time.time(), 0
    loss_sum, loss_n = torch.zeros((), device=device), 0
    model.train()
    while step < t["steps"]:
        images, mask, texts = next(loader)
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast():
            lp, lens = model(images.to(device, non_blocking=cuda),
                             mask.to(device, non_blocking=cuda))
        loss = ctc_loss(lp.float(), lens, texts)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += len(texts)
        loss_sum += loss.detach()                 # no .item(): no GPU sync every step
        loss_n += 1
        if step % t["eval_every"] == 0 or step == t["steps"]:
            metrics = evaluate(model, Path(d["eval_dir"]), t["eval_n"], device, autocast)
            session = time.time() - t0
            entry = {"step": step, "loss": round((loss_sum / loss_n).item(), 4), "lr": lr,
                     "elapsed_s": round(elapsed0 + session, 1),
                     "lines_per_s": round(seen / session, 1)}
            if scaler.is_enabled():
                entry["scale"] = scaler.get_scale()
            entry.update(metrics)
            loss_sum, loss_n = torch.zeros((), device=device), 0
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            improved = metrics["cer_printed"] < best
            if improved:
                best, bad = metrics["cer_printed"], 0
            else:
                bad += 1
            meta = {"vit": vars(vit), "step": step, "precision": t["precision"],
                    **ckpt_compat()}
            if improved:          # before latest.pt; resume takes min(latest, best.pt)
                save(ckpt_dir / "best.pt", {"model": model.state_dict(), "cer_printed": best,
                                            **meta})
            save(ckpt_dir / "latest.pt", {"model": model.state_dict(), "opt": opt.state_dict(),
                                          "scaler": scaler.state_dict(), "best_cer": best,
                                          "bad_evals": bad, "elapsed_s": elapsed0 + session,
                                          **meta})
            if bad >= t["patience"]:
                print(f"stopping: printed CER has not improved for {bad} evals")
                break
    del loader                                # shuts the workers down
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
