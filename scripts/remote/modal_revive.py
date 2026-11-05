"""Run scripts/revive_experts.py on the stable checkpoint on Volume quipu-ckpt, in a
CPU-only Modal container next to the Volume (no 8 GB download).

Run it with plain Python (the Python that has the `modal` package), NOT `modal run`:

    python scripts/remote/modal_revive.py --plan-only     # what would run, the cost
    python scripts/remote/modal_revive.py --dry-run       # measure + revise, write nothing
    python scripts/remote/modal_revive.py                 # revive layer 0 in place

In place means: the revived checkpoint is written to /stable/.revive_tmp/ first; only
when that finished is the original moved to /stable/pre-revive/<name> and the revived
file renamed to <name>, so latest.pt (which names the file) needs no change and the
next `modal_train.py` launch resumes from it. The report <name>.revive.json is kept in
/stable/pre-revive/. Refused if /stable/pre-revive/<name> already exists (revived once).

Load is measured on one pool shard (mixed/, the CPT training mix). Budget guard as in
modal_train.py: metered this month + this launch's worst case must stay within $27.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

LOCAL = modal is None or modal.is_local()
HERE = Path(__file__).resolve().parents[2] if LOCAL else Path("/src")
sys.path.insert(0, str(HERE / "scripts" / "remote"))
sys.path.insert(0, str(HERE))

if LOCAL:
    # Pricing and the budget guard live in modal_train.py; the container does not need
    # (or import) it, so its own image and app definitions never run there.
    from modal_train import (CAP_USD, CPU_USD_PER_CORE_HOUR, MEM_USD_PER_GIB_HOUR,  # noqa: E402
                             check_budget, read_metered_cost)

# Copies of modal_train.CONFIG and .PIP (tests check they match).
CONFIG = "configs/quipu-moe-cpt.toml"
PIP = ["torch==2.11.0", "numpy==2.5.3", "tokenizers==0.23.2", "tiktoken==0.14.0",
       "safetensors==0.8.0", "torchao==0.18.0"]
CPU, MEMORY_GIB, TIMEOUT_S = 4.0, 32, 40 * 60
POOL_SHARD = "mixed/shard_00100.bin"
CKPT_MOUNT, POOL_MOUNT = "/ckpt", "/pool"


def worst_case_usd() -> float:
    return (CPU * CPU_USD_PER_CORE_HOUR + MEMORY_GIB * MEM_USD_PER_GIB_HOUR) * TIMEOUT_S / 3600


def _pointer_file(ckpt_dir: Path) -> Path | None:
    """The checkpoint latest.pt points at, or None (modal_train._pointer_file)."""
    latest = ckpt_dir / "latest.pt"
    if not latest.is_file():
        return None
    import torch

    path = ckpt_dir / torch.load(latest, map_location="cpu", weights_only=True)["file"]
    if not path.is_file():
        raise FileNotFoundError(f"{latest} points at {path}, which is missing")
    return path


def revive_in_place(stable: Path, tokens: Path, dry_run: bool, layer: int = 0,
                    config: str = CONFIG, cwd: Path = HERE) -> dict:
    """The container's work, as a plain function (tests call it on the CPU)."""
    import json
    import importlib.util

    spec = importlib.util.spec_from_file_location("revive_experts",
                                                  cwd / "scripts" / "revive_experts.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)

    src = _pointer_file(stable)
    if src is None:
        raise FileNotFoundError(f"no latest.pt in {stable}")
    aside = stable / "pre-revive" / src.name
    if aside.exists():
        raise FileExistsError(f"{aside} exists: {src.name} was revived already")
    tmp_dir = stable / ".revive_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    out = (Path("/tmp/revive") if dry_run else tmp_dir) / src.name
    if dry_run and out.parent.exists():
        shutil.rmtree(out.parent)
    argv = ["--config", str(cwd / config), "--checkpoint", str(src), "--tokens", str(tokens),
            "--out", str(out), "--layer", str(layer)]
    code = rv.main(argv)
    if code != 0:
        raise RuntimeError(f"revive_experts exited {code}")
    report_path = Path(str(out) + ".revive.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {"pairs": []}
    if dry_run or not out.exists():
        return {"checkpoint": src.name, "written": False, **report}
    aside.parent.mkdir(parents=True, exist_ok=True)
    src.rename(aside)
    out.rename(src)
    report_path.rename(aside.parent / report_path.name)
    shutil.rmtree(tmp_dir)
    return {"checkpoint": src.name, "written": True, "original": str(aside), **report}


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(*PIP)
        .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/src"})
        .add_local_dir(HERE / "quipu", "/src/quipu", ignore=["__pycache__", "memory"])
        .add_local_file(HERE / "configs" / "quipu-moe.toml", "/src/configs/quipu-moe.toml")
        .add_local_file(HERE / CONFIG, f"/src/{CONFIG}")
        .add_local_file(HERE / "scripts" / "revive_experts.py", "/src/scripts/revive_experts.py")
    )
    ckpt_vol = modal.Volume.from_name("quipu-ckpt")
    pool_vol = modal.Volume.from_name("quipu-pool")
    app = modal.App("quipu-revive", image=image)

    @app.function(cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT_S,
                  volumes={CKPT_MOUNT: ckpt_vol, POOL_MOUNT: pool_vol})
    def revive(dry_run: bool) -> dict:
        ckpt_vol.reload()
        result = revive_in_place(Path(CKPT_MOUNT) / "stable", Path(POOL_MOUNT) / POOL_SHARD,
                                 dry_run)
        if result["written"]:
            ckpt_vol.commit()
        return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    usd_h = CPU * CPU_USD_PER_CORE_HOUR + MEMORY_GIB * MEM_USD_PER_GIB_HOUR
    print(f"CPU only: {CPU:g} cores, {MEMORY_GIB} GiB, ${usd_h:.2f}/h list; timeout "
          f"{TIMEOUT_S // 60} min -> worst case ${worst_case_usd():.2f}")
    print(f"revive layer 0 of /stable (latest.pt's file) on Volume quipu-ckpt, load "
          f"measured on quipu-pool {POOL_SHARD}" + (" (dry run: writes nothing)" if a.dry_run else ""))
    ok, msg = check_budget(read_metered_cost(), worst_case_usd(), CAP_USD)
    print(msg)
    if not ok:
        return 4
    if a.plan_only:
        return 0
    with modal.enable_output(), app.run():
        result = revive.remote(a.dry_run)
    pairs = result.pop("pairs", [])
    for k in ("load_before", "load_after"):
        v = result.pop(k, None)
        if v:
            dead = sum(x < result.get("threshold", 0.1) for x in v)
            print(f"{k}: dead {dead}, min {min(v):.2f}, max {max(v):.2f}")
    print(f"{len(pairs)} experts revived: {pairs}")
    print({k: v for k, v in result.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
