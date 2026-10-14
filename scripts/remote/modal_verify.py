"""Distillation v2 verification on Modal CPUs: scripts/distill_verify.py, one container per part.

Runs the same verifier (quipu.sandbox) as the laptop, on the Volume "quipu-distill-v2"
(see modal_distill.py). Parts whose prompts or generations are not on the Volume yet are
uploaded first; results are copied back to data/distill-v2/{verified,kept}/ and the
summary is rebuilt locally.

    modal run scripts/remote/modal_verify.py --parts "1,2,3,4,12,13,14,15,16,17,18,19"

A part without a .done file (generation stopped early, e.g. part 4) is verified as is:
the answers it has are complete rows, only the part is short. The sandbox's Python has
the same third-party packages as the laptop's (the answers' tests import some of them)."""
import hashlib
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/src")
DATA = HERE / "data" / "distill-v2"
SANDBOX_PKGS = ["numpy", "pandas", "sympy", "regex", "requests", "beautifulsoup4", "networkx",
                "python-dateutil", "pytz", "pyyaml"]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*SANDBOX_PKGS)
    .add_local_file(HERE / "quipu" / "__init__.py", "/src/quipu/__init__.py")
    .add_local_file(HERE / "quipu" / "sandbox.py", "/src/quipu/sandbox.py")
    .add_local_file(HERE / "scripts" / "distill_verify.py", "/src/scripts/distill_verify.py")
)
vol = modal.Volume.from_name("quipu-distill-v2", create_if_missing=True)
app = modal.App("quipu-distill-verify", image=image)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(cpu=16.0, memory=16384, volumes={"/vol": vol}, timeout=3 * 3600)
def verify(part: int, workers: int = 32, timeout: float = 10.0) -> dict:
    sys.path.insert(0, "/src")
    from scripts.distill_verify import verify_part

    # A private root: generation files linked in from the Volume, outputs written to it.
    root = Path("/tmp/root")
    gens = root / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    (root / "prompts").symlink_to("/vol/prompts")
    for d in ("verified", "kept"):
        Path("/vol", d).mkdir(exist_ok=True)
        (root / d).symlink_to(f"/vol/{d}")
    files = sorted(Path("/vol/generations").glob(f"part-{part:04d}.*.jsonl"))
    if not files:
        raise RuntimeError(f"part {part}: no generations on the Volume")
    hashes = {}
    for f in files:
        (gens / f.name).symlink_to(f)
        hashes[f.name] = sha256(f)
        done = f.with_suffix(".done")
        (gens / done.name).write_text(done.read_text() if done.exists() else "{}")
    meta = Path("/vol/verified") / f"part-{part:04d}.meta.json"
    meta.unlink(missing_ok=True)                      # always re-verify what was asked
    s = verify_part(root, part, workers, timeout)
    vol.commit()
    print(f"part {part}: {s}", flush=True)
    return {"part": part, **s, "sha256": hashes}


@app.local_entrypoint()
def main(parts: str = "", workers: int = 32):
    todo = [int(p) for p in parts.split(",") if p.strip()]
    if not todo:
        raise SystemExit("--parts \"2,3,4\" is required")
    have = {e.path for d in ("prompts", "generations") for e in vol.listdir(d)}
    with vol.batch_upload() as up:
        for p in todo:
            src = [DATA / "prompts" / f"part-{p:04d}.jsonl",
                   *sorted((DATA / "generations").glob(f"part-{p:04d}.*.jsonl")),
                   *sorted((DATA / "generations").glob(f"part-{p:04d}.*.done"))]
            for f in src:
                rel = f"{f.parent.name}/{f.name}"
                if rel not in have:
                    print("uploading", rel, flush=True)
                    up.put_file(str(f), rel)

    for out in verify.starmap([(p, workers) for p in todo]):
        p = out["part"]
        for name, digest in out["sha256"].items():     # the Volume's copy is the laptop's
            local = DATA / "generations" / name
            if sha256(local) != digest:
                raise SystemExit(f"part {p}: {name} differs from the laptop's copy")
        for d in ("verified", "kept"):
            dest = DATA / d / f"part-{p:04d}.jsonl"
            dest.parent.mkdir(exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in vol.read_file(f"{d}/part-{p:04d}.jsonl"):
                    f.write(chunk)
            os.replace(tmp, dest)
        # Stamp with the laptop's files, so the local verifier treats the part as up to date.
        srcs = sorted((DATA / "generations").glob(f"part-{p:04d}.*.jsonl"))
        stamp = {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in srcs}
        (DATA / "verified" / f"part-{p:04d}.meta.json").write_text(json.dumps(stamp), encoding="utf-8")
        print(f"part {p}: {out['kept']}/{out['answers']} kept "
              f"({100 * out['kept'] / out['answers']:.1f}%)", flush=True)

    sys.path.insert(0, str(HERE))
    from scripts.distill_verify import verify_parts
    summary = verify_parts(DATA, workers=1, timeout=10.0)   # every part is up to date: summary only
    print(json.dumps({k: summary[k] for k in ("answers", "kept", "kept_pct")}))
