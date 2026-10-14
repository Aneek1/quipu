"""Distillation v2 verification on Modal CPUs: scripts/distill_verify.py, one container per part.

Runs the same verifier (quipu.sandbox) as the laptop, on the Volume "quipu-distill-v2"
(see modal_distill.py). Parts whose prompts or generations are not on the Volume yet are
uploaded first; results come back to data/distill-v2/{verified,kept}/ and the
summary is rebuilt locally.

    modal run scripts/remote/modal_verify.py --parts "1,2,3,4,12,13,14,15,16,17,18,19"

A part without a .done file (generation stopped early, e.g. part 4) is verified as is:
the answers it has are complete rows, only the part is short. The sandbox's Python has
the same third-party packages as the laptop's (the answers' tests import some of them)."""
import gzip
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


# Read-only: parallel containers committing to one Volume once lost its prompts/ and
# generations/ folders. Results come back in the return value (gzip) instead.
@app.function(cpu=16.0, memory=16384, volumes={"/vol": vol.read_only()}, timeout=3 * 3600)
def verify(part: int, workers: int = 32, timeout: float = 10.0) -> dict:
    sys.path.insert(0, "/src")
    from scripts.distill_verify import verify_part

    # A fresh private root per call (a container can serve several parts): generation
    # files linked in from the Volume, outputs written locally.
    import gzip
    import tempfile
    root = Path(tempfile.mkdtemp(prefix=f"part{part}-"))
    gens = root / "generations"
    gens.mkdir()
    (root / "prompts").symlink_to("/vol/prompts")
    files = sorted(Path("/vol/generations").glob(f"part-{part:04d}.*.jsonl"))
    if not files:
        raise RuntimeError(f"part {part}: no generations on the Volume")
    hashes = {}
    for f in files:
        (gens / f.name).symlink_to(f)
        hashes[f.name] = sha256(f)
        done = f.with_suffix(".done")
        (gens / done.name).write_text(done.read_text() if done.exists() else "{}")
    s = verify_part(root, part, workers, timeout)
    print(f"part {part}: {s}", flush=True)
    # Only the verdicts come back (the kept file is ~15x larger and is rebuilt locally
    # from the laptop's own prompts and generations: build_kept).
    files = {"verified": gzip.compress((root / "verified" / f"part-{part:04d}.jsonl").read_bytes())}
    return {"part": part, **s, "sha256": hashes, "files": files}


@app.local_entrypoint()
def main(parts: str = "", workers: int = 32):
    todo = [int(p) for p in parts.split(",") if p.strip()]
    if not todo:
        raise SystemExit("--parts \"2,3,4\" is required")
    have = set()
    for d in ("prompts", "generations"):
        try:
            have |= {e.path for e in vol.listdir(d)}
        except Exception:                                    # folder not there yet
            pass
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

    failed = []
    sys.path.insert(0, str(HERE))
    from scripts.distill_verify import build_kept

    for p, out in zip(todo, verify.starmap([(p, workers) for p in todo], return_exceptions=True)):
        if isinstance(out, BaseException):
            print(f"part {p}: FAILED {out!r}", flush=True)
            failed.append(p)
            continue
        for name, digest in out["sha256"].items():     # the Volume's copy is the laptop's
            local = DATA / "generations" / name
            if sha256(local) != digest:
                raise SystemExit(f"part {p}: {name} differs from the laptop's copy")
        dest = DATA / "verified" / f"part-{p:04d}.jsonl"
        dest.parent.mkdir(exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(gzip.decompress(out["files"]["verified"]))
        os.replace(tmp, dest)
        if build_kept(DATA, p) != out["kept"]:
            raise SystemExit(f"part {p}: rebuilt kept file has a different row count")
        # Stamp with the laptop's files, so the local verifier treats the part as up to date.
        srcs = sorted((DATA / "generations").glob(f"part-{p:04d}.*.jsonl"))
        stamp = {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in srcs}
        (DATA / "verified" / f"part-{p:04d}.meta.json").write_text(json.dumps(stamp), encoding="utf-8")
        print(f"part {p}: {out['kept']}/{out['answers']} kept "
              f"({100 * out['kept'] / out['answers']:.1f}%)", flush=True)

    sys.path.insert(0, str(HERE))
    from scripts.distill_verify import summarize
    summary = summarize(DATA)
    if failed:
        print("failed parts (rerun them):", ",".join(map(str, failed)))
    print(json.dumps({k: summary[k] for k in ("answers", "kept", "kept_pct")}))
