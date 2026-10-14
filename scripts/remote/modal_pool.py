"""Data v3 pool on Modal CPUs: scripts/build_pool.py, one container per source.

The pool lives on the Modal Volume "quipu-pool", mounted at /pool:
    /pool/sources/<name>/...   each source's unique tokens, written ONLY by its container
    /pool/mixed/shard_*.bin    the training shards, written ONLY by the merge container
    /pool/inputs/distill-v2/kept/part-*.jsonl   the verified answers, uploaded once
    /pool/pool.json, /pool/manifest.json        merge container only

Why the split: parallel containers committing the same Volume paths once lost whole
folders (the distill-v2 incident, 11 containers). Here every source directory is
created first by one container (prepare), each build container then writes only
inside its own sources/<name>/ and commits (every few minutes and at the end), and a
single merge container reads them all and writes mixed/ and the manifests.

One chunk (default 10B tokens): plan -> build each source that needs tokens, in
parallel -> merge. A source that runs dry gives its share to the others, which may
then need more: the loop builds again (at most --max-rounds rounds). Every build
resumes where the last one stopped, so a failed or interrupted run is rerun as is.

    modal run scripts/remote/modal_pool.py --chunk-tokens 10e9
    modal run scripts/remote/modal_pool.py --plan-only      # what would be built

Hugging Face: bigcode/starcoderdata is gated; the build containers read the token
from the Modal secret "huggingface" (HF_TOKEN; create it once with
`modal secret create huggingface HF_TOKEN=...`, a read-only fine-grained token).

Local check, no Modal and no Volume (the same per-source function on a temp dir):
    python scripts/remote/modal_pool.py --local-smoke [--docs 200] [--sources a,b]

Cost control: CPU only (cpu=16 per build container), each call capped by its timeout;
the Modal workspace budget limit is the hard cap."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # --local-smoke runs without Modal installed
    modal = None

LOCAL = modal is None or modal.is_local()
HERE = Path(__file__).resolve().parents[2] if LOCAL else Path("/src")
CONFIG = HERE / "configs" / "data-v3.toml"
POOL_ROOT = Path("/pool")
VERIFIED = "inputs/distill-v2/kept"          # on the Volume
VERIFIED_LOCAL = HERE / "data" / "distill-v2" / "kept"
CPU = 16.0
MEMORY_MB = 16384
WORKERS = 15                                 # tokenising processes; the 16th core reads
COMMIT_EVERY_S = 300
SRC_FILES = ["quipu/__init__.py", "quipu/data.py", "quipu/tokenizer.py", "quipu/bpe.py",
             "quipu/fsio.py", "quipu/shard_mix.py", "quipu/decontam.py", "quipu/pool.py",
             "quipu/pool_sources.py", "scripts/build_pool.py", "configs/data-v3.toml",
             "artifacts/tokenizer/tokenizer.json"]
# The laptop venv's versions (the ones the tests ran with); fastText as in uv.lock.
PIP = ["numpy==2.5.3", "pyarrow==25.0.1", "huggingface_hub==1.32.0", "fsspec==2026.6.0",
       "tokenizers==0.23.2", "tiktoken==0.14.0", "fasttext-numpy2==0.10.4"]


def _bp() -> Any:
    """scripts/build_pool.py, imported from the repo (or /src in the container)."""
    scripts = str(HERE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import build_pool
    return build_pool


def _env(lid_filter: bool, local_paths: list[str]) -> tuple[Any, Any]:
    bp = _bp()
    cfg = bp.pool.load_pool_config(CONFIG)
    args = argparse.Namespace(lid_filter=lid_filter, local_path=local_paths)
    return bp, bp.Env(cfg, args)


def build_in(root: Path, name: str, target: int, workers: int, *, limit_docs: int | None = None,
             commit: Any = None, lid_filter: bool = True,
             local_paths: list[str] | None = None) -> dict[str, Any]:
    """The per-source step (a Modal container, or --local-smoke): build `target` new
    tokens of source `name` under root/sources/<name>/ and nowhere else; commit() at
    most every COMMIT_EVERY_S seconds at checkpoints, and once at the end."""
    bp, env = _env(lid_filter, local_paths or [])
    last = [time.time()]

    def tick() -> None:
        if commit is not None and time.time() - last[0] >= COMMIT_EVERY_S:
            commit()
            last[0] = time.time()

    res = bp.build_one(env, Path(root), name, int(target), workers=workers,
                       limit_docs=limit_docs, on_checkpoint=tick)
    if commit is not None:
        commit()
    return res


def plan_in(root: Path, chunk_tokens: int) -> dict[str, int]:
    bp = _bp()
    cfg = bp.pool.load_pool_config(CONFIG)
    for n in cfg.sources:  # every source directory exists before any build container runs
        bp.pool.source_dir(root, n).mkdir(parents=True, exist_ok=True)
    return bp.plan(cfg, Path(root), int(chunk_tokens))


def merge_in(root: Path, chunk_tokens: int | None, *, fit: bool = False, lid_filter: bool = True,
             cfg: Any = None) -> dict[str, Any]:
    """The merge step: mix the next chunk. Returns the chunk's record, or {"need":
    {source: tokens}} when live sources must build more first."""
    bp = _bp()
    cfg = cfg or bp.pool.load_pool_config(CONFIG)
    _, tok = bp.load_tokenizer(cfg)
    _, dc_info = bp.load_decontam(cfg)
    prov = {"tokenizer": tok, "decontamination": dc_info,
            "lid": {"enabled": lid_filter, "model": cfg.lid_model, "revision": cfg.lid_revision,
                    "sources": bp.lid_sources(cfg)},
            "stackexchange_cutoff": cfg.stackexchange_cutoff, "built_on": "modal" if not LOCAL
            else "local"}
    T = bp.pool.fit_chunk_tokens(root, cfg) if fit else int(chunk_tokens)
    try:
        return bp.pool.merge_chunk(Path(root), cfg, T, eot=tok["eot"], provenance=prov)
    except bp.pool.NeedMore as exc:
        return {"need": exc.targets}


def local_smoke(docs: int = 200, sources: str = "", out: str | None = None,
                lid_filter: bool = False) -> dict[str, Any]:
    """build_in for each source on `docs` documents into a temp root (no Modal), then
    the merge of the largest chunk they fill; prints the per-source table."""
    bp = _bp()
    cfg = bp.pool.load_pool_config(CONFIG)
    root = Path(out or tempfile.mkdtemp(prefix="quipu-pool-smoke-"))
    names = [n for n in sources.split(",") if n] or list(cfg.sources)
    local = [f"verified-answers={VERIFIED_LOCAL}"] if "verified-answers" in names else []
    print(f"local smoke: {len(names)} sources, {docs} documents each, into {root}", flush=True)
    report: dict[str, Any] = {"root": str(root), "sources": {}, "errors": {}}
    for n in names:
        try:
            r = build_in(root, n, 10**15, 0, limit_docs=docs, lid_filter=lid_filter,
                         local_paths=local)
        except Exception as exc:
            report["errors"][n] = repr(exc)[:300]
            print(f"  {n}: FAILED {exc!r}"[:300], flush=True)
            continue
        report["sources"][n] = r
        print(f"  {n}: {r['documents']} docs, {r['tokens']:,} tokens", flush=True)
    if report["sources"]:
        sub = bp.subset(cfg, list(report["sources"]))
        rec = merge_in(root, None, fit=True, lid_filter=lid_filter, cfg=sub)
        report["chunk"] = rec
        for n, s in rec.get("by_source", {}).items():
            print(f"  {n:24} {s['tokens']:>9,} tokens  share {100 * s['share']:6.2f}% "
                  f"(target {100 * s['target_share']:.2f}%)")
    return report


if modal is not None:
    image = modal.Image.debian_slim(python_version="3.12").pip_install(*PIP).env(
        {"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"})
    for f in SRC_FILES:
        image = image.add_local_file(HERE / f, f"/src/{f}")
    vol = modal.Volume.from_name("quipu-pool", create_if_missing=True)
    hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
    app = modal.App("quipu-pool", image=image)

    @app.function(cpu=2.0, memory=4096, volumes={"/pool": vol}, timeout=30 * 60)
    def plan(chunk_tokens: int) -> dict[str, int]:
        vol.reload()
        out = plan_in(POOL_ROOT, chunk_tokens)
        vol.commit()  # the source directories, created here once
        return out

    @app.function(cpu=CPU, memory=MEMORY_MB, volumes={"/pool": vol}, secrets=[hf_secret],
                  timeout=12 * 3600)
    def build(name: str, target: int, workers: int = WORKERS, lid_filter: bool = True) -> dict:
        vol.reload()
        local = [f"verified-answers={POOL_ROOT / VERIFIED}"]
        return build_in(POOL_ROOT, name, target, workers, commit=vol.commit,
                        lid_filter=lid_filter, local_paths=local)

    @app.function(cpu=4.0, memory=MEMORY_MB, volumes={"/pool": vol}, timeout=6 * 3600)
    def merge(chunk_tokens: int, lid_filter: bool = True) -> dict:
        vol.reload()
        out = merge_in(POOL_ROOT, chunk_tokens, lid_filter=lid_filter)
        vol.commit()
        return out

    def upload_verified() -> None:
        """The verified answers (~1.4 GB, local only) onto the Volume once."""
        try:
            have = {Path(e.path).name for e in vol.listdir(VERIFIED)}
        except Exception:
            have = set()
        todo = [f for f in sorted(VERIFIED_LOCAL.glob("part-*.jsonl")) if f.name not in have]
        if todo:
            with vol.batch_upload() as up:
                for f in todo:
                    print("uploading", f.name, flush=True)
                    up.put_file(str(f), f"/{VERIFIED}/{f.name}")

    @app.local_entrypoint()
    def main(chunk_tokens: float = 10e9, workers: int = WORKERS, max_rounds: int = 4,
             plan_only: bool = False, lid_filter: bool = True):
        T = int(chunk_tokens)
        targets = plan.remote(T)
        print(f"plan for a {T:,}-token chunk: {targets}", flush=True)
        if plan_only:
            return
        if "verified-answers" in targets:
            upload_verified()
        for rnd in range(max_rounds):
            if targets:
                jobs = [(n, t, workers, lid_filter) for n, t in sorted(targets.items())]
                print(f"round {rnd + 1}: {len(jobs)} build containers", flush=True)
                failed = []
                for (n, *_), out in zip(jobs, build.starmap(jobs, return_exceptions=True)):
                    if isinstance(out, BaseException):
                        print(f"{n}: FAILED {out!r}", flush=True)
                        failed.append(n)
                    else:
                        print(f"{n}: {out['tokens']:,} tokens, {out['documents']:,} docs, "
                              f"exhausted={out['exhausted']}, {out.get('wall_seconds')}s",
                              flush=True)
                if failed:
                    raise SystemExit(f"builds failed: {','.join(failed)} (rerun the same "
                                     "command: every source resumes where it stopped)")
            out = merge.remote(T, lid_filter)
            if "need" in out:
                targets = out["need"]
                print(f"merge needs more: {targets}", flush=True)
                continue
            print(f"chunk {out['index']}: {out['tokens']:,} tokens, shards "
                  f"{out['shards'][0]}..{out['shards'][1]}", flush=True)
            return
        raise SystemExit(f"chunk not mixed after {max_rounds} rounds; still needed {targets}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="local checks of the Modal pool runner")
    p.add_argument("--local-smoke", action="store_true", required=True)
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--sources", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--lid-filter", action=argparse.BooleanOptionalAction,
                   default=sys.platform != "win32")
    a = p.parse_args()
    local_smoke(a.docs, a.sources, a.out, a.lid_filter)
