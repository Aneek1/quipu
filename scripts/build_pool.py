"""Build the data v3 pool (configs/data-v3.toml; quipu/pool.py, quipu/pool_sources.py).

The pool grows in chunks (default [pool] chunk_tokens, 10B tokens). Each source is built
on its own (its unique tokens, resumable: a second run continues its stream where the
first stopped), then the chunk is mixed into 100M-token uint16 shards under
<out>/mixed/, indices continuing from the previous chunk. On Modal the same steps run
one container per source (scripts/remote/modal_pool.py).

    # what the next chunk needs from each source (tokens to build)
    python scripts/build_pool.py plan  --out /pool
    # one source (what a Modal container runs)
    python scripts/build_pool.py build --out /pool --source fineweb-edu --target 2e9
    # mix the next chunk from what is built (refuses if a live source is short)
    python scripts/build_pool.py merge --out /pool [--chunk-tokens 10e9 | --fit]
    # all of it, sources one after another (a local or single-box build)
    python scripts/build_pool.py run   --out /pool --chunk-tokens 10e9
    # a tiny end-to-end pool: a few hundred documents per source, few MB
    python scripts/build_pool.py smoke --out <tmp> --docs 200 --no-lid-filter

Language ID (the FineWeb-2 sources) needs fastText, which has no Windows wheel:
--no-lid-filter builds without it (recorded in the manifest; for smoke runs only).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quipu import pool  # noqa: E402
from quipu import pool_sources as psrc  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "data-v3.toml"
LID_FILE = "lid-specialist-fasttext.ftz"
SMOKE_REPORT = "smoke_report.json"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_tokenizer(cfg: pool.PoolConfig) -> tuple[Any, dict[str, Any]]:
    """The tokenizer factory (picklable) and its provenance; the file must be the
    pinned one and fit uint16 (and the model's vocabulary)."""
    from quipu.tokenizer import make_tokenizer

    path = _resolve(cfg.tokenizer)
    sha = _sha256_file(path)
    if cfg.tokenizer_sha256 and sha != cfg.tokenizer_sha256:
        raise SystemExit(f"{path}: sha256 {sha} is not the pinned tokenizer "
                         f"{cfg.tokenizer_sha256}")
    factory = functools.partial(make_tokenizer, str(path))
    tok = factory()
    if cfg.vocab_size and tok.vocab_size != cfg.vocab_size:
        raise SystemExit(f"{path}: vocabulary {tok.vocab_size} != [pool] vocab_size "
                         f"{cfg.vocab_size}")
    if tok.vocab_size > 65536:
        raise SystemExit(f"vocabulary {tok.vocab_size} does not fit uint16 shards")
    return factory, {"path": cfg.tokenizer, "sha256": sha, "vocab_size": tok.vocab_size,
                     "eot": tok.eot}


def load_decontam(cfg: pool.PoolConfig) -> tuple[Any, dict[str, Any]]:
    """HumanEval + MBPP at the pinned commits (~0.2 MB), as for data v2."""
    from quipu import decontam as dc

    d, info = dc.load_benchmarks({dc.HUMANEVAL: cfg.humaneval_revision,
                                  dc.MBPP: cfg.mbpp_revision})
    return d, {"index": d.summary(), "benchmarks": info,
               "applied_to": "every document of every source, before tokenization"}


def lid_sources(cfg: pool.PoolConfig) -> list[str]:
    return [n for n, s in cfg.sources.items() if s.get("lid")]


def load_lid(cfg: pool.PoolConfig, enabled: bool) -> tuple[Any, dict[str, Any] | None]:
    if not enabled or not lid_sources(cfg):
        return None, ({"enabled": False, "reason": "--no-lid-filter (smoke / Windows): the "
                       "FineWeb-2 sources are NOT language-filtered in this build",
                       "sources": lid_sources(cfg)} if lid_sources(cfg) else None)
    from huggingface_hub import hf_hub_download

    from quipu import shard_mix as sm

    path = hf_hub_download(cfg.lid_model, LID_FILE, revision=cfg.lid_revision)
    lid = sm.FastTextLid(path)
    labels = lid.labels()  # loads fastText now: a missing install fails here
    needed = sm.lid_labels_needed(cfg.sources[n].get("lid") for n in lid_sources(cfg))
    missing = sorted(needed - set(labels))
    if missing:
        raise SystemExit(f"{cfg.lid_model}@{cfg.lid_revision} has no label for {missing}")
    return lid, {"enabled": True, "model": cfg.lid_model, "revision": cfg.lid_revision,
                 "file": LID_FILE, "sha256": _sha256_file(path), "labels": labels,
                 "threshold": 0.0, "sources": lid_sources(cfg)}


class Env:
    """What every build step needs, loaded once."""

    def __init__(self, cfg: pool.PoolConfig, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.tok_factory, tok_info = load_tokenizer(cfg)
        self.eot = tok_info["eot"]
        self.decontam, dc_info = load_decontam(cfg)
        self.lid, lid_info = load_lid(cfg, args.lid_filter)
        self.setup = pool.BuildSetup(tokenizer=self.tok_factory, decontam=self.decontam,
                                     lid=self.lid, max_doc_tokens=cfg.max_doc_tokens)
        self.provenance = {"tokenizer": tok_info, "decontamination": dc_info, "lid": lid_info,
                           "stackexchange_cutoff": cfg.stackexchange_cutoff}
        self.local_paths = dict(kv.split("=", 1) for kv in (args.local_path or []))
        self.fingerprint = {"tokenizer_sha256": tok_info["sha256"],
                            "decontam": self.decontam.fingerprint(),
                            "lid": (lid_info or {}).get("sha256") if (lid_info or {}).get(
                                "enabled") else None}

    def spec(self, name: str) -> pool.SourceSpec:
        s = self.cfg.sources[name]
        if s.kind == "local_jsonl":
            path = self.local_paths.get(name, s.get("path"))
            s = pool.SourceSpec(s.name, s.kind, s.slice, s.weight, s.share,
                                {**s.opts, "path": str(_resolve(path))})
        return s

    def open(self, name: str) -> psrc.Source:
        return psrc.open_source(self.spec(name), cutoff=self.cfg.stackexchange_cutoff)

    def identity(self, src: psrc.Source, spec: pool.SourceSpec) -> dict[str, Any]:
        ident = dict(src.identity)
        if spec.kind == "local_jsonl":
            ident.pop("path", None)  # the same files may sit elsewhere (a Modal Volume)
        uses_lid = bool(spec.get("lid"))
        return {**ident, "tokenizer_sha256": self.fingerprint["tokenizer_sha256"],
                "decontam": self.fingerprint["decontam"],
                "lid": self.fingerprint["lid"] if uses_lid else None}


def _progress(every: float = 30.0) -> Any:
    last = [0.0]

    def log(name: str, built: int, target: int, rows: int) -> None:
        now = time.time()
        if now - last[0] >= every:
            last[0] = now
            print(f"  {name}: {built:,} / {target:,} tokens ({rows:,} rows)", flush=True)
    return log


def build_one(env: Env, root: Path, name: str, target: int, *, workers: int,
              limit_docs: int | None = None, on_checkpoint: Any = None) -> dict[str, Any]:
    spec = env.spec(name)
    src = env.open(name)
    return pool.build_source(pool.source_dir(root, name), spec, int(target), setup=env.setup,
                             open_rows=src.open_rows, shard_tokens=env.cfg.shard_tokens,
                             workers=workers, limit_docs=limit_docs,
                             identity=env.identity(src, spec), on_checkpoint=on_checkpoint,
                             log=_progress())


def plan(cfg: pool.PoolConfig, root: Path, chunk_tokens: int) -> dict[str, int]:
    return pool.build_targets(chunk_tokens, cfg.shares(), pool.pool_totals(root, cfg),
                              max_epochs=cfg.max_epochs)


def subset(cfg: pool.PoolConfig, names: list[str]) -> pool.PoolConfig:
    """cfg with only `names`, shares renormalized over them (smoke runs)."""
    from dataclasses import replace
    total = sum(cfg.sources[n].share for n in names)
    sources = {n: replace(cfg.sources[n], share=cfg.sources[n].share / total) for n in names}
    return replace(cfg, sources=sources)


# ------------------------------------------------------------------ smoke

def smoke(env: Env, root: Path, names: list[str], docs: int) -> dict[str, Any]:
    """Build `docs` documents of each source in two halves (the second resumes the
    stream where the first stopped), mix the largest chunk they fill, print per-source
    tokens, achieved shares and per-core throughput."""
    report: dict[str, Any] = {"docs_per_source": docs, "sources": {}, "errors": {}}
    first = docs // 2
    for name in names:
        t0 = time.time()
        try:
            a = build_one(env, root, name, 10**15, workers=0, limit_docs=first)
            b = build_one(env, root, name, 10**15, workers=0, limit_docs=docs - first)
        except Exception as exc:  # one broken source must not hide the others
            report["errors"][name] = repr(exc)[:300]
            print(f"  {name}: FAILED {exc!r}"[:300], flush=True)
            continue
        st = pool.SourceState.load(pool.source_dir(root, name))
        # Rates over everything processed (the batches in flight at the stop included),
        # not just the documents kept: tokens through decontam + LID + BPE per second of
        # one core, and per second of the main process's own work (reading, parquet
        # decoding, hashing, writing).
        proc = (a["process_seconds"] or 0) + (b["process_seconds"] or 0)
        main = a["cpu_seconds_main"] + b["cpu_seconds_main"] - proc
        wall = a["wall_seconds"] + b["wall_seconds"]
        ptok = (a["processed_tokens"] or 0) + (b["processed_tokens"] or 0)
        fetched = a["rows_fetched"] + b["rows_fetched"]
        report["sources"][name] = {
            "documents": st.documents, "tokens": st.unique_tokens, "rows": st.stats.get("rows"),
            "exhausted": st.exhausted, "resumed": len(st.builds) == 2 and b["rows"] > 0,
            "duplicates": st.stats.get("dropped_duplicate", 0),
            "contaminated": st.stats.get("dropped_contamination", 0),
            "wall_seconds": round(wall, 2), "process_seconds": round(proc, 3),
            "main_seconds": round(main, 3),
            "docs_per_s_wall": round(st.documents / wall, 1) if wall else None,
            "processed_tokens": ptok, "rows_fetched": fetched,
            "tok_per_s_core": round(ptok / proc) if proc else None,
            "tok_per_s_main": round(ptok / main) if main > 0 else None,
            "tokens_per_doc": round(st.unique_tokens / st.documents) if st.documents else None,
            "share_target": env.cfg.sources[name].share,
            "seconds_total": round(time.time() - t0, 1)}
        r = report["sources"][name]
        print(f"  {name}: {r['documents']} docs, {r['tokens']:,} tokens, {wall:.1f}s wall",
              flush=True)
    ok = list(report["sources"])
    if ok:
        sub = subset(env.cfg, ok)
        t = pool.fit_chunk_tokens(root, sub)
        rec = pool.merge_chunk(root, sub, t, eot=env.eot, provenance=env.provenance)
        report["chunk"] = {"tokens": rec["tokens"], "shards": rec["shards"],
                           "by_source": rec["by_source"]}
    print_smoke(report, env.cfg)
    pool.write_json_atomic(root / SMOKE_REPORT, report)
    return report


def print_smoke(report: dict[str, Any], cfg: pool.PoolConfig) -> None:
    by = (report.get("chunk") or {}).get("by_source", {})
    hdr = (f"{'source':24} {'docs':>5} {'tokens':>10} {'target%':>8} {'mix tok':>9} "
           f"{'share%':>7} {'docs/s':>7} {'tok/s/core':>10} {'tok/s main':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for n, r in report["sources"].items():
        m = by.get(n, {})
        print(f"{n:24} {r['documents']:>5} {r['tokens']:>10,} {100 * cfg.sources[n].share:>8.3f} "
              f"{m.get('tokens', 0):>9,} {100 * m.get('share', 0):>7.3f} "
              f"{r['docs_per_s_wall'] or 0:>7.1f} {r['tok_per_s_core'] or 0:>10,} "
              f"{r['tok_per_s_main'] or 0:>10,}")
    for n, e in report["errors"].items():
        print(f"{n:24} FAILED: {e}")
    c = report.get("chunk")
    if c:
        print(f"\nmixed chunk: {c['tokens']:,} tokens, shards {c['shards'][0]}..{c['shards'][1]}"
              " (target% = config share; share% = achieved, renormalized over the sources "
              "built)")


# ------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default=str(DEFAULT_CONFIG))
        sp.add_argument("--out", required=True, help="pool root (e.g. /pool on Modal)")
        sp.add_argument("--lid-filter", action=argparse.BooleanOptionalAction,
                        default=sys.platform != "win32",
                        help="language-ID the FineWeb-2 sources (needs fastText; default on, "
                             "off on Windows)")
        sp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="processes that decontaminate / LID / tokenize (0: inline)")
        sp.add_argument("--local-path", action="append", metavar="SOURCE=PATH",
                        help="where a local source's files are (default: its config path)")

    b = sub.add_parser("build", help="build one source")
    common(b)
    b.add_argument("--source", required=True)
    b.add_argument("--target", type=float, required=True, help="new unique tokens")
    b.add_argument("--limit-docs", type=int, default=None)
    pl = sub.add_parser("plan", help="tokens each source must build for the next chunk")
    pl.add_argument("--config", default=str(DEFAULT_CONFIG))
    pl.add_argument("--out", required=True)
    pl.add_argument("--chunk-tokens", type=float, default=None)
    m = sub.add_parser("merge", help="mix the next chunk from what is built")
    m.add_argument("--config", default=str(DEFAULT_CONFIG))
    m.add_argument("--out", required=True)
    g = m.add_mutually_exclusive_group()
    g.add_argument("--chunk-tokens", type=float, default=None)
    g.add_argument("--fit", action="store_true", help="the largest chunk the built tokens fill")
    r = sub.add_parser("run", help="build every source the chunk needs, then merge")
    common(r)
    r.add_argument("--chunk-tokens", type=float, default=None)
    s = sub.add_parser("smoke", help="tiny end-to-end pool")
    common(s)
    s.add_argument("--docs", type=int, default=200)
    s.add_argument("--sources", default="", help="comma-separated (default: all)")
    s.set_defaults(workers=0)
    return p


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    cfg = pool.load_pool_config(args.config)
    root = Path(args.out)
    if args.cmd == "plan":
        t = plan(cfg, root, int(args.chunk_tokens or cfg.chunk_tokens))
        print(json.dumps(t, indent=1))
        return t
    if args.cmd == "merge":
        tok_factory, tok_info = load_tokenizer(cfg)
        T = pool.fit_chunk_tokens(root, cfg) if args.fit else int(args.chunk_tokens or
                                                                  cfg.chunk_tokens)
        try:
            rec = pool.merge_chunk(root, cfg, T, eot=tok_info["eot"])
        except pool.NeedMore as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(3) from None
        print(json.dumps({k: rec[k] for k in ("index", "tokens", "shards")}))
        return rec
    env = Env(cfg, args)
    if args.cmd == "build":
        res = build_one(env, root, args.source, int(args.target), workers=args.workers,
                        limit_docs=args.limit_docs)
        print(json.dumps(res))
        return res
    if args.cmd == "run":
        def build(targets: dict[str, int]) -> None:
            for name, t in targets.items():
                print(f"{name}: building {t:,} tokens", flush=True)
                print(json.dumps(build_one(env, root, name, t, workers=args.workers)),
                      flush=True)
        rec = pool.run_chunk(root, cfg, build, chunk_tokens=int(args.chunk_tokens or
                                                                cfg.chunk_tokens),
                             eot=env.eot, provenance=env.provenance)
        print(json.dumps({k: rec[k] for k in ("index", "tokens", "shards")}))
        return rec
    if args.cmd == "smoke":
        names = [n for n in args.sources.split(",") if n] or list(cfg.sources)
        unknown = sorted(set(names) - set(cfg.sources))
        if unknown:
            raise SystemExit(f"unknown sources: {unknown}")
        return smoke(env, root, names, args.docs)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
