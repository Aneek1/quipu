"""Distillation v2 prompt set: all of OpenCodeInstruct, in 50k-prompt part files.

Same per-row rules as v1 (scripts/build_distill_prompts.function_problem: reference
passed every test, 3-15 asserts calling one non-builtin function, problem 150-2500
chars), same decontamination (quipu.decontam, HumanEval and MBPP at the pinned
commits) and whitespace-collapsed exact dedup. APPS is not used (the v1 teacher
passed 31% of it). Files are read one at a time, in a seeded order, and shuffled
within each file, so memory stays at about one parquet file; the dedup set holds
16-byte hashes, not text. Stops at --max-prompts.

Output (default data/distill-v2/prompts/):
- part-NNNN.jsonl: id (dv2-NNNNNNN), part, split (every 20th is "val"), source,
  source_id, licence, kind, prompt, entry_point, tests;
- manifest.json: source files with sha256, rules, counts per drop reason,
  decontamination hits, every part's rows and sha256.

Run:  uv run python scripts/build_distill_prompts_v2.py [--max-prompts 1000000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.decontam import collapse, load_benchmarks  # noqa: E402
from scripts.build_distill_prompts import (HUMANEVAL_REV, MBPP_REV, OCI,  # noqa: E402
                                           function_problem, sha256)

OCI_FILES = [f"data/train-{i:05d}-of-00050.parquet" for i in range(50)]
VAL_EVERY = 20


class PartWriter:
    """Writes prompts to part-NNNN.jsonl files of part_size rows, numbering ids."""

    def __init__(self, out: Path, part_size: int):
        self.out, self.part_size, self.n = Path(out), part_size, 0
        self.out.mkdir(parents=True, exist_ok=True)
        self.f = None
        self.parts: list[dict] = []

    def _path(self, part: int) -> Path:
        return self.out / f"part-{part:04d}.jsonl"

    def _finish(self) -> None:
        if self.f:
            self.f.close()
            part = len(self.parts)
            p = self._path(part)
            rows = sum(1 for _ in open(p, encoding="utf-8"))
            self.parts.append({"part": part, "file": p.name, "rows": rows, "sha256": sha256(p)})
            self.f = None

    def add(self, p: dict) -> None:
        part = self.n // self.part_size
        if self.f is None or part != len(self.parts):
            self._finish()
            self.f = open(self._path(part), "w", encoding="utf-8", newline="\n")
        rec = {"id": f"dv2-{self.n:07d}", "part": part,
               "split": "val" if self.n % VAL_EVERY == 0 else "train",
               "source": p["source"], "source_id": p["source_id"], "licence": p["licence"],
               "kind": p["kind"], "prompt": p["prompt"],
               "entry_point": p["entry_point"], "tests": p["tests"]}
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> list[dict]:
        self._finish()
        return self.parts


def take_unique(problems: Iterable[dict], decontam, seen: set[bytes], drops: Counter,
                hits: Counter) -> Iterator[dict]:
    """Yield problems not seen before (by collapsed text) and not contaminated."""
    for p in problems:
        key = hashlib.blake2b(collapse(p["problem"]).lower().encode("utf-8"),
                              digest_size=16).digest()
        if key in seen:
            drops[f"{p['source']}:duplicate"] += 1
            continue
        found = decontam.find(p["prompt"] + "\n" + "\n".join(p["tests"])) if decontam else None
        if found:
            drops[f"{p['source']}:decontam_{found[0]}"] += 1
            hits[found[0]] += 1
            continue
        seen.add(key)
        yield p


def publish(tmp: Path, out: Path) -> None:
    """Replace out with tmp wholesale, so out never holds a mix of old and new files."""
    tmp, out = Path(tmp), Path(out)
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-prompts", type=int, default=1_000_000)
    ap.add_argument("--part-size", type=int, default=50_000)
    ap.add_argument("--files", type=int, default=len(OCI_FILES), help="first N files (tests)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/distill-v2/prompts")
    ap.add_argument("--cache", default="data/distill-src")
    ap.add_argument("--no-decontam", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    if args.part_size < 1:
        ap.error("--part-size must be >= 1")
    if args.max_prompts < 1:
        ap.error("--max-prompts must be >= 1")
    if not 1 <= args.files <= len(OCI_FILES):
        ap.error(f"--files must be between 1 and {len(OCI_FILES)}")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    files = OCI_FILES[:args.files]
    rng.shuffle(files)
    decontam, bench = (None, {}) if args.no_decontam else load_benchmarks(
        {"humaneval": HUMANEVAL_REV, "mbpp": MBPP_REV})
    out = ROOT / args.out
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    writer = PartWriter(tmp, args.part_size)
    seen: set[bytes] = set()
    drops: Counter = Counter()
    hits: Counter = Counter()
    shas, rows_read = {}, 0
    try:
        for f in files:
            if writer.n >= args.max_prompts:
                break
            p = Path(hf_hub_download(OCI["dataset"], f, repo_type="dataset",
                                     revision=OCI["revision"],
                                     local_dir=str(ROOT / args.cache / "oci")))
            shas[f] = sha256(p)
            rows = pq.read_table(p, columns=["id", "input", "unit_tests",
                                             "average_test_score"]).to_pylist()
            rows_read += len(rows)
            cands = []
            for r in rows:
                prob, why = function_problem(r)
                if prob is None:
                    drops[f"opencodeinstruct:{why}"] += 1
                else:
                    cands.append(prob)
            del rows
            rng.shuffle(cands)
            pulled = 0

            def counted():
                nonlocal pulled
                for c in cands:
                    pulled += 1
                    yield c

            for prob in take_unique(counted(), decontam, seen, drops, hits):
                writer.add(prob)
                if writer.n >= args.max_prompts:
                    break
            if pulled < len(cands):
                drops["opencodeinstruct:over_cap"] += len(cands) - pulled
            print(f"{f}: {writer.n} prompts so far", flush=True)
    finally:
        if writer.f:
            writer.f.close()
    parts = writer.close()

    manifest = {
        "version": 2, "seed": args.seed,
        "source": {**OCI, "files": list(shas), "sha256": shas, "rows_read": rows_read},
        "benchmarks_decontaminated": ({k: {"dataset": v["dataset"], "revision": v["revision"],
                                           "problems": v["problems"]} for k, v in bench.items()}
                                      if bench else "skipped (--no-decontam)"),
        "rules": {"rows": "v1 function_problem: average_test_score == 1.0; 3-15 asserts "
                          "calling one non-builtin function; problem 150-2500 chars",
                  "dedup": "whitespace-collapsed lowercase text, blake2b-128",
                  "decontam": "prompt + tests through quipu.decontam",
                  "val": f"every {VAL_EVERY}th prompt", "max_prompts": args.max_prompts,
                  "part_size": args.part_size},
        "total": writer.n, "parts": parts,
        "drops": dict(sorted(drops.items())), "decontam_hits": dict(hits),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                       encoding="utf-8")
    publish(tmp, out)
    print(json.dumps({"total": writer.n, "parts": len(parts), "decontam_hits": dict(hits)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
