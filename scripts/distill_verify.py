"""Distillation CUDA arm, step 2: check every teacher answer by running its tests.

Reads the prompt set (data/distill-prompts/prompts.jsonl) and the teacher's
replies (data/distill-cuda/generations.jsonl), extracts each reply's code
(quipu.sandbox.extract_code: the fenced block defining the entry point) and runs
it against the problem's asserts in a separate interpreter (quipu.sandbox).

Writes, beside the generations:
- verified.jsonl: id, split, source, passed, total, error, kept (every test
  passed and the reply finished on its own, not at the token cap);
- verify_summary.json: counts kept and rejected, per source, split and reason.
The kept answers are the CUDA arm's training data.

Run:  uv run python scripts/distill_verify.py [--workers 4] [--timeout 10] [--parts-dir data/distill-v2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.sandbox import extract_code, run_tests  # noqa: E402


def check(problem: dict, gen: dict, timeout: float) -> dict:
    rec = {"id": problem["id"], "split": problem["split"], "source": problem["source"],
           "total": len(problem["tests"])}
    if gen.get("finish") != "eos":
        return rec | {"passed": 0, "error": "hit_token_cap", "kept": False}
    code = extract_code(gen["response"], problem["entry_point"])
    if code is None:
        return rec | {"passed": 0, "error": "no_code", "kept": False}
    r = run_tests(code, problem["tests"], timeout=timeout)
    return rec | {"passed": r.passed, "error": r.error, "kept": r.all_passed}


TAG_PRIORITY = ["bf16-vllm", "q4km-llamacpp"]  # first answer per id wins, in this order


def _tag_rank(tag: str) -> int:
    return TAG_PRIORITY.index(tag) if tag in TAG_PRIORITY else len(TAG_PRIORITY)


def _safe_check(problem: dict, gen: dict, timeout: float) -> dict:
    try:
        return check(problem, gen, timeout)
    except Exception as e:  # e.g. OSError spawning the subprocess
        return {"id": problem["id"], "split": problem["split"], "source": problem["source"],
                "total": len(problem["tests"]), "passed": 0,
                "error": f"verifier:{type(e).__name__}", "kept": False}


def verify_part(root: Path, part: int, workers: int, timeout: float) -> dict | None:
    """Verify one part's finished generations; None when not done or already up to date."""
    root = Path(root)
    gdir = root / "generations"
    done = sorted(gdir.glob(f"part-{part:04d}.*.done"))
    if not done:
        return None
    finished = {d.name.split(".")[1] for d in done}  # partial files of other tags are ignored
    srcs = sorted((p for p in gdir.glob(f"part-{part:04d}.*.jsonl")
                   if p.name.split(".")[1] in finished),
                  key=lambda p: _tag_rank(p.name.split(".")[1]))
    stamp = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in srcs}
    meta_path = root / "verified" / f"part-{part:04d}.meta.json"
    try:
        if json.loads(meta_path.read_text(encoding="utf-8")) == stamp:
            return None
    except (OSError, ValueError):
        pass  # missing or corrupt meta: treat as stale
    problems: dict[str, dict] = {}
    with open(root / "prompts" / f"part-{part:04d}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                problems[row["id"]] = row
    chosen: dict[str, dict] = {}
    for src in srcs:
        with open(src, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    g = json.loads(line)
                except ValueError:
                    continue
                if isinstance(g, dict) and g.get("id") in problems and g["id"] not in chosen:
                    chosen[g["id"]] = g
    gens = [chosen[i] for i in sorted(chosen)]
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda g: _safe_check(problems[g["id"]], g, timeout), gens))
    (root / "verified").mkdir(exist_ok=True)
    (root / "kept").mkdir(exist_ok=True)
    vfinal = root / "verified" / f"part-{part:04d}.jsonl"
    kfinal = root / "kept" / f"part-{part:04d}.jsonl"
    vtmp, ktmp = vfinal.with_suffix(".jsonl.tmp"), kfinal.with_suffix(".jsonl.tmp")
    with open(vtmp, "w", encoding="utf-8", newline="\n") as fv, \
            open(ktmp, "w", encoding="utf-8", newline="\n") as fk:
        for g, r in zip(gens, results):
            fv.write(json.dumps(r | {"teacher_tag": g.get("teacher_tag")}) + "\n")
            if r["kept"]:
                p = problems[g["id"]]
                fk.write(json.dumps({"id": g["id"], "split": p["split"], "prompt": p["prompt"],
                                     "response": g["response"], "entry_point": p["entry_point"],
                                     "teacher": g.get("teacher"),
                                     "teacher_tag": g.get("teacher_tag"),
                                     "revision": g.get("revision")},
                                    ensure_ascii=False) + "\n")
    os.replace(vtmp, vfinal)
    os.replace(ktmp, kfinal)
    meta_path.write_text(json.dumps(stamp), encoding="utf-8")
    return {"answers": len(results), "kept": sum(r["kept"] for r in results)}


def verify_parts(root: Path, workers: int, timeout: float) -> dict:
    root = Path(root)
    for p in sorted((root / "prompts").glob("part-*.jsonl")):
        part = int(p.stem.split("-")[1])
        s = verify_part(root, part, workers, timeout)
        if s:
            print(f"part {part}: {s['kept']}/{s['answers']} kept", flush=True)
    answers = kept = 0
    by_tag: dict[str, dict[str, int]] = {}
    outcomes: Counter = Counter()
    for f in sorted((root / "verified").glob("part-*.jsonl")):  # empty if dir is missing
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                t = by_tag.setdefault(r.get("teacher_tag") or "?", {"answers": 0, "kept": 0})
                t["answers"] += 1
                t["kept"] += bool(r["kept"])
                answers += 1
                kept += bool(r["kept"])
                outcomes["passed" if r["kept"] else
                         ((r.get("error") or "").split(":")[0] or "some_tests_failed")] += 1
    summary = {"answers": answers, "kept": kept,
               "kept_pct": round(100 * kept / answers, 1) if answers else 0.0,
               "by_teacher_tag": dict(sorted(by_tag.items())),
               "outcomes": dict(outcomes.most_common()),
               "rule": "kept = the reply ended on its own (eos), has a code block defining "
                       "the entry point, and every assert passed in quipu.sandbox"}
    (root / "verify_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                              encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", default="data/distill-prompts/prompts.jsonl")
    ap.add_argument("--generations", default="data/distill-cuda/generations.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--parts-dir", default=None, help="v2: verify part files under this dir")
    args = ap.parse_args(argv)
    if args.parts_dir:
        print(json.dumps(verify_parts(ROOT / args.parts_dir, args.workers, args.timeout)))
        return 0

    problems = {json.loads(l)["id"]: json.loads(l)
                for l in open(ROOT / args.prompts, encoding="utf-8") if l.strip()}
    gens = []
    for line in open(ROOT / args.generations, encoding="utf-8"):
        try:
            gens.append(json.loads(line))
        except ValueError:
            continue
    gens = [g for g in gens if g["id"] in problems]

    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda g: check(problems[g["id"]], g, args.timeout), gens):
            results.append(r)
            if len(results) % 250 == 0 or len(results) == len(gens):
                print(f"{len(results)}/{len(gens)} checked, "
                      f"{sum(x['kept'] for x in results)} kept", flush=True)

    out = (ROOT / args.generations).parent
    with open(out / "verified.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")

    def tally(rows):
        kept = sum(r["kept"] for r in rows)
        return {"answers": len(rows), "kept": kept,
                "kept_pct": round(100 * kept / len(rows), 1) if rows else 0.0}

    reasons = Counter(("passed" if r["kept"] else
                       ("some_tests_failed" if r["error"] in ("AssertionError",) or
                        (r["passed"] and not r["kept"]) else r["error"].split(":")[0] or "unknown"))
                      for r in results)
    summary = {"teacher": gens[0]["teacher"] if gens else None,
               "revision": gens[0]["revision"] if gens else None,
               "problems_in_set": len(problems), **tally(results),
               "by_source": {s: tally([r for r in results if r["source"] == s])
                             for s in sorted({r["source"] for r in results})},
               "by_split": {s: tally([r for r in results if r["split"] == s])
                            for s in ("train", "val")},
               "outcomes": dict(reasons.most_common()),
               "rule": "kept = the reply ended on its own (eos), has a code block defining the "
                       "entry point, and every assert passed in quipu.sandbox"}
    (out / "verify_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("answers", "kept", "kept_pct", "by_source",
                                               "outcomes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
