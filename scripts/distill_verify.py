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

Run:  uv run python scripts/distill_verify.py [--workers 4] [--timeout 10]
"""
from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", default="data/distill-prompts/prompts.jsonl")
    ap.add_argument("--generations", default="data/distill-cuda/generations.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args(argv)

    problems = {json.loads(l)["id"]: json.loads(l)
                for l in open(ROOT / args.prompts, encoding="utf-8") if l.strip()}
    gens = []
    for line in open(ROOT / args.generations, encoding="utf-8"):
        try:
            gens.append(json.loads(line))
        except ValueError:
            continue
    gens = [g for g in gens if g["id"] in problems]

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda g: check(problems[g["id"]], g, args.timeout), gens))

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
