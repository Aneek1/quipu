"""Build the distillation prompt set: Python coding problems that each carry tests.

Both distillation arms read the same file, so the only difference between them is
how a teacher answer is checked (the Mac arm: Nemotron judges it; the CUDA arm: the
answer is executed against these tests). Only the problems and their tests are
used; the datasets' own reference answers are never written out.

Sources (pinned):
- nvidia/OpenCodeInstruct (CC-BY-4.0): function-level problems with assert tests.
  Kept only when the dataset's reference solution passed every test
  (average_test_score == 1.0), so the tests are known to be satisfiable, and the
  asserts all call one function, whose name the prompt then states.
- codeparrot/apps (MIT), train split, introductory difficulty, call-based
  problems (Codewars style; LeetCode `class Solution` ones skipped): each case
  becomes `assert fn(args) == expected`; problems whose expected values hold a
  float are skipped (exact comparison would be unreliable).

Every prompt (with its tests) goes through quipu.decontam (HumanEval and MBPP at
the commits pinned in configs/quipu-moe.toml, the pretraining shards' rules); a
hit is dropped. Exact duplicates (whitespace-collapsed problem text) are dropped.

Output (default data/distill-prompts/):
- prompts.jsonl: one problem per line: id, split ("train" | "val"), source,
  source_id, licence, kind ("function"), prompt (the text both teachers get),
  entry_point, tests (assert strings);
- manifest.json: sources with revisions and file sha256, the selection rules,
  counts kept and dropped per reason, decontamination hits, the output's sha256.

Run:  uv run python scripts/build_distill_prompts.py [--function N] [--apps N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.decontam import collapse, load_benchmarks  # noqa: E402

OCI = {"dataset": "nvidia/OpenCodeInstruct",
       "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
       "files": ["data/train-00000-of-00050.parquet", "data/train-00001-of-00050.parquet"],
       "license": "CC-BY-4.0"}
APPS = {"dataset": "codeparrot/apps",
        "revision": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
        "files": ["train.jsonl"], "license": "MIT"}
HUMANEVAL_REV = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
MBPP_REV = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

FUNCTION_SUFFIX = ("\n\nWrite it in Python as a function named `{name}`. "
                   "Reply with the code in one ```python block.")
STARTER = "\n\nStart from this signature:\n```python\n{starter}\n```"

ASSERT_CALL = re.compile(r"^\s*assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(")
NOT_ENTRY = {"abs", "all", "any", "len", "round", "sorted", "set", "list", "tuple", "dict",
             "isinstance", "sum", "max", "min", "str", "int", "float", "bool", "math", "type"}

PROMPT_CHARS = (150, 2500)
FUNCTION_TESTS = (3, 15)
APPS_CASES = (3, 30)
APPS_TEST_CHARS = 20_000
VAL_FRACTION = 0.05


def entry_point(tests: list[str]) -> str | None:
    """The one function every assert calls, or None (none, several, a builtin)."""
    names = set()
    for t in tests:
        m = ASSERT_CALL.match(t.strip())
        if not m:
            return None
        names.add(m.group(1))
    if len(names) != 1:
        return None
    name = names.pop()
    return None if name in NOT_ENTRY else name


def function_problem(row: dict) -> tuple[dict | None, str]:
    """An OpenCodeInstruct row as a problem, or (None, drop reason)."""
    if row.get("average_test_score") not in (1.0, "1.0", 1):
        return None, "reference_failed_tests"
    try:
        raw = row["unit_tests"]
        tests = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, ValueError):
        return None, "tests_unreadable"
    tests = [t.strip() for t in tests if isinstance(t, str) and t.strip()]
    if not FUNCTION_TESTS[0] <= len(tests) <= FUNCTION_TESTS[1]:
        return None, "test_count"
    name = entry_point(tests)
    if name is None:
        return None, "no_single_entry_point"
    text = (row.get("input") or "").strip()
    if not PROMPT_CHARS[0] <= len(text) <= PROMPT_CHARS[1]:
        return None, "prompt_length"
    return {"source": "opencodeinstruct", "licence": OCI["license"], "kind": "function",
            "source_id": row["id"], "problem": text, "entry_point": name, "tests": tests,
            "prompt": text + FUNCTION_SUFFIX.format(name=name)}, ""


def has_float(v) -> bool:
    if isinstance(v, float):
        return True
    if isinstance(v, (list, tuple)):
        return any(has_float(x) for x in v)
    if isinstance(v, dict):
        return any(has_float(x) for x in v.values())
    return False


def apps_problem(row: dict) -> tuple[dict | None, str]:
    """An APPS call-based row as a problem with assert tests, or (None, drop reason)."""
    if row.get("difficulty") != "introductory":
        return None, "difficulty"
    starter = (row.get("starter_code") or "").strip()
    if "class Solution" in starter:
        return None, "class_solution"
    try:
        io = json.loads(row.get("input_output") or "")
    except ValueError:
        return None, "io_unreadable"
    name = io.get("fn_name") if isinstance(io, dict) else None
    if not name or not re.fullmatch(r"[A-Za-z_]\w*", name) or name in NOT_ENTRY:
        return None, "not_call_based"
    ins, outs = io.get("inputs") or [], io.get("outputs") or []
    if len(ins) != len(outs) or not APPS_CASES[0] <= len(ins) <= APPS_CASES[1]:
        return None, "case_count"
    if (not all(isinstance(a, list) for a in ins)
            or not all(isinstance(o, list) and len(o) == 1 for o in outs)):
        return None, "io_shape"
    if has_float(outs):
        return None, "float_output"
    tests = [f"assert {name}({', '.join(repr(a) for a in args)}) == {out[0]!r}"
             for args, out in zip(ins, outs)]
    if sum(len(t) for t in tests) > APPS_TEST_CHARS:
        return None, "tests_too_large"
    text = (row.get("question") or "").strip()
    if not PROMPT_CHARS[0] <= len(text) <= PROMPT_CHARS[1]:
        return None, "prompt_length"
    prompt = text + (STARTER.format(starter=starter.replace("\t", "    ")) if starter else "")
    return {"source": "apps", "licence": APPS["license"], "kind": "function",
            "source_id": str(row.get("id", row.get("problem_id"))), "problem": text,
            "entry_point": name, "tests": tests,
            "prompt": prompt + FUNCTION_SUFFIX.format(name=name)}, ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select(problems: list[dict], want: int, decontam, seen: set[str], drops: Counter,
           hits: Counter, rng: random.Random) -> list[dict]:
    """Shuffle, then keep problems in order until `want`: decontaminated, not seen."""
    rng.shuffle(problems)
    out = []
    for p in problems:
        if len(out) >= want:
            break
        key = collapse(p["problem"]).lower()
        if key in seen:
            drops[f"{p['source']}:duplicate"] += 1
            continue
        found = decontam.find(p["prompt"] + "\n" + "\n".join(p["tests"])) if decontam else None
        if found:
            drops[f"{p['source']}:decontam_{found[0]}"] += 1
            hits[found[0]] += 1
            continue
        seen.add(key)
        out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--function", type=int, default=4000, help="OpenCodeInstruct problems")
    ap.add_argument("--apps", type=int, default=1000, help="APPS problems")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/distill-prompts")
    ap.add_argument("--cache", default="data/distill-src")
    ap.add_argument("--no-decontam", action="store_true", help="tests only")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    drops: Counter = Counter()
    hits: Counter = Counter()
    sources = {}

    oci_rows: list[dict] = []
    shas = {}
    for f in OCI["files"]:
        p = Path(hf_hub_download(OCI["dataset"], f, repo_type="dataset", revision=OCI["revision"],
                                 local_dir=str(ROOT / args.cache / "oci")))
        shas[f] = sha256(p)
        oci_rows += pq.read_table(p, columns=["id", "input", "unit_tests",
                                              "average_test_score"]).to_pylist()
    sources["opencodeinstruct"] = {**OCI, "sha256": shas, "rows_read": len(oci_rows)}

    apps_path = Path(hf_hub_download(APPS["dataset"], APPS["files"][0], repo_type="dataset",
                                     revision=APPS["revision"],
                                     local_dir=str(ROOT / args.cache / "apps")))
    sys.set_int_max_str_digits(0)  # some APPS cases hold very large integers
    apps_rows = [json.loads(line) for line in open(apps_path, encoding="utf-8") if line.strip()]
    sources["apps"] = {**APPS, "sha256": {APPS["files"][0]: sha256(apps_path)},
                       "rows_read": len(apps_rows)}

    candidates: dict[str, list[dict]] = {"opencodeinstruct": [], "apps": []}
    for src, rows, make in (("opencodeinstruct", oci_rows, function_problem),
                            ("apps", apps_rows, apps_problem)):
        for r in rows:
            p, why = make(r)
            if p is None:
                drops[f"{src}:{why}"] += 1
            else:
                candidates[src].append(p)

    decontam, bench = (None, {}) if args.no_decontam else load_benchmarks(
        {"humaneval": HUMANEVAL_REV, "mbpp": MBPP_REV})
    seen: set[str] = set()
    chosen = (select(candidates["opencodeinstruct"], args.function, decontam, seen, drops, hits, rng)
              + select(candidates["apps"], args.apps, decontam, seen, drops, hits, rng))
    rng.shuffle(chosen)
    n_val = round(len(chosen) * VAL_FRACTION)

    path = out / "prompts.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, p in enumerate(chosen):
            rec = {"id": f"dp-{i:05d}", "split": "val" if i < n_val else "train",
                   "source": p["source"], "source_id": p["source_id"], "licence": p["licence"],
                   "kind": p["kind"], "prompt": p["prompt"],
                   "entry_point": p["entry_point"], "tests": p["tests"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    kept = Counter(p["source"] for p in chosen)
    requested = {"opencodeinstruct": args.function, "apps": args.apps}
    manifest = {
        "version": 1, "seed": args.seed, "sources": sources,
        "benchmarks_decontaminated": ({k: {"dataset": v["dataset"], "revision": v["revision"],
                                           "problems": v["problems"]} for k, v in bench.items()}
                                      if bench else "skipped (--no-decontam)"),
        "rules": {"opencodeinstruct": "average_test_score == 1.0; 3-15 assert tests calling one "
                                      "non-builtin function; problem 150-2500 chars; the prompt "
                                      "names the function",
                  "apps": "train split, introductory, call-based (no class Solution), 3-30 cases "
                          "as asserts <= 20k chars, no float expected values, problem 150-2500 "
                          "chars; the prompt shows the starter signature and names the function",
                  "both": "whitespace-collapsed exact duplicates dropped; prompt + tests through "
                          "quipu.decontam; reference answers never written",
                  "val_fraction": VAL_FRACTION},
        "requested": requested,
        "candidates": {k: len(v) for k, v in candidates.items()},
        "kept": {"total": len(chosen), **{k: kept[k] for k in requested},
                 "train": len(chosen) - n_val, "val": n_val},
        "drops": dict(sorted(drops.items())),
        "decontam_hits": dict(hits),
        "output": {"file": "prompts.jsonl", "sha256": sha256(path)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("candidates", "kept", "decontam_hits")}))
    short = [k for k in requested if kept[k] < requested[k]]
    if short:
        print(f"WARNING: fewer than requested: {short}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
