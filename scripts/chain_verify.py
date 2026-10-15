"""Re-verify generated chains from scratch and write the kept ones as chat messages.

For every chain in data/chains/gen/*.jsonl, each turn the generator marked ok is checked
again in a fresh sandbox (its code against its own asserts and every earlier turn's).
The chain is cut at the first turn that fails; it is kept when 2+ turns survive. Kept
chains go to data/chains/kept/{train,val}/part-NNNN.jsonl as rows with "messages"; counts go
to data/chains/verify_summary.json. When a chain id appears under several teacher
tags, the first by scripts.distill_verify's tag priority wins. Re-checks cut short by a
sandbox crash (infrastructure, not the answer) are counted apart as recheck_sandbox_error.
Fine-tuning reads kept/train only; the held-out eval (later chain_eval) reads gen/ for
the val seeds' tests, so val chains never reach training.

    .venv/Scripts/python.exe scripts/chain_verify.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.chains import check_turn                       # noqa: E402
from quipu.fsio import write_text_atomic                   # noqa: E402
from quipu.sandbox import extract_code                     # noqa: E402
from scripts.chain_generate import SETTINGS               # noqa: E402
from scripts.distill_verify import _tag_rank               # noqa: E402


def recheck(chain: dict, timeout: float = 10.0) -> tuple[int, str]:
    """(how many leading turns pass a fresh check, why it stopped). The reason is the
    failing check_turn result, "no_code", "no_tests" (a turn with nothing to check) or
    "not_ok" (generator didn't mark the turn ok), or "" when every turn passed. Turn 1 is
    checked against its own tests. A sandbox_error is retried once before it cuts the chain."""
    entry, earlier = chain["entry_point"], []
    for i, t in enumerate(chain["turns"]):
        if t["status"] != "ok":
            return i, "not_ok"
        if not t["tests"]:
            return i, "no_tests"
        code = extract_code(t["assistant"], entry)
        if code is None:
            return i, "no_code"
        res = check_turn(code, t["tests"], earlier, timeout)
        if res == "sandbox_error":
            res = check_turn(code, t["tests"], earlier, timeout)
        if res != "ok":
            return i, res
        earlier += t["tests"]
    return len(chain["turns"]), ""


def _safe_recheck(chain: dict, timeout: float) -> tuple[int, str]:
    try:
        return recheck(chain, timeout)
    except Exception as e:  # one odd chain must not kill the run
        return 0, "verifier:" + type(e).__name__


FILE_NAME = re.compile(r"^(part-\d+)\.(.+)\.jsonl$")


def _load(gen_dir: Path, stats: Counter) -> dict[str, list[dict]]:
    """Chains per part, first by teacher-tag priority for a repeated id. Ledger files
    (*.failures.jsonl) are skipped; torn lines and records without turns are counted."""
    found = []
    for p in gen_dir.glob("part-*.jsonl"):
        m = FILE_NAME.match(p.name)
        if m and not p.name.endswith(".failures.jsonl"):
            found.append((int(m[1][5:]), _tag_rank(m[2]), p.name, m[1], p))
    seen: set[str] = set()
    parts: dict[str, list[dict]] = {}
    for _, _, _, part, p in sorted(found):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except ValueError:
                stats["bad_lines"] += 1
                continue
            if not isinstance(c, dict) or not {"id", "turns", "entry_point"} <= c.keys():
                stats["bad_records"] += 1
                continue
            if c["id"] not in seen:
                seen.add(c["id"])
                parts.setdefault(part, []).append(c)
    return parts


def verify(root: Path, workers: int = 8, timeout: float = 10.0) -> dict:
    root = Path(root)
    summary = {"chains": 0, "kept": 0, "kept_3_turns": 0, "recheck_failed": 0,
               "recheck_sandbox_error": 0, "verifier_errors": 0, "bad_lines": 0,
               "bad_records": 0, "stop": Counter(), "kept_by_split": Counter()}
    stats: Counter = Counter()
    parts = _load(root / "gen", stats)
    summary["bad_lines"], summary["bad_records"] = stats["bad_lines"], stats["bad_records"]
    for part, chains in sorted(parts.items(), key=lambda kv: int(kv[0][5:])):
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: _safe_recheck(c, timeout), chains))
        rows: dict[str, list[dict]] = {"train": [], "val": []}
        for c, (n, why) in zip(chains, results):
            summary["chains"] += 1
            summary["stop"][c.get("stop") or "max_turns"] += 1
            if why.startswith("verifier:"):
                summary["verifier_errors"] += 1
            elif why == "sandbox_error":
                summary["recheck_sandbox_error"] += 1
            elif n < c.get("kept_turns", 0):
                summary["recheck_failed"] += 1
            if n < 2:
                continue
            msgs = []
            for t in c["turns"][:n]:
                msgs += [{"role": "user", "content": t["user"]},
                         {"role": "assistant", "content": t["assistant"]}]
            split = c.get("split")
            rows["val" if split == "val" else "train"].append(
                {"id": c["id"], "seed_id": c.get("seed_id"), "split": split,
                 "teacher": c.get("teacher"), "revision": c.get("revision"),
                 "teacher_tag": c.get("teacher_tag"), "turns": n,
                 **{k: c[k] for k in SETTINGS if k in c}, "messages": msgs})
            summary["kept"] += 1
            summary["kept_3_turns"] += n >= 3
            summary["kept_by_split"][split] += 1
        for sp, rs in rows.items():
            (root / "kept" / sp).mkdir(parents=True, exist_ok=True)
            write_text_atomic(root / "kept" / sp / f"{part}.jsonl",
                              "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rs))
    summary = json.loads(json.dumps(summary))               # Counters -> plain dicts
    write_text_atomic(root / "verify_summary.json", json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/chains")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    print(json.dumps(verify(ROOT / args.root, args.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
