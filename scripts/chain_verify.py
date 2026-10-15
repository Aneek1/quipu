"""Re-verify generated chains from scratch and write the kept ones as chat messages.

For every chain in data/chains/gen/*.jsonl, each turn the generator marked ok is checked
again in a fresh sandbox (its code against its own asserts and every earlier turn's).
The chain is cut at the first turn that fails; it is kept when 2+ turns survive. Kept
chains go to data/chains/kept/part-NNNN.jsonl as {"messages": [...]} rows; counts go
to data/chains/verify_summary.json. When a chain id appears under several teacher
tags, the first by scripts.distill_verify's tag priority wins. Re-checks cut short by a
sandbox crash (infrastructure, not the answer) are counted apart as recheck_sandbox_error.

    .venv/Scripts/python.exe scripts/chain_verify.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.chains import check_turn                       # noqa: E402
from quipu.fsio import write_text_atomic                   # noqa: E402
from quipu.sandbox import extract_code                     # noqa: E402
from scripts.distill_verify import _tag_rank               # noqa: E402

SETTINGS = ("max_turns", "min_new_tests", "max_new_tokens", "timeout")


def recheck(chain: dict, timeout: float = 10.0) -> tuple[int, str]:
    """(how many leading turns pass a fresh check, why it stopped). The reason is the
    failing check_turn result, "no_code" or "not_ok" (generator didn't mark the turn ok),
    or "" when every turn passed. Turn 1 is checked against its own tests."""
    entry, earlier = chain["entry_point"], []
    for i, t in enumerate(chain["turns"]):
        if t["status"] != "ok":
            return i, "not_ok"
        code = extract_code(t["assistant"], entry)
        if code is None:
            return i, "no_code"
        res = check_turn(code, t["tests"], earlier, timeout)
        if res != "ok":
            return i, res
        earlier += t["tests"]
    return len(chain["turns"]), ""


def _load(gen_dir: Path) -> dict[str, list[dict]]:
    """Chains per part, first by teacher-tag priority for a repeated id."""
    files = sorted(gen_dir.glob("part-*.jsonl"), key=lambda p: (p.name[:9], _tag_rank(p.name[10:-6])))
    seen: set[str] = set()
    parts: dict[str, list[dict]] = {}
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c["id"] not in seen:
                seen.add(c["id"])
                parts.setdefault(p.name[:9], []).append(c)
    return parts


def verify(root: Path, workers: int = 8, timeout: float = 10.0) -> dict:
    root = Path(root)
    summary = {"chains": 0, "kept": 0, "kept_3_turns": 0, "recheck_failed": 0,
               "recheck_sandbox_error": 0, "stop": Counter(), "kept_by_split": Counter()}
    for part, chains in sorted(_load(root / "gen").items()):
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: recheck(c, timeout), chains))
        rows = []
        for c, (n, why) in zip(chains, results):
            summary["chains"] += 1
            summary["stop"][c["stop"] or "max_turns"] += 1
            if why == "sandbox_error":
                summary["recheck_sandbox_error"] += 1
            elif n < c["kept_turns"]:
                summary["recheck_failed"] += 1
            if n < 2:
                continue
            msgs = []
            for t in c["turns"][:n]:
                msgs += [{"role": "user", "content": t["user"]},
                         {"role": "assistant", "content": t["assistant"]}]
            rows.append({"id": c["id"], "seed_id": c["seed_id"], "split": c["split"],
                         "teacher": c.get("teacher"), "revision": c.get("revision"),
                         "teacher_tag": c.get("teacher_tag"), "turns": n,
                         **{k: c[k] for k in SETTINGS if k in c}, "messages": msgs})
            summary["kept"] += 1
            summary["kept_3_turns"] += n >= 3
            summary["kept_by_split"][c["split"]] += 1
        (root / "kept").mkdir(parents=True, exist_ok=True)
        write_text_atomic(root / "kept" / f"{part}.jsonl",
                          "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
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
