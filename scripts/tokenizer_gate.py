"""The quipu-moe tokenizer gate (spec sections 4.3 and 11), against GPT-2, on held-out documents.

Held-out documents are ones train_tokenizer.py never sampled, all read from
sample_manifest.json beside the tokenizer:
  - code: the held-out github-code-clean files (840..879; same licence/path/length
    filters and language weights as the sample, counted in documents), minus any
    exact duplicate of a sampled code document;
  - English: the last sample-10BT parquet file, which the sample never opens (the
    gate refuses to run if it was trained on);
  - the nine other languages: FineWeb-2's test split (the sample reads train only).

Criteria, all must pass (exit 0; otherwise exit 1):
  1. whitespace-only share of code tokens < 15%. A token is whitespace-only when
     decoding it on its own gives a non-empty string of whitespace characters;
  2. characters per token on code better than GPT-2's;
  3. characters per token on English within 5% of GPT-2's (at least 0.95x);
  4. characters per token better than GPT-2's for EACH of the nine other languages;
  5. decode(encode(doc)) == doc for every held-out document.

Run: uv run python scripts/tokenizer_gate.py --tokenizer artifacts/tokenizer/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tokenizer as tt  # noqa: E402

from quipu.fsio import write_text_atomic  # noqa: E402

WHITESPACE_SHARE_MAX = 0.15
TEXT_RATIO_MIN = 0.95


def whitespace_table(tok: Any) -> np.ndarray:
    """is_ws[i]: token i decoded on its own is non-empty and all whitespace."""
    out = np.zeros(tok.vocab_size, dtype=bool)
    for i in range(tok.vocab_size):
        try:
            s = tok.decode([i])
        except Exception:  # ids with no mapping (gaps in a vocabulary)
            continue
        out[i] = bool(s) and s.isspace()
    return out


def _encode_all(tok: Any, docs: Sequence[str]) -> list[list[int]]:
    if hasattr(tok, "encode_batch"):
        return tok.encode_batch(list(docs))
    return [tok.encode(d) for d in docs]


def measure(tok: Any, docs: Sequence[str], is_ws: np.ndarray) -> dict[str, Any]:
    """Token count, characters, whitespace-only tokens and round-trip failures."""
    tokens = chars = ws = 0
    failures: list[int] = []
    for i, (doc, ids) in enumerate(zip(docs, _encode_all(tok, docs))):
        tokens += len(ids)
        chars += len(doc)
        if ids:
            ws += int(is_ws[np.asarray(ids)].sum())
        if tok.decode(ids) != doc:
            failures.append(i)
    return {"documents": len(docs), "tokens": tokens, "chars": chars,
            "whitespace_tokens": ws,
            "whitespace_share": ws / tokens if tokens else 0.0,
            "chars_per_token": chars / tokens if tokens else 0.0,
            "round_trip_failures": failures}


def judge(new: dict[str, dict], gpt2: dict[str, dict],
          english: str = tt.ENGLISH) -> list[dict]:
    """The criteria as rows {measure, gpt2, new, required, passed}.

    new and gpt2 map a set name ("code", english, or a FineWeb-2 language) to its
    measure() result; every set other than code and english must beat GPT-2.
    """
    code_n, code_g, eng_n, eng_g = new["code"], gpt2["code"], new[english], gpt2[english]
    ratio = eng_n["chars_per_token"] / eng_g["chars_per_token"]
    rows = [
        {"measure": "Whitespace-only share of code tokens",
         "gpt2": f"{code_g['whitespace_share']:.1%}",
         "new": f"{code_n['whitespace_share']:.1%}",
         "required": f"< {WHITESPACE_SHARE_MAX:.0%}",
         "passed": code_n["whitespace_share"] < WHITESPACE_SHARE_MAX},
        {"measure": "Characters per token, code",
         "gpt2": f"{code_g['chars_per_token']:.3f}",
         "new": f"{code_n['chars_per_token']:.3f}",
         "required": "better than GPT-2",
         "passed": code_n["chars_per_token"] > code_g["chars_per_token"]},
        {"measure": f"Characters per token, {english}",
         "gpt2": f"{eng_g['chars_per_token']:.3f}",
         "new": f"{eng_n['chars_per_token']:.3f} ({ratio:.3f}x)",
         "required": f">= {TEXT_RATIO_MIN}x GPT-2",
         "passed": ratio >= TEXT_RATIO_MIN},
    ]
    for name in new:
        if name in ("code", english):
            continue
        n, g = new[name]["chars_per_token"], gpt2[name]["chars_per_token"]
        rows.append({"measure": f"Characters per token, {name}", "gpt2": f"{g:.3f}",
                     "new": f"{n:.3f} ({n / g:.2f}x)" if g else f"{n:.3f}",
                     "required": "better than GPT-2", "passed": n > g})
    fails = sum(len(m["round_trip_failures"]) for m in new.values())
    docs = sum(m["documents"] for m in new.values())
    rows.append({"measure": "Round trip decode(encode(x)) == x", "gpt2": "-",
                 "new": f"{docs - fails}/{docs} exact", "required": "every document",
                 "passed": fails == 0})
    return rows


def render(rows: list[dict], info: dict[str, Any]) -> str:
    verdict = "PASS" if all(r["passed"] for r in rows) else "FAIL"
    lines = [f"# Tokenizer gate: {verdict}", "",
             f"Tokenizer `{info['tokenizer']}` (sha256 `{info['sha256']}`, vocab "
             f"{info['vocab_size']:,}), measured {info['when']}.", "",
             "| Measure | GPT-2 | quipu-moe | Required | Result |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['measure']} | {r['gpt2']} | {r['new']} | {r['required']} | "
                     f"{'pass' if r['passed'] else 'FAIL'} |")
    lines += ["", "Held-out documents (never in the training sample):", ""]
    for name, d in info["sets"].items():
        lines.append(f"- {name}: {d}")
    lines += ["", "Token counts:", "", "| Set | Tokenizer | Documents | Characters | Tokens | "
              "Whitespace-only tokens |", "|---|---|---|---|---|---|"]
    for (set_name, tok_name), m in info["measurements"].items():
        lines.append(f"| {set_name} | {tok_name} | {m['documents']:,} | {m['chars']:,} | "
                     f"{m['tokens']:,} | {m['whitespace_tokens']:,} |")
    return "\n".join(lines) + "\n"

# ------------------------------------------------------------------ held-out data

def heldout_code(manifest: dict, n: int, exclude: set[int]) -> tuple[list[str], dict]:
    from huggingface_hub import HfFileSystem

    code = manifest["code"]
    first, last = code["heldout_files"]
    trained_last = code["files_read"][1]
    if trained_last is not None and trained_last >= first:
        raise SystemExit(f"code file {trained_last} was sampled for training; "
                         f"held-out starts at {first}")
    sampler = tt.QuotaSampler(tt.LANGUAGE_WEIGHTS, n, window=max(n, 1000))
    stats: Counter = Counter()
    rows = tt.code_rows(HfFileSystem(), code["revision"], range(first, last + 1))
    docs = [c for _, c in tt.code_documents(rows, sampler, size=lambda _: 1,
                                            exclude=frozenset(exclude), stats=stats)]
    by_lang = {k: v["documents"] for k, v in sampler.summary()["by_language"].items()}
    return docs, {"files": f"{first}..{stats.get('last_file')}", "documents": len(docs),
                  "dropped_as_duplicate_of_sample": stats.get("dropped_as_duplicate", 0),
                  "by_language": by_lang}


def heldout_english(manifest: dict, n: int) -> tuple[list[str], dict]:
    text = manifest["text"]
    files = text["heldout_files"]
    if set(files) & set(text["train_files"]):
        raise SystemExit(f"held-out text {files} overlaps the training files")
    docs = list(tt.text_documents(tt.text_rows(text["revision"], files), n,
                                  size=lambda _: 1))
    return docs, {"files": files, "documents": len(docs)}


def heldout_fineweb2(fs: Any, manifest: dict, lang: str, n: int) -> tuple[list[str], dict]:
    fw2 = manifest["fineweb2"]
    files = fw2["heldout_files"][lang]
    if set(files) & set(fw2["train_files"][lang]):
        raise SystemExit(f"held-out {lang} {files} overlaps the training files")
    docs = list(tt.text_documents(tt.iter_parquet_text(fs, files), n, size=lambda _: 1))
    return docs, {"files": ["/".join(f.split("/")[-3:]) for f in files],
                  "documents": len(docs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--manifest", default=None,
                        help="default: sample_manifest.json beside the tokenizer")
    parser.add_argument("--text-sample", type=int, default=2000, help="English documents")
    parser.add_argument("--code-sample", type=int, default=2000, help="documents")
    parser.add_argument("--lang-sample", type=int, default=500,
                        help="documents per FineWeb-2 language")
    parser.add_argument("--out", default="results/tokenizer_gate.md")
    args = parser.parse_args()

    from huggingface_hub import HfFileSystem

    from quipu.bpe import BPETokenizer
    from quipu.tokenizer import Tokenizer

    tok_path = Path(args.tokenizer)
    manifest_path = Path(args.manifest) if args.manifest else \
        tok_path.with_name("sample_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes_path = manifest_path.with_name(manifest["code"]["content_hashes"])
    exclude = {int(h) for h in np.load(hashes_path)}

    new, gpt2 = BPETokenizer(tok_path), Tokenizer()
    fs = HfFileSystem()
    sets: dict[str, list[str]] = {}
    set_info: dict[str, dict] = {}
    sets["code"], set_info["code"] = heldout_code(manifest, args.code_sample, exclude)
    sets[tt.ENGLISH], set_info[tt.ENGLISH] = heldout_english(manifest, args.text_sample)
    for lang in manifest["fineweb2"]["languages"]:
        sets[lang], set_info[lang] = heldout_fineweb2(fs, manifest, lang, args.lang_sample)
    for name, docs in sets.items():
        print(f"{name}: {len(docs)} held-out documents", flush=True)
        if not docs:
            raise SystemExit(f"no held-out documents for {name}")

    ws_new, ws_gpt2 = whitespace_table(new), whitespace_table(gpt2)
    m_new = {name: measure(new, docs, ws_new) for name, docs in sets.items()}
    m_gpt2 = {name: measure(gpt2, docs, ws_gpt2) for name, docs in sets.items()}
    rows = judge(m_new, m_gpt2)
    measurements = {}
    for name in sets:
        measurements[name, "quipu-moe"] = m_new[name]
        measurements[name, "gpt2"] = m_gpt2[name]
    info = {"tokenizer": str(tok_path).replace("\\", "/"), "sha256": new.sha256(),
            "vocab_size": new.vocab_size,
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sets": set_info, "measurements": measurements}
    report = render(rows, info)
    write_text_atomic(args.out, report)
    print(report, flush=True)
    sys.exit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
