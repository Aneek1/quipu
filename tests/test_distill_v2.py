"""Distillation v2: prompt parts, generator, verifier (no network, no GPU)."""
import json
from pathlib import Path

from quipu.sandbox import extract_code

STYLED = (
    "We need the sum of the even numbers in a list; one pass with a filter does it.\n\n"
    "1. Walk the list.\n2. Keep values divisible by 2.\n3. Add them up.\n\n"
    "```python\ndef sum_even(xs):\n    \"\"\"Sum of the even values in xs.\"\"\"\n"
    "    return sum(x for x in xs if x % 2 == 0)\n```\n\n"
    "Example:\n```python\nprint(sum_even([1, 2, 4]))\n```\n\n"
    "Edge cases: an empty list gives 0; negatives work; O(n) time, O(1) space."
)


def test_extract_code_explain_first_reply():
    code = extract_code(STYLED, "sum_even")
    assert code.startswith("def sum_even(xs):")
    assert "print(" not in code  # the usage block is not taken


from scripts.build_distill_prompts_v2 import PartWriter, take_unique


def test_part_writer_rolls_parts_and_ids(tmp_path):
    w = PartWriter(tmp_path, part_size=3)
    for i in range(7):
        w.add({"source": "opencodeinstruct", "source_id": f"s{i}", "licence": "CC-BY-4.0",
               "kind": "function", "prompt": f"p{i}", "entry_point": "f", "tests": ["assert f()"]})
    parts = w.close()
    assert [p["rows"] for p in parts] == [3, 3, 1]
    rows = [json.loads(l) for l in open(tmp_path / "part-0001.jsonl", encoding="utf-8")]
    assert [r["id"] for r in rows] == ["dv2-0000003", "dv2-0000004", "dv2-0000005"]
    assert all(r["part"] == 1 for r in rows)
    assert rows[0]["split"] == "train"
    first = json.loads(open(tmp_path / "part-0000.jsonl", encoding="utf-8").readline())
    assert first["split"] == "val"  # every 20th prompt (index % 20 == 0) is val
    assert len(parts[0]["sha256"]) == 64


def test_take_unique_drops_seen_and_contaminated():
    from collections import Counter

    class Fake:
        def find(self, text):
            return ("mbpp", "1", "substring") if "LEAK" in text else None

    seen, drops, hits = set(), Counter(), Counter()
    probs = [{"source": "opencodeinstruct", "problem": t, "prompt": t, "tests": ["assert f()"]}
             for t in ("a  b", "a b", "LEAK", "c")]
    kept = list(take_unique(probs, Fake(), seen, drops, hits))
    assert [p["problem"] for p in kept] == ["a  b", "c"]
    assert drops["opencodeinstruct:duplicate"] == 1 and hits["mbpp"] == 1
    assert len(seen) == 2 and all(isinstance(k, bytes) for k in seen)  # hashes, not text
