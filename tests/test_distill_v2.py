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
