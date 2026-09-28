"""Pick dataset examples to show the model for a step.

The library is the train split of the step dataset (spec §3.1): each example is a
STEP instruction and the FILE-block reply that carried it out. For a benchmark
step the caller queries with the step title plus the app spec, and the top two
examples go into the prompt.

BM25 over GPT-2 token ids (quipu.memory.bm25, tokenized with quipu's tiktoken
wrapper) rather than embeddings: it needs no model, is deterministic, and rewards
exact identifiers (a route name, a field, `create_app`) that matter most when
choosing code to imitate. Each example is indexed on step + reply, so a query
matches both what was asked and what was written.

Ranking is deterministic, with ties going to the earlier example, so a benchmark
run is reproducible. Only examples that share at least one token with the query
are returned; an empty library returns [] instead of failing, so the harness runs
unchanged before any dataset exists.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Sequence

from quipu.memory.bm25 import BM25Index
from quipu.tokenizer import Tokenizer

_STEP_PREFIX = "STEP: "


@dataclasses.dataclass(frozen=True)
class Example:
    step: str   # the STEP instruction
    reply: str  # FILE blocks


class ExampleLibrary:
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples: tuple[Example, ...] = tuple(examples)
        bad = [type(e).__name__ for e in self.examples if not isinstance(e, Example)]
        if bad:
            raise TypeError(f"examples must be Example instances, got {bad}")
        self._tok = Tokenizer("gpt2")
        # BM25Index refuses an empty corpus; an empty library simply never matches.
        self._index = (
            BM25Index([self._tok.encode(e.step + "\n" + e.reply) for e in self.examples])
            if self.examples
            else None
        )

    @classmethod
    def from_jsonl(cls, paths: Sequence[Path], split: str = "train") -> "ExampleLibrary":
        """Load the examples of one split from dataset JSONL shards (spec §3.1 format).

        The step is the text after "STEP: " on the user message, up to the first
        blank line (where CONTEXT FILES begins); the reply is the assistant message.
        Blank lines are skipped; anything malformed raises ValueError naming the
        file and line, because a silently skipped row would bias retrieval unseen.
        """
        examples: list[Example] = []
        for path in paths:
            path = Path(path)
            with path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    where = f"{path.name}:{lineno}"
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{where}: not valid JSON ({e.msg})") from None
                    example_split = row.get("split") if isinstance(row, dict) else None
                    if not isinstance(example_split, str):
                        raise ValueError(f"{where}: row has no 'split' string")
                    if example_split != split:
                        continue
                    examples.append(_parse_row(row, where))
        return cls(examples)

    def top(self, query: str, k: int = 2) -> list[Example]:
        """The k best-matching examples, best first; [] if nothing matches."""
        if self._index is None or k <= 0:
            return []
        hits = self._index.search(self._tok.encode(query), k)
        return [self.examples[i] for i in hits]


def _parse_row(row: dict, where: str) -> Example:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{where}: row has no 'messages' list")
    user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    reply = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"), None
    )
    if user is None or reply is None:
        raise ValueError(f"{where}: needs a user and an assistant message")
    content, answer = user.get("content"), reply.get("content")
    if not isinstance(content, str) or not isinstance(answer, str):
        raise ValueError(f"{where}: message contents must be strings")
    if not content.startswith(_STEP_PREFIX):
        raise ValueError(f"{where}: user message must start with {_STEP_PREFIX!r}")
    step = content[len(_STEP_PREFIX):].split("\n\n", 1)[0].strip()
    if not step:
        raise ValueError(f"{where}: empty STEP instruction")
    return Example(step=step, reply=answer)
