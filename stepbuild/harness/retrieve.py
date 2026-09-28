"""Pick dataset examples to show the model for a step.

The library is the train split of the step dataset (spec §3.1): each example is a
STEP instruction (a commit message) and the FILE-block reply that carried it out.

Callers query with the STEP TITLE ONLY, not the app spec, and pass the step's
allowed files as `prefer_paths`. Both choices come from measuring on real
commit-message data: a title + spec query over step + reply picked an example of
the same kind of step (model, routes, tests, frontend) first only 12 times in 50,
because the spec's many nouns match everything; the title alone plus a preference
for examples that write the same kind of file got 47 in 50.

File roles (the preference): a Python test file (`test_*.py`, or any .py under a
`tests/` directory), other Python (.py), and frontend source (.js .jsx .ts .tsx
.css). Examples whose reply writes a file with a role of any preferred path come
first, in BM25 order; the other matching examples fill the remaining places, also
in BM25 order. A preferred path with no role (README.md) prefers nothing.

BM25 over GPT-2 token ids (quipu.memory.bm25, tokenized with quipu's tiktoken
wrapper) rather than embeddings: it needs no model, is deterministic, and rewards
exact identifiers (a route name, a field, `create_app`) that matter most when
choosing code to imitate. Each example is indexed on step + reply.

Ranking is deterministic, with ties going to the earlier example, so a benchmark
run is reproducible. Only examples that share at least one token with the query
are returned; an empty library returns [] instead of failing, so the harness runs
unchanged before any dataset exists.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Collection, Sequence

from quipu.memory.bm25 import BM25Index
from quipu.tokenizer import Tokenizer

_STEP_PREFIX = "STEP: "
_STEP_END = "\n\nCONTEXT FILES:"
_FILE_LINE = re.compile(r"^=== FILE: (.+?) ===[ \t]*$", re.MULTILINE)
_FRONTEND_EXTS = (".js", ".jsx", ".ts", ".tsx", ".css")


def file_role(path: str) -> str | None:
    """"test", "python", "frontend", or None for anything else."""
    segments = path.split("/")
    name = segments[-1]
    if name.endswith(".py"):
        if name.startswith("test_") or "tests" in segments[:-1]:
            return "test"
        return "python"
    if name.endswith(_FRONTEND_EXTS):
        return "frontend"
    return None


def _written_roles(reply: str) -> frozenset[str]:
    roles = (file_role(p.strip()) for p in _FILE_LINE.findall(reply))
    return frozenset(r for r in roles if r is not None)


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
        self._roles = [_written_roles(e.reply) for e in self.examples]
        # BM25Index refuses an empty corpus; an empty library simply never matches.
        self._index = (
            BM25Index([self._tok.encode(e.step + "\n" + e.reply) for e in self.examples])
            if self.examples
            else None
        )

    @classmethod
    def from_jsonl(cls, paths: Sequence[Path], split: str = "train") -> "ExampleLibrary":
        """Load the examples of one split from dataset JSONL shards (spec §3.1 format).

        The step is everything between "STEP: " and "\\n\\nCONTEXT FILES:" in the
        user message, so a multi-paragraph commit message keeps its body; the reply
        is the assistant message. CRLF and lone CR become LF in both. Blank lines are
        skipped; anything malformed raises ValueError naming the file and line,
        because a silently skipped row would bias retrieval unseen.
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

    def top(
        self, query: str, k: int = 2, prefer_paths: Collection[str] | None = None
    ) -> list[Example]:
        """The k best examples for `query` (the step title), best first; [] if
        nothing matches. Examples writing a file of the same role as any of
        `prefer_paths` rank ahead of the rest (see the module docstring)."""
        if self._index is None or k <= 0:
            return []
        if isinstance(prefer_paths, str):
            raise TypeError("prefer_paths must be a collection of paths, not a single str")
        ranked = self._index.search(self._tok.encode(query), self._index.n)
        wanted = {file_role(p) for p in prefer_paths or ()} - {None}
        if wanted:
            preferred = [i for i in ranked if self._roles[i] & wanted]
            ranked = preferred + [i for i in ranked if not self._roles[i] & wanted]
        return [self.examples[i] for i in ranked[:k]]


def _to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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
    content = _to_lf(content)
    if not content.startswith(_STEP_PREFIX):
        raise ValueError(f"{where}: user message must start with {_STEP_PREFIX!r}")
    end = content.find(_STEP_END)
    if end < 0:
        raise ValueError(f"{where}: user message has no {_STEP_END.strip()!r} section")
    step = content[len(_STEP_PREFIX):end].strip()
    if not step:
        raise ValueError(f"{where}: empty STEP instruction")
    return Example(step=step, reply=_to_lf(answer))
