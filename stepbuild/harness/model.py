"""The one interface between the harness and a model.

A model turns a chat (a list of {"role", "content"} messages) into one reply
string. That is all the runner needs, so any backend (a local Qwen, Quipu, an API)
fits behind it later without the harness changing. Only ScriptedModel lives here:
it replays canned replies so the runner and the benchmark can be tested
end-to-end (the reference solutions are played through it) without a GPU.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class StepModel(Protocol):
    name: str  # recorded in traces and results, so runs can be told apart

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class ScriptedModel:
    """Returns canned replies in order; raises RuntimeError if asked more times than
    scripted, so a runner that retries too often fails loudly instead of reusing a
    reply. Records a copy of every messages list it was given (for assertions): a
    copy, because the caller may keep appending to the list it passed."""

    def __init__(self, replies: Sequence[str], name: str = "scripted") -> None:
        if isinstance(replies, str):
            raise TypeError("replies must be a sequence of strings, not a single str")
        replies = tuple(replies)
        bad = [type(r).__name__ for r in replies if not isinstance(r, str)]
        if bad:
            raise TypeError(f"every reply must be a str, got {bad}")
        self.name = name
        self._replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        n = len(self.calls)
        if n >= len(self._replies):
            raise RuntimeError(
                f"ScriptedModel {self.name!r} has {len(self._replies)} scripted replies "
                f"and was asked for reply {n + 1}"
            )
        self.calls.append([dict(m) for m in messages])
        return self._replies[n]
