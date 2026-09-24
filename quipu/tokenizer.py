"""GPT-2 BPE, wrapped.

The wrapper exists for one reason: sub-project 2 may train a tokenizer on a
code-heavy mixture, and every call site should keep working when it does.
"""
from __future__ import annotations

import tiktoken


class Tokenizer:
    def __init__(self, encoding: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eot(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
