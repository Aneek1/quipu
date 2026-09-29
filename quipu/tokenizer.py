"""GPT-2 BPE, wrapped.

The wrapper exists for one reason: sub-project 2 may train a tokenizer on a
code-heavy mixture, and every call site should keep working when it does.
"""
from __future__ import annotations

from pathlib import Path

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

    def token_byte_lengths(self) -> list[int]:
        """Raw UTF-8 byte length of every token id (bits per byte counts with it);
        special tokens (<|endoftext|>) are 0 bytes of text."""
        enc = self._enc
        special = {enc.encode_single_token(s) for s in enc.special_tokens_set}
        return [0 if i in special else len(enc.decode_single_token_bytes(i))
                for i in range(enc.n_vocab)]


def make_tokenizer(tokenizer: str = "gpt2"):
    """The tokenizer a config names: "gpt2" (quipu-114m) or a path to a tokenizer.json
    trained by scripts/train_tokenizer.py (quipu-moe). Returns Tokenizer | BPETokenizer."""
    if tokenizer == "gpt2":
        return Tokenizer()
    from quipu.bpe import BPETokenizer  # the tokenizers package is not needed for GPT-2

    path = Path(tokenizer)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer {tokenizer!r} is neither 'gpt2' nor a file")
    return BPETokenizer(path)
