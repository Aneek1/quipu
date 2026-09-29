"""A code-aware byte-level BPE tokenizer (quipu-moe spec section 4), wrapped.

GPT-2's tokenizer spends 36.5% of quipu-114m's code validation tokens on pure
whitespace: indentation is split into many small space tokens. This backend is
trained on the quipu-moe mix with a pre-tokeniser that gives whitespace structure
its own pieces, so BPE can learn one token per common indent:

  1. runs of newlines are their own pieces ([\\r\\n]+, so CRLF stays together);
  2. runs of 2 to 16 spaces are their own pieces (a longer run is cut every 16);
     a SINGLE space is left to the word regex, so " the" stays one token and
     prose costs no more tokens than with GPT-2;
  3. runs of 1 to 16 tabs are their own pieces;
  4. every digit is its own piece;
  5. what is left is split by the GPT-4 (cl100k) word/punctuation regex.

The pieces are then mapped to bytes (ByteLevel, no prefix space, no second regex),
and the 256-byte alphabet is always in the vocabulary, so any string round-trips
exactly: decode(encode(x)) == x.

Special tokens sit at fixed low ids 0..6 in SPECIAL_TOKENS order, the same in
every training run. `encode` never produces them from text (a document containing
the literal "<|user|>" is encoded as ordinary text, exactly as the GPT-2 wrapper's
encode_ordinary does); `encode_with_special` parses them, for chat formatting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import Regex, decoders, models, pre_tokenizers, trainers
from tokenizers import Tokenizer as HFTokenizer

from quipu.fsio import write_text_atomic

SPECIAL_TOKENS = ("<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "=== FILE: ", "=== END FILE ===")
EOT = SPECIAL_TOKENS[0]
MAX_VOCAB = 65536  # shards are uint16

NEWLINE_RUN = r"[\r\n]+"
SPACE_RUN = r" {2,16}"
TAB_RUN = r"\t{1,16}"
DIGIT = r"\p{N}"
# GPT-4's cl100k pattern (digits are already single pieces by the time it runs).
WORD = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_TOKENIZER_RULES = (NEWLINE_RUN, SPACE_RUN, TAB_RUN, DIGIT, WORD)


def build_pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    """The section 4.1 pre-tokeniser: the rules above in order, then bytes."""
    splits = [pre_tokenizers.Split(Regex(rule), behavior="isolated")
              for rule in PRE_TOKENIZER_RULES]
    return pre_tokenizers.Sequence(
        splits + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


def train_bpe(texts: Iterable[str] | Iterable[list[str]], vocab_size: int,
              out_path: str | Path, *, show_progress: bool = False,
              min_frequency: int = 2) -> "BPETokenizer":
    """Train byte-level BPE on `texts` (strings, or batches of strings) and save it.

    `texts` is consumed once, lazily, so a large sample can be streamed from disk
    or the network without being held in memory. The special tokens are given to
    the trainer first, which places them at ids 0..len(SPECIAL_TOKENS)-1.
    """
    min_vocab = len(SPECIAL_TOKENS) + 256
    if not min_vocab < vocab_size <= MAX_VOCAB:
        raise ValueError(f"vocab_size must be in ({min_vocab}, {MAX_VOCAB}]; got {vocab_size}")
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = build_pre_tokenizer()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress)
    tok.train_from_iterator(texts, trainer=trainer)
    write_text_atomic(out_path, tok.to_str())
    return BPETokenizer(out_path)


class BPETokenizer:
    """tokenizer.json behind the same interface as quipu.tokenizer.Tokenizer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Two instances of the same file: one that treats special-token strings as
        # text (encode), one that parses them (encode_with_special). Flipping one
        # shared flag per call would not be thread-safe.
        self._plain = HFTokenizer.from_file(str(self.path))
        self._plain.encode_special_tokens = True
        self._special = HFTokenizer.from_file(str(self.path))
        for i, name in enumerate(SPECIAL_TOKENS):
            got = self._plain.token_to_id(name)
            if got != i:
                raise ValueError(f"{self.path}: special token {name!r} has id {got}, "
                                 f"expected {i}")
        if self.vocab_size > MAX_VOCAB:
            raise ValueError(f"{self.path}: vocab {self.vocab_size} does not fit uint16")

    @property
    def vocab_size(self) -> int:
        return self._plain.get_vocab_size(with_added_tokens=True)

    @property
    def eot(self) -> int:
        return self.special_id(EOT)

    def special_id(self, name: str) -> int:
        if name not in SPECIAL_TOKENS:
            raise KeyError(f"not a special token: {name!r}")
        return SPECIAL_TOKENS.index(name)

    def encode(self, text: str) -> list[int]:
        return self._plain.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """encode() for many texts at once (parallel in the Rust backend)."""
        return [e.ids for e in self._plain.encode_batch(texts, add_special_tokens=False)]

    def encode_with_special(self, text: str) -> list[int]:
        return self._special.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._plain.decode(list(ids), skip_special_tokens=False)

    def sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def describe(self) -> dict:
        return {"path": str(self.path), "vocab_size": self.vocab_size,
                "sha256": self.sha256(),
                "pre_tokenizer": json.loads(self._plain.to_str())["pre_tokenizer"]}
