"""A code-aware, multilingual byte-level BPE tokenizer (quipu-moe spec 4 and 11), wrapped.

GPT-2's tokenizer spends 36.5% of quipu-114m's code validation tokens on pure
whitespace: indentation is split into many small space tokens. This backend is
trained on the quipu-moe mix (code, English and nine other languages) with a
pre-tokeniser that gives whitespace structure its own pieces, so BPE can learn one
token per line break plus indent. It is ONE regex whose alternatives are tried in
this order at each position (the leftmost alternative that matches wins):

  1. NEWLINE_INDENT  a run of newlines plus the up to 16 spaces/tabs after it
                     (CRLF stays together), so a code line costs one whitespace
                     token rather than two;
  2. SPACE_RUN       2 to 16 spaces (a longer run is cut every 16). A SINGLE space
                     is left to WORD, so " the" stays one token and prose costs no
                     more tokens than with GPT-2;
  3. TAB_RUN         1 to 16 tabs;
  4. DIGIT           every digit alone;
  5. CJK             1 to 3 Han/Hiragana/Katakana/Hangul characters (an optional
                     leading space kept, so Korean words keep their space the way
                     " the" does); a longer run is cut left to right. See CJK_CLASS;
  6. GPT-4o-style word and punctuation alternatives: contractions; an optional
     non-letter then (non-CJK) letters AND combining marks (so Tamil and Devanagari vowel
     signs stay inside their word); punctuation runs, which never swallow a
     following newline (it belongs to NEWLINE_INDENT); any other non-newline
     whitespace.

One regex rather than a Sequence of Splits: a later Split would cut apart the
pieces an earlier one made (the space rule would split "\\n" + indent again).

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

NEWLINE_INDENT = r"[\r\n]+[ \t]{0,16}"
SPACE_RUN = r" {2,16}"
TAB_RUN = r"\t{1,16}"
DIGIT = r"\p{N}"
CONTRACTION = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
# Han, Hiragana, Katakana (with the prolonged sound mark U+30FC, which is script
# Common) and Hangul. Chinese and Japanese are written without spaces, so an
# uncapped run is a whole clause: nearly every pre-token is unique and the
# trainer's word table outgrew this laptop's RAM (3.5 GB and rising at 467 of
# 500 MB). Capped at 3 characters, cut left to right, the pieces repeat and BPE
# can still learn every 1-3 character word.
CJK_CLASS = r"\p{Han}\p{Hiragana}\p{Katakana}\x{30FC}\p{Hangul}"
CJK_MAX = 3
CJK = r" ?[" + CJK_CLASS + r"]{1," + str(CJK_MAX) + "}"
WORD = r"[^\r\n\p{L}\p{M}\p{N}]?[[\p{L}\p{M}]&&[^" + CJK_CLASS + r"]]+"
PUNCT = r" ?[^\s\p{L}\p{M}\p{N}]+"
OTHER_SPACE = r"[^\S\r\n]+(?!\S)|[^\S\r\n]+"
PRE_TOKENIZER_RULES = (NEWLINE_INDENT, SPACE_RUN, TAB_RUN, DIGIT, CONTRACTION, CJK, WORD,
                       PUNCT, OTHER_SPACE)


def build_pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    """The pre-tokeniser described above: one split on the joined rules, then bytes."""
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex("|".join(PRE_TOKENIZER_RULES)), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


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
