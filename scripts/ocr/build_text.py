"""Cut OCR text lines from Quipu's token shards (spec section 4.1).

    .venv/Scripts/python scripts/ocr/build_text.py --shards data/shards-moe/train \
        --out data/ocr/text --lines 10000000

Documents are the EOT-separated runs of a shard, decoded with Quipu's tokenizer. Shards
are visited in a seeded random order and each document is kept with probability
--doc-keep, so the lines come from the whole corpus rather than the first shard. Each
kept document is cut into lines of 5-60 characters at word boundaries, keeping only
characters in the OCR set. About 15% of lines are random strings (codes, digits,
punctuation) so a model cannot read by guessing language. One document in 20 goes to
held.txt, chosen by a hash of "shard_name:doc_index" (so the choice does not depend on
sampling), the rest to train.txt; a line of 12+ characters that lands in both is removed
from train. Files are written with LF line endings."""
from __future__ import annotations

import argparse
import hashlib
import random
import string
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quipu.ocr_ctc import CHARSET, in_charset  # noqa: E402

MIN_CHARS, MAX_CHARS = 5, 60
LEAK_MIN_CHARS = 12      # shorter train lines that also occur in held are not leakage


def cut_lines(doc: str, rng: random.Random, min_chars: int = MIN_CHARS,
              max_chars: int = MAX_CHARS) -> list[str]:
    words = [w for w in doc.split() if in_charset(w)]
    lines, cur = [], []
    target = rng.randint(min_chars, max_chars)
    for w in words:
        if len(w) > max_chars:
            continue
        cand = " ".join(cur + [w])
        if len(cand) > target and cur:
            line = " ".join(cur)
            if len(line) >= min_chars:
                lines.append(line)
            cur, target = [w], rng.randint(min_chars, max_chars)
        else:
            cur.append(w)
    line = " ".join(cur)
    if min_chars <= len(line) <= max_chars:
        lines.append(line)
    return lines


_KINDS = (string.digits, string.ascii_uppercase + string.digits, CHARSET,
          string.ascii_letters + string.digits + "-_/.:#")


def random_line(rng: random.Random, max_chars: int = MAX_CHARS) -> str:
    alphabet = rng.choice(_KINDS)
    n = rng.randint(MIN_CHARS, max_chars)
    s = "".join(rng.choice(alphabet) for _ in range(n)).strip()
    return s if len(s) >= MIN_CHARS else "".join(rng.choice(string.digits) for _ in range(MIN_CHARS))


def is_held(doc_key: str) -> bool:
    return int(hashlib.sha256(doc_key.encode()).hexdigest(), 16) % 20 == 0


def build(docs: Iterable[tuple[str, str]], out: Path, random_frac: float, seed: int,
          max_lines: int | None = None) -> tuple[int, int]:
    rng = random.Random(seed)
    train, held = [], []
    for key, doc in docs:
        dest = held if is_held(key) else train
        for line in cut_lines(doc, rng):
            dest.append(line)
            if rng.random() < random_frac / (1 - random_frac):
                dest.append(random_line(rng))
        if max_lines and len(train) + len(held) >= max_lines:
            break
    held_set = set(held)
    train = [s for s in train if len(s) < LEAK_MIN_CHARS or s not in held_set]
    out.mkdir(parents=True, exist_ok=True)
    for name, lines in (("train.txt", train), ("held.txt", held)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return len(train), len(held)


def shard_documents(shard_dir: Path, tokenizer: str, max_shards: int | None = None,
                    seed: int = 0, doc_keep: float = 1.0, tok=None) -> Iterator[tuple[str, str]]:
    """Yield (key, text) for a random sample of documents. Shards come in a seeded random
    order (the first max_shards of it), each document is kept with probability doc_keep,
    and the key "shard_name:doc_index" names a document whether or not it is sampled."""
    from quipu.data import read_shard

    if tok is None:
        from quipu.tokenizer import make_tokenizer
        tok = make_tokenizer(tokenizer)
    rng = random.Random(seed)
    shards = sorted(shard_dir.glob("shard_*.bin"))
    rng.shuffle(shards)
    for shard in shards[:max_shards]:
        ids = read_shard(shard)
        cuts = [-1, *(int(j) for j in (ids == tok.eot).nonzero()[0]), len(ids)]
        index = 0
        for a, b in zip(cuts, cuts[1:]):
            if b - a <= 1:
                continue
            key = f"{shard.name}:{index}"
            index += 1
            if rng.random() < doc_keep:
                yield key, tok.decode(ids[a + 1:b].tolist())


def _fraction(text: str) -> float:
    v = float(text)
    if not 0.0 <= v < 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", default=str(ROOT / "data" / "shards-moe" / "train"))
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "text"))
    ap.add_argument("--lines", type=int, default=10_000_000)
    ap.add_argument("--random-frac", type=_fraction, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doc-keep", type=float, default=0.12,
                    help="probability of keeping each document, in (0, 1]")
    ap.add_argument("--max-shards", type=int, default=None, help="use only the first N shards")
    args = ap.parse_args(argv)
    if not 0.0 < args.doc_keep <= 1.0:
        ap.error("--doc-keep must be in (0, 1]")
    n_train, n_held = build(shard_documents(Path(args.shards), args.tokenizer, args.max_shards, args.seed,
                                  args.doc_keep), Path(args.out),
                            args.random_frac, args.seed, args.lines)
    print(f"train {n_train:,} lines, held {n_held:,} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
