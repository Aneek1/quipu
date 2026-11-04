"""Cut OCR text lines from Quipu's token shards (spec section 4.1).

    .venv/Scripts/python scripts/ocr/build_text.py --shards data/shards-moe/train \
        --out data/ocr/text --lines 10000000

Documents are the EOT-separated runs of a shard, decoded with Quipu's tokenizer. Each
is cut into lines of 5-60 characters at word boundaries, keeping only characters in
the OCR set. About 15% of lines are random strings (codes, digits, punctuation) so a
model cannot read by guessing language. Every 20th document (by index hash) goes to
held.txt, the rest to train.txt; a line that lands in both is removed from train."""
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
    n = rng.randint(1, max_chars)
    s = "".join(rng.choice(alphabet) for _ in range(n)).strip()
    return s or rng.choice(string.digits)


def is_held(doc_index: int) -> bool:
    return int(hashlib.sha256(str(doc_index).encode()).hexdigest(), 16) % 20 == 0


def build(docs: Iterable[str], out: Path, random_frac: float, seed: int,
          max_lines: int | None = None) -> tuple[int, int]:
    rng = random.Random(seed)
    train, held = [], []
    for i, doc in enumerate(docs):
        dest = held if is_held(i) else train
        for line in cut_lines(doc, rng):
            dest.append(line)
            if rng.random() < random_frac / (1 - random_frac):
                dest.append(random_line(rng))
        if max_lines and len(train) + len(held) >= max_lines:
            break
    held_set = set(held)
    train = [s for s in train if s not in held_set]
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "held.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    return len(train), len(held)


def shard_documents(shard_dir: Path, tokenizer: str,
                    max_shards: int | None = None) -> Iterator[str]:
    from quipu.data import read_shard
    from quipu.tokenizer import make_tokenizer

    tok = make_tokenizer(tokenizer)
    for shard in sorted(shard_dir.glob("shard_*.bin"))[:max_shards]:
        ids = read_shard(shard)
        cuts = [-1, *(int(j) for j in (ids == tok.eot).nonzero()[0]), len(ids)]
        for a, b in zip(cuts, cuts[1:]):
            if b - a > 1:
                yield tok.decode(ids[a + 1:b].tolist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", default=str(ROOT / "data" / "shards-moe" / "train"))
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts" / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr" / "text"))
    ap.add_argument("--lines", type=int, default=10_000_000)
    ap.add_argument("--random-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=None, help="use only the first N shards")
    args = ap.parse_args(argv)
    n_train, n_held = build(shard_documents(Path(args.shards), args.tokenizer, args.max_shards), Path(args.out),
                            args.random_frac, args.seed, args.lines)
    print(f"train {n_train:,} lines, held {n_held:,} lines -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
