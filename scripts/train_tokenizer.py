"""Sample the quipu-moe data mix and train its byte-level BPE tokenizer (spec 4 and 11).

The sample is the section 11 mix in miniature, by bytes:
  - code_share (60%) from codeparrot/github-code-clean (permissive licences, the
    per-language weights below, the shard builder's minified/vendored filters);
  - english_share (28%) from FineWeb-Edu sample-10BT;
  - the rest (12%) from FineWeb-2, the nine other languages in about equal shares.
    Languages are read one after another; one that runs out of documents leaves
    its shortfall to the languages after it (the last one absorbs the rest). No
    language-ID filter: the tokenizer only needs text in roughly the right script.
It is streamed straight into the trainer, a batch of documents at a time; nothing
but the trainer's own word counts is held in memory.

Held out for scripts/tokenizer_gate.py, by construction:
  - English: the LAST parquet file of sample-10BT (training reads only the others);
  - other languages: FineWeb-2's test split (training reads only train);
  - code: files code_heldout_first_file.. (840..879, the shard builder's code val
    files); training reads files from 0 upward and never reaches them.
The content hash of every code document in the sample is also written, so the gate
can drop exact duplicates that happen to live in a held-out file.

Language weights: each language gets weight * code budget bytes. A language that
runs short (see QuotaSampler: it would not fill within max_windows windows of about
one parquet file each) is marked exhausted and its remainder redistributed to the
other languages in proportion to their weights, never to HTML (a hard cap). For a
tokenizer sample a few percent off the target mix is harmless; the achieved shares
and the exhausted languages are written to the manifest.

Outputs, beside --out: tokenizer.json, sample_manifest.json, sample_code_hashes.npy.

Run: uv run python scripts/train_tokenizer.py --out artifacts/tokenizer/tokenizer.json \
         --sample-bytes 1000000000 --vocab 49152
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_shards as bs  # noqa: E402  (the shard builder's streaming and filters)

from quipu.fsio import write_text_atomic  # noqa: E402

TEXT_DATASET = "HuggingFaceFW/fineweb-edu"
TEXT_DIR = "sample/10BT"
CODE_DATASET = "codeparrot/github-code-clean"
CODE_FILES_TOTAL = 880
CODE_HELDOUT_FIRST_FILE = 840
CODE_LICENSES = ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0",
                 "unlicense")
CODE_SHARE = 0.6
ENGLISH_SHARE = 0.28
FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
# Spec section 11: the nine non-English languages of the owner's LID model, as
# FineWeb-2 subsets. cmn_Hani is sampled as-is here (the zh-Hans/zh-Hant split is a
# shard-building step).
FINEWEB2_LANGUAGES = ("ind_Latn", "zsm_Latn", "cmn_Hani", "jpn_Jpan", "kor_Hang",
                      "tam_Taml", "hin_Deva", "hin_Latn", "urd_Latn")
ENGLISH = "eng_Latn"
# Spec section 5. "All others ~15%" is spelled out per language so the weights have
# the same shape as DataConfig.code_language_weights (Task M2). Names are exactly as
# they appear in github-code-clean (note GO).
LANGUAGE_WEIGHTS = {
    "Python": 0.30, "JavaScript": 0.25, "TypeScript": 0.12, "HTML": 0.08, "CSS": 0.05,
    "SQL": 0.05,
    "PHP": 0.03, "Java": 0.03, "GO": 0.03, "Shell": 0.02, "Dockerfile": 0.01,
    "C": 0.01, "C++": 0.01, "Rust": 0.01,
}
CAPPED_LANGUAGES = ("HTML",)
# The shard builder skips code files over 16,000 tokens; at ~4 bytes per GPT-2 token
# of code that is about this many bytes.
MAX_CODE_DOC_BYTES = 64_000
# About one parquet file's worth of rows that pass the licence/path/length filters.
WINDOW_OFFERS = 40_000
MAX_WINDOWS = 6          # no language is scanned for more than ~6 files' worth
MIN_WINDOW_GAIN = 0.01   # under 1% of its quota in a window: gone from the stream
MAX_CODE_FILES = 60
BATCH_DOCS = 500


class QuotaSampler:
    """Admit documents by language until each language's quota of `budget` is met.

    Units are whatever the caller counts (bytes for training, documents for the
    gate). A document is admitted while its language is under quota, so a language
    overshoots by at most one document. `done` is true once the admitted total
    reaches the budget.

    Short languages. Offers are counted in windows of `window` offers (about one
    parquet file of eligible rows at the default). At the end of each window, an
    unfilled language is marked exhausted when its gain in that window was under
    min_gain of its original quota (it has left the stream), or when, at that rate,
    it would not be filled by the end of window max_windows (it is too rare to be
    worth the download). Its quota is frozen at what it has taken, and the remainder
    is added to the quotas of the languages that are neither exhausted nor capped,
    in proportion to their weights. The price is a few percent off the target mix,
    reported per language in summary().
    """

    def __init__(self, weights: dict[str, float], budget: float, *,
                 capped: Iterable[str] = CAPPED_LANGUAGES, window: int = WINDOW_OFFERS,
                 max_windows: int = MAX_WINDOWS,
                 min_gain: float = MIN_WINDOW_GAIN) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if not weights or any(w <= 0 for w in weights.values()):
            raise ValueError("language weights must be positive")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"language weights sum to {sum(weights.values())}, not 1")
        self.weights = dict(weights)
        self.budget = budget
        self.capped = frozenset(capped)
        if window <= 0 or max_windows <= 0 or not 0 <= min_gain < 1:
            raise ValueError("window and max_windows must be positive, min_gain in [0, 1)")
        self.window = window
        self.max_windows = max_windows
        self.min_gain = min_gain
        self.windows = 0
        self.quota = {lang: w * budget for lang, w in weights.items()}
        self.taken: dict[str, float] = {lang: 0 for lang in weights}
        self.documents: Counter[str] = Counter()
        self.offers = 0
        self._window_start = dict(self.taken)
        self.exhausted: list[str] = []
        self.redistributions = 0

    @property
    def total(self) -> float:
        return sum(self.taken.values())

    @property
    def done(self) -> bool:
        """The budget is met, or no language may take more (all filled or exhausted)."""
        return self.total >= self.budget or all(
            self.taken[lang] >= self.quota[lang] for lang in self.quota)

    def offer(self, language: str, n: float) -> bool:
        if language not in self.quota:
            return False
        self.offers += 1
        admitted = self.taken[language] < self.quota[language]
        if admitted:
            self.taken[language] += n
            self.documents[language] += 1
        if self.offers % self.window == 0:
            self._end_window()
        return admitted

    def _end_window(self) -> None:
        self.windows += 1
        windows_left = self.max_windows - self.windows
        slow = []
        for lang in self.quota:
            remaining = self.quota[lang] - self.taken[lang]
            if lang in self.exhausted or remaining <= 0:
                continue
            gain = self.taken[lang] - self._window_start[lang]
            if (gain < self.min_gain * self.weights[lang] * self.budget
                    or gain * windows_left < remaining):
                slow.append(lang)
        self._window_start = dict(self.taken)
        for lang in slow:
            self._exhaust(lang)

    def _exhaust(self, lang: str) -> None:
        remainder = self.quota[lang] - self.taken[lang]
        self.quota[lang] = self.taken[lang]
        self.exhausted.append(lang)
        takers = [t for t in self.quota if t not in self.exhausted and t not in self.capped]
        if remainder <= 0 or not takers:
            return
        w = sum(self.weights[t] for t in takers)
        for t in takers:
            self.quota[t] += remainder * self.weights[t] / w
        self.redistributions += 1

    def summary(self) -> dict[str, Any]:
        total = self.total
        return {
            "budget": self.budget,
            "taken": total,
            "redistributions": self.redistributions,
            "exhausted": list(self.exhausted),
            "rule": (f"window {self.window} offers; exhausted below {self.min_gain:.0%} of "
                     f"quota per window or not filled within {self.max_windows} windows"),
            "windows": self.windows,
            "by_language": {lang: {"target_share": self.weights[lang],
                                   "achieved_share": round(self.taken[lang] / total, 6)
                                   if total else 0.0,
                                   "units": self.taken[lang],
                                   "documents": self.documents[lang]}
                            for lang in self.weights},
        }


def code_documents(rows: Iterable[dict], sampler: QuotaSampler, *,
                   size: Callable[[str], float], licenses: Iterable[str] = CODE_LICENSES,
                   max_doc_bytes: int = MAX_CODE_DOC_BYTES,
                   exclude: frozenset[int] | set[int] = frozenset(),
                   stats: Counter | None = None,
                   hashes: list[int] | None = None) -> Iterator[tuple[str, str]]:
    """(language, code) for rows that pass the shard builder's filters and the sampler.

    Filters in order: licence, blank, minified/vendored path, longer than
    max_doc_bytes, content hash in `exclude`; then the language quota. Stops as
    soon as the sampler is done.
    """
    licenses = frozenset(licenses)
    stats = stats if stats is not None else Counter()
    for row in rows:
        if sampler.done:
            return
        stats["rows_scanned"] += 1
        language = row.get("language")
        if language not in sampler.quota:
            stats["dropped_language"] += 1
            continue
        if row.get("license") not in licenses:
            stats["dropped_license"] += 1
            continue
        code = row.get("code") or ""
        if not code.strip():
            stats["skipped_blank"] += 1
            continue
        reason = bs.path_skip_reason(row.get("path"))
        nbytes = len(code.encode("utf-8", "surrogatepass"))
        if reason is None and nbytes > max_doc_bytes:
            reason = "too_long"
        if reason is not None:
            stats[f"skipped_{reason}"] += 1
            continue
        h = bs.content_hash(code)
        if h in exclude:
            stats["dropped_as_duplicate"] += 1
            continue
        if not sampler.offer(language, size(code)):
            stats["skipped_quota"] += 1
            continue
        if hashes is not None:
            hashes.append(h)
        if "file" in row:
            stats["last_file"] = row["file"]
        yield language, code


def text_documents(rows: Iterable[dict], budget: float, *, size: Callable[[str], float],
                   stats: Counter | None = None) -> Iterator[str]:
    """Non-blank FineWeb-Edu texts until `budget` units have been taken."""
    stats = stats if stats is not None else Counter()
    taken = 0.0
    for row in rows:
        if taken >= budget:
            return
        stats["rows_scanned"] += 1
        text = row.get("text") or ""
        if not text.strip():
            stats["skipped_blank"] += 1
            continue
        n = size(text)
        taken += n
        stats["documents"] += 1
        stats["units"] += n
        yield text


def multilingual_documents(streams: dict[str, Callable[[], Iterable[dict]]], budget: float, *,
                           size: Callable[[str], float],
                           summary: dict[str, dict] | None = None) -> Iterator[tuple[str, str]]:
    """(language, text) from each language's stream in turn, about budget/len each.

    Each language is opened only when its turn comes (streams maps it to a
    zero-argument callable) and gets an equal share of what the budget still lacks,
    so a language that runs dry passes its shortfall on to the languages after it.
    `summary` receives, per language, the target, units taken and documents.
    """
    summary = summary if summary is not None else {}
    langs = list(streams)
    left = float(budget)
    for i, lang in enumerate(langs):
        target = left / (len(langs) - i)
        stats: Counter = Counter()
        for text in text_documents(streams[lang](), target, size=size, stats=stats):
            yield lang, text
        summary[lang] = {"target": target, "units": stats["units"],
                         "documents": stats["documents"], "short": stats["units"] < target}
        left -= stats["units"]


def iter_parquet_text(fs: Any, paths: Iterable[str], retries: int = 5) -> Iterator[dict]:
    """{"text", "file"} rows of each parquet file's "text" column, one row group at a time.

    Like build_shards.iter_code_row_groups: a failed read is retried from the same
    (file, row group), so a network blip changes nothing in the output.
    """
    import pyarrow.parquet as pq

    for path in paths:
        rg, n_groups, attempt = 0, None, 0
        while n_groups is None or rg < n_groups:
            try:
                with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f:
                    pf = pq.ParquetFile(f)
                    n_groups = pf.metadata.num_row_groups
                    while rg < n_groups:
                        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                        rg += 1
                        attempt = 0
                        for t in texts:
                            yield {"text": t, "file": path}
            except Exception as exc:
                attempt += 1
                if attempt > retries:
                    raise
                wait = 5 * 2 ** (attempt - 1)
                print(f"\nread failed ({path}, row group {rg}): {exc!r}; "
                      f"retry {attempt}/{retries} in {wait}s", file=sys.stderr, flush=True)
                time.sleep(wait)


def fineweb2_files(fs: Any, revision: str, lang: str, split: str) -> list[str]:
    """Full HfFileSystem paths of one FineWeb-2 language's split, sorted."""
    listed = fs.ls(f"datasets/{FINEWEB2_DATASET}@{revision}/data/{lang}/{split}", detail=False)
    paths = sorted(p for p in listed if p.endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"{FINEWEB2_DATASET} {lang}/{split} has no parquet files")
    return paths


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8", "surrogatepass"))


def batches_with_digest(docs: Iterable[str], digest: Any, size: int = BATCH_DOCS,
                        progress: Callable[[int], None] | None = None) -> Iterator[list[str]]:
    """Batches of documents for the trainer; every document also feeds `digest`."""
    batch: list[str] = []
    for doc in docs:
        data = doc.encode("utf-8", "surrogatepass")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
        if progress is not None:
            progress(len(data))
        batch.append(doc)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------------ real sources

def text_files(fs: Any, revision: str) -> list[str]:
    """sample-10BT's parquet files (repo-relative), sorted."""
    listed = fs.ls(f"datasets/{TEXT_DATASET}@{revision}/{TEXT_DIR}", detail=False)
    names = sorted(p.rsplit("/", 1)[-1] for p in listed if p.endswith(".parquet"))
    if len(names) < 2:
        raise RuntimeError(f"expected several parquet files under {TEXT_DIR}; got {names}")
    return [f"{TEXT_DIR}/{n}" for n in names]


def text_rows(revision: str, files: list[str]) -> Iterator[dict]:
    from datasets import load_dataset

    stream = load_dataset(TEXT_DATASET, data_files={"train": files}, revision=revision,
                          split="train", streaming=True).select_columns(["text"])
    return bs.flatten(bs.prefetch(bs.batched(stream, 1000), depth=8))


def code_rows(fs: Any, revision: str, files: range) -> Iterator[dict]:
    return bs.flatten(bs.prefetch(bs.iter_code_row_groups(fs, CODE_DATASET, revision, files,
                                                          CODE_FILES_TOTAL), depth=2))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--sample-bytes", type=float, default=1_000_000_000)
    parser.add_argument("--vocab", type=int, default=49152)
    parser.add_argument("--code-share", type=float, default=CODE_SHARE)
    parser.add_argument("--english-share", type=float, default=ENGLISH_SHARE)
    parser.add_argument("--max-code-files", type=int, default=MAX_CODE_FILES)
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS,
                        help="code sampler: a language not filled by this window is "
                             "exhausted (about one code file per window)")
    parser.add_argument("--min-window-gain", type=float, default=MIN_WINDOW_GAIN,
                        help="code sampler: a language gaining under this fraction of its "
                             "quota in a window is exhausted")
    parser.add_argument("--low-priority", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4,
                        help="RAYON_NUM_THREADS for the trainer (default 4; see below)")
    args = parser.parse_args()
    # The tokenizers trainer pre-tokenises and counts words on a rayon pool, and each
    # thread keeps its own partial word counts until they are merged, so peak memory
    # grows with the thread count. 4 threads bounded the laptop run; the pool reads
    # this once, when it is first used, so it must be set before training starts.
    # An explicit RAYON_NUM_THREADS in the environment wins.
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    print(f"RAYON_NUM_THREADS={os.environ['RAYON_NUM_THREADS']}", flush=True)
    if args.low_priority:
        print(f"below-normal priority set: {bs.lower_priority()}", flush=True)
    other_share = 1 - args.code_share - args.english_share
    if not (0 < args.code_share < 1 and 0 < args.english_share < 1 and other_share > 0):
        raise SystemExit("--code-share and --english-share must be in (0, 1) and sum below 1")
    max_files = min(args.max_code_files, CODE_HELDOUT_FIRST_FILE)

    from huggingface_hub import HfApi, HfFileSystem
    from tqdm import tqdm

    from quipu.bpe import PRE_TOKENIZER_RULES, SPECIAL_TOKENS, train_bpe

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    api, fs = HfApi(), HfFileSystem()
    text_rev = api.dataset_info(TEXT_DATASET).sha
    code_rev = api.dataset_info(CODE_DATASET).sha
    fw2_rev = api.dataset_info(FINEWEB2_DATASET).sha
    all_text = text_files(fs, text_rev)
    train_text, heldout_text = all_text[:-1], all_text[-1:]
    fw2_train = {lang: fineweb2_files(fs, fw2_rev, lang, "train") for lang in FINEWEB2_LANGUAGES}
    fw2_test = {lang: fineweb2_files(fs, fw2_rev, lang, "test") for lang in FINEWEB2_LANGUAGES}

    code_budget = args.sample_bytes * args.code_share
    sampler = QuotaSampler(LANGUAGE_WEIGHTS, code_budget, max_windows=args.max_windows,
                           min_gain=args.min_window_gain)
    code_stats: Counter = Counter()
    text_stats: Counter = Counter()
    fw2_summary: dict[str, dict] = {}
    hashes: list[int] = []

    def docs() -> Iterator[str]:
        for _, code in code_documents(code_rows(fs, code_rev, range(max_files)), sampler,
                                      size=utf8_len, stats=code_stats, hashes=hashes):
            yield code
        if not sampler.done:
            raise RuntimeError(f"code ran short: {sampler.total:,.0f} of {code_budget:,.0f} "
                               f"bytes after {max_files} files; raise --max-code-files")
        print(f"\ncode done: {sampler.total:,.0f} bytes, {sum(sampler.documents.values()):,} "
              f"documents, last file {code_stats.get('last_file')}", flush=True)
        # If every language filled or was exhausted short of the code budget, text is
        # scaled down with it so the sample keeps the mix.
        scale = min(1.0, sampler.total / code_budget)
        yield from text_documents(text_rows(text_rev, train_text),
                                  args.sample_bytes * args.english_share * scale,
                                  size=utf8_len, stats=text_stats)
        print(f"\nenglish done: {text_stats['units']:,} bytes", flush=True)
        streams = {lang: (lambda p=fw2_train[lang]: iter_parquet_text(fs, p))
                   for lang in FINEWEB2_LANGUAGES}
        for _, text in multilingual_documents(streams, args.sample_bytes * other_share * scale,
                                              size=utf8_len, summary=fw2_summary):
            yield text

    digest = hashlib.sha256()
    bar = tqdm(total=int(args.sample_bytes), unit="B", unit_scale=True, desc="sample",
               mininterval=5.0)
    t0 = time.monotonic()
    tok = train_bpe(batches_with_digest(docs(), digest, progress=bar.update), args.vocab, out,
                    show_progress=True)
    bar.close()
    seconds = round(time.monotonic() - t0, 1)

    hashes_path = out.with_name("sample_code_hashes.npy")
    np.save(hashes_path, np.asarray(sorted(set(hashes)), dtype=np.uint64))
    code_summary = sampler.summary()
    fw2_bytes = sum(v["units"] for v in fw2_summary.values())
    total = code_summary["taken"] + text_stats["units"] + fw2_bytes
    manifest = {
        "tokenizer": {"path": out.name, "sha256": sha256_file(out),
                      "vocab_size": tok.vocab_size, "special_tokens": list(SPECIAL_TOKENS),
                      "pre_tokenizer_rules": list(PRE_TOKENIZER_RULES)},
        "sample": {"bytes_target": int(args.sample_bytes),
                   "bytes": int(total),
                   "sha256": digest.hexdigest(),
                   "sha256_of": "for each document in order: 8-byte LE length + UTF-8 bytes",
                   "target_shares": {"code": args.code_share, ENGLISH: args.english_share,
                                     **{lang: round(other_share / len(FINEWEB2_LANGUAGES), 6)
                                        for lang in FINEWEB2_LANGUAGES}},
                   "achieved_shares": {
                       "code": round(code_summary["taken"] / total, 6),
                       ENGLISH: round(text_stats["units"] / total, 6),
                       **{lang: round(v["units"] / total, 6)
                          for lang, v in fw2_summary.items()}}},
        "code": {"dataset": CODE_DATASET, "revision": code_rev, "licenses": list(CODE_LICENSES),
                 "files_read": [0, code_stats.get("last_file")],
                 "heldout_files": [CODE_HELDOUT_FIRST_FILE, CODE_FILES_TOTAL - 1],
                 "max_doc_bytes": MAX_CODE_DOC_BYTES, "path_skip_rules": bs.CODE_PATH_SKIP_RULES,
                 "bytes": int(code_summary["taken"]),
                 "stats": {k: v for k, v in code_stats.items() if k != "last_file"},
                 "languages": code_summary,
                 "content_hashes": hashes_path.name},
        "text": {"dataset": TEXT_DATASET, "revision": text_rev, "train_files": train_text,
                 "heldout_files": heldout_text, "bytes": int(text_stats["units"]),
                 "documents": text_stats["documents"],
                 "rows_scanned": text_stats["rows_scanned"]},
        "fineweb2": {"dataset": FINEWEB2_DATASET, "revision": fw2_rev,
                     "languages": list(FINEWEB2_LANGUAGES),
                     "train_files": fw2_train, "heldout_files": fw2_test,
                     "by_language": fw2_summary, "bytes": int(fw2_bytes),
                     "lid_filter": None},
        "train_seconds": seconds,
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
    }
    write_text_atomic(out.with_name("sample_manifest.json"), json.dumps(manifest, indent=2))
    print(json.dumps({"tokenizer_sha256": manifest["tokenizer"]["sha256"],
                      "vocab": tok.vocab_size, "sample_bytes": manifest["sample"]["bytes"],
                      "seconds": seconds}), flush=True)


if __name__ == "__main__":
    main()
