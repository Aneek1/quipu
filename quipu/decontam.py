"""Benchmark decontamination: find documents that contain a HumanEval or MBPP problem,
so that training data (the shard build, and later the M12 SFT set) never teaches the
answers the code evaluations ask for.

A document is contaminated by a problem when either rule holds:

- substring: it contains one of the problem's texts (HumanEval: the prompt and the
  canonical solution; MBPP: the code and the task text), both sides with every run
  of whitespace collapsed to one space, and the text is at least MIN_NEEDLE_CHARS
  characters long after collapsing (shorter ones, such as `return a + b`, would
  match ordinary code);
- ngrams: for one of the problem's n-gram texts (its solution, HumanEval: the
  canonical_solution, MBPP: the code; and HumanEval's prompt and each docstring in
  it, so a copied docstring under a changed signature is caught), at least
  MIN_GRAM_FRACTION of that text's distinct NGRAM-grams, and at least MIN_GRAMS of
  them, occur in the document. A text with fewer than MIN_GRAMS n-grams takes no part
  in this rule (a solution of one 13-gram, such as HumanEval/13's gcd, would drop
  ordinary files on that one n-gram; its substring rule still holds). Tokens are \\w+
  runs and single punctuation characters (tokens(), not the BPE), so the rule
  ignores whitespace and formatting too.

find() reports the FIRST problem (in index order: HumanEval, then MBPP, each in file
order) that matches, so the answer never depends on how the check was sped up.
Internally each problem is one or more index entries (its needles with its solution's
n-grams, then one entry per further n-gram text), in problem order.

DECONTAM_VERSION is part of the fingerprint: bump it whenever a rule changes what is
dropped, so a build saved under the old rules is not resumed under the new.

Speed (the check runs on every code document, next to the BPE tokenizer): both rules
imply that certain strings occur in the document with all whitespace removed (its
"spaceless" form): the spaceless needle, or the spaceless form of each matched
n-gram (its tokens joined). If a pattern of at least 2 * BLOCK - 1 bytes occurs in a
text, every BLOCK-byte block of the text at an offset that is a multiple of BLOCK
and lies inside the occurrence is a block of the pattern at one fixed residue r
(0 <= r < BLOCK). So the index keeps, for every pattern and every residue, ONE block
(the one fewest benchmark problems share), and a document is cut into its aligned
BLOCK-byte blocks in one step (numpy's view of the UTF-8 bytes as uint64). A problem
is looked at only when those blocks show one of its needles, or evidence for at
least as many of its n-grams as the rule needs (patterns too short to index count
as evidence always); then its patterns are searched in the spaceless document (C
string search) and, only if enough are there, the document is tokenized and the rule
applied exactly. The prefilter changes the work, never the answer (find(...,
prefilter=False) checks every problem).

The Decontaminator pickles (it is sent to the shard builder's worker processes) and
has a fingerprint() of its problems and rules, recorded with every build.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

# 2: at least MIN_GRAMS n-grams; HumanEval prompt and docstring n-grams; train text.
DECONTAM_VERSION = 2
NGRAM = 13
MIN_NEEDLE_CHARS = 60
MIN_GRAM_FRACTION = 0.5
MIN_GRAMS = 2
BLOCK = 8  # bytes; one uint64

HUMANEVAL = "humaneval"
MBPP = "mbpp"
HUMANEVAL_DATASET = "openai/openai_humaneval"
MBPP_DATASET = "google-research-datasets/mbpp"
BENCHMARKS: dict[str, dict[str, Any]] = {
    HUMANEVAL: {"dataset": HUMANEVAL_DATASET, "config": "openai_humaneval", "license": "MIT",
                "files": ["openai_humaneval/test-00000-of-00001.parquet"]},
    MBPP: {"dataset": MBPP_DATASET, "config": "sanitized", "license": "CC-BY-4.0",
           "files": [f"sanitized/{s}-00000-of-00001.parquet"
                     for s in ("prompt", "test", "train", "validation")]},
}

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def collapse(text: str) -> str:
    """Every run of whitespace as one space, no leading or trailing space."""
    return " ".join(text.split())


def spaceless(text: str) -> str:
    """The text with all whitespace removed."""
    return "".join(text.split())


def tokens(text: str) -> list[str]:
    """\\w+ runs and single punctuation characters (whitespace dropped)."""
    return TOKEN_RE.findall(text)


def ngrams(toks: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return set()
    return set(zip(*(toks[i:] for i in range(n))))


SHORT_BLOCK = 4  # bytes; one uint32, for patterns shorter than 2 * BLOCK - 1 bytes
# Blocks made of these are common in any code, so a pattern's index block avoids them
# when it can (speed only: which block is indexed never changes an answer).
COMMON_CODE = tuple(w.encode() for w in (
    "return", "append", "range", "isinstance", "None", "True", "False", "else", "self",
    "import", "while", "print", "len(", "for", "if", "in", "def", "elif", "not", "and",
    "lambda", "split", "join", "items", "keys", "values", "sorted", "int(", "str(",
    "list(", "dict(", "set(", "max(", "min(", "sum(", "abs(", "):", "()", "[i]", "i+1",
    "==", "+=", "0", "1"))


def _blocks(data: bytes, start: int, size: int) -> list[bytes]:
    return [data[k:k + size] for k in range(start, len(data) - size + 1, size)]


def _key(block: bytes) -> int:
    return int.from_bytes(block, "little")  # as numpy's "<u8" / "<u4" view reads it


class _BlockIndex:
    """Sorted block keys and their postings; hits(data) looks up every block of
    `data` at the offsets that are multiples of the block size."""

    def __init__(self, size: int, postings: dict[int, list]) -> None:
        self.size = size
        self.dtype = "<u8" if size == 8 else "<u4"
        keys = sorted(postings)
        self.keys = np.array(keys, dtype=np.uint64 if size == 8 else np.uint32)
        self.postings = [tuple(postings[k]) for k in keys]

    def hits(self, data: bytes) -> list[tuple]:
        n = len(data) // self.size
        if not n or not len(self.keys):
            return []
        blocks = np.frombuffer(data, dtype=self.dtype, count=n)
        idx = np.searchsorted(self.keys, blocks)
        np.minimum(idx, len(self.keys) - 1, out=idx)
        found = idx[self.keys[idx] == blocks].tolist()
        return [self.postings[i] for i in dict.fromkeys(found)]


def _at_least(need: int, patterns: tuple[str, ...], text: str) -> bool:
    """True when at least `need` of `patterns` occur in `text` (stops early)."""
    found, left = 0, len(patterns)
    for p in patterns:
        left -= 1
        if p in text:
            found += 1
            if found >= need:
                return True
        elif found + left < need:
            return False
    return found >= need


class Problem(NamedTuple):
    benchmark: str
    task_id: str
    needles: tuple[str, ...]  # texts matched as whitespace-collapsed substrings
    solution: str             # the text whose n-grams are counted
    gram_texts: tuple[str, ...] = ()  # further texts whose n-grams are counted, each alone


class Decontaminator:
    """The decontamination index over a list of Problems (see the module docstring)."""

    def __init__(self, problems: Iterable[Problem], *, revisions: dict[str, str] | None = None,
                 ngram: int = NGRAM, min_chars: int = MIN_NEEDLE_CHARS,
                 fraction: float = MIN_GRAM_FRACTION, min_grams: int = MIN_GRAMS) -> None:
        self.problems = [Problem(p.benchmark, str(p.task_id), tuple(p.needles), p.solution,
                                 tuple(p.gram_texts)) for p in problems]
        self.revisions = dict(revisions or {})
        self.ngram, self.min_chars, self.fraction = ngram, min_chars, fraction
        self.min_grams = min_grams
        named = {p.benchmark for p in self.problems}
        self.benchmarks = ([b for b in BENCHMARKS if b in named]
                           + sorted(named - set(BENCHMARKS)))
        # Per entry (a problem's needles and solution, then each further n-gram text):
        # its problem; collapsed needles and their spaceless forms; the distinct
        # n-grams (sorted, for a stable index; none when fewer than min_grams) and
        # their spaceless forms; the count the n-gram rule needs.
        self._owner: list[int] = []
        self._needles: list[tuple[str, ...]] = []
        self._needles_sl: list[tuple[str, ...]] = []
        self._grams: list[frozenset] = []
        self._grams_sl: list[tuple[str, ...]] = []
        self._need: list[int] = []
        for pi, p in enumerate(self.problems):
            needles = tuple(dict.fromkeys(c for c in (collapse(t) for t in p.needles)
                                          if len(c) >= min_chars))
            for needed, text in [(needles, p.solution)] + [((), t) for t in p.gram_texts]:
                grams = sorted(ngrams(tokens(text), ngram))
                if len(grams) < min_grams:
                    grams = []
                if not needed and not grams:
                    continue
                self._owner.append(pi)
                self._needles.append(needed)
                self._needles_sl.append(tuple(spaceless(c) for c in needed))
                self._grams.append(frozenset(grams))
                self._grams_sl.append(tuple("".join(g) for g in grams))
                self._need.append(max(min_grams, 1, math.ceil(fraction * len(grams)))
                                  if grams else 0)
        self._build_index()

    def _build_index(self) -> None:
        """Two block indexes (BLOCK bytes, and SHORT_BLOCK bytes for the patterns too
        short for BLOCK), each block key -> ((kind, problem, pattern), ...), kind 0
        needle, 1 n-gram: one block per pattern and residue, preferring blocks free
        of COMMON_CODE and shared by the fewest entries. (Here i is an entry.)"""
        entries = range(len(self._owner))
        df: dict[bytes, int] = {}
        for i in entries:
            seen: set[bytes] = set()
            for sl in self._needles_sl[i] + self._grams_sl[i]:
                b = sl.encode("utf-8")
                for size in (BLOCK, SHORT_BLOCK):
                    seen.update(b[k:k + size] for k in range(len(b) - size + 1))
            for blk in seen:
                df[blk] = df.get(blk, 0) + 1

        @functools.lru_cache(maxsize=None)
        def rank(blk: bytes) -> tuple:
            return (sum(w in blk for w in COMMON_CODE), df[blk], blk)

        # size -> key -> problem -> ("needle" seen, n-gram ids)
        postings: dict[int, dict[int, dict[int, list]]] = {BLOCK: {}, SHORT_BLOCK: {}}
        self._always_needles: list[int] = []   # entries with an unindexable needle
        self._free_grams = [0] * len(entries)  # n-grams too short to index
        free_sl: list[list[str]] = [[] for _ in entries]
        for i in entries:
            patterns = [(0, j, sl) for j, sl in enumerate(self._needles_sl[i])] + \
                       [(1, j, sl) for j, sl in enumerate(self._grams_sl[i])]
            for kind, j, sl in patterns:
                b = sl.encode("utf-8")
                size = next((s for s in (BLOCK, SHORT_BLOCK) if len(b) >= 2 * s - 1), None)
                if size is None:
                    if kind == 0:
                        self._always_needles.append(i)
                    else:
                        self._free_grams[i] += 1
                        free_sl[i].append(sl)
                    continue
                for r in range(size):
                    best = min(_blocks(b, r, size), key=rank)
                    entry = postings[size].setdefault(_key(best), {}).setdefault(
                        i, [False, set()])
                    if kind == 0:
                        entry[0] = True
                    else:
                        entry[1].add(j)
        # Each posting: ((entry, has a needle, n-gram ids), ...) in entry order.
        self._index = [_BlockIndex(size, {k: [(i, e[0], tuple(sorted(e[1])))
                                              for i, e in sorted(v.items())]
                                          for k, v in postings[size].items()})
                       for size in (BLOCK, SHORT_BLOCK) if postings[size]]
        self._free_sl = [tuple(x) for x in free_sl]
        self._always_needles = sorted(set(self._always_needles))
        # Entries the n-gram rule may hold for whatever the blocks show.
        self._always_grams = [i for i, n in enumerate(self._free_grams)
                              if self._grams[i] and n >= self._need[i]]

    # -------------------------------------------------------------- matching

    def find(self, text: str, *, prefilter: bool = True) -> tuple[str, str, str] | None:
        """(benchmark, task_id, "substring" | "ngrams") of the first problem `text`
        contains, or None."""
        if not text:
            return None
        sl = spaceless(text)
        if prefilter:
            data = sl.encode("utf-8", "surrogatepass")
            needle_c: set[int] = set(self._always_needles)
            evidence: dict[int, set[int]] = {}
            for index in self._index:
                for posting in index.hits(data):
                    for i, needle, js in posting:
                        if needle:
                            needle_c.add(i)
                        if js:
                            ev = evidence.get(i)
                            if ev is None:
                                evidence[i] = set(js)
                            else:
                                ev.update(js)
            gram_c = {i for i, ev in evidence.items()
                      if len(ev) + self._free_grams[i] >= self._need[i]}
            gram_c.update(self._always_grams)
            order = sorted(needle_c | gram_c)
            if not order:
                return None
        else:
            order = range(len(self._owner))
            needle_c = gram_c = None
        flat: str | None = None
        doc_grams: set | None = None
        for i in order:
            if needle_c is None or i in needle_c:
                for c_sl, c in zip(self._needles_sl[i], self._needles[i]):
                    if c_sl in sl:
                        if flat is None:
                            flat = collapse(text)
                        if c in flat:
                            p = self.problems[self._owner[i]]
                            return p.benchmark, p.task_id, "substring"
            if self._grams[i] and (gram_c is None or i in gram_c):
                need = self._need[i]
                if gram_c is None:
                    maybe = self._grams_sl[i]
                else:  # an indexed n-gram without evidence is not in the document
                    grams_sl = self._grams_sl[i]
                    maybe = tuple(grams_sl[j] for j in sorted(evidence.get(i, ()))) + \
                        self._free_sl[i]
                if not _at_least(need, maybe, sl):
                    continue
                if doc_grams is None:
                    doc_grams = ngrams(tokens(text), self.ngram)
                if len(self._grams[i] & doc_grams) >= need:
                    p = self.problems[self._owner[i]]
                    return p.benchmark, p.task_id, "ngrams"
        return None

    # -------------------------------------------------------------- provenance

    def fingerprint(self) -> str:
        """sha256 of the rule version, the problems, the benchmark revisions and the
        rules."""
        blob = json.dumps({"version": DECONTAM_VERSION,
                           "rules": {"ngram": self.ngram, "min_chars": self.min_chars,
                                     "fraction": self.fraction, "min_grams": self.min_grams},
                           "revisions": self.revisions,
                           "problems": [list(p) for p in self.problems]},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        counts = {b: sum(p.benchmark == b for p in self.problems) for b in self.benchmarks}
        checkable = set(self._owner)  # an entry exists only with a needle or n-grams
        uncheckable = [p.task_id for i, p in enumerate(self.problems) if i not in checkable]
        return {
            "version": DECONTAM_VERSION,
            "problems": counts,
            "revisions": dict(self.revisions),
            "index_sha256": self.fingerprint(),
            "rule": (f"a document is dropped when it contains a problem's text (HumanEval: "
                     f"prompt, canonical_solution; MBPP: code, task text) as a "
                     f"whitespace-collapsed substring of at least {self.min_chars} "
                     f"characters, or, for one of the problem's n-gram texts (its solution; "
                     f"HumanEval also its prompt and each docstring in it), at least "
                     f"{self.fraction:.0%} and at least {self.min_grams} of that text's "
                     f"distinct {self.ngram}-grams (tokens: \\w+ runs and single "
                     "punctuation characters; a text with fewer than "
                     f"{self.min_grams} {self.ngram}-grams takes no part)"),
            "ngram": self.ngram, "min_chars": self.min_chars, "fraction": self.fraction,
            "min_grams": self.min_grams,
            # Every text of these is under min_chars and every n-gram text under
            # min_grams n-grams: neither rule can see them (too short to be told from
            # ordinary code).
            "too_short_to_check": uncheckable,
        }


# ------------------------------------------------------------------ loading

DOCSTRING_RE = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)


def docstrings(source: str) -> list[str]:
    """The triple-quoted strings in `source`, in order (their contents)."""
    return [a or b for a, b in DOCSTRING_RE.findall(source) if (a or b).strip()]


def humaneval_problems(rows: Iterable[dict]) -> list[Problem]:
    """Needles: prompt and canonical_solution. N-gram texts: the solution, the prompt
    and each docstring in the prompt (a repo that copies the docstring under its own
    signature is caught by the docstring's n-grams)."""
    return [Problem(HUMANEVAL, r["task_id"], (r["prompt"], r["canonical_solution"]),
                    r["canonical_solution"],
                    tuple(dict.fromkeys([r["prompt"], *docstrings(r["prompt"])])))
            for r in rows]


def mbpp_problems(rows: Iterable[dict]) -> list[Problem]:
    return [Problem(MBPP, str(r["task_id"]), (r["code"], r["prompt"]), r["code"])
            for r in rows]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmarks(revisions: dict[str, str],
                    download: Callable[..., str] | None = None
                    ) -> tuple[Decontaminator, dict[str, dict[str, Any]]]:
    """HumanEval (test split) and MBPP (sanitized config, every split) at the pinned
    commits, as a Decontaminator and each benchmark's provenance (dataset, revision,
    files and their sha256, licence, problems). `download` is hf_hub_download's
    signature (the default); the files are small (~0.2 MB in all)."""
    import pyarrow.parquet as pq

    if download is None:
        from huggingface_hub import hf_hub_download as download
    readers = {HUMANEVAL: humaneval_problems, MBPP: mbpp_problems}
    problems: list[Problem] = []
    info: dict[str, dict[str, Any]] = {}
    for name, b in BENCHMARKS.items():
        rev = revisions.get(name) or ""
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise ValueError(f"the {name} revision must be pinned to a 40-hex commit of "
                             f"{b['dataset']}; got {rev!r}")
        rows: list[dict] = []
        shas = {}
        for f in b["files"]:
            path = download(b["dataset"], f, repo_type="dataset", revision=rev)
            shas[f] = _sha256(path)
            rows += pq.read_table(path).to_pylist()
        got = readers[name](rows)
        problems += got
        info[name] = {"dataset": b["dataset"], "config": b["config"], "revision": rev,
                      "files": list(b["files"]), "sha256": shas, "license": b["license"],
                      "problems": len(got)}
    return Decontaminator(problems, revisions={k: info[k]["revision"] for k in info}), info
