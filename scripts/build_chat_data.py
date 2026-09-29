"""Build the chat fine-tune's data (spec section 13, plan Task M12).

    python -m uv run python scripts/build_chat_data.py --config configs/quipu-moe-sft.toml
        [--out DIR] [--limit N] [--sources aya,oasst2,stepbuild] [--no-pack]

Sources (human-written, permissive; the revisions are pinned in the config's [data]):
- aya: CohereForAI/aya_dataset (Apache-2.0), train split, one user turn (inputs) and
  one assistant turn (targets) per row, kept when its language is one of spec 11's
  ten (AYA_LANGUAGES below; the mapping is also written to the manifest);
- oasst2: OpenAssistant/oasst2 (Apache-2.0), train and validation files, one
  conversation per message tree along its top-ranked path (oasst2_paths): from the
  root prompt, the rank-0 reply at every assistant turn (an only reply, unranked,
  counts as top), and at every user turn the best-ranked follow-up (rank, then
  creation time). Deleted, negatively reviewed and synthetic messages are left out;
  the tree's language is its root's (OASST2_LANGUAGES), and a path stops at the
  first message in another language;
- stepbuild: the step-builder dataset's train split (--stepbuild-dir, default
  data.stepbuild_dir): system + user + an assistant reply of FILE blocks. A row is
  dropped when the benchmark's LeakageGuard finds a bench app's reference in its
  reply, or when its reply copies a test-split reply (whitespace ignored) or has a
  FILE block that is a near-copy of a test-split block for the same path (the same
  5-shingle Jaccard >= 0.9 the LeakageGuard uses): HeldOutSplitGuard. Its prompt
  (most are several thousand tokens: context files and the project tree) is then
  fitted to the context beside the whole reply, as the benchmark harness fits its
  prompts (fit_stepbuild_prompt): tree paths go first, then whole context files
  (never part of one; the files the reply rewrites last); a row is dropped only
  when its reply alone leaves no room ("reply_too_long") or not even the STEP line
  fits beside it ("prompt_too_long"). The manifest counts the cut prompts under
  "prompts_fitted".

Every example, whatever its source, then goes through, in order (each drop counted
per source in the manifest):
- the chat format (quipu.chat.validate): content holding a chat special string
  (<|user|> ...) is dropped ("special_tokens"), as are empty turns ("empty");
  content is encoded as plain text, so the FILE markers stay the plain-text tokens
  pretraining taught (quipu.chat);
- decontamination (quipu.decontam, the pretraining shards' rules): the whole
  conversation's text containing a HumanEval or MBPP problem ("decontam_humaneval",
  "decontam_mbpp");
- exact duplicates of an earlier example's rendered text ("duplicate");
- the context: a conversation longer than model.context tokens is cut at a turn
  boundary after its last assistant turn that fits ("truncated", kept) or dropped
  when not even its first assistant turn fits ("too_long").

A kept conversation goes to validation when a hash of its source and id falls below
--val-fraction, else to training. Training conversations are shuffled (--seed) and
packed greedily into blocks of exactly model.context tokens, each conversation
followed by <|endoftext|> when there is room and a block padded with <|endoftext|>;
--no-pack puts one conversation per block. Blocks go to <out>/train/shard_NNN.bin
(uint16) with the loss mask beside each (shard_NNN.mask, uint8, 1 on assistant
content and its <|end|>: quipu.chat), read by quipu.loader.MaskedBlockStream;
validation likewise to <out>/val/ and, per source, <out>/val_by_source/<source>/
(held-out chat loss per source). <out>/manifest.json records the tokenizer, the
context, every source's dataset, revision, licence, rows read, drops per filter,
conversations and tokens (all and assistant) per language, the decontamination
index and the splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quipu import chat  # noqa: E402
from quipu.data import write_mask, write_shard  # noqa: E402
from quipu.fsio import write_text_atomic  # noqa: E402

MANIFEST_VERSION = 1

AYA = "aya"
OASST2 = "oasst2"
STEPBUILD = "stepbuild"
SOURCES = (AYA, OASST2, STEPBUILD)

AYA_DATASET = "CohereForAI/aya_dataset"
AYA_FILES = ("data/train-00000-of-00001.parquet",)
AYA_COLUMNS = ["inputs", "targets", "language", "language_code"]
OASST2_DATASET = "OpenAssistant/oasst2"
OASST2_COLUMNS = ["message_id", "parent_id", "message_tree_id", "text", "role", "lang",
                  "deleted", "review_result", "rank", "synthetic", "created_date"]
LICENSES = {AYA: "Apache-2.0", OASST2: "Apache-2.0",
            STEPBUILD: "per repository (permissive; the row's licence field)"}

# Spec 11's ten languages, as the rest of the build labels them (cmn_Hani split into
# Simplified and Traditional; Hindi in Devanagari and romanised; Urdu romanised only).
LANGUAGES = ("eng_Latn", "ind_Latn", "zsm_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan",
             "kor_Hang", "tam_Taml", "hin_Deva", "hin_Latn", "urd_Latn")
CODE = "code"   # the stepbuild examples' "language"

# Aya's language_code (ISO 639-3; the card lists Malay as `msa`, the data uses `zsm`)
# -> ours. A str is the label; a tuple names a rule: ("zh",) Simplified / Traditional
# from Aya's language name ("Simplified Chinese" / "Traditional Chinese"; by script
# otherwise); ("script", {script: label}) the label from the text's dominant script,
# and a script not listed is dropped ("script"). Aya's Urdu is Perso-Arabic script
# in the main, and spec 11 covers only romanised Urdu, so most of it is dropped.
AYA_LANGUAGES: dict[str, Any] = {
    "eng": "eng_Latn", "ind": "ind_Latn", "zsm": "zsm_Latn", "msa": "zsm_Latn",
    "jpn": "jpn_Jpan", "kor": "kor_Hang", "tam": "tam_Taml",
    "zho": ("zh",),
    "hin": ("script", {"Deva": "hin_Deva", "Latn": "hin_Latn"}),
    "urd": ("script", {"Latn": "urd_Latn"}),
}
AYA_ZH_NAMES = {"Simplified Chinese": "zho_Hans", "Traditional Chinese": "zho_Hant"}
# oasst2's lang (BCP-47-ish) -> ours; its train split has en, zh, ja, ko and id of
# these (no ms, ta, hi or ur).
OASST2_LANGUAGES: dict[str, Any] = {
    "en": "eng_Latn", "id": "ind_Latn", "ms": "zsm_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "ta": "tam_Taml", "zh": ("zh",),
    "hi": ("script", {"Deva": "hin_Deva", "Latn": "hin_Latn"}),
    "ur": ("script", {"Latn": "urd_Latn"}),
}

# Characters written differently in Simplified and Traditional Chinese (common ones),
# for Chinese text that carries no variant label: more Traditional-only than
# Simplified-only characters means zho_Hant, else zho_Hans.
_SIMPLIFIED = set("这们个说时国会为来对发学过还没样经问题关实现点动种与开长门见东车书语让请电话认识应该头买卖听爱让钱习写读号")
_TRADITIONAL = set("這們個說時國會為來對發學過還沒樣經問題關實現點動種與開長門見東車書語讓請電話認識應該頭買賣聽愛錢習寫讀號")
_SIMPLIFIED -= _TRADITIONAL   # 题/題 etc. differ; any shared character counts for neither
_TRADITIONAL -= _SIMPLIFIED

VAL_FRACTION = 0.02
SEED = 1337
# A turn can hardly fit when its characters exceed context x this (no tokenizer at
# 16+ characters a token): the conversation is cut there before tokenizing, so a
# multi-megabyte Aya row costs nothing.
MAX_CHARS_PER_TOKEN = 16


# ---- languages -----------------------------------------------------------------------------

def script_of(text: str) -> str | None:
    """The dominant script of the letters in `text`: "Latn", "Deva", "Arab", "Taml",
    "Hani", "Kana", "Hang", or None (no letters / other)."""
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250:
            counts["Latn"] += 1
        elif 0x900 <= o <= 0x97F:
            counts["Deva"] += 1
        elif 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFEFF:
            counts["Arab"] += 1
        elif 0xB80 <= o <= 0xBFF:
            counts["Taml"] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts["Kana"] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts["Hang"] += 1
        elif "CJK" in unicodedata.name(ch, ""):
            counts["Hani"] += 1
        else:
            counts["other"] += 1
    if not counts:
        return None
    best = counts.most_common(1)[0][0]
    return None if best == "other" else best


def zh_variant(text: str) -> str:
    simp = sum(ch in _SIMPLIFIED for ch in text)
    trad = sum(ch in _TRADITIONAL for ch in text)
    return "zho_Hant" if trad > simp else "zho_Hans"


def map_language(table: dict[str, Any], code: str, text: str,
                 zh_name: str | None = None) -> tuple[str | None, str | None]:
    """(our label, None) or (None, drop reason: "language" | "script")."""
    rule = table.get(code)
    if rule is None:
        return None, "language"
    if isinstance(rule, str):
        return rule, None
    if rule[0] == "zh":
        return (AYA_ZH_NAMES.get(zh_name or "") or zh_variant(text)), None
    label = rule[1].get(script_of(text) or "")
    return (label, None) if label else (None, "script")


def language_map_doc() -> dict[str, Any]:
    def doc(table: dict[str, Any]) -> dict[str, str]:
        out = {}
        for code, rule in table.items():
            if isinstance(rule, str):
                out[code] = rule
            elif rule[0] == "zh":
                out[code] = ("zho_Hans / zho_Hant (Aya's language name; else by "
                             "Simplified- vs Traditional-only characters)")
            else:
                out[code] = ("by dominant script: " + ", ".join(
                    f"{s} -> {lab}" for s, lab in rule[1].items()) + "; other scripts dropped")
        return out
    return {AYA: doc(AYA_LANGUAGES), OASST2: doc(OASST2_LANGUAGES), STEPBUILD: {"*": CODE}}


# ---- examples ------------------------------------------------------------------------------

@dataclass
class Example:
    source: str
    key: str                      # stable id within the source (the val split hashes it)
    language: str
    messages: list[dict[str, str]]
    licence: str = ""


@dataclass
class SourceStats:
    rows_read: int = 0
    drops: Counter = field(default_factory=Counter)
    truncated: int = 0
    languages: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    licences: Counter = field(default_factory=Counter)
    # stepbuild: prompts cut to fit the context (fit_stepbuild_prompt)
    prompts: Counter = field(default_factory=Counter)

    def keep(self, ex: Example, ids: Sequence[int], mask: Sequence[int], split: str) -> None:
        c = self.languages[ex.language]
        c[f"{split}_conversations"] += 1
        c[f"{split}_tokens"] += len(ids)
        c[f"{split}_assistant_tokens"] += int(sum(mask))
        if ex.licence:
            self.licences[ex.licence] += 1

    def as_dict(self) -> dict[str, Any]:
        langs = {k: dict(sorted(v.items())) for k, v in sorted(self.languages.items())}
        tot: Counter = Counter()
        for v in self.languages.values():
            tot.update(v)
        return {"rows_read": self.rows_read, "drops": dict(sorted(self.drops.items())),
                "truncated_at_turn_boundary": self.truncated,
                "kept": dict(sorted(tot.items())), "languages": langs,
                **({"licences": dict(sorted(self.licences.items()))} if self.licences else {}),
                **({"prompts_fitted": dict(sorted(self.prompts.items()))}
                   if self.prompts else {})}


def aya_examples(rows: Iterable[dict[str, Any]], stats: SourceStats) -> Iterator[Example]:
    """One user + one assistant turn per Aya row in our languages."""
    for n, r in enumerate(rows):
        stats.rows_read += 1
        user, reply = (r.get("inputs") or "").strip(), (r.get("targets") or "").strip()
        lang, why = map_language(AYA_LANGUAGES, r.get("language_code") or "",
                                 user + "\n" + reply, r.get("language"))
        if lang is None:
            stats.drops[why] += 1
            continue
        if not user or not reply:
            stats.drops["empty"] += 1
            continue
        key = hashlib.sha256(f"{user}\x00{reply}".encode("utf-8")).hexdigest()[:24]
        yield Example(AYA, key, lang, [{"role": "user", "content": user},
                                       {"role": "assistant", "content": reply}])


def _usable(m: dict[str, Any]) -> bool:
    return (not m.get("deleted") and m.get("review_result") is not False
            and not m.get("synthetic") and bool((m.get("text") or "").strip()))


def _rank_key(m: dict[str, Any]) -> tuple:
    rank = m.get("rank")
    return (rank if rank is not None else float("inf"), str(m.get("created_date") or ""),
            str(m["message_id"]))


def oasst2_paths(rows: Iterable[dict[str, Any]], stats: SourceStats
                 ) -> Iterator[tuple[str, str, list[dict[str, Any]]]]:
    """(tree id, root lang, top-ranked path of messages) per message tree."""
    by_id: dict[str, dict[str, Any]] = {}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: list[dict[str, Any]] = []
    for m in rows:
        stats.rows_read += 1
        if not _usable(m):
            stats.drops["message_unusable"] += 1
            continue
        by_id[m["message_id"]] = m
    for m in by_id.values():
        parent = m.get("parent_id")
        if parent is None:
            if m.get("role") == "prompter":
                roots.append(m)
        elif parent in by_id:
            children[parent].append(m)
    roots.sort(key=lambda m: (str(m.get("message_tree_id") or m["message_id"])))
    for root in roots:
        path = [root]
        node = root
        while True:
            replies = [c for c in children[node["message_id"]] if c.get("role") == "assistant"]
            top = [c for c in replies if c.get("rank") == 0]
            if not top and len(replies) == 1 and replies[0].get("rank") is None:
                top = replies
            if not top:
                break
            reply = min(top, key=_rank_key)
            path.append(reply)
            follow = [c for c in children[reply["message_id"]] if c.get("role") == "prompter"]
            if not follow:
                break
            node = min(follow, key=_rank_key)
            path.append(node)
        yield str(root.get("message_tree_id") or root["message_id"]), root.get("lang") or "", path


def oasst2_examples(rows: Iterable[dict[str, Any]], stats: SourceStats) -> Iterator[Example]:
    for tree, lang_code, path in oasst2_paths(rows, stats):
        lang, why = map_language(OASST2_LANGUAGES, lang_code, path[0]["text"])
        if lang is None:
            stats.drops[f"tree_{why}"] += 1
            continue
        same = []
        for m in path:
            if (m.get("lang") or "") != lang_code:
                stats.drops["path_cut_language_switch"] += 1
                break
            same.append(m)
        while same and same[-1]["role"] != "assistant":
            same.pop()
        if not same:
            stats.drops["tree_no_ranked_reply"] += 1
            continue
        messages = [{"role": "user" if m["role"] == "prompter" else "assistant",
                     "content": m["text"].strip()} for m in same]
        yield Example(OASST2, tree, lang, messages)


class HeldOutSplitGuard:
    """Refuses a train reply that copies a stepbuild test-split reply (whitespace
    collapsed), or has a FILE block that is a near-copy (5-shingle Jaccard >= the
    LeakageGuard's LEAK_JACCARD) of a test-split block for the same path."""

    def __init__(self, test_rows: Iterable[dict[str, Any]]) -> None:
        from stepbuild.bench.run import _blocks, _collapse, _shingles
        self._whole: dict[str, str] = {}
        self._by_path: dict[str, list[tuple[str, frozenset]]] = defaultdict(list)
        self.rows = 0
        for r in test_rows:
            reply = _reply(r)
            if reply is None:
                continue
            self.rows += 1
            where = f"{r.get('repo')}@{str(r.get('commit'))[:12]}"
            self._whole.setdefault(_collapse(reply), where)
            for block in _blocks(reply):
                self._by_path[block.path].append((where, _shingles(block.content)))

    def find(self, reply: str) -> str | None:
        from stepbuild.bench.run import LEAK_JACCARD, _blocks, _collapse, _shingles, jaccard
        where = self._whole.get(_collapse(reply))
        if where is not None:
            return f"copies test-split reply {where}"
        for block in _blocks(reply):
            refs = self._by_path.get(block.path)
            if not refs:
                continue
            mine = _shingles(block.content)
            for where, theirs in refs:
                if jaccard(mine, theirs) >= LEAK_JACCARD:
                    return f"{block.path} near-copies test-split {where}"
        return None


def _reply(row: dict[str, Any]) -> str | None:
    msgs = row.get("messages") or []
    return msgs[-1]["content"] if msgs and msgs[-1].get("role") == "assistant" else None


# ---- fitting a stepbuild prompt to the context ----------------------------------------------------

# Tokens kept free below the context when a stepbuild prompt is fitted: the
# <|endoftext|> that follows a packed conversation, and a little slack.
STEPBUILD_RESERVE = 8
_SB_CONTEXT = "\n\nCONTEXT FILES:\n"
_SB_TREE = "\nPROJECT TREE:\n"
_SB_NO_CONTEXT = "(none yet)\n"


@dataclass
class _SbPrompt:
    """A stepbuild user message taken apart (stepbuild.dataset.format's layout:
    "STEP: ...", CONTEXT FILES as FILE blocks, PROJECT TREE as tree text)."""
    head: str                     # "STEP: ..." up to the CONTEXT FILES section
    blocks: list[Any]             # FileBlocks, in the row's order
    paths: list[str]              # the tree's paths, de-duplicated, in tree_order
    hidden: int                   # paths the row itself had left out ("... (N more")

    @property
    def max_shown(self) -> int:
        from stepbuild.harness.prompt import TREE_MAX_FILES
        return min(len(self.paths), TREE_MAX_FILES)

    def render(self, blocks: Sequence[Any], shown: int) -> str:
        """The message with these context blocks and the first `shown` tree paths,
        in the harness's own layout: render_blocks, and render_tree's text (tree
        order, at most TREE_MAX_FILES, then "... (N more files not shown)" counting
        the paths the row had already left out too)."""
        from stepbuild.harness.blocks import render_blocks
        from stepbuild.harness.prompt import render_tree
        ctx = render_blocks(blocks) if blocks else _SB_NO_CONTEXT
        tree = render_tree(self.paths, shown)
        if self.hidden:
            lines = tree.splitlines()[:shown]
            lines.append(f"... ({len(self.paths) - shown + self.hidden} more files not shown)")
            tree = "".join(line + "\n" for line in lines)
        return f"{self.head}{_SB_CONTEXT}{ctx}{_SB_TREE}{tree}"


def _parse_sb_user(user: str) -> _SbPrompt | None:
    """The row's user message taken apart, or None when it is not the dataset's
    layout (then it is not re-rendered at all). The context must re-render byte for
    byte (render_blocks); the tree is read as one path per line. Rows mined before
    the tree cap list their whole tree sorted; a cut prompt shows it the way the
    harness does now (tree order, capped)."""
    import re
    from stepbuild.harness.blocks import BlockError, render_blocks, parse_blocks
    from stepbuild.harness.prompt import tree_order
    i = user.find(_SB_CONTEXT)
    j = user.rfind(_SB_TREE)
    if not user.startswith("STEP: ") or i < 0 or j < i + len(_SB_CONTEXT) - 1:
        return None
    head, ctx, tree = user[:i], user[i + len(_SB_CONTEXT):j], user[j + len(_SB_TREE):]
    if ctx == _SB_NO_CONTEXT:
        blocks = []
    else:
        try:
            blocks = parse_blocks(ctx)
        except BlockError:
            return None
        if render_blocks(blocks) != ctx:
            return None
    lines = tree.splitlines()
    hidden = 0
    m = re.fullmatch(r"\.\.\. \((\d+) more files not shown\)", lines[-1]) if lines else None
    if m:
        hidden = int(m.group(1))
        lines = lines[:-1]
    if any(not line.strip() for line in lines):
        return None
    return _SbPrompt(head, blocks, tree_order(lines), hidden)


def fit_stepbuild_prompt(tok: Any, messages: list[dict[str, str]], context: int,
                         reserve: int = STEPBUILD_RESERVE
                         ) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    """The stepbuild conversation (system, user, assistant) with its user message cut
    to fit `context` - `reserve` tokens beside the whole reply, the way the benchmark
    harness fits a prompt (stepbuild.harness.prompt.build_messages): the PROJECT
    TREE is cut first (the largest prefix that fits, the harness's render_tree and
    its "... (N more files not shown)"), and FILE blocks are never cut in the
    middle. Where build_messages would then give up (PromptTooLong), a training row
    drops WHOLE context files instead, the files the reply does not rewrite first
    (the last shown first), and the tree is refitted to the room that frees. The
    reply is never touched. Counted with the real tokenizer, exactly as training
    encodes the conversation.

    Returns (messages, info): messages unchanged when they fit, the cut ones, or
    None with info["drop"] = "reply_too_long" (the reply alone leaves no room) or
    "prompt_too_long" (not even the STEP line fits beside it). info also says
    whether anything was cut ("trimmed") and how many files and tree paths went."""
    info: dict[str, Any] = {"drop": None, "trimmed": False, "files_dropped": 0,
                            "tree_paths_dropped": 0}
    if (len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]):
        return list(messages), info
    system, user, reply = messages
    reply_n = len(chat.encode_turn(tok, reply)[0])
    if reply_n + reserve > context:
        info["drop"] = "reply_too_long"
        return None, info
    budget = context - reserve - reply_n - len(chat.encode_turn(tok, system)[0])

    def size(text: str) -> int:
        return len(chat.encode_turn(tok, {"role": "user", "content": text})[0])

    if size(user["content"]) <= budget:
        return list(messages), info
    prompt = _parse_sb_user(user["content"])
    if prompt is None:
        return list(messages), info     # not the dataset's layout: fit() decides
    written = {b.path for b in _reply_blocks(reply["content"])}
    order = ([b for b in reversed(prompt.blocks) if b.path not in written]
             + [b for b in reversed(prompt.blocks) if b.path in written])
    for dropped in range(len(order) + 1):
        gone = {id(b) for b in order[:dropped]}
        blocks = [b for b in prompt.blocks if id(b) not in gone]
        if size(prompt.render(blocks, 0)) > budget:
            continue
        lo, hi = 0, prompt.max_shown         # the largest tree prefix that fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if size(prompt.render(blocks, mid)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        info.update(trimmed=True, files_dropped=dropped,
                    tree_paths_dropped=len(prompt.paths) - lo + prompt.hidden)
        return [dict(system), {"role": "user", "content": prompt.render(blocks, lo)},
                dict(reply)], info
    info["drop"] = "prompt_too_long"
    return None, info


def _reply_blocks(reply: str) -> list[Any]:
    from stepbuild.harness.blocks import BlockError, parse_blocks
    try:
        return parse_blocks(reply)
    except BlockError:
        return []


def stepbuild_examples(rows: Iterable[dict[str, Any]], stats: SourceStats, *,
                       bench_guard: Any, test_guard: HeldOutSplitGuard | None,
                       tok: Any = None, context: int | None = None) -> Iterator[Example]:
    """The stepbuild train split, screened by the bench's LeakageGuard and the
    test-split guard; with `tok` and `context`, each prompt fitted to the context
    (fit_stepbuild_prompt; a row it cannot fit is dropped as "reply_too_long" or
    "prompt_too_long")."""
    for r in rows:
        stats.rows_read += 1
        if r.get("split") != "train":
            stats.drops["not_train_split"] += 1
            continue
        reply = _reply(r)
        if reply is None:
            stats.drops["no_reply"] += 1
            continue
        if bench_guard is not None and bench_guard.find(reply) is not None:
            stats.drops["leakage_bench"] += 1
            continue
        if test_guard is not None and test_guard.find(reply) is not None:
            stats.drops["leakage_test_split"] += 1
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        if tok is not None and context is not None:
            fitted, info = fit_stepbuild_prompt(tok, messages, context)
            if fitted is None:
                stats.drops[info["drop"]] += 1
                continue
            if info["trimmed"]:
                stats.prompts["trimmed"] += 1
                stats.prompts["context_files_dropped"] += info["files_dropped"]
            messages = fitted
        yield Example(STEPBUILD, f"{r.get('repo')}@{r.get('commit')}", CODE, messages,
                      licence=str(r.get("licence") or ""))


# ---- screening, splitting, packing ------------------------------------------------------------

def is_val(ex: Example, fraction: float) -> bool:
    h = hashlib.sha256(f"{ex.source}\x00{ex.key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < fraction


def _cut_chars(messages: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """The messages up to (not including) the first that takes the running character
    count past `limit`: nothing after it could fit the context."""
    total = 0
    for i, m in enumerate(messages):
        total += len(m["content"])
        if total > limit:
            return messages[:i]
    return messages


@dataclass
class Kept:
    source: str
    ids: list[int]
    mask: list[int]


def screen(examples: Iterable[Example], stats: dict[str, SourceStats], *, tok: Any,
           context: int, decontam: Any, val_fraction: float
           ) -> tuple[list[Kept], list[Kept], Counter]:
    """Chat format, decontamination, dedupe, context fit and the val split, in that
    order. Returns (train, val, decontamination hits per benchmark)."""
    seen: set[str] = set()
    train: list[Kept] = []
    val: list[Kept] = []
    hits: Counter = Counter()
    for ex in examples:
        st = stats[ex.source]
        if any(not m["content"].strip() for m in ex.messages):
            st.drops["empty"] += 1
            continue
        try:
            text = chat.render(ex.messages)
        except chat.ChatFormatError:
            st.drops["special_tokens" if any(s in m["content"] for m in ex.messages
                                             for s in chat.CHAT_SPECIALS) else "format"] += 1
            continue
        if decontam is not None:
            found = decontam.find("\n".join(m["content"] for m in ex.messages))
            if found is not None:
                st.drops[f"decontam_{found[0]}"] += 1
                hits[found[0]] += 1
                continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            st.drops["duplicate"] += 1
            continue
        seen.add(digest)
        messages = _cut_chars(ex.messages, context * MAX_CHARS_PER_TOKEN)
        fitted = chat.fit(tok, messages, context) if messages else None
        if fitted is None:
            st.drops["too_long"] += 1
            continue
        if fitted.truncated or len(messages) < len(ex.messages):
            st.truncated += 1
        split = "val" if is_val(ex, val_fraction) else "train"
        st.keep(ex, fitted.ids, fitted.mask, split)
        (val if split == "val" else train).append(Kept(ex.source, fitted.ids, fitted.mask))
    return train, val, hits


def pack(convs: Sequence[Kept], context: int, eot: int, packed: bool = True
         ) -> tuple[np.ndarray, np.ndarray]:
    """Blocks of exactly `context` tokens (flattened) and their masks. Greedy: a
    conversation goes into the current block when it fits, followed by <|endoftext|>
    when there is room; otherwise the block is padded with <|endoftext|> and the
    conversation starts the next one. packed=False: one conversation per block."""
    blocks_t: list[list[int]] = []
    blocks_m: list[list[int]] = []
    cur_t: list[int] = []
    cur_m: list[int] = []

    def close() -> None:
        pad = context - len(cur_t)
        blocks_t.append(cur_t + [eot] * pad)
        blocks_m.append(cur_m + [0] * pad)

    for c in convs:
        if len(c.ids) > context:
            raise ValueError(f"a conversation of {len(c.ids)} tokens is over the context")
        if cur_t and (not packed or len(cur_t) + len(c.ids) > context):
            close()
            cur_t, cur_m = [], []
        cur_t += c.ids
        cur_m += c.mask
        if len(cur_t) < context:
            cur_t.append(eot)
            cur_m.append(0)
    if cur_t:
        close()
    tokens = np.array([t for b in blocks_t for t in b], dtype=np.uint16)
    mask = np.array([m for b in blocks_m for m in b], dtype=np.uint8)
    return tokens, mask


def write_blocks(out_dir: Path, tokens: np.ndarray, mask: np.ndarray, context: int,
                 shard_tokens: int) -> list[str]:
    """shard_NNN.bin + shard_NNN.mask, each shard a whole number of blocks; stale
    shards of an earlier build in out_dir are removed first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in list(out_dir.glob("shard_*.bin")) + list(out_dir.glob("shard_*.mask")):
        p.unlink()
    per = max(1, shard_tokens // context) * context
    names = []
    for i, lo in enumerate(range(0, len(tokens), per)):
        name = f"shard_{i:03d}"
        write_shard(out_dir / f"{name}.bin", tokens[lo:lo + per])
        write_mask(out_dir / f"{name}.mask", mask[lo:lo + per])
        names.append(name + ".bin")
    return names


def _split_summary(convs: Sequence[Kept], tokens: np.ndarray, mask: np.ndarray,
                   context: int, shards: list[str]) -> dict[str, Any]:
    return {"conversations": len(convs), "blocks": len(tokens) // context,
            "tokens": int(len(tokens)),
            "conversation_tokens": int(sum(len(c.ids) for c in convs)),
            "assistant_tokens": int(mask.sum()), "shards": shards}


def build(sources: dict[str, Callable[[SourceStats], Iterable[Example]]], out: Path, *,
          tok: Any, context: int, decontam: Any = None, val_fraction: float = VAL_FRACTION,
          seed: int = SEED, packed: bool = True, shard_tokens: int = 50_000_000,
          provenance: dict[str, dict[str, Any]] | None = None,
          extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screen, split, pack and write every source; returns the manifest (also
    written to out/manifest.json). sources maps a source name to a function that
    yields its Examples, counting what it reads and drops in the SourceStats it is
    given."""
    stats = {name: SourceStats() for name in sources}

    def every() -> Iterator[Example]:
        for name, make in sources.items():
            yield from make(stats[name])

    train, val, hits = screen(every(), stats, tok=tok, context=context, decontam=decontam,
                              val_fraction=val_fraction)
    if not train:
        raise ValueError("no training conversations survived the filters")
    random.Random(seed).shuffle(train)
    eot = tok.special_id(chat.EOT)
    splits: dict[str, Any] = {}
    t_tok, t_mask = pack(train, context, eot, packed)
    splits["train"] = _split_summary(train, t_tok, t_mask, context,
                                     write_blocks(out / "train", t_tok, t_mask, context,
                                                  shard_tokens))
    if val:
        v_tok, v_mask = pack(val, context, eot, packed)
        splits["val"] = _split_summary(val, v_tok, v_mask, context,
                                       write_blocks(out / "val", v_tok, v_mask, context,
                                                    shard_tokens))
        splits["val_by_source"] = {}
        for name in sources:
            mine = [c for c in val if c.source == name]
            if mine:
                s_tok, s_mask = pack(mine, context, eot, packed)
                splits["val_by_source"][name] = _split_summary(
                    mine, s_tok, s_mask, context,
                    write_blocks(out / "val_by_source" / name, s_tok, s_mask, context,
                                 shard_tokens))
    prov = provenance or {}
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": context,
        "packing": ("greedy, conversations separated by <|endoftext|>, blocks padded with it"
                    if packed else "one conversation per block, padded with <|endoftext|>"),
        "loss_mask": "1 on assistant content tokens and their <|end|>",
        "seed": seed, "val_fraction": val_fraction,
        "sources": {name: {**prov.get(name, {}), "license": LICENSES.get(name, ""),
                           **stats[name].as_dict()} for name in sources},
        "language_map": {k: v for k, v in language_map_doc().items() if k in sources},
        "decontamination": ({**decontam.summary(), "dropped": dict(hits)}
                            if decontam is not None else "off"),
        "splits": splits,
        **(extra or {}),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


# ---- the real sources (network) ---------------------------------------------------------------

def _pinned(name: str, rev: str) -> str:
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", rev or ""):
        raise SystemExit(f"error: the {name} revision must be a pinned 40-hex commit "
                         f"(config [data]); got {rev!r}")
    return rev


def _parquet_rows(paths: Sequence[str], columns: list[str], limit: int | None
                  ) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    n = 0
    for p in paths:
        f = pq.ParquetFile(p)
        cols = [c for c in columns if c in f.schema_arrow.names]
        for batch in f.iter_batches(batch_size=4096, columns=cols):
            for row in batch.to_pylist():
                if limit is not None and n >= limit:
                    return
                n += 1
                yield row


def _jsonl_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/quipu-moe-sft.toml")
    ap.add_argument("--out", default=None, help="default: the config's data.shard_dir")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--stepbuild-dir", default=None, help="default: data.stepbuild_dir")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per source (trials)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-pack", action="store_true", help="one conversation per block")
    ap.add_argument("--no-decontam", action="store_true", help="trials only; never for the run")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    from quipu.config import load_config
    from quipu.decontam import load_benchmarks
    from quipu.tokenizer import make_tokenizer

    cfg = load_config(args.config)
    tok = make_tokenizer(cfg.data.tokenizer)
    if not hasattr(tok, "encode_with_special"):
        print("error: the chat data needs the BPE tokenizer with chat tokens", file=sys.stderr)
        return 2
    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = sorted(set(wanted) - set(SOURCES))
    if unknown:
        print(f"error: unknown source(s) {unknown}; choose from {list(SOURCES)}", file=sys.stderr)
        return 2
    out = Path(args.out or cfg.data.shard_dir)
    context = cfg.model.context
    sources: dict[str, Callable[[SourceStats], Iterable[Example]]] = {}
    provenance: dict[str, dict[str, Any]] = {}

    if AYA in wanted:
        rev = _pinned("aya", cfg.data.aya_revision)
        files = [hf_hub_download(AYA_DATASET, f, repo_type="dataset", revision=rev)
                 for f in AYA_FILES]
        provenance[AYA] = {"dataset": AYA_DATASET, "revision": rev, "files": list(AYA_FILES)}
        sources[AYA] = lambda st, files=files: aya_examples(
            _parquet_rows(files, AYA_COLUMNS, args.limit), st)
    if OASST2 in wanted:
        rev = _pinned("oasst2", cfg.data.oasst2_revision)
        names = sorted(f for f in HfApi().list_repo_files(OASST2_DATASET, repo_type="dataset",
                                                           revision=rev)
                       if f.startswith("data/") and f.endswith(".parquet"))
        files = [hf_hub_download(OASST2_DATASET, f, repo_type="dataset", revision=rev)
                 for f in names]
        provenance[OASST2] = {"dataset": OASST2_DATASET, "revision": rev, "files": names}
        sources[OASST2] = lambda st, files=files: oasst2_examples(
            _parquet_rows(files, OASST2_COLUMNS, args.limit), st)
    if STEPBUILD in wanted:
        from stepbuild.bench.run import LeakageGuard
        sdir = Path(args.stepbuild_dir or cfg.data.stepbuild_dir)
        train_files = sorted(sdir.glob("train-*.jsonl"))
        test_files = sorted(sdir.glob("test-*.jsonl"))
        if not train_files:
            print(f"error: no train-*.jsonl in {sdir}", file=sys.stderr)
            return 2
        bench = LeakageGuard()
        test_guard = HeldOutSplitGuard(_jsonl_rows(test_files))
        provenance[STEPBUILD] = {
            "dataset": str(sdir), "files": [p.name for p in train_files],
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in train_files + test_files},
            "leakage_guard": {"bench_reference_files": bench.reference_files,
                              "test_split_rows": test_guard.rows,
                              "test_split_files": [p.name for p in test_files]}}

        def stepbuild_source(st: SourceStats) -> Iterator[Example]:
            rows = _jsonl_rows(train_files)
            if args.limit is not None:
                rows = (r for i, r in zip(range(args.limit), rows))
            return stepbuild_examples(rows, st, bench_guard=bench, test_guard=test_guard,
                                      tok=tok, context=context)
        sources[STEPBUILD] = stepbuild_source

    decontam = None
    if not args.no_decontam:
        decontam, _ = load_benchmarks({"humaneval": cfg.data.humaneval_revision,
                                       "mbpp": cfg.data.mbpp_revision})
    extra = {"config": str(args.config), "config_layers": list(cfg.layers),
             "tokenizer": {"path": cfg.data.tokenizer, "sha256": tok.sha256(),
                           "vocab_size": tok.vocab_size}}
    manifest = build(sources, out, tok=tok, context=context, decontam=decontam,
                     val_fraction=args.val_fraction, seed=args.seed, packed=not args.no_pack,
                     shard_tokens=cfg.data.shard_tokens, provenance=provenance, extra=extra)
    tr = manifest["splits"]["train"]
    print(f"train: {tr['conversations']:,} conversations, {tr['blocks']:,} blocks, "
          f"{tr['tokens']:,} tokens ({tr['assistant_tokens']:,} assistant)")
    for name, s in manifest["sources"].items():
        print(f"  {name}: kept {s['kept']}, drops {s['drops']}, "
              f"truncated {s['truncated_at_turn_boundary']}")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
