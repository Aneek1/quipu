"""quipu.bpe, make_tokenizer, and the pure parts of the trainer and gate; no network."""
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from quipu.bpe import SPECIAL_TOKENS, BPETokenizer, train_bpe
from quipu.tokenizer import Tokenizer, make_tokenizer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tt = _load("train_tokenizer")
gate = _load("tokenizer_gate")

CODE = '''def area(width, height):
    """Rectangle area."""
    if width < 0:
        raise ValueError("negative width")
    for i in range(10):
        total = width * height + i
        if total > 100:
            return total
    return width * height


class Shape:
    def __init__(self, name):
        self.name = name

    def describe(self):
        return f"{self.name} has 4 sides"
'''
TEXT = ("The quipu was a recording device made of knotted cords. Historians still "
        "debate how much of its content was numerical and how much was narrative. "
        "Each cord could carry several knots, and the position of a knot mattered.\n")


def corpus(variant=0):
    return [CODE.replace("area", f"area{variant}")] * 40 + [TEXT] * 40


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    path = tmp_path_factory.mktemp("bpe") / "tokenizer.json"
    return train_bpe(corpus(), 512, path)


@pytest.mark.parametrize("text", [
    "naïve café, 東京, Ελληνικά, привет",
    "line one\r\nline two\r\n\r\n",
    "\tindented\n\t\tdeeper\n",
    " " * 40 + "return x\n",
    "emoji 🎉🚀 and a ZWJ family 👨‍👩‍👧 and a lone  nbsp",
    "",
    "trailing spaces   \n  \n",
    "简体中文和繁體中文，日本語のテキスト、한국어 문장입니다。",
    "தமிழ் ஒரு திராவிட மொழி ஆகும்.",
    "हिन्दी भारत की एक भाषा है। संख्या १२३",
    "mujhe nahin pata ki woh kab aayega, bhai!",
    "Saya tidak tahu bila dia akan datang.",
    "\n\n\t\t  \r\n" + " " * 20 + "x\n" + "\t" * 20 + "y",
])
def test_round_trip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_encode_treats_special_strings_as_text_but_encode_with_special_parses_them(tok):
    user = tok.special_id("<|user|>")
    assert user not in tok.encode("hello <|user|> there")
    assert tok.decode(tok.encode("hello <|user|> there")) == "hello <|user|> there"
    assert user in tok.encode_with_special("hello <|user|> there")
    assert tok.encode_with_special("<|user|>") == [user]


def test_special_tokens_have_fixed_low_ids_in_every_training(tmp_path, tok):
    other = train_bpe(corpus(variant=7) + ["something else entirely"], 400,
                      tmp_path / "other.json")
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.special_id(name) == i == other.special_id(name)
        assert tok.encode_with_special(name) == [i]
    assert tok.eot == 0


def _pieces(text):
    from quipu.bpe import build_pre_tokenizer
    from tokenizers import decoders

    return [decoders.ByteLevel().decode([p]) for p, _ in
            build_pre_tokenizer().pre_tokenize_str(text)]


def test_space_runs_are_own_pieces_but_a_single_space_stays_on_its_word():
    # Indentation now travels with its newline (below); a space run elsewhere
    # (alignment, a file that starts indented) is still its own piece, cut at 16.
    assert _pieces("a        = b") == ["a", " " * 8, "=", " b"]
    assert _pieces(" " * 20 + "x") == [" " * 16, " " * 4, "x"]
    assert _pieces("\n" + " " * 20 + "x") == ["\n" + " " * 16, " " * 4, "x"]
    assert _pieces("\t\tx") == ["\t\t", "x"]


def test_a_newline_plus_8_spaces_is_one_token_after_training_on_indented_code(tok):
    ids = tok.encode("x = 1\n        return total\n")
    assert "\n" + " " * 8 in [tok.decode([i]) for i in ids]


def test_punctuation_does_not_swallow_the_newline_and_indent_that_follow(tok):
    ids = tok.encode("):\n        return total")
    assert [tok.decode([i]) for i in ids][-3:] == ["\n" + " " * 8, "return", " total"]


def test_indic_combining_marks_stay_inside_their_word():
    assert _pieces("தமிழ் மொழி हिन्दी भाषा") == ["தமிழ்", " மொழி", " हिन्दी", " भाषा"]


def test_digits_split_individually(tok):
    ids = tok.encode("12345")
    assert [tok.decode([i]) for i in ids] == list("12345")


def test_vocab_fits_uint16_and_loader_rejects_moved_special_tokens(tmp_path, tok):
    assert tok.vocab_size <= 65536
    bad = tmp_path / "bad.json"
    bad.write_text(tok.path.read_text(encoding="utf-8").replace('"<|user|>"', '"<|usr|>"'),
                   encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("<|user|>")):
        BPETokenizer(bad)


def test_train_rejects_a_vocab_too_small_for_bytes_and_specials(tmp_path):
    with pytest.raises(ValueError):
        train_bpe(["x"], 200, tmp_path / "t.json")


def test_make_tokenizer_gpt2_is_the_old_tokenizer_and_a_path_is_bpe(tok):
    old = make_tokenizer("gpt2")
    assert type(old) is Tokenizer and old.vocab_size == 50257 and old.eot == 50256
    new = make_tokenizer(str(tok.path))
    assert isinstance(new, BPETokenizer) and new.vocab_size == tok.vocab_size
    with pytest.raises(FileNotFoundError):
        make_tokenizer("no/such/tokenizer.json")


# ------------------------------------------------------------------ gate metrics

class FakeTok:
    """ids: 0 'a', 1 ' ', 2 '\\n', 3 'bc', 4 '  ' ; each character-run is one token."""
    pieces = ["a", " ", "\n", "bc", "  "]
    vocab_size = 5

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for j in (4, 3, 0, 1, 2):
                p = self.pieces[j]
                if text.startswith(p, i):
                    out.append(j)
                    i += len(p)
                    break
            else:
                raise ValueError(text[i])
        return out

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_whitespace_share_is_computed_by_decoding_each_token():
    fake = FakeTok()
    is_ws = gate.whitespace_table(fake)
    assert is_ws.tolist() == [False, True, True, False, True]
    m = gate.measure(fake, ["a  bc\n", "bc a"], is_ws)
    # tokens: a,'  ',bc,\n  |  bc,' ',a  -> 7 tokens, 3 whitespace-only
    assert m["tokens"] == 7 and m["whitespace_tokens"] == 3
    assert m["whitespace_share"] == pytest.approx(3 / 7)
    assert m["chars"] == 10 and m["chars_per_token"] == pytest.approx(10 / 7)
    assert m["round_trip_failures"] == []


def test_measure_reports_round_trip_failures():
    class Lossy(FakeTok):
        def decode(self, ids):
            return super().decode(ids).replace("\n", "")
    m = gate.measure(Lossy(), ["a\n", "a"], gate.whitespace_table(FakeTok()))
    assert m["round_trip_failures"] == [0]


def _m(ws, cpt, fails=()):
    return {"documents": 10, "whitespace_share": ws, "chars_per_token": cpt,
            "round_trip_failures": list(fails)}


def test_judge_applies_every_criterion_including_each_other_language():
    gpt2 = {"code": _m(0.365, 2.5), "eng_Latn": _m(0.02, 4.4), "tam_Taml": _m(0.0, 0.6),
            "cmn_Hani": _m(0.0, 0.8)}
    good = {"code": _m(0.10, 3.0), "eng_Latn": _m(0.02, 4.3), "tam_Taml": _m(0.0, 3.1),
            "cmn_Hani": _m(0.0, 1.4)}
    rows = gate.judge(good, gpt2)
    assert [r["measure"] for r in rows] == [
        "Whitespace-only share of code tokens", "Characters per token, code",
        "Characters per token, eng_Latn", "Characters per token, tam_Taml",
        "Characters per token, cmn_Hani", "Round trip decode(encode(x)) == x"]
    assert all(r["passed"] for r in rows)
    bad = {"code": _m(0.16, 2.4, [3]), "eng_Latn": _m(0.02, 4.1), "tam_Taml": _m(0.0, 3.1),
           "cmn_Hani": _m(0.0, 0.7)}
    rows = gate.judge(bad, gpt2)
    assert [r["passed"] for r in rows] == [False, False, False, True, False, False]
    assert "FAIL" in gate.render(rows, {"tokenizer": "t", "sha256": "0", "vocab_size": 1,
                                        "when": "now", "sets": {}, "measurements": {}})


# ------------------------------------------------------------------ trainer sampling

def test_quota_sampler_redistributes_a_language_that_never_appears():
    s = tt.QuotaSampler({"Python": 0.5, "SQL": 0.3, "HTML": 0.2}, 100, window=10,
                        max_windows=100)
    for lang in ["Python", "HTML"] * 200:  # SQL never appears
        if s.done:
            break
        s.offer(lang, 1)
    assert s.done and s.total >= 100 and s.exhausted == ["SQL"]
    assert s.taken["HTML"] == 20  # capped: never a taker
    assert s.taken["SQL"] == 0 and s.taken["Python"] == 80


def test_quota_sampler_cuts_a_language_too_rare_to_fill_in_max_windows():
    # Rust trickles in at 1 unit per 20-offer window; filling its 30 would take 30
    # windows, beyond max_windows=8, so after window 1 it is exhausted at 1 and its
    # remaining 29 go to Python. Without the rule this loop would crawl for 600 offers.
    s = tt.QuotaSampler({"Python": 0.7, "Rust": 0.3}, 100, window=20, max_windows=8,
                        capped=())
    stream = (["Python"] * 19 + ["Rust"]) * 50
    offered = 0
    for lang in stream:
        if s.done:
            break
        s.offer(lang, 1)
        offered += 1
    assert s.exhausted == ["Rust"] and s.taken["Rust"] == 1
    assert s.done and s.taken["Python"] == 99 and offered < 120
    summary = s.summary()
    assert summary["exhausted"] == ["Rust"]
    assert summary["by_language"]["Rust"]["achieved_share"] == pytest.approx(0.01)
    assert summary["by_language"]["Python"]["target_share"] == 0.7


def test_quota_sampler_keeps_a_common_language_that_fills_within_max_windows():
    s = tt.QuotaSampler({"Python": 0.5, "JavaScript": 0.5}, 50, window=10, max_windows=6,
                        capped=())
    for lang in ["Python", "JavaScript"] * 100:
        if s.done:
            break
        s.offer(lang, 1)
    assert s.exhausted == [] and s.taken == {"Python": 25, "JavaScript": 25}


def test_quota_sampler_is_done_when_every_language_is_filled_or_exhausted():
    s = tt.QuotaSampler({"HTML": 0.5, "SQL": 0.5}, 100, window=5, max_windows=100)
    for lang in ["HTML"] * 100:  # HTML capped, SQL absent: nothing can take the rest
        if s.done:
            break
        s.offer(lang, 1)
    assert s.done and s.taken["HTML"] == 50 and s.exhausted == ["SQL"]


def test_quota_sampler_rejects_bad_weights():
    with pytest.raises(ValueError):
        tt.QuotaSampler({"Python": 0.5}, 10)


def test_code_documents_apply_licence_path_length_and_duplicate_filters():
    rows = [
        {"code": "print(1)", "language": "Python", "license": "mit", "path": "a.py"},
        {"code": "print(2)", "language": "Python", "license": "gpl-3.0", "path": "b.py"},
        {"code": "x", "language": "JavaScript", "license": "mit", "path": "app.min.js"},
        {"code": "y" * 50, "language": "JavaScript", "license": "mit", "path": "big.js"},
        {"code": "dup", "language": "Python", "license": "mit", "path": "c.py"},
        {"code": "  ", "language": "Python", "license": "mit", "path": "d.py"},
        {"code": "ok", "language": "Brainfuck", "license": "mit", "path": "e.bf"},
        {"code": "let x", "language": "JavaScript", "license": "mit", "path": "src/x.js"},
    ]
    s = tt.QuotaSampler({"Python": 0.5, "JavaScript": 0.3, "HTML": 0.2}, 1000)
    stats, hashes = Counter(), []
    got = list(tt.code_documents(rows, s, size=len, max_doc_bytes=40, stats=stats,
                                 exclude={tt.bs.content_hash("dup")}, hashes=hashes))
    assert got == [("Python", "print(1)"), ("JavaScript", "let x")]
    assert stats["dropped_license"] == 1 and stats["skipped_minified"] == 1
    assert stats["skipped_too_long"] == 1 and stats["dropped_as_duplicate"] == 1
    assert stats["skipped_blank"] == 1 and stats["dropped_language"] == 1
    assert hashes == [tt.bs.content_hash("print(1)"), tt.bs.content_hash("let x")]


def test_text_documents_stop_at_the_budget_and_digest_is_order_sensitive():
    rows = [{"text": "abcd"}, {"text": " "}, {"text": "efgh"}, {"text": "ijkl"}]
    assert list(tt.text_documents(rows, 8, size=len)) == ["abcd", "efgh"]
    import hashlib
    d1, d2 = hashlib.sha256(), hashlib.sha256()
    list(tt.batches_with_digest(["ab", "c"], d1, size=1))
    list(tt.batches_with_digest(["a", "bc"], d2, size=1))
    assert d1.hexdigest() != d2.hexdigest()
    assert list(tt.batches_with_digest(iter("abcde"), hashlib.sha256(), size=2)) == \
        [["a", "b"], ["c", "d"], ["e"]]


def test_language_weights_follow_the_spec():
    w = tt.LANGUAGE_WEIGHTS
    assert sum(w.values()) == pytest.approx(1.0)
    assert (w["Python"], w["JavaScript"], w["TypeScript"], w["HTML"], w["CSS"], w["SQL"]) == \
        (0.30, 0.25, 0.12, 0.08, 0.05, 0.05)
    assert "HTML" in tt.CAPPED_LANGUAGES
    np.testing.assert_allclose(sum(v for k, v in w.items() if k not in
                                   ("Python", "JavaScript", "TypeScript", "HTML", "CSS",
                                    "SQL")), 0.15)


def test_multilingual_documents_share_the_budget_and_pass_a_shortfall_on():
    opened = []

    def stream(lang, n):
        def open_():
            opened.append(lang)
            return iter([{"text": f"{lang}{i}"} for i in range(n)])
        return open_

    # Each doc is 1 unit; budget 30 over three languages. "b" has only 4 docs, so
    # its shortfall of 6 is shared: c gets (30 - 10 - 4) = 16.
    streams = {"a": stream("a", 100), "b": stream("b", 4), "c": stream("c", 100)}
    summary = {}
    got = list(tt.multilingual_documents(streams, 30, size=lambda _: 1, summary=summary))
    counts = Counter(lang for lang, _ in got)
    assert counts == {"a": 10, "b": 4, "c": 16}
    assert summary["b"]["short"] and not summary["a"]["short"]
    assert opened == ["a", "b", "c"]


def test_iter_parquet_text_reads_every_row_group_in_order(tmp_path):
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"text": [f"t{i}" for i in range(25)], "id": list(range(25))}),
                   path, row_group_size=10)
    rows = list(tt.iter_parquet_text(fsspec.filesystem("file"), [str(path)]))
    assert [r["text"] for r in rows] == [f"t{i}" for i in range(25)]


def test_the_multilingual_mix_follows_spec_section_11():
    assert tt.FINEWEB2_LANGUAGES == ("ind_Latn", "zsm_Latn", "cmn_Hani", "jpn_Jpan",
                                     "kor_Hang", "tam_Taml", "hin_Deva", "hin_Latn",
                                     "urd_Latn")
    assert (tt.CODE_SHARE, tt.ENGLISH_SHARE) == (0.6, 0.28)
