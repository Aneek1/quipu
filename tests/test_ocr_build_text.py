"""scripts/ocr/build_text.py: cutting lines, random lines, the held-out split."""
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_text", ROOT / "scripts" / "ocr" / "build_text.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def test_cut_lines_respects_lengths_and_word_boundaries():
    doc = "The quick brown fox jumps over the lazy dog. " * 5
    lines = bt.cut_lines(doc, random.Random(0), min_chars=5, max_chars=20)
    assert lines
    for s in lines:
        assert 5 <= len(s) <= 20
        assert s == s.strip()
        assert all(w in doc for w in s.split())


def test_cut_lines_drops_characters_outside_the_set():
    lines = bt.cut_lines("café crème naïve plain words here", random.Random(0), 5, 60)
    assert all("é" not in s and "ï" not in s for s in lines)


def test_random_line_is_in_charset_and_bounded():
    from quipu.ocr_ctc import in_charset
    rng = random.Random(1)
    for _ in range(200):
        s = bt.random_line(rng, max_chars=40)
        assert 1 <= len(s) <= 40 and in_charset(s) and s == s.strip()


def test_is_held_is_stable_and_about_one_in_twenty():
    held = [bt.is_held(i) for i in range(20000)]
    assert held == [bt.is_held(i) for i in range(20000)]
    assert 0.03 < sum(held) / len(held) < 0.07


def test_build_writes_disjoint_files(tmp_path):
    docs = [f"document number {i} has some ordinary words in it for lines" for i in range(400)]
    bt.build(docs, tmp_path, random_frac=0.15, seed=0)
    train = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()
    held = (tmp_path / "held.txt").read_text(encoding="utf-8").splitlines()
    assert train and held
    assert not set(train) & set(held)
