"""quipu/ocr_metrics.py."""
import pytest

from quipu.ocr_metrics import cer, edit_distance, exact_match, wer


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_cer_is_total_edits_over_total_reference_chars():
    assert cer(["abcd", "xy"], ["abcd", "xz"]) == 1 / 6
    assert cer(["ab"], [""]) == 1.0


def test_wer_counts_words():
    assert wer(["the cat sat"], ["the bat sat"]) == 1 / 3


def test_exact_match():
    assert exact_match(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3


def test_empty_reference_set_raises():
    with pytest.raises(ValueError):
        cer([], [])


def test_exact_match_length_mismatch_raises():
    with pytest.raises(ValueError, match="references"):
        exact_match(["a", "b"], ["a"])
