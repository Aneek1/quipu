"""quipu/ocr_ctc.py: character set, labels, CTC length check, greedy decode."""
import pytest
import torch

from quipu import ocr_ctc as oc


def test_charset_is_printable_ascii_plus_extras():
    assert oc.CHARSET[:95] == "".join(chr(c) for c in range(32, 127))
    assert oc.CHARSET[95:] == "°£€×–‘’"
    assert oc.N_CLASSES == len(oc.CHARSET) + 1
    assert oc.BLANK == 0


def test_encode_text_is_one_based_and_rejects_unknown():
    assert oc.encode_text(" !") == [1, 2]
    assert oc.encode_text("a") == [oc.CHARSET.index("a") + 1]
    with pytest.raises(ValueError, match="not in the OCR character set"):
        oc.encode_text("naïve")


def test_in_charset():
    assert oc.in_charset("Hello, world! £5")
    assert not oc.in_charset("tab\there")


def test_ctc_steps_needed_counts_repeats():
    assert oc.ctc_steps_needed("abc") == 3
    assert oc.ctc_steps_needed("hello") == 6      # one "ll"
    assert oc.ctc_steps_needed("aaa") == 5
    assert oc.ctc_steps_needed("") == 0


def _one_hot_path(path, n_classes):
    lp = torch.full((len(path), 1, n_classes), -30.0)
    for t, k in enumerate(path):
        lp[t, 0, k] = 0.0
    return lp


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    a, b = oc.encode_text("a")[0], oc.encode_text("b")[0]
    lp = _one_hot_path([a, a, oc.BLANK, a, b, b, oc.BLANK], oc.N_CLASSES)
    assert oc.ctc_greedy_decode(lp, torch.tensor([7])) == ["aab"]


def test_greedy_decode_respects_lengths():
    a, b = oc.encode_text("a")[0], oc.encode_text("b")[0]
    lp = _one_hot_path([a, oc.BLANK, b, b], oc.N_CLASSES)
    assert oc.ctc_greedy_decode(lp, torch.tensor([2])) == ["a"]
