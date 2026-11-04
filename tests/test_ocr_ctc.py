"""quipu/ocr_ctc.py: character set, labels, CTC length check, greedy decode."""
import pytest
import torch

from quipu import ocr_ctc as oc
from quipu.vision import ViTConfig

TINY = ViTConfig(d_model=32, n_layer=2, n_head=4, mlp=64)


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
def test_line_ctc_shapes():
    m = oc.LineCTC(TINY)
    images = torch.rand(2, 1, 64, 128)
    mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    lp, lens = m(images, mask)
    assert oc.STEPS_PER_COL == 2
    assert lp.shape == (16, 2, oc.N_CLASSES)
    assert lp.dtype == torch.float32
    assert lens.tolist() == [16, 10]
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(16, 2))


def test_ctc_loss_goes_down_when_overfitting_two_lines():
    torch.manual_seed(0)
    m = oc.LineCTC(TINY)
    images = torch.rand(2, 1, 64, 128)
    mask = torch.ones(2, 8, dtype=torch.bool)
    texts = ["ab", "cab"]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        lp, lens = m(images, mask)
        loss = oc.ctc_loss(lp, lens, texts)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first
    lp, lens = m(images, mask)
    assert oc.ctc_greedy_decode(lp, lens) == texts


def test_ctc_loss_raises_when_text_cannot_fit():
    lp = torch.randn(5, 2, oc.N_CLASSES).log_softmax(-1)
    with pytest.raises(ValueError, match=r"item 1.*needs 6"):
        oc.ctc_loss(lp, torch.tensor([5, 5]), ["abc", "hello"])


def test_ctc_loss_on_padded_batch_is_finite():
    torch.manual_seed(0)
    m = oc.LineCTC(TINY)
    images = torch.rand(2, 1, 64, 128)
    mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    lp, lens = m(images, mask)
    loss = oc.ctc_loss(lp, lens, ["ab", "cab"])
    assert torch.isfinite(loss) and loss.item() > 0


def test_checkpoint_compat_fields_round_trip_and_mismatch_is_refused():
    ck = {"model": {}, **oc.ckpt_compat()}
    oc.check_ckpt_compat(ck, "x.pt")                       # does not raise
    with pytest.raises(ValueError, match="charset"):
        oc.check_ckpt_compat({**ck, "charset_sha256": "0" * 64}, "x.pt")
    with pytest.raises(ValueError, match="steps"):
        oc.check_ckpt_compat({**ck, "steps_per_col": 1}, "x.pt")
    with pytest.raises(ValueError, match="charset"):
        oc.check_ckpt_compat({"model": {}}, "old.pt")      # written before the fields existed
