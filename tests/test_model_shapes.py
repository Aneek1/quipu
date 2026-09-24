import torch

from quipu.config import ModelConfig, load_config
from quipu.model import Quipu


def tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=32, rope_base=10000.0, norm_eps=1e-6,
    )


def test_forward_returns_logits_over_the_vocabulary():
    cfg = tiny()
    model = Quipu(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (2, 8)))
    assert out.shape == (2, 8, cfg.vocab_size)


def test_embeddings_are_tied():
    # Tied weights save 38.6M parameters at the real scale. If the tie silently
    # breaks, the parameter count test below is the only thing that notices.
    model = Quipu(tiny())
    assert model.lm_head.weight is model.embed.weight


def test_parameter_count_is_exactly_the_documented_figure():
    cfg = load_config("configs/quipu-114m.toml").model
    total = sum(p.numel() for p in Quipu(cfg).parameters())
    assert total == 114_114_048


def test_causal_mask_does_not_leak():
    """The single most important test in this sub-project.

    A mask that lets position t see position t+1 produces a beautiful loss curve
    and a model that cannot generate. Changing the LAST token must leave every
    earlier position bit-identical.
    """
    torch.manual_seed(0)
    cfg = tiny()
    model = Quipu(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        before = model(ids)
    ids2 = ids.clone()
    ids2[0, -1] = (int(ids2[0, -1]) + 1) % cfg.vocab_size
    with torch.no_grad():
        after = model(ids2)

    assert torch.equal(before[0, :-1], after[0, :-1])
    # And the changed position really did change, or the test proves nothing.
    assert not torch.equal(before[0, -1], after[0, -1])


def test_accepts_a_sequence_shorter_than_the_context():
    cfg = tiny()
    assert Quipu(cfg)(torch.randint(0, cfg.vocab_size, (1, 5))).shape == (1, 5, cfg.vocab_size)
