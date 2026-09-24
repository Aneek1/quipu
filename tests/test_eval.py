import numpy as np
import torch

from quipu.config import ModelConfig
from quipu.data import write_shard
from quipu.eval import estimate_loss, generate
from quipu.loader import TokenStream
from quipu.model import Quipu


def tiny() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, d_model=64, n_layer=2, n_head=4, n_kv_head=2,
        ffn_hidden=128, context=32, rope_base=10000.0, norm_eps=1e-6,
    )


def test_estimate_loss_is_near_ln_vocab_at_initialisation(tmp_path):
    # An untrained model is uniform over the vocabulary, so cross-entropy should be
    # about ln(128) = 4.85. A number far from this means the head is mis-initialised.
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    loss = estimate_loss(Quipu(tiny()), stream, batches=5, device="cpu")
    assert 4.0 < loss < 5.6


def test_estimate_loss_leaves_the_model_in_training_mode(tmp_path):
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    model = Quipu(tiny())
    model.train()
    estimate_loss(model, stream, batches=2, device="cpu")
    assert model.training, "estimate_loss must restore training mode"


def test_estimate_loss_does_not_move_the_stream_position(tmp_path):
    # Evaluation must not consume training tokens.
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 4096).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    stream.load_state_dict({"position": 100})
    estimate_loss(Quipu(tiny()), stream, batches=3, device="cpu")
    assert stream.position == 100


def test_estimate_loss_does_not_move_the_stream_wraps(tmp_path):
    # Use a shard small enough that the eval batches themselves would wrap
    # the stream: each batch needs micro_batch * context + 1 = 33 tokens, and
    # the shard below holds only 40, so a second eval batch must wrap. That
    # wrap must be undone along with the position, or the trainer's wrap
    # count silently drifts every time it evaluates.
    write_shard(tmp_path / "shard_000.bin", np.random.randint(0, 128, 40).astype(np.uint16))
    stream = TokenStream(tmp_path, micro_batch=2, context=16)
    stream.load_state_dict({"position": 0, "wraps": 5})
    estimate_loss(Quipu(tiny()), stream, batches=3, device="cpu")
    assert stream.position == 0
    assert stream.wraps == 5, "estimate_loss must restore the wrap count too"


def test_generate_returns_the_requested_number_of_new_tokens():
    model = Quipu(tiny()).eval()
    prompt = torch.randint(0, 128, (1, 4))
    out = generate(model, prompt, max_new_tokens=6, device="cpu")
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


def test_generate_never_exceeds_the_context():
    cfg = tiny()
    model = Quipu(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.context))
    # Asking for more tokens than the context must crop, not crash.
    out = generate(model, prompt, max_new_tokens=5, device="cpu")
    assert out.shape == (1, cfg.context + 5)
