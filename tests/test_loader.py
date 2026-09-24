import numpy as np
import torch

from quipu.data import write_shard
from quipu.loader import TokenStream


def make_shards(tmp_path, n_shards=3, per_shard=1000):
    for i in range(n_shards):
        start = i * per_shard
        write_shard(
            tmp_path / f"shard_{i:03d}.bin",
            np.arange(start, start + per_shard, dtype=np.uint16),
        )
    return tmp_path


def test_yields_inputs_and_targets_offset_by_one(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    x, y = stream.next_batch()
    assert x.shape == (2, 8) and y.shape == (2, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_position_advances(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    assert stream.position == 0
    stream.next_batch()
    # Each batch consumes micro_batch x context tokens, plus the one-token lookahead.
    assert stream.position == 2 * 8


def test_state_round_trip_resumes_the_same_batch(tmp_path):
    a = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    for _ in range(5):
        a.next_batch()
    state = a.state_dict()
    expected_x, expected_y = a.next_batch()

    b = TokenStream(tmp_path, micro_batch=2, context=8)
    b.load_state_dict(state)
    got_x, got_y = b.next_batch()

    assert torch.equal(expected_x, got_x)
    assert torch.equal(expected_y, got_y)


def test_wraps_around_at_the_end_of_the_data(tmp_path):
    stream = TokenStream(make_shards(tmp_path, n_shards=1, per_shard=64),
                         micro_batch=2, context=8)
    for _ in range(20):          # far past the 64 tokens available
        x, _ = stream.next_batch()
        assert x.shape == (2, 8)


def test_crosses_a_shard_boundary_without_a_gap(tmp_path):
    # Shards hold 0..999, 1000..1999, 2000..2999. Reading across the join must be
    # contiguous, or the model sees a discontinuity it will happily learn.
    stream = TokenStream(make_shards(tmp_path), micro_batch=1, context=4)
    stream.load_state_dict({"position": 998})
    x, _ = stream.next_batch()
    assert x[0].tolist() == [998, 999, 1000, 1001]


def test_ignores_leftover_tmp_files_from_atomic_writes(tmp_path):
    # write_shard's atomic-write path leaves a shard_*.bin.tmp behind if a
    # process is killed mid-write. The loader's glob must never pick it up.
    make_shards(tmp_path)
    (tmp_path / "shard_003.bin.tmp").write_bytes(b"not a real shard, garbage bytes")

    stream = TokenStream(tmp_path, micro_batch=2, context=8)
    assert len(stream) == 3000
