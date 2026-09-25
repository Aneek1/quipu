import ctypes
import gc
import sys

import numpy as np
import pytest
import torch

from quipu.data import write_shard
from quipu.loader import TokenStream


def _rss_bytes() -> int:
    """Current process resident set size, in bytes.

    No psutil in this environment; on Windows the reliable way to get RSS
    without a third-party dependency is GetProcessMemoryInfo via ctypes.
    """
    if sys.platform != "win32":
        raise OSError("no RSS measurement implemented for this platform")

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo failed, error {ctypes.get_last_error()}")
    return counters.WorkingSetSize


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


def test_wraps_increments_exactly_once_when_a_batch_runs_past_the_end(tmp_path):
    # 64 tokens, need = micro_batch*context + 1 = 17 per batch, position steps by 16.
    # 0 -> 16 -> 32 -> 48 -> (48+17=65 > 64) wrap to 0 -> 16.
    stream = TokenStream(make_shards(tmp_path, n_shards=1, per_shard=64),
                         micro_batch=2, context=8)
    for _ in range(3):
        stream.next_batch()
        assert stream.wraps == 0
    stream.next_batch()
    assert stream.wraps == 1
    stream.next_batch()
    assert stream.wraps == 1


def test_wraps_round_trips_through_state_dict(tmp_path):
    a = TokenStream(make_shards(tmp_path, n_shards=1, per_shard=64),
                    micro_batch=2, context=8)
    for _ in range(4):
        a.next_batch()
    assert a.wraps == 1
    state = a.state_dict()
    assert state["wraps"] == 1

    b = TokenStream(tmp_path, micro_batch=2, context=8)
    b.load_state_dict(state)
    assert b.wraps == 1


def test_load_state_dict_rejects_position_at_or_past_the_end(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    with pytest.raises(ValueError):
        stream.load_state_dict({"position": len(stream)})


def test_load_state_dict_rejects_negative_position(tmp_path):
    stream = TokenStream(make_shards(tmp_path), micro_batch=2, context=8)
    with pytest.raises(ValueError):
        stream.load_state_dict({"position": -1})


def test_ignores_leftover_tmp_files_from_atomic_writes(tmp_path):
    # write_shard's atomic-write path leaves a shard_*.bin.tmp behind if a
    # process is killed mid-write. The loader's glob must never pick it up.
    make_shards(tmp_path)
    (tmp_path / "shard_003.bin.tmp").write_bytes(b"not a real shard, garbage bytes")

    stream = TokenStream(tmp_path, micro_batch=2, context=8)
    assert len(stream) == 3000


def test_crosses_three_shards_without_a_gap(tmp_path):
    # Three 3-token shards: 0,1,2 | 3,4,5 | 6,7,8. A 7-token batch starting at
    # position 2 pulls the last token of shard 0, all of shard 1, and the
    # first two tokens of shard 2 -- three shards in one read.
    make_shards(tmp_path, n_shards=3, per_shard=3)
    stream = TokenStream(tmp_path, micro_batch=1, context=6)
    stream.load_state_dict({"position": 2})
    x, y = stream.next_batch()
    assert x[0].tolist() == [2, 3, 4, 5, 6, 7]
    assert y[0].tolist() == [3, 4, 5, 6, 7, 8]


def test_opening_large_shard_set_does_not_load_it_into_rss(tmp_path):
    # ~200 MB across 4 shards. If the loader still concatenated shards into
    # one in-RAM array, constructing TokenStream alone would grow RSS by
    # roughly that much; memmapping should barely move it.
    n_shards = 4
    per_shard = 25_000_000  # 50 MB/shard * 4 = 200 MB total
    for i in range(n_shards):
        write_shard(
            tmp_path / f"shard_{i:03d}.bin",
            np.zeros(per_shard, dtype=np.uint16),
        )
    gc.collect()

    try:
        rss_before = _rss_bytes()
    except OSError as exc:
        pytest.skip(f"no reliable RSS measurement available: {exc}")

    stream = TokenStream(tmp_path, micro_batch=2, context=8)
    rss_after = _rss_bytes()

    assert len(stream) == n_shards * per_shard
    grew_by = rss_after - rss_before
    assert grew_by < 50 * 1024 * 1024, f"RSS grew by {grew_by / 1024 / 1024:.1f} MB opening shards"
    stream.close()
