import numpy as np
import pytest

from quipu.data import read_shard, tokenize_documents, write_shard
from quipu.tokenizer import Tokenizer


def test_shard_round_trips_exactly(tmp_path):
    tokens = np.array([1, 2, 3, 50256, 4, 5], dtype=np.uint16)
    path = tmp_path / "shard_000.bin"
    write_shard(path, tokens)
    assert np.array_equal(read_shard(path), tokens)


def test_shard_reads_back_as_uint16(tmp_path):
    path = tmp_path / "shard_000.bin"
    write_shard(path, np.array([7, 8, 9], dtype=np.uint16))
    assert read_shard(path).dtype == np.uint16


def test_shard_length_is_exact(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "shard_000.bin"
    write_shard(path, tokens)
    assert len(read_shard(path)) == 1000


def test_write_rejects_the_wrong_dtype(tmp_path):
    # int32 tokens would write twice the bytes and every later read would be garbage.
    with pytest.raises(ValueError, match="uint16"):
        write_shard(tmp_path / "x.bin", np.array([1, 2, 3], dtype=np.int32))


def test_documents_are_separated_by_eot():
    tok = Tokenizer()
    out = tokenize_documents(["hello", "world"], tok)
    # Each document is followed by exactly one EOT.
    assert out.tolist().count(tok.eot) == 2
    assert out[-1] == tok.eot


def test_tokenized_output_is_uint16():
    out = tokenize_documents(["hello"], Tokenizer())
    assert out.dtype == np.uint16


def test_empty_documents_are_skipped():
    tok = Tokenizer()
    # A blank document would contribute a bare EOT and teach the model that EOT
    # follows EOT, which is not a pattern in the corpus.
    out = tokenize_documents(["hello", "", "   ", "world"], tok)
    assert out.tolist().count(tok.eot) == 2


def test_shard_is_raw_little_endian_no_header(tmp_path):
    # Pins the on-disk format: two bytes per token, little-endian, nothing else.
    # Task 11 writes ~2.5B tokens against this exact contract.
    path = tmp_path / "shard_000.bin"
    write_shard(path, np.array([1, 256], dtype=np.uint16))
    assert path.read_bytes() == b"\x01\x00\x00\x01"
