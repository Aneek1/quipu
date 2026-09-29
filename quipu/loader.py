"""A position in the token stream, which is checkpoint state.

Shards are concatenated into one logical array and read sequentially. The position
is a token offset into that array, so resuming is exact and crossing a shard
boundary is not a special case.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import torch

from quipu.data import shard_token_count


class TokenStream:
    def __init__(self, shard_dir: str | Path, micro_batch: int, context: int) -> None:
        self.shard_dir = Path(shard_dir)
        self.micro_batch = micro_batch
        self.context = context

        paths = sorted(self.shard_dir.glob("shard_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no shards in {self.shard_dir}")
        # At 3B tokens a shard set is 6 GB; loading it into RAM (or worse,
        # np.concatenate-ing copies of it) doesn't scale on this machine. Each
        # shard is memory-mapped instead, so the OS pages in only the bytes a
        # batch actually touches. The shards are logically one array: _offsets
        # holds the cumulative token count before each shard, so a position is
        # still a single flat index and a batch that spans a boundary is just
        # a slice from two (or more) shards concatenated in _read.
        self._shards: list[np.memmap] = []
        offsets = [0]
        for p in paths:
            shard_token_count(p)  # validates; raises ValueError on odd byte count
            self._shards.append(np.memmap(p, dtype="<u2", mode="r"))
            offsets.append(offsets[-1] + len(self._shards[-1]))
        self._offsets = offsets

        self.position = 0
        # Wrapping means the stream is about to re-serve tokens the model has
        # already seen. That is fine at small scale but must not happen
        # silently, so the trainer can log it instead of the loss curve just
        # looking a little too good.
        self.wraps = 0

    def __len__(self) -> int:
        return self._offsets[-1]

    def close(self) -> None:
        """Release the shard file handles.

        Not required: dropping the last reference to a TokenStream lets the
        garbage collector close the underlying mmaps on its own. This exists
        for callers (tests, mainly) that want the handles released
        deterministically, e.g. so a temp directory can be cleaned up right
        after.
        """
        for shard in self._shards:
            mm = getattr(shard, "_mmap", None)
            if mm is not None:
                mm.close()
        self._shards = []

    def _read(self, start: int, count: int) -> np.ndarray:
        end = start + count
        idx = bisect.bisect_right(self._offsets, start) - 1
        pieces = []
        pos = start
        while pos < end:
            shard_start = self._offsets[idx]
            shard_end = self._offsets[idx + 1]
            local_start = pos - shard_start
            local_end = min(end, shard_end) - shard_start
            pieces.append(self._shards[idx][local_start:local_end])
            pos = shard_start + local_end
            idx += 1
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.micro_batch * self.context + 1   # +1 for the shifted target
        if self.position + need > len(self):
            self.position = 0
            self.wraps += 1
        chunk = self._read(self.position, need).astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.micro_batch, self.context)
        y = torch.from_numpy(chunk[1:]).view(self.micro_batch, self.context)
        self.position += self.micro_batch * self.context
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position, "wraps": self.wraps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = state["position"]
        if not isinstance(position, int) or isinstance(position, bool) or not (0 <= position < len(self)):
            raise ValueError(
                f"corrupted checkpoint: position {position!r} out of range "
                f"[0, {len(self)})"
            )
        self.position = position
        self.wraps = int(state.get("wraps", 0))


MASK_SUFFIX = ".mask"
IGNORE_INDEX = -100     # quipu.chat.IGNORE_INDEX; cross_entropy's default ignore_index


def mask_path(shard: str | Path) -> Path:
    """The loss-mask file beside a shard: shard_000.bin -> shard_000.mask."""
    return Path(shard).with_suffix(MASK_SUFFIX)


class MaskedBlockStream:
    """The chat fine-tune's data (train.mode "sft"; scripts/build_chat_data.py):
    shards cut into blocks of exactly `context` tokens, each block one or more whole
    conversations packed with <|endoftext|> separators and padding, with a parallel
    uint8 loss mask (shard_NNN.mask, 1 = the token is a training target: assistant
    content and its <|end|>).

    A row of a batch is one block: x = the block, y = the block shifted by one with
    every target whose mask is 0 set to IGNORE_INDEX, and the last position's target
    ignored (it would be the next block's first token, another conversation). So a
    block never reads across into another, and blocks may be served in any order.
    Same interface as TokenStream: next_batch, state_dict / load_state_dict
    (position counts TOKENS, a multiple of context, so checkpoints look alike),
    wraps, len() in tokens, close().

    Checked at open: every shard has its mask of the same length, a whole number of
    blocks, mask values 0/1, and every block at least one target (a block with none
    would make a micro-batch's mean loss 0/0). When <shard_dir>/../manifest.json
    records a "context", it must equal `context`: blocks built for another context
    would cut conversations in half."""

    def __init__(self, shard_dir: str | Path, micro_batch: int, context: int) -> None:
        import json

        self.shard_dir = Path(shard_dir)
        self.micro_batch = micro_batch
        self.context = context
        paths = sorted(self.shard_dir.glob("shard_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no shards in {self.shard_dir}")
        manifest = self.shard_dir.parent / "manifest.json"
        if manifest.is_file():
            built = json.loads(manifest.read_text(encoding="utf-8")).get("context")
            if built is not None and built != context:
                raise ValueError(f"{self.shard_dir}: the chat data was built for context "
                                 f"{built}, the model's is {context}; rebuild it")
        self._tokens: list[np.memmap] = []
        self._masks: list[np.memmap] = []
        offsets = [0]                      # in blocks
        for p in paths:
            n = shard_token_count(p)
            mp = mask_path(p)
            if not mp.is_file() or mp.stat().st_size != n:
                raise ValueError(f"{p}: no loss mask {mp.name} of {n} bytes beside it")
            if n % context:
                raise ValueError(f"{p}: {n} tokens is not a whole number of "
                                 f"{context}-token blocks")
            tokens = np.memmap(p, dtype="<u2", mode="r")
            mask = np.memmap(mp, dtype=np.uint8, mode="r")
            blocks = mask.reshape(-1, context)
            if blocks.size and int(blocks.max()) > 1:
                raise ValueError(f"{mp}: mask values must be 0 or 1")
            # Targets are positions 1..context-1 of each block.
            empty = np.flatnonzero(blocks[:, 1:].max(axis=1) == 0) if blocks.size else []
            if len(empty):
                raise ValueError(f"{p}: block {int(empty[0])} has no training target")
            self._tokens.append(tokens)
            self._masks.append(mask)
            offsets.append(offsets[-1] + n // context)
        self._offsets = offsets
        if self.n_blocks < micro_batch:
            raise ValueError(f"{self.shard_dir}: {self.n_blocks} blocks, fewer than one "
                             f"micro-batch ({micro_batch})")
        self.position = 0
        self.wraps = 0

    @property
    def n_blocks(self) -> int:
        return self._offsets[-1]

    def __len__(self) -> int:
        return self.n_blocks * self.context

    def close(self) -> None:
        for arr in self._tokens + self._masks:
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()
        self._tokens, self._masks = [], []

    def _block(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        s = bisect.bisect_right(self._offsets, b) - 1
        lo = (b - self._offsets[s]) * self.context
        return (self._tokens[s][lo:lo + self.context], self._masks[s][lo:lo + self.context])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.position // self.context
        if first + self.micro_batch > self.n_blocks:
            first = 0
            self.wraps += 1
        toks = np.stack([self._block(b)[0] for b in range(first, first + self.micro_batch)])
        mask = np.stack([self._block(b)[1] for b in range(first, first + self.micro_batch)])
        x = torch.from_numpy(toks.astype(np.int64))
        y = torch.full_like(x, IGNORE_INDEX)
        keep = torch.from_numpy(mask[:, 1:].astype(bool))
        y[:, :-1] = torch.where(keep, x[:, 1:], IGNORE_INDEX)
        self.position = (first + self.micro_batch) * self.context
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position, "wraps": self.wraps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = state["position"]
        if (not isinstance(position, int) or isinstance(position, bool)
                or not 0 <= position <= len(self) or position % self.context):
            raise ValueError(f"corrupted checkpoint: position {position!r} is not a block "
                             f"boundary in [0, {len(self)}]")
        self.position = position
        self.wraps = int(state.get("wraps", 0))
