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
