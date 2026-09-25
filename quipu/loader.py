"""A position in the token stream, which is checkpoint state.

Shards are concatenated into one logical array and read sequentially. The position
is a token offset into that array, so resuming is exact and crossing a shard
boundary is not a special case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from quipu.data import read_shard


class TokenStream:
    def __init__(self, shard_dir: str | Path, micro_batch: int, context: int) -> None:
        self.shard_dir = Path(shard_dir)
        self.micro_batch = micro_batch
        self.context = context

        paths = sorted(self.shard_dir.glob("shard_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no shards in {self.shard_dir}")
        # Concatenating keeps the boundary logic in one place. At 1.5B tokens this
        # is 3 GB (peak ~6 GB while concatenating), which fits in 31.7 GB of RAM;
        # if it ever does not, this becomes a memmap and nothing else changes.
        self.tokens = np.concatenate([read_shard(p) for p in paths])
        self.position = 0
        # Wrapping means the stream is about to re-serve tokens the model has
        # already seen. That is fine at small scale but must not happen
        # silently, so the trainer can log it instead of the loss curve just
        # looking a little too good.
        self.wraps = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.micro_batch * self.context + 1   # +1 for the shifted target
        if self.position + need > len(self.tokens):
            self.position = 0
            self.wraps += 1
        chunk = self.tokens[self.position : self.position + need].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.micro_batch, self.context)
        y = torch.from_numpy(chunk[1:]).view(self.micro_batch, self.context)
        self.position += self.micro_batch * self.context
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position, "wraps": self.wraps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = state["position"]
        if not isinstance(position, int) or isinstance(position, bool) or not (0 <= position < len(self.tokens)):
            raise ValueError(
                f"corrupted checkpoint: position {position!r} out of range "
                f"[0, {len(self.tokens)})"
            )
        self.position = position
        self.wraps = int(state.get("wraps", 0))
