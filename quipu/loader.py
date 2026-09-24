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
        # Concatenating keeps the boundary logic in one place. At 2.5B tokens this
        # is 5 GB, which fits in 31.7 GB of RAM; if it ever does not, this becomes
        # a memmap and nothing else changes.
        self.tokens = np.concatenate([read_shard(p) for p in paths])
        self.position = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.micro_batch * self.context + 1   # +1 for the shifted target
        if self.position + need > len(self.tokens):
            self.position = 0
        chunk = self.tokens[self.position : self.position + need].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).view(self.micro_batch, self.context)
        y = torch.from_numpy(chunk[1:]).view(self.micro_batch, self.context)
        self.position += self.micro_batch * self.context
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = int(state["position"])
