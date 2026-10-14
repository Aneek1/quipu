"""Data-parallel training across processes (torchrun): the few collectives the
trainer needs, and RankStream, which splits one data stream over the ranks.

Every rank holds the whole model and takes an equal share of each optimizer step's
micro-batches. After the backward passes the gradients are all-reduced, so every
rank steps the same optimizer on the same gradients and the replicas stay
identical without ever sending weights (save_checkpoint checks that they did:
params_match). No DistributedDataParallel: the trainer's gradient accumulation,
non-finite skip and SFT target count want the reduction at one known point, once per
step, and a hand-rolled bucketed all-reduce there is all DDP would add.

Launch (see quipu.train): torchrun sets RANK, WORLD_SIZE and LOCAL_RANK, and
init_from_env() joins the process group (NCCL on CUDA, gloo on CPU) and pins the
process to cuda:LOCAL_RANK. Without torchrun (WORLD_SIZE unset or 1) nothing is
initialised and every helper here is a no-op that returns its input, so a
single-process run executes exactly the code it did before this module existed.
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Iterable

import torch
import torch.distributed as tdist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# Gradients are all-reduced in flat buckets of about this many bytes: one collective
# per bucket rather than one per tensor (quipu-moe has ~200 parameter tensors).
BUCKET_BYTES = 256 * 2**20
# How long a collective may wait. Rank 0 evaluates, writes checkpoints and milestones
# while the others wait at a barrier; a multi-GB checkpoint on a slow disk must not
# trip the default (10 minutes for NCCL).
TIMEOUT = datetime.timedelta(minutes=60)


class ReplicaMismatch(RuntimeError):
    """The ranks' weights differ (params_match): they took different steps, so a
    checkpoint of rank 0's copy would not be the run's state. Never expected; it
    means a nondeterministic kernel or a bug, and continuing would train on it."""


def _on() -> bool:
    return tdist.is_available() and tdist.is_initialized()


def rank() -> int:
    return tdist.get_rank() if _on() else 0


def world_size() -> int:
    return tdist.get_world_size() if _on() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0")) if _on() else 0


def is_main() -> bool:
    return rank() == 0


def init_from_env(cuda: bool | None = None) -> int:
    """Join the torchrun process group when WORLD_SIZE > 1; returns the world size.
    `cuda` picks the backend: NCCL (and torch.cuda.set_device(LOCAL_RANK), before
    the group exists, so NCCL binds the right GPU) or gloo; None = CUDA if usable.
    Already initialised: nothing more is done."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or _on():
        return world_size()
    if cuda is None:
        cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    tdist.init_process_group(backend="nccl" if cuda else "gloo", timeout=TIMEOUT)
    return world_size()


def shutdown() -> None:
    if _on():
        tdist.destroy_process_group()


def barrier() -> None:
    if _on() and world_size() > 1:
        tdist.barrier()


_OPS = {"sum": "SUM", "max": "MAX", "min": "MIN"}


def all_reduce_(t: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """In place over the ranks ("sum" | "max" | "min"); returns t. The tensor must be
    on the backend's device (the training device under NCCL)."""
    if world_size() > 1:
        tdist.all_reduce(t, op=getattr(tdist.ReduceOp, _OPS[op]))
    return t


def all_reduce_gradients(params: Iterable[nn.Parameter], op: str = "mean") -> None:
    """Every rank's .grad becomes the sum ("sum") or the mean ("mean") over the
    ranks. A parameter whose gradient is None on some ranks but not on others gets a
    zero gradient on those first (one small all-reduce of a has-gradient mask finds
    them), so the buckets line up; None everywhere stays None."""
    if world_size() == 1:
        return
    if op not in ("sum", "mean"):
        raise ValueError(f"op must be 'sum' or 'mean', got {op!r}")
    params = [p for p in params if p.requires_grad]
    if not params:
        return
    has = torch.tensor([p.grad is not None for p in params], dtype=torch.int32,
                       device=params[0].device)
    all_reduce_(has, "max")
    grads = []
    for p, h in zip(params, has.tolist()):
        if not h:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    world = world_size()
    for bucket in _buckets(grads):
        flat = _flatten_dense_tensors(bucket)
        tdist.all_reduce(flat)
        if op == "mean":
            flat.div_(world)
        for g, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
            g.copy_(synced)


def average_gradients(params: Iterable[nn.Parameter]) -> None:
    """Every rank's .grad becomes the mean over the ranks."""
    all_reduce_gradients(params, op="mean")


def _buckets(tensors: list[torch.Tensor]) -> list[list[torch.Tensor]]:
    """Group by (dtype, device) and cut into runs of about BUCKET_BYTES, in the
    given order (the same on every rank)."""
    groups: dict[tuple[torch.dtype, torch.device], list[list[torch.Tensor]]] = {}
    for t in tensors:
        runs = groups.setdefault((t.dtype, t.device), [[]])
        size = sum(x.numel() * x.element_size() for x in runs[-1])
        if runs[-1] and size + t.numel() * t.element_size() > BUCKET_BYTES:
            runs.append([])
        runs[-1].append(t)
    return [run for runs in groups.values() for run in runs]


def all_gather_cat(t: torch.Tensor) -> torch.Tensor:
    """Every rank's t, concatenated along dim 0 in rank order. The ranks may hold
    different numbers of rows (the other dims must agree): the sizes are gathered
    first and every piece is padded to the longest."""
    world = world_size()
    if world == 1:
        return t
    n = torch.tensor([t.shape[0]], dtype=torch.int64, device=t.device)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    tdist.all_gather(sizes, n)
    sizes = [int(s) for s in sizes]
    longest = max(sizes)
    padded = t.new_zeros((longest, *t.shape[1:]))
    padded[: t.shape[0]] = t
    pieces = [torch.empty_like(padded) for _ in range(world)]
    tdist.all_gather(pieces, padded.contiguous())
    return torch.cat([p[:s] for p, s in zip(pieces, sizes)])


def _tensors(module: nn.Module) -> list[torch.Tensor]:
    """Parameters then buffers, each once (the tied embedding / lm_head is one)."""
    seen: set[int] = set()
    out = []
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


@torch.no_grad()
def broadcast_params(module: nn.Module) -> None:
    """Rank 0's parameters and buffers (the balancer biases are buffers) to every
    rank, in place, so optimizers built on them keep pointing at the same tensors."""
    if world_size() == 1:
        return
    for t in _tensors(module):
        tdist.broadcast(t.data, src=0)


@torch.no_grad()
def params_match(module: nn.Module) -> bool:
    """Whether every rank holds the same parameters and buffers: per tensor, its
    float64 sum and sum of squares, compared by an all-reduce of the max and of the
    min (equal everywhere iff max == min). Cheap (two vectors of 2 x tensors), and a
    real divergence moves the sums. True on one process."""
    if world_size() == 1:
        return True
    tensors = _tensors(module)
    device = tensors[0].device
    sums = []
    for t in tensors:
        x = t.detach().double()
        sums += [x.sum(), (x * x).sum()]
    hi = torch.stack(sums).to(device)
    lo = hi.clone()
    all_reduce_(hi, "max")
    all_reduce_(lo, "min")
    return bool(torch.equal(hi, lo))


class RankStream:
    """One data stream split over the ranks: next_batch() reads `world` batches from
    the inner stream and returns the rank-th. Every rank reads the same batches in
    the same order, so the stream position (and so its checkpoint state, resume and
    wraps) is identical on all of them, and the global batch is exactly the batches
    one process would have read: rank r's j-th micro-batch of a step is the step's
    micro-batch j * world + r. Everything but next_batch is the inner stream's."""

    def __init__(self, stream: Any, rank: int, world: int) -> None:
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} is not in [0, {world})")
        self.stream = stream
        self.rank = rank
        self.world = world

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        mine = None
        for r in range(self.world):
            batch = self.stream.next_batch()
            if r == self.rank:
                mine = batch
        return mine

    def state_dict(self) -> dict[str, int]:
        return self.stream.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.stream.load_state_dict(state)

    def __len__(self) -> int:
        return len(self.stream)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes RankStream lacks: position, wraps, close, ...
        if name == "stream":
            raise AttributeError(name)
        return getattr(self.stream, name)
