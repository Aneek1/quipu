"""Multi-process data parallel training (quipu.dist; quipu.train under torchrun).

CPU only: gloo process groups of two processes started with torch.multiprocessing
(spawn, the only start method on Windows), MASTER_ADDR 127.0.0.1 and a free port.
Every worker is a module-level function so spawn can pickle it, and every one tears
its process group down before it returns.

The key tests: two processes splitting one global batch end on the same weights,
balancer biases, stream position and logged losses as one process taking the whole
batch, for quipu-moe (Muon), the dense model and the SFT (whose loss divides by the
step's GLOBAL count of assistant targets). The single-process references run in the
pytest process itself, world size 1, through exactly the code a laptop run uses.
"""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import quipu.train as train_mod
from quipu import dist
from quipu.config import load_config
from quipu.data import write_mask, write_shard
from quipu.dist import RankStream
from quipu.loader import TokenStream
from quipu.train import LATEST, Trainer, UsageError

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "quipu-moe-smoke.toml"
CONTEXT = 32
MICRO = 2
GLOBAL_ACCUM = 4
STEPS = 4
WORLD = 2


# ---- process plumbing --------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(fn, *args) -> None:
    mp.spawn(fn, args=(_free_port(), *args), nprocs=WORLD, join=True)


def _init(rank: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank))
    dist.init_from_env(cuda=False)


# ---- configs and data ------------------------------------------------------------------

DENSE = {"kind": "dense", "n_experts": 0, "top_k": 0, "expert_hidden": 0,
         "shared_experts": 0, "shared_hidden": 0, "attnres_blocks": 0,
         "activation": "swiglu", "ffn_hidden": 128}


def _cfg(root: Path, name: str, scenario: str, **train):
    model = {"context": CONTEXT, **(DENSE if scenario == "dense" else {})}
    sft = scenario == "sft"
    over = {
        "model": model,
        "data": {"shard_dir": str(root / ("chat" if sft else "shards"))},
        "train": {"ckpt_dir": str(root / "ckpt" / name), "micro_batch": MICRO,
                  "batch_tokens": MICRO * CONTEXT * GLOBAL_ACCUM,
                  "total_tokens": MICRO * CONTEXT * GLOBAL_ACCUM * STEPS,
                  "warmup_steps": 1, "eval_every": 2, "ckpt_every": 2, "eval_batches": 1,
                  "milestones": [2], "optimizer": "adamw" if sft else "muon",
                  "mode": "sft" if sft else "pretrain", **train},
    }
    return load_config(SMOKE, over)


def _trainer(root: Path, name: str, scenario: str, resume: bool = False, **train) -> Trainer:
    cfg = _cfg(root, name, scenario, **train)
    shard = Path(cfg.data.shard_dir)
    val = shard / "val" if scenario != "sft" else None
    return Trainer(model_cfg=cfg.model, train_cfg=cfg.train, shard_dir=shard / "train",
                   val_dir=val, device="cpu", run_dir=root / "runs", run_id=name,
                   resume=resume)


def _write_data(root: Path) -> None:
    rng = np.random.RandomState(0)
    write_shard(root / "shards" / "train" / "shard_000.bin",
                rng.randint(0, 512, 6000).astype(np.uint16))
    write_shard(root / "shards" / "val" / "shard_000.bin",
                rng.randint(0, 512, 2000).astype(np.uint16))
    # Chat blocks: random tokens, a third of them the <|endoftext|> padding (left out
    # of the balance update, so the ranks stash different numbers of score rows), and
    # a random loss mask with a very uneven number of targets per block, so a mean
    # per rank would differ from the mean over the step.
    blocks = 48
    toks = rng.randint(0, 512, (blocks, CONTEXT))
    toks[rng.rand(blocks, CONTEXT) < 0.3] = train_mod.SFT_PAD_ID
    mask = (rng.rand(blocks, CONTEXT) < rng.choice([0.05, 0.9], blocks)[:, None])
    mask[:, 1] = True                                    # every block has a target
    write_shard(root / "chat" / "train" / "shard_000.bin",
                toks.reshape(-1).astype(np.uint16))
    write_mask(root / "chat" / "train" / "shard_000.mask",
               mask.reshape(-1).astype(np.uint8))


def _final(root: Path, name: str) -> dict:
    ckpt = root / "ckpt" / name
    pointer = torch.load(ckpt / LATEST, weights_only=False)
    return torch.load(ckpt / pointer["file"], map_location="cpu", weights_only=False)


def _losses(root: Path, name: str, key: str = "train_loss") -> list[float]:
    log = json.loads((root / "runs" / f"{name}.json").read_text(encoding="utf-8"))
    return [s[key] for s in log["steps"]]


# ---- the two-process scenarios ---------------------------------------------------------

def _record(root: Path, rank: int, name: str, outcome: str) -> None:
    out = root / "outcomes"
    out.mkdir(exist_ok=True)
    (out / f"{name}.{rank}").write_text(outcome, encoding="utf-8")


def _scenarios_worker(rank: int, port: int, root: str) -> None:
    root = Path(root)
    _init(rank, port)
    try:
        for scenario in ("moe", "dense", "sft"):
            _trainer(root, f"{scenario}-2", scenario).run()

        # Resume: two steps, a checkpoint, a fresh pair of processes' worth of
        # trainers resuming it and running to the end.
        tr = _trainer(root, "moe-resume", "moe")
        tr.train_step()
        tr.train_step()
        tr.save_checkpoint()
        del tr
        tr = _trainer(root, "moe-resume", "moe", resume=True)
        tr.resume_from_latest()
        tr.run()

        # The spend backstop is rank 0's clock alone: rank 0 sees it spent from step
        # 3 on, rank 1's own clock never gets there. Both ranks must stop together.
        saved_clock = train_mod._clock
        holder: dict = {}
        if rank == 0:
            train_mod._clock = lambda: 1e9 if "tr" in holder and holder["tr"].step >= 3 else 0.0
        holder["tr"] = tr = _trainer(root, "budget", "moe", budget_usd=1.0, usd_per_hour=1.0,
                                     ckpt_every=100)
        try:
            tr.run()
            _record(root, rank, "budget", "completed")
        except train_mod.BudgetStop:
            _record(root, rank, "budget", f"BudgetStop {tr.step}")
        finally:
            train_mod._clock = saved_clock
            train_mod._stopping = False

        # A stop signal reaching only rank 1 (the handler sets the flag under
        # torchrun): both ranks take the interrupt path at the next step.
        tr = _trainer(root, "signal", "moe", ckpt_every=100)
        if rank == 1:
            train_mod._stop_requested = True
        try:
            tr.run()
            _record(root, rank, "signal", "completed")
        except KeyboardInterrupt:
            _record(root, rank, "signal", f"KeyboardInterrupt {tr.step}")
        finally:
            train_mod._stop_requested = False

        # Replicas that drifted apart: the checkpoint refuses.
        tr = _trainer(root, "drift", "moe")
        if rank == 1:
            with torch.no_grad():
                next(tr.model.parameters()).add_(1e-3)
        try:
            tr.save_checkpoint()
            _record(root, rank, "drift", "saved")
        except dist.ReplicaMismatch:
            _record(root, rank, "drift", "ReplicaMismatch")

        # A global grad_accum that does not split over the ranks.
        try:
            _trainer(root, "odd", "moe", batch_tokens=MICRO * CONTEXT * 3)
            _record(root, rank, "odd", "built")
        except UsageError as exc:
            _record(root, rank, "odd", f"UsageError {exc}")
    finally:
        dist.shutdown()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The two-process scenarios (one spawn) and the single-process references."""
    root = tmp_path_factory.mktemp("dist")
    _write_data(root)
    _spawn(_scenarios_worker, str(root))
    for scenario in ("moe", "dense", "sft"):
        _trainer(root, f"{scenario}-1", scenario).run()
    return root


def _outcome(root: Path, name: str) -> list[str]:
    return [(root / "outcomes" / f"{name}.{r}").read_text(encoding="utf-8")
            for r in range(WORLD)]


def _assert_same_state(a: dict, b: dict) -> None:
    assert a["step"] == b["step"] == STEPS
    assert a["stream"] == b["stream"]
    assert a["model"].keys() == b["model"].keys()
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k], rtol=1e-5, atol=1e-6,
                                   msg=lambda m, k=k: f"{k}: {m}")


@pytest.mark.parametrize("scenario", ["moe", "dense", "sft"])
def test_two_processes_train_exactly_like_one(runs, scenario):
    one, two = _final(runs, f"{scenario}-1"), _final(runs, f"{scenario}-2")
    _assert_same_state(one, two)
    if scenario == "moe":
        # The balancer biases are buffers in the state_dict, compared above; they
        # must also have moved, or the comparison would prove nothing about them.
        biases = [v for k, v in two["model"].items() if k.endswith("balancer.bias")]
        assert biases and all(b.abs().sum() > 0 for b in biases)
        torch.testing.assert_close(one["moe_health_streak"], two["moe_health_streak"])
        # The expert counts are summed over the ranks: the logged load statistics
        # (every eval_every steps) are the single process's.
        logs = [json.loads((runs / "runs" / f"moe-{n}.json").read_text(encoding="utf-8"))
                for n in (1, 2)]
        assert [m["step"] for m in logs[1]["moe"]] == [2, 4]
        assert logs[1]["moe"] == logs[0]["moe"]
    # The logged gradient norm is taken before clipping: with AdamW (the SFT's), a
    # gradient off by a constant factor (averaged where it must be summed) would
    # give nearly the same weights, but not the same norm.
    for key in ("train_loss", "grad_norm"):
        assert _losses(runs, f"{scenario}-2", key) == pytest.approx(
            _losses(runs, f"{scenario}-1", key), rel=1e-5), key


def test_only_rank_0_writes_and_evaluates(runs):
    log = json.loads((runs / "runs" / "moe-2.json").read_text(encoding="utf-8"))
    assert [s["step"] for s in log["steps"]] == list(range(1, STEPS + 1))
    assert [e["step"] for e in log["evals"]] == [2, 4]
    assert log["status"] == "completed"
    assert (runs / "ckpt" / "moe-2" / "milestones" / "step_000002.pt").is_file()


def test_a_two_process_resume_continues_like_the_uninterrupted_run(runs):
    _assert_same_state(_final(runs, "moe-1"), _final(runs, "moe-resume"))


def test_rank_0s_budget_stop_stops_every_rank_at_the_same_step(runs):
    assert _outcome(runs, "budget") == ["BudgetStop 3", "BudgetStop 3"]
    assert _final(runs, "budget")["step"] == 3
    log = json.loads((runs / "runs" / "budget.json").read_text(encoding="utf-8"))
    assert log["status"] == "stopped_budget"


def test_a_signal_on_one_rank_interrupts_every_rank(runs):
    assert _outcome(runs, "signal") == ["KeyboardInterrupt 0", "KeyboardInterrupt 0"]
    assert _final(runs, "signal")["step"] == 0
    log = json.loads((runs / "runs" / "signal.json").read_text(encoding="utf-8"))
    assert log["status"] == "interrupted"


def test_a_checkpoint_of_drifted_replicas_is_refused(runs):
    assert _outcome(runs, "drift") == ["ReplicaMismatch", "ReplicaMismatch"]
    assert not (runs / "ckpt" / "drift" / LATEST).exists()


def test_a_grad_accum_that_does_not_split_over_the_ranks_is_a_usage_error(runs):
    outcomes = _outcome(runs, "odd")
    assert all(o.startswith("UsageError") and "grad_accum 3" in o for o in outcomes), outcomes


# ---- the helpers ---------------------------------------------------------------------------

def test_every_helper_is_a_no_op_in_one_process(tmp_path):
    assert (dist.rank(), dist.world_size(), dist.is_main()) == (0, 1, True)
    lin = nn.Linear(3, 2)
    lin.weight.grad = torch.full_like(lin.weight, 2.0)
    before = lin.weight.grad.clone()
    dist.average_gradients(lin.parameters())
    dist.all_reduce_gradients(lin.parameters(), op="sum")
    assert torch.equal(lin.weight.grad, before) and lin.bias.grad is None
    state = {k: v.clone() for k, v in lin.state_dict().items()}
    dist.broadcast_params(lin)
    assert all(torch.equal(v, lin.state_dict()[k]) for k, v in state.items())
    assert dist.params_match(lin)
    t = torch.arange(3.0)
    assert dist.all_gather_cat(t) is t
    assert dist.all_reduce_(t, "sum") is t and t.tolist() == [0.0, 1.0, 2.0]
    dist.barrier()
    assert dist.init_from_env(cuda=False) == 1          # no torchrun env: one process


def test_rank_stream_takes_every_world_th_batch_and_shares_the_position(tmp_path):
    write_shard(tmp_path / "d" / "shard_000.bin", np.arange(100, dtype=np.uint16))
    ref = TokenStream(tmp_path / "d", 1, 4)
    want = [ref.next_batch()[0] for _ in range(6)]
    for rank in range(3):
        s = RankStream(TokenStream(tmp_path / "d", 1, 4), rank, 3)
        got = [s.next_batch()[0] for _ in range(2)]
        assert [g.tolist() for g in got] == [want[rank].tolist(), want[3 + rank].tolist()]
        assert s.state_dict() == ref.state_dict() and s.position == ref.position
        assert len(s) == len(ref) and s.wraps == 0
        s.load_state_dict({"position": 8, "wraps": 1})
        assert s.next_batch()[0].tolist() == [[8 + 4 * rank + i for i in range(4)]]
        assert s.wraps == 1
    with pytest.raises(ValueError):
        RankStream(ref, 3, 3)


def _helpers_worker(rank: int, port: int, out: str) -> None:
    _init(rank, port)
    try:
        res: dict = {"rank": dist.rank(), "world": dist.world_size(), "main": dist.is_main()}
        # Gradients: fp32 and fp64 buckets, and a gradient only rank 1 has (rank 0
        # sees zeros for it, as DDP would).
        a = nn.Parameter(torch.zeros(3))
        b = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        c = nn.Parameter(torch.zeros(2))
        d = nn.Parameter(torch.zeros(1))                 # no gradient anywhere
        a.grad = torch.full((3,), float(rank + 1))
        b.grad = torch.full((2,), float(10 * (rank + 1)), dtype=torch.float64)
        if rank == 1:
            c.grad = torch.full((2,), 4.0)
        dist.average_gradients([a, b, c, d])
        res["mean"] = [a.grad.tolist(), b.grad.tolist(), c.grad.tolist(), d.grad]
        a.grad = torch.full((3,), float(rank + 1))
        dist.all_reduce_gradients([a], op="sum")
        res["sum"] = a.grad.tolist()
        # Variable-length gather, in rank order.
        res["gather"] = dist.all_gather_cat(torch.full((rank + 1, 2), float(rank))).tolist()
        # Parameters and buffers from rank 0; the checksum catches a difference.
        torch.manual_seed(rank)
        bn = nn.BatchNorm1d(3)
        with torch.no_grad():
            bn.weight.normal_()
        bn.running_mean.fill_(float(rank))
        res["match_before"] = dist.params_match(bn)
        dist.broadcast_params(bn)
        res["match_after"] = dist.params_match(bn)
        res["bn"] = [bn.weight.tolist(), bn.running_mean.tolist()]
        Path(f"{out}.{rank}").write_text(json.dumps(res), encoding="utf-8")
    finally:
        dist.shutdown()


def test_the_helpers_across_two_processes(tmp_path):
    out = tmp_path / "helpers"
    _spawn(_helpers_worker, str(out))
    res = [json.loads(Path(f"{out}.{r}").read_text(encoding="utf-8")) for r in range(WORLD)]
    assert [(r["rank"], r["world"], r["main"]) for r in res] == [(0, 2, True), (1, 2, False)]
    for r in res:
        assert r["mean"] == [[1.5] * 3, [15.0] * 2, [2.0] * 2, None]
        assert r["sum"] == [3.0] * 3
        assert r["gather"] == [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
        assert r["match_before"] is False and r["match_after"] is True
    assert res[0]["bn"] == res[1]["bn"] and res[0]["bn"][1] == [0.0] * 3


def test_the_ranks_together_stash_the_balance_rows_one_process_would():
    # A step of 4 micro-batches of 12 rows, capped at 3 rows each (stride 4): one
    # process stashes micro-batch i's rows from offset i % 4; ranks 0 and 1 of two,
    # passing the step-wide index j * 2 + rank, stash exactly the same rows.
    from quipu.moe import MoELayer
    cfg = _cfg(Path("."), "x", "moe").model
    batches = [torch.arange(12 * 8, dtype=torch.float32).view(12, 8) + 1000 * i
               for i in range(4)]

    def stash(layer, items):
        for i, b in items:
            layer._last_scores = b
            layer.accumulate_balance_scores(3, index=i)
        return torch.cat(layer._step_scores)

    one = MoELayer(cfg)
    for b in batches:                                   # the default index
        one._last_scores = b
        one.accumulate_balance_scores(3)
    ranks = [stash(MoELayer(cfg), [(j * 2 + r, batches[j * 2 + r]) for j in range(2)])
             for r in range(2)]
    key = lambda t: sorted(map(tuple, t.tolist()))
    assert key(torch.cat(ranks)) == key(torch.cat(one._step_scores))
