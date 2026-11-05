"""Revive dead routed experts in one MoE layer of a quipu-moe checkpoint by splitting
the busiest experts into the dead slots.

    .venv/Scripts/python scripts/revive_experts.py --config CONFIG --checkpoint IN.pt \
        --tokens SHARD.bin --out OUT.pt [--layer 0] [--threshold 0.1] [--seqs 32]

Why (5 Nov 2026): in layer 0 the router drives some experts' softmax scores to ~1e-8
on every token. Quantile Balancing can still push such an expert into a token's Top-k,
but select_experts weights it by its unbiased score, so it adds ~0 to the output and
gets ~0 gradient: it never recovers. Layer 0 sees little more than the token
embedding, which makes this collapse easy there.

What it does, on the raw checkpoint dicts (memory-mapped; nothing is built at full
size, so it runs on a laptop):
  1. measure: run the model up to `layer` on --seqs random windows of --tokens and
     count each expert's Top-k assignments with the checkpoint's balancing bias.
  2. plan: an expert below `threshold` x its target load is dead. Each dead slot takes
     a copy of the expert that currently has the most load, and that donor's expected
     load is split between its copies, so one hot expert can fill several slots.
  3. apply, for every (dead, donor) pair: the expert's gate / up / down rows and their
     Muon momentum are copied from the donor; the router row and its AdamW moments are
     copied too, plus a little noise on each copy's router row so the balancer can
     tell donor and copies apart. The router has no bias term, so the group's softmax
     mass grows (m copies take about m times the donor's share of the softmax
     denominator); select_experts renormalises the weights over each token's Top-k,
     so the output moves a little, not a lot. Each copy takes its donor's balancing
     bias (re-centred to mean 0), and the layer's health streak and interval counts
     are zeroed.
  4. calibrate: the router alone is far from balanced (the bias did all the work),
     so --calib-updates steps of the trainer's own Quantile Balancing update are run
     on the new layer's scores and the result is stored as the bias.
  5. write OUT (never IN) and OUT.revive.json with the plan and loads before / after.

A training checkpoint ("model", "optimizers", ...) and a model-only milestone (the
state dict at top level) both work; the optimizer part is skipped for a milestone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quipu.config import Config, load_config  # noqa: E402
from quipu.data import read_shard  # noqa: E402
from quipu.model import build_rope_cache  # noqa: E402
from quipu.model_factory import build_model  # noqa: E402
from quipu.moe import QuantileBalancer, select_experts  # noqa: E402
from quipu.optim.groups import build_optimizers  # noqa: E402

COMPILE_PREFIX = "_orig_mod."


def model_state(ck: dict[str, Any]) -> dict[str, torch.Tensor]:
    return ck["model"] if "model" in ck and isinstance(ck["model"], dict) else ck


def _key(sd: dict[str, torch.Tensor], name: str) -> str:
    """The state-dict key for parameter `name`, with or without torch.compile's prefix."""
    for k in (name, COMPILE_PREFIX + name):
        if k in sd:
            return k
    raise KeyError(name)


def build_meta(cfg: Config) -> torch.nn.Module:
    with torch.device("meta"):
        return build_model(cfg.model)


def param_indices(cfg: Config) -> dict[str, tuple[int, int]]:
    """name -> (optimizer number, index in that optimizer's state_dict), in the order
    the trainer's build_optimizers lays the parameters out."""
    model = build_meta(cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, tuple[int, int]] = {}
    for o, opt in enumerate(build_optimizers(model, cfg.train)):
        i = 0
        for group in opt.param_groups:
            for p in group["params"]:
                out[names[id(p)]] = (o, i)
                i += 1
    return out


@torch.no_grad()
def layer_scores(cfg: Config, sd: dict[str, torch.Tensor], layer: int,
                 tokens: np.ndarray, seqs: int, seq_len: int, seed: int) -> torch.Tensor:
    """Unbiased router scores (T, n_experts) of `layer` on `seqs` random windows of
    `tokens`, from the weights in sd."""
    model = build_meta(cfg)
    model.load_state_dict({k.removeprefix(COMPILE_PREFIX): v for k, v in sd.items()},
                          assign=True, strict=False)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq_len, seqs)
    idx = torch.tensor(np.stack([tokens[s:s + seq_len] for s in starts]).astype(np.int64))
    cos, sin = build_rope_cache(cfg.model.context, cfg.model.head_dim, cfg.model.rope_base)
    cos, sin = cos[:seq_len], sin[:seq_len]
    x = model.embed.float()(idx)
    for blk in model.blocks[:layer]:
        blk = blk.float()
        blk.moe.dispatch = "loop"
        x, _ = blk(x, cos, sin)
        x = x.float()
    blk = model.blocks[layer].float()
    x = x + blk.attn(blk.norm1(x), cos, sin).float()
    h = blk.norm2(x).reshape(-1, cfg.model.d_model)
    return torch.softmax(h @ blk.moe.router.weight.float().T, -1)


@torch.no_grad()
def eval_loss(cfg: Config, sd: dict[str, torch.Tensor], tokens: np.ndarray, seqs: int,
              seq_len: int, seed: int) -> float:
    """Mean next-token cross-entropy of the whole model (fp32, loop dispatch, one
    window at a time) on `seqs` random windows of `tokens`."""
    model = build_meta(cfg)
    model.load_state_dict({k.removeprefix(COMPILE_PREFIX): v for k, v in sd.items()},
                          assign=True, strict=False)
    model = model.float().eval()
    model.rope_cos, model.rope_sin = build_rope_cache(cfg.model.context, cfg.model.head_dim,
                                                      cfg.model.rope_base)
    for blk in model.blocks:
        blk.moe.dispatch = "loop"
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq_len - 1, seqs)
    losses = []
    for s in starts:
        row = torch.tensor(tokens[s:s + seq_len + 1].astype(np.int64))
        logits = model(row[None, :-1]).float()
        losses.append(float(torch.nn.functional.cross_entropy(logits[0], row[1:])))
    return sum(losses) / len(losses)


def load_of(scores: torch.Tensor, bias: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k assignments per expert as a fraction of the balanced target."""
    n = scores.shape[1]
    _, ix = select_experts(scores, bias.float(), top_k)
    counts = torch.bincount(ix.reshape(-1), minlength=n).double()
    return counts / (counts.sum() / n)


def measure_load(cfg: Config, sd: dict[str, torch.Tensor], layer: int,
                 tokens: np.ndarray, seqs: int, seq_len: int, seed: int,
                 bias: torch.Tensor | None = None) -> torch.Tensor:
    """load_of(layer_scores(...)) with the checkpoint's balancing bias by default."""
    scores = layer_scores(cfg, sd, layer, tokens, seqs, seq_len, seed)
    if bias is None:
        bias = sd[_key(sd, f"blocks.{layer}.moe.balancer.bias")]
    return load_of(scores, bias, cfg.model.top_k)


def calibrate_bias(scores: torch.Tensor, bias: torch.Tensor, cfg: Config,
                   updates: int, rows: int, seed: int) -> torch.Tensor:
    """`updates` Quantile Balancing steps (the trainer's own update, at the config's
    rate) on random `rows`-row samples of `scores`, starting from `bias`."""
    qb = QuantileBalancer(cfg.model.n_experts, cfg.model.top_k, cfg.model.balance_update_rate)
    qb.bias.copy_(bias.float())
    gen = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        pick = torch.randperm(scores.shape[0], generator=gen)[:rows]
        qb.update(scores[pick])
    return qb.bias.clone()


def plan_splits(load: torch.Tensor, threshold: float) -> list[tuple[int, int]]:
    """(dead, donor) pairs. Dead: load < threshold. Each dead slot goes to the donor
    whose load per copy is highest at that moment; a donor with m copies counts as
    load / m each."""
    load = load.double()
    dead = [int(e) for e in (load < threshold).nonzero().flatten()]
    copies = {int(e): 1 for e in range(load.numel()) if int(e) not in dead}
    if not copies:
        raise ValueError("every expert is dead; nothing to copy from")
    pairs = []
    for d in dead:
        donor = max(copies, key=lambda e: (float(load[e]) / copies[e], -e))
        copies[donor] += 1
        pairs.append((d, donor))
    return pairs


def apply_splits(ck: dict[str, Any], cfg: Config, layer: int,
                 pairs: list[tuple[int, int]], noise: float, seed: int) -> None:
    """Edit ck in place (see the module docstring, step 3)."""
    sd = model_state(ck)
    pre = f"blocks.{layer}.moe."
    names = {w: pre + f"experts.{w}" for w in ("gate", "up", "down")}
    router = pre + "router.weight"
    bias = pre + "balancer.bias"
    gen = torch.Generator().manual_seed(seed)

    def own(k: str) -> torch.Tensor:
        # mmap'd tensors are read-only views of the file: work on a private copy.
        sd[k] = sd[k].clone()
        return sd[k]

    for name in names.values():
        w = own(_key(sd, name))
        for d, s in pairs:
            w[d] = w[s]
    r = own(_key(sd, router))
    groups: dict[int, list[int]] = {}
    for d, s in pairs:
        groups.setdefault(s, [s]).append(d)
        r[d] = r[s]
    row_scale = r.float().norm(dim=1).mean() / math.sqrt(r.shape[1])
    for s, members in groups.items():
        for d in members[1:]:
            r[d] = (r[d].float() + noise * row_scale *
                    torch.randn(r.shape[1], generator=gen)).to(r.dtype)
    b = own(_key(sd, bias))
    for d, s in pairs:
        b[d] = b[s]
    b.sub_(b.mean())

    if "optimizers" in ck:
        idx = param_indices(cfg)
        for name in list(names.values()) + [router]:
            o, i = idx[name]
            st = ck["optimizers"][o]["state"].get(i)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == cfg.model.n_experts:
                    v = v.clone()
                    for d, s in pairs:
                        v[d] = v[s]
                    st[k] = v
    streak = ck.get("moe_health_streak")
    if torch.is_tensor(streak):
        streak = streak.clone()
        streak[layer] = 0
        ck["moe_health_streak"] = streak
    interval = ck.get("moe_interval")
    if interval is not None:
        counts, dropped = (t.clone() for t in interval)
        counts[layer] = 0
        dropped[layer] = 0
        ck["moe_interval"] = (counts, dropped)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokens", required=True, help="a token shard (.bin) to measure load on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="dead below this fraction of the target load (trainer's MOE_HEALTH_LOW)")
    ap.add_argument("--noise", type=float, default=0.3,
                    help="router-row noise on each copy, as a fraction of the mean row scale. "
                         "On step_008129 layer 0: 0.01 left 1 of 23 dead after calibration, "
                         "0.3 left none (max load 1.8x)")
    ap.add_argument("--calib-updates", type=int, default=400)
    ap.add_argument("--calib-rows", type=int, default=16_384)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-seqs", type=int, default=0,
                    help="if > 0: whole-model loss on this many windows before and after "
                         "(same windows; slow on a CPU, ~10 s per 1024-token window at 1B)")
    a = ap.parse_args(argv)

    src, out = Path(a.checkpoint).resolve(), Path(a.out).resolve()
    if src == out:
        sys.exit("--out must differ from --checkpoint")
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    cfg = load_config(a.config)
    if cfg.model.kind != "moe" or not 0 <= a.layer < cfg.model.n_layer:
        sys.exit("need a moe config and 0 <= --layer < n_layer")
    if cfg.model.attnres_blocks and a.layer > 0:
        sys.exit("measure_load runs blocks one after another; with attnres_blocks > 0 "
                 "that is only right for --layer 0")
    a.seq_len = min(a.seq_len, cfg.model.context)
    tokens = read_shard(a.tokens)
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    sd = model_state(ck)

    before = measure_load(cfg, sd, a.layer, tokens, a.seqs, a.seq_len, a.seed)
    pairs = plan_splits(before, a.threshold)
    print(f"layer {a.layer}: {len(pairs)} dead (< {a.threshold:.0%} of target), "
          f"load max {before.max():.2f}x")
    if not pairs:
        print("nothing to revive; no file written")
        return 0
    loss_seed = a.seed + 2
    loss_before = (eval_loss(cfg, sd, tokens, a.loss_seqs, a.seq_len, loss_seed)
                   if a.loss_seqs > 0 else None)
    apply_splits(ck, cfg, a.layer, pairs, a.noise, a.seed)
    bkey = _key(sd, f"blocks.{a.layer}.moe.balancer.bias")
    if sd[bkey].dtype != torch.float32:
        print(f"warning: the balancing bias is stored as {sd[bkey].dtype}; the trainer keeps "
              f"it in fp32 and rounding it changes routing (a milestone export?)")
    scores = layer_scores(cfg, sd, a.layer, tokens, a.seqs, a.seq_len, a.seed)
    sd[bkey] = calibrate_bias(scores, sd[bkey], cfg, a.calib_updates, a.calib_rows,
                              a.seed).to(sd[bkey].dtype)
    after = measure_load(cfg, sd, a.layer, tokens, a.seqs, a.seq_len, a.seed + 1)
    print(f"after (calibrated bias, fresh windows): dead {int((after < a.threshold).sum())}, "
          f"load {after.min():.2f}-{after.max():.2f}x")
    loss_after = None
    if loss_before is not None:
        loss_after = eval_loss(cfg, sd, tokens, a.loss_seqs, a.seq_len, loss_seed)
        print(f"loss on {a.loss_seqs} windows: before {loss_before:.4f}, after "
              f"{loss_after:.4f} ({loss_after - loss_before:+.4f})")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(out)
    report = {"checkpoint": str(src), "layer": a.layer, "threshold": a.threshold,
              "noise": a.noise, "seed": a.seed, "pairs": pairs,
              "calib_updates": a.calib_updates, "calib_rows": a.calib_rows,
              "load_before": [round(x, 4) for x in before.tolist()],
              "load_after": [round(x, 4) for x in after.tolist()],
              "loss_seqs": a.loss_seqs, "loss_before": loss_before, "loss_after": loss_after}
    Path(str(out) + ".revive.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
