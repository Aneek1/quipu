"""Expert routing frequency per layer, per expert, per language (spec 7: expert
specialisation) for a quipu-moe checkpoint.

    python -m uv run python scripts/expert_usage.py --config results/moe/run_config.toml \
        [--checkpoint latest|PATH] [--out results/experts] [--batches 20] [--device auto]

For every evaluation split of the shard set (quipu.evalsets.eval_splits: English val,
each val_lang/<language>, code_val) the model reads the first --batches batches with
loop dispatch (nothing dropped, nothing batch-dependent) and every routed assignment
is counted from the layers' MoEStats. frequency = assignments to the expert / (tokens
x top_k), so each layer's row sums to 1 and the balanced target is 1 / n_experts.

Flags, per split and layer (the same bounds as the trainer's health alert): an expert
is "dead" below 10% of the target and "overloaded" above 300% of it.

Writes (atomically) to --out:
    usage.json   counts, frequencies and flags per split
    summary.md   flags per split, then one table per split: one row per expert, one
                 column per layer, the expert's share of that layer's assignments (%)
    heatmap.png  one panel per split, layers x experts, colour = log2(frequency /
                 target): blue under-used, grey on target, red over-used (clipped at
                 1/4x and 4x)

code_val mixes programming languages (the shards carry no per-document label), so
code is one split here; the Python vs JavaScript split of spec 7 needs labelled
code validation shards, which the shard builder does not write.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quipu import evalsets  # noqa: E402
from quipu.config import Config, load_config  # noqa: E402
from quipu.eval import loop_dispatch  # noqa: E402
from quipu.fsio import write_text_atomic  # noqa: E402

DEAD_BELOW = 0.10       # x target
OVERLOADED_ABOVE = 3.0  # x target
CLIP = 2.0              # heatmap: log2 ratio clipped to [-2, 2] (1/4x .. 4x)

# Diverging pair (dataviz reference palette): blue <-> red, neutral grey midpoint.
UNDER, MID, OVER = "#2a78d6", "#f0efec", "#e34948"
SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1e", "#6b6b66"


@torch.no_grad()
def count_usage(model: nn.Module, batches) -> tuple[torch.Tensor, int]:
    """(counts [n_layer, n_experts] int64, tokens) over the batches, loop dispatch.
    Each token contributes top_k assignments per layer."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    counts = None
    tokens = 0
    try:
        with loop_dispatch(model):
            for x, _ in batches:
                model(x.to(device))
                layer = torch.stack([s.counts.to("cpu", torch.int64) for s in model.last_stats])
                counts = layer if counts is None else counts + layer
                tokens += x.numel()
    finally:
        if was_training:
            model.train()
    if counts is None:
        raise ValueError("count_usage: no batches")
    return counts, tokens


def analyse(counts: torch.Tensor, tokens: int, top_k: int) -> dict:
    """Frequencies, ratio to target and dead / overloaded flags for one split."""
    n_layer, n = counts.shape
    freq = counts.double() / (tokens * top_k)
    ratio = freq * n
    flags = {"dead": [], "overloaded": []}
    for l in range(n_layer):
        for e in range(n):
            r = float(ratio[l, e])
            if r < DEAD_BELOW:
                flags["dead"].append({"layer": l, "expert": e, "x_target": round(r, 4)})
            elif r > OVERLOADED_ABOVE:
                flags["overloaded"].append({"layer": l, "expert": e, "x_target": round(r, 4)})
    return {"tokens": tokens, "counts": counts.tolist(), "frequency": freq.tolist(),
            "x_target": ratio.tolist(), **flags}


def measure(model: nn.Module, cfg: Config, batches: int) -> dict[str, dict]:
    out = {}
    for label, split in evalsets.eval_splits(cfg.data.shard_dir).items():
        data = evalsets.split_batches(split, cfg.train.micro_batch, cfg.model.context, batches)
        counts, tokens = count_usage(model, data)
        out[label] = analyse(counts, tokens, cfg.model.top_k)
    if not out:
        raise FileNotFoundError(f"no evaluation splits under {cfg.data.shard_dir}")
    return out


def summary_md(usage: dict[str, dict], cfg: Config, checkpoint: str) -> str:
    m = cfg.model
    lines = [
        f"# Expert usage: {cfg.name}", "",
        f"Checkpoint `{checkpoint}`; {m.n_layer} layers x {m.n_experts} routed experts, "
        f"top-{m.top_k}. Share of each layer's routed assignments (loop dispatch); the "
        f"balanced target is {100 / m.n_experts:.2f}% per expert. Dead: < "
        f"{DEAD_BELOW:.0%} of target; overloaded: > {OVERLOADED_ABOVE:.0%} of target.",
        "", "## Flags", "",
        "| split | tokens | dead | overloaded |", "|---|---:|---|---|",
    ]

    def fmt(items):
        if not items:
            return "none"
        return ", ".join(f"L{d['layer']}/E{d['expert']} ({d['x_target']:.2f}x)" for d in items)

    for label, u in usage.items():
        lines.append(f"| {evalsets.LANGUAGES.get(label, label)} | {u['tokens']:,} | "
                     f"{fmt(u['dead'])} | {fmt(u['overloaded'])} |")
    for label, u in usage.items():
        freq = u["frequency"]
        lines += ["", f"## {evalsets.LANGUAGES.get(label, label)} (`{label}`)", "",
                  "| expert | " + " | ".join(f"L{l}" for l in range(m.n_layer)) + " |",
                  "|---:|" + "---:|" * m.n_layer]
        for e in range(m.n_experts):
            lines.append(f"| {e} | " + " | ".join(f"{100 * freq[l][e]:.2f}"
                                                  for l in range(m.n_layer)) + " |")
    return "\n".join(lines) + "\n"


def heatmap(usage: dict[str, dict], path: Path, title: str) -> None:
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("under_over", [UNDER, MID, OVER])
    labels = list(usage)
    cols = min(3, len(labels))
    rows = math.ceil(len(labels) / cols)
    n_layer = len(usage[labels[0]]["x_target"])
    n_exp = len(usage[labels[0]]["x_target"][0])
    fig, axes = plt.subplots(rows, cols, squeeze=False, facecolor=SURFACE, layout="constrained",
                             figsize=(4.2 * cols, (0.15 * n_layer + 0.9) * rows + 0.5))
    norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0.0, vmax=CLIP)
    im = None
    for i, ax in enumerate(axes.flat):
        if i >= len(labels):
            ax.axis("off")
            continue
        ratio = torch.tensor(usage[labels[i]]["x_target"], dtype=torch.float64)
        val = torch.log2(ratio.clamp_min(1e-9)).clamp(-CLIP, CLIP).numpy()
        im = ax.imshow(val, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_facecolor(SURFACE)
        ax.set_title(evalsets.LANGUAGES.get(labels[i], labels[i]), color=INK, fontsize=10,
                     loc="left")
        ax.set_xlabel("expert", color=MUTED, fontsize=8)
        ax.set_ylabel("layer", color=MUTED, fontsize=8)
        ax.set_yticks(range(n_layer))
        ax.set_xticks(range(0, n_exp, max(1, n_exp // 8)))
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["1/4x", "1/2x", "target", "2x", "4x"])
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    cb.outline.set_visible(False)
    cb.set_label("routing frequency vs balanced target", color=MUTED, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    tmp = path.with_name(path.name + ".tmp.png")
    fig.savefig(tmp, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    from quipu.fsio import replace_with_retry
    replace_with_retry(tmp, path)


def write_outputs(usage: dict[str, dict], cfg: Config, checkpoint: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg.name, "checkpoint": checkpoint,
               "n_layer": cfg.model.n_layer, "n_experts": cfg.model.n_experts,
               "top_k": cfg.model.top_k, "dead_below_x_target": DEAD_BELOW,
               "overloaded_above_x_target": OVERLOADED_ABOVE, "splits": usage}
    write_text_atomic(out / "usage.json", json.dumps(payload, indent=1))
    write_text_atomic(out / "summary.md", summary_md(usage, cfg, checkpoint))
    heatmap(usage, out / "heatmap.png", f"{cfg.name}: expert routing frequency by language")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="latest",
                    help="'latest' (ckpt_dir/latest.pt) or a milestone / checkpoint path")
    ap.add_argument("--out", default="results/experts")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.model.kind != "moe":
        print("error: expert usage needs a model.kind = 'moe' config", file=sys.stderr)
        return 2
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    path = (evalsets.latest_checkpoint(cfg.train.ckpt_dir) if args.checkpoint == "latest"
            else Path(args.checkpoint))
    model = evalsets.load_model(cfg.model, path, device)
    usage = measure(model, cfg, args.batches)
    write_outputs(usage, cfg, str(path), Path(args.out))
    for label, u in usage.items():
        print(f"{label}: {u['tokens']:,} tokens, {len(u['dead'])} dead, "
              f"{len(u['overloaded'])} overloaded")
    print(f"wrote {args.out}/usage.json, summary.md, heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
