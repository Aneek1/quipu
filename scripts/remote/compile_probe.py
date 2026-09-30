"""Short real-trainer probes on the GPU box, before the A/B runs and the throughput gate
(scripts/remote/preflight.sh --gpu-box runs it; scripts/remote/README.md).

    uv run python scripts/remote/compile_probe.py --usd-per-hour R [--steps 20]
        [--only NAME,...] [--out results/preflight]
    uv run python scripts/remote/compile_probe.py --dynamo-report     # CPU, no GPU needed

Each probe is `python -m quipu.train` on the real shards for --steps optimizer steps
(checkpoints and milestones in a scratch directory, deleted afterwards; run logs in
<out>/runs), with TORCH_LOGS=recompiles, while nvidia-smi samples the GPU's memory:

    full-mb8-c1   configs/quipu-moe.toml     micro_batch 8, compile on
    full-mb8-c0   configs/quipu-moe.toml     micro_batch 8, compile off
    full-mb4-c1   configs/quipu-moe.toml     micro_batch 4, compile on
    full-mb4-c0   configs/quipu-moe.toml     micro_batch 4, compile off
    ab-c0         configs/quipu-moe-ab.toml  micro_batch 8, compile off, no AttnRes
    ab-attn-c0    configs/quipu-moe-ab.toml  micro_batch 8, compile off, attnres_blocks 4

Per probe (parse_probe): exit code, out of memory, peak VRAM (MiB) against the card's
total, tokens/s (median of the trainer's step lines), the compile outcome and dynamo's
counters from the run log ("compile", "compile_stats": unique graphs, graph breaks,
recompile_limit hits), and from the output the "Recompiling function" lines
(TORCH_LOGS=recompiles) and recompile_limit warnings. Everything goes to
<out>/probes.json with the decisions (decide):

- micro_batch: the largest of 8 / 4 whose eager probe finished with peak VRAM under
  VRAM_HEADROOM of the card and ran at least as fast as the other; 4 (the config's)
  when 8 does not fit or is not faster. It must be settled BEFORE run_moe.py: once
  results/moe/plan.json exists, another micro_batch is refused (exit 2).
- compile (the full run only; the A/B arms are always eager): keep it only if the
  compiled probe at the chosen micro_batch is >= COMPILE_MIN_GAIN (10%) faster than
  eager AND its log shows no storm: no recompile_limit hit and at most
  STORM_RECOMPILES_PER_LAYER x n_layer recompiles. Otherwise run_moe.py (and the SFT)
  get `--override train.compile=false`.
- the A/B throughput: ab-c0's tokens/s, printed as the ab_runs.py --dry-run command
  to cost the A/B plan against its $3; ab-attn-c0 against it gives AttnRes's
  throughput cost (the A/B's own rule allows <= 10%).

--dynamo-report is the reviewer's CPU diagnostic (no GPU, no Triton): the smoke-size
MoE under torch.compile with the aot_eager backend for a few steps with varying and
then skewed routing, printing graph breaks by reason, recompiles and unique graphs;
it shows why compile is off for the A/B (loop dispatch breaks the graph at bincount /
.tolist(), and per-layer frames recompile on new expert sizes).

Exit: 0 probes ran (whatever they found), 1 no probe could run, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FULL = "configs/quipu-moe.toml"
AB = "configs/quipu-moe-ab.toml"
PROBES: list[dict[str, Any]] = [
    {"name": "full-mb8-c1", "config": FULL, "micro_batch": 8, "compile": True},
    {"name": "full-mb8-c0", "config": FULL, "micro_batch": 8, "compile": False},
    {"name": "full-mb4-c1", "config": FULL, "micro_batch": 4, "compile": True},
    {"name": "full-mb4-c0", "config": FULL, "micro_batch": 4, "compile": False},
    {"name": "ab-c0", "config": AB, "micro_batch": 8, "compile": False},
    {"name": "ab-attn-c0", "config": AB, "micro_batch": 8, "compile": False,
     "overrides": ["model.attnres_blocks=4"]},
]
STEPS = 20
VRAM_HEADROOM = 0.95            # peak VRAM must stay under this share of the card
COMPILE_MIN_GAIN = 0.10         # compile must be >= 10% faster to be kept
STORM_RECOMPILES_PER_LAYER = 4  # more recompiles than this x n_layer = a storm
PROBE_TIMEOUT_S = 1800

STEP_LINE = re.compile(r"^step (\d+)/(\d+)\s+loss (\S+).*?([\d,.]+) tok/s")
SAVE_LINE = re.compile(r"^checkpoint step (\d+) saved in ([\d.]+) s")
OOM = ("out of memory", "outofmemoryerror")


# ---- parsing (pure) -----------------------------------------------------------------------

def parse_probe(log_text: str, run_log: dict[str, Any] | None, exit_code: int,
                vram_samples: list[int] | None = None,
                vram_total_mib: int | None = None) -> dict[str, Any]:
    """One probe's measurements from its output, its run log and the VRAM samples."""
    tok_s = []
    saves = []
    for line in log_text.splitlines():
        m = STEP_LINE.match(line.strip())
        if m:
            tok_s.append(float(m.group(4).replace(",", "")))
        s = SAVE_LINE.match(line.strip())
        if s:
            saves.append(float(s.group(2)))
    lower = log_text.lower()
    rec = run_log or {}
    peak = max(vram_samples) if vram_samples else None
    return {
        "exit_code": exit_code,
        "ok": exit_code == 0,
        "oom": any(m in lower for m in OOM),
        # The first step line can still carry a lazy recompile; the median of the rest.
        "tokens_per_s": (statistics.median(tok_s[1:]) if len(tok_s) > 1
                         else (tok_s[0] if tok_s else None)),
        "step_tokens_per_s": tok_s,
        "peak_vram_mib": peak,
        "total_vram_mib": vram_total_mib,
        "vram_share": (peak / vram_total_mib) if peak and vram_total_mib else None,
        "compile": rec.get("compile"),
        "compile_stats": rec.get("compile_stats"),
        "recompile_lines": log_text.count("Recompiling function"),
        "recompile_limit_lines": len(re.findall(r"recompile.limit", log_text)),
        "max_save_s": max(saves) if saves else None,
    }


def _fits(p: dict[str, Any] | None) -> bool:
    return bool(p and p.get("ok") and p.get("tokens_per_s")
                and (p.get("vram_share") is None or p["vram_share"] < VRAM_HEADROOM))


def storm(p: dict[str, Any], n_layer: int) -> str | None:
    """Why a compiled probe's log counts as a recompile storm, or None."""
    stats = p.get("compile_stats") or {}
    hits = max(int(stats.get("recompile_limit_hits") or 0), int(p.get("recompile_limit_lines") or 0))
    if hits:
        return f"{hits} recompile_limit hit(s)"
    if p.get("recompile_lines", 0) > STORM_RECOMPILES_PER_LAYER * n_layer:
        return (f"{p['recompile_lines']} recompiles (> {STORM_RECOMPILES_PER_LAYER} x "
                f"{n_layer} layers)")
    return None


def decide(results: dict[str, dict[str, Any]], n_layer: int = 16,
           config_micro_batch: int = 4) -> dict[str, Any]:
    """micro_batch and compile for the full run, the A/B tokens/s and AttnRes's cost."""
    out: dict[str, Any] = {"notes": []}
    eager = {mb: results.get(f"full-mb{mb}-c0") for mb in (8, 4)}
    fitting = [mb for mb in (8, 4) if _fits(eager[mb])]
    if not fitting:
        mb = config_micro_batch
        out["notes"].append("no eager full-model probe finished within the VRAM headroom: "
                            f"micro_batch stays {mb}; do NOT start the run before this is "
                            "understood (see the probe logs)")
        out["micro_batch_ok"] = False
    else:
        mb = max(fitting, key=lambda b: (eager[b]["tokens_per_s"], b == config_micro_batch))
        out["micro_batch_ok"] = True
        if 8 in fitting and mb == 4:
            out["notes"].append("micro_batch 8 fits but is not faster than 4 eagerly: 4 kept")
        if 8 not in fitting:
            why = ("out of memory" if eager[8] and eager[8].get("oom")
                   else "over the VRAM headroom or failed" if eager[8] else "not probed")
            out["notes"].append(f"micro_batch 8: {why}")
    out["micro_batch"] = mb
    c0, c1 = results.get(f"full-mb{mb}-c0"), results.get(f"full-mb{mb}-c1")
    keep, why = False, ""
    if not _fits(c1):
        why = "the compiled probe did not finish within the VRAM headroom"
    elif not str(c1.get("compile") or "").startswith("on"):
        why = f"compile did not stay on ({c1.get('compile')})"
    elif not _fits(c0):
        why = "no eager probe to compare with"
    else:
        gain = c1["tokens_per_s"] / c0["tokens_per_s"] - 1
        s = storm(c1, n_layer)
        if s:
            why = f"a recompile storm in the log ({s}); gain {gain:+.1%}"
        elif gain < COMPILE_MIN_GAIN:
            why = f"only {gain:+.1%} tokens/s over eager (needs >= {COMPILE_MIN_GAIN:.0%})"
        else:
            keep, why = True, f"{gain:+.1%} tokens/s over eager, no storm"
    out["compile"] = keep
    out["compile_reason"] = why
    best = c1 if keep else c0
    out["full_tokens_per_s"] = best.get("tokens_per_s") if best else None
    overrides = []
    if mb != config_micro_batch:
        overrides.append(f"--override train.micro_batch={mb}")
    if not keep:
        overrides.append("--override train.compile=false")
    out["run_moe_overrides"] = overrides
    ab, attn = results.get("ab-c0"), results.get("ab-attn-c0")
    out["ab_tokens_per_s"] = ab.get("tokens_per_s") if _fits(ab) else None
    if _fits(ab) and _fits(attn):
        out["attnres_cost"] = 1 - attn["tokens_per_s"] / ab["tokens_per_s"]
    return out


def ab_dry_run_command(tokens_per_s: float | None, usd_per_hour: float) -> str | None:
    if not tokens_per_s:
        return None
    return ("uv run python scripts/ab_runs.py --config configs/quipu-moe-ab.toml "
            f"--out results/ab --dry-run --tokens-per-second {tokens_per_s:.0f} "
            f"--usd-per-hour {usd_per_hour:g}")


# ---- running (box only) -------------------------------------------------------------------

class VramSampler:
    """nvidia-smi's memory.used for GPU 0 every `every` s while a probe runs."""

    def __init__(self, every: float = 1.0) -> None:
        self.every = every
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def query(field: str) -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={field}",
                                  "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(float(out.strip().splitlines()[0]))
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            v = self.query("memory.used")
            if v is not None:
                self.samples.append(v)

    def __enter__(self) -> "VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def run_probe(probe: dict[str, Any], steps: int, usd_per_hour: float, out: Path,
              scratch: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from quipu.config import load_config

    cfg = load_config(ROOT / probe["config"])
    batch = cfg.train.batch_tokens
    ckpt = scratch / probe["name"]
    shutil.rmtree(ckpt, ignore_errors=True)
    run_dir = out / "runs"
    run_log = run_dir / f"pf-{probe['name']}.json"
    run_log.unlink(missing_ok=True)
    overrides = [f"train.total_tokens={steps * batch}", f"train.warmup_steps={min(5, steps - 1)}",
                 f"train.eval_every={steps}", f"train.ckpt_every={steps}", "train.milestones=[]",
                 f"train.ckpt_dir={ckpt.as_posix()}", f"train.micro_batch={probe['micro_batch']}",
                 f"train.compile={'true' if probe['compile'] else 'false'}",
                 "train.budget_usd=1.0", f"train.usd_per_hour={usd_per_hour!r}",
                 *probe.get("overrides", [])]
    cmd = [sys.executable, "-m", "quipu.train", "--config", probe["config"], "--run-id",
           f"pf-{probe['name']}", "--run-dir", str(run_dir), "--device", "cuda"]
    for o in overrides:
        cmd += ["--override", o]
    log_path = out / "logs" / f"{probe['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TORCH_LOGS="recompiles",
               TORCHINDUCTOR_CACHE_DIR=str(scratch / "inductor-cache"))
    total = VramSampler.query("memory.total")
    t0 = time.monotonic()
    print(f"[probe] {probe['name']}: {steps} steps of {probe['config']} at micro_batch "
          f"{probe['micro_batch']}, compile {'on' if probe['compile'] else 'off'}", flush=True)
    with VramSampler() as vram, open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=PROBE_TIMEOUT_S)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -9
    wall = time.monotonic() - t0
    shutil.rmtree(ckpt, ignore_errors=True)
    try:
        record = json.loads(run_log.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    result = parse_probe(log_path.read_text(encoding="utf-8", errors="replace"), record, code,
                         vram.samples, total)
    result.update(name=probe["name"], config=probe["config"],
                  micro_batch=probe["micro_batch"], compile_requested=probe["compile"],
                  overrides=probe.get("overrides", []), steps=steps, wall_s=round(wall, 1),
                  log=str(log_path))
    tps = result["tokens_per_s"]
    print(f"[probe] {probe['name']}: exit {code}"
          + (" (OUT OF MEMORY)" if result["oom"] else "")
          + f", peak {result['peak_vram_mib']} / {total} MiB"
          + (f", {tps:,.0f} tok/s" if tps else "")
          + f", compile {result['compile']}, {result['recompile_lines']} recompiles, "
          f"{result['recompile_limit_lines']} recompile_limit lines, {wall:.0f} s", flush=True)
    return result


def dynamo_report(steps: int = 6, n_layer: int = 2) -> int:
    """The reviewer's CPU diagnostic: graph breaks and recompiles of the smoke MoE."""
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    from torch._dynamo.utils import counters

    from quipu.config import load_config
    from quipu.model_factory import build_model

    cfg = load_config(ROOT / "configs" / "quipu-moe-smoke.toml",
                      {"model": {"moe_dispatch": "loop", "attnres_blocks": 0,
                                 "n_layer": n_layer, "n_experts": 16, "top_k": 4}})
    torch.manual_seed(0)
    model = build_model(cfg.model).train()
    cm = torch.compile(model, backend="aot_eager")
    for s in range(steps):
        x = torch.randint(cfg.model.vocab_size, (2, 128))
        if s == steps - 1:
            x = torch.full((2, 128), 7)          # skewed routing: some experts empty
        logits = cm(x)
        F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1)).backward()
        model.zero_grad(set_to_none=True)
        model.update_balance()
        print(f"step {s}: frames {dict(counters['stats'])}", flush=True)
    print("graph breaks by reason:")
    for k, v in counters["graph_break"].items():
        print(f"  {v} x {str(k)[:160]}")
    print("unique graphs:", counters["stats"].get("unique_graphs"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--usd-per-hour", type=float, default=None,
                    help="the box's rate (each probe's backstop; the A/B dry-run command)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--only", default="", help="comma list of probe names")
    ap.add_argument("--out", default="results/preflight")
    ap.add_argument("--scratch", default="/workspace/preflight-scratch",
                    help="probe checkpoints and inductor cache (deleted per probe)")
    ap.add_argument("--dynamo-report", action="store_true")
    args = ap.parse_args(argv)
    if args.dynamo_report:
        return dynamo_report()
    if args.usd_per_hour is None or not args.usd_per_hour > 0 or args.steps < 2:
        print("error: pass --usd-per-hour R (> 0) and --steps >= 2", file=sys.stderr)
        return 2
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = set(names) - {p["name"] for p in PROBES}
    if unknown:
        print(f"error: unknown probe(s) {sorted(unknown)}", file=sys.stderr)
        return 2
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    scratch = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for probe in PROBES:
        if names and probe["name"] not in names:
            continue
        results[probe["name"]] = run_probe(probe, args.steps, args.usd_per_hour, out, scratch)
    shutil.rmtree(scratch / "inductor-cache", ignore_errors=True)
    if not any(r["ok"] for r in results.values()):
        print("no probe finished: see the logs in " + str(out / "logs"), file=sys.stderr)
    sys.path.insert(0, str(ROOT))
    from quipu.config import load_config

    full = load_config(ROOT / FULL)
    decision = decide(results, full.model.n_layer, full.train.micro_batch)
    decision["ab_dry_run"] = ab_dry_run_command(decision.get("ab_tokens_per_s"),
                                                args.usd_per_hour)
    (out / "probes.json").write_text(json.dumps(
        {"steps": args.steps, "usd_per_hour": args.usd_per_hour, "probes": results,
         "decision": decision}, indent=2), encoding="utf-8")
    print(f"\n[probe] results in {out / 'probes.json'}")
    print(f"[probe] micro_batch {decision['micro_batch']}; compile "
          f"{'ON' if decision['compile'] else 'OFF'} ({decision['compile_reason']})")
    for note in decision["notes"]:
        print(f"[probe] note: {note}")
    if decision.get("full_tokens_per_s"):
        print(f"[probe] full model: {decision['full_tokens_per_s']:,.0f} tok/s at that setting")
    if decision["run_moe_overrides"]:
        print("[probe] pass to run_moe.py (and the SFT): "
              + " ".join(decision["run_moe_overrides"]))
    if decision.get("attnres_cost") is not None:
        print(f"[probe] AttnRes throughput cost on the A/B model: {decision['attnres_cost']:.1%}")
    if decision["ab_dry_run"]:
        print("[probe] for the owner, the A/B plan at the measured rate:\n  "
              + decision["ab_dry_run"])
    return 0 if any(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
