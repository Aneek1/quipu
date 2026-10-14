"""Continued pretraining of quipu-moe on Modal GPUs (data v3 spec, sections 2 and 3).

Run it with plain Python (the Python that has the `modal` package), NOT `modal run`:
the budget guard has to run before Modal builds or starts anything.

    python scripts/remote/modal_train.py --smoke --plan-only     # what would run, the cost
    python scripts/remote/modal_train.py --smoke                 # 2x L40S, ~10 min
    python scripts/remote/modal_train.py --smoke --gpu-type H100 --dispatch grouped
    python scripts/remote/modal_train.py --plan-only             # the monthly run
    python scripts/remote/modal_train.py --gpus 1 --hours 3.5    # the monthly run, 1x B200
    python scripts/remote/modal_train.py --decay-steps 600       # a decay branch (release)

Budget guard (both modes, before any launch): `modal billing summary --json` gives the
month's metered cost; the launch is refused when metered + this launch's worst case
(GPUs x list $/h + CPU + memory, for the whole function timeout) is over CAP_USD ($27;
the $3 under Modal's $30 hard limit cover the demo and storage). The prices in
GPU_USD_PER_HOUR are Modal's list prices read with `modal billing rates` on 14 Oct 2026:
estimates, not quotes. The trainer's own wall-clock backstop (train.budget_usd /
train.usd_per_hour) is the second guard, the workspace's hard limit the third.

--smoke: N GPUs (default 2 x L40S) in one container. torchrun trains configs/quipu-moe-cpt.toml
from scratch (random init; the full 1B-total model) for a few steps at a small micro
batch on the pool, checkpoints, then a SECOND torchrun resumes it for a few more
steps. Every rank writes its own record (quipu.rankrun), and the smoke passes only if:
both torchruns and every rank exit 0, the loss is finite and goes down, params_match
held on every rank at every checkpoint, the resume starts at the saved step and runs
on to the end, and the ranks read disjoint data (and the resume no data the first run
read). Prints PASS/FAIL and tokens/s. Function timeout 15 min.

Normal run: 1..8 B200 in one container. The trainer's shard view (quipu.shard_view)
is built in the container: train/ links to the pool's mixed/ shards on Volume
quipu-pool, val/ to the old English validation shard on Volume quipu-shards. State
lives on Volume quipu-ckpt:

    /init/step_008129.pt         the pretrain's final checkpoint (uploaded once, below)
    /stable/                     the continued run (ckpt_dir; latest.pt, step_*.pt)
    /decay/step_<S>/             a decay branch from stable step S (its milestones/ is
                                 the release)
    /runs/<run id>.json          run logs (written in /tmp during the run, copied here)
    /logs/<run id>-<time>/       torchrun output and the rank records of each launch

The first launch starts from /init (--init-from, weights AND optimizer states, new wsd
schedule from step 0, pool from its first token); every later one resumes /stable.
The trainer stops itself after --hours (its budget backstop, exit 4) and checkpoints;
the function's timeout is --hours + TIMEOUT_MARGIN_S. Upload the checkpoint once
(~8.1 GB; ~47 min at the laptop's ~2.9 MB/s):

    modal volume create quipu-ckpt
    modal volume put quipu-ckpt remote-runs/quipu-moe/checkpoints/quipu-moe/step_008129.pt /init/step_008129.pt

(From Git Bash set MSYS_NO_PATHCONV=1 first, or /init/... becomes a Windows path.)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import modal
except ImportError:  # the unit tests and plan logic run without Modal
    modal = None

LOCAL = modal is None or modal.is_local()
HERE = Path(__file__).resolve().parents[2] if LOCAL else Path("/src")

CONFIG = "configs/quipu-moe-cpt.toml"
TOKENIZER = "artifacts/tokenizer/tokenizer.json"
# The 48k tokenizer the model and the pool were built with (data/shards-moe/manifest.json,
# the pool manifest's provenance.tokenizer, configs/data-v3.toml all record it).
TOKENIZER_SHA256 = "9f3e33d074f3a0d7da234eec11a59de1e20aa756855d1ef552fc9706b9006943"
CONTEXT = 2048                 # configs/quipu-moe.toml [model].context
BATCH_TOKENS = 524_288         # configs/quipu-moe-cpt.toml [train].batch_tokens
RUN_ID = "quipu-moe-cpt"

CAP_USD = 27.0
HARD_LIMIT_USD = 30.0
# Modal list prices, $/h, from `modal billing rates` on 14 Oct 2026. Estimates.
GPU_USD_PER_HOUR = {"B200": 6.25, "H100": 3.95, "L40S": 1.95, "L4": 0.80}
CPU_USD_PER_CORE_HOUR = 0.0473
MEM_USD_PER_GIB_HOUR = 0.008

SMOKE_TIMEOUT_S = 15 * 60
TIMEOUT_MARGIN_S = 20 * 60     # run mode: startup, the final checkpoint, the copy-back
COMMIT_EVERY_S = 10 * 60
FAIL_GRACE_S = 20             # run_processes: the other ranks after one fails

POOL_MOUNT, SHARDS_MOUNT, CKPT_MOUNT = "/pool", "/shards", "/ckpt"
VIEW_DIR = "/tmp/shards"

# ---- cost and the budget guard (no Modal needed) ----------------------------------------


def resources(gpus: int) -> tuple[float, int]:
    """(CPU cores, memory MiB) for a container with `gpus` GPUs. Every rank loads a
    whole ~8.1 GB training checkpoint into CPU memory on resume, plus the model it
    builds on the CPU first: 24 GiB a rank, 16 GiB for the rest."""
    return 2.0 + 4.0 * gpus, (16 + 24 * gpus) * 1024


def usd_per_hour(gpu_type: str, gpus: int, cpu: float, memory_mib: int) -> float:
    if gpu_type not in GPU_USD_PER_HOUR:
        raise ValueError(f"no price for GPU {gpu_type!r}; known: {sorted(GPU_USD_PER_HOUR)}")
    return (gpus * GPU_USD_PER_HOUR[gpu_type] + cpu * CPU_USD_PER_CORE_HOUR
            + memory_mib / 1024 * MEM_USD_PER_GIB_HOUR)


_TEXT_METERED = re.compile(r"Metered Cost:\s*\|?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def parse_metered_cost(text: str) -> float:
    """The month's metered cost (USD) from `modal billing summary` output: the --json
    form ({"metered_cost": "25.03546599", ...}) or the table ("| Metered Cost: | 25.04 |").
    ValueError when neither is there: the guard never guesses."""
    s = text.strip()
    try:
        obj = json.loads(s)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and "metered_cost" in obj:
        value = float(obj["metered_cost"])
    else:
        m = _TEXT_METERED.search(s)
        if not m:
            raise ValueError(f"no metered cost in the billing summary: {s[:200]!r}")
        value = float(m.group(1).replace(",", ""))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"metered cost {value!r} is not a cost")
    return value


def read_metered_cost(run: Callable[..., Any] = subprocess.run) -> float:
    """`modal billing summary --json` (this month), through this Python's modal CLI."""
    r = run([sys.executable, "-m", "modal", "billing", "summary", "--json"],
            capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"modal billing summary failed (exit {r.returncode}): "
                           f"{(r.stderr or r.stdout)[-500:]}")
    return parse_metered_cost(r.stdout)


def check_budget(metered: float, planned: float, cap: float = CAP_USD) -> tuple[bool, str]:
    """Allowed only when the month's metered cost plus this launch's worst case stays
    within the cap."""
    total = metered + planned
    ok = total <= cap
    verdict = "OK" if ok else "REFUSED"
    msg = (f"budget guard: metered this month ${metered:.2f} + this launch at most "
           f"${planned:.2f} = ${total:.2f} vs cap ${cap:.2f} -> {verdict}"
           + ("" if ok else f" (over by ${total - cap:.2f})"))
    return ok, msg


# ---- the plan --------------------------------------------------------------------------


@dataclass
class Plan:
    mode: str                  # "smoke" | "run" | "decay"
    gpu_type: str
    gpus: int
    timeout_s: int
    hours: float               # run/decay: the trainer's wall-clock budget
    micro_batch: int
    dispatch: str
    compile: bool
    steps1: int = 6
    steps2: int = 4
    decay_steps: int = 0
    cpu: float = 0.0
    memory_mib: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_spec(self) -> str:
        return self.gpu_type if self.gpus == 1 else f"{self.gpu_type}:{self.gpus}"

    @property
    def usd_per_hour(self) -> float:
        return usd_per_hour(self.gpu_type, self.gpus, self.cpu, self.memory_mib)

    @property
    def worst_case_usd(self) -> float:
        """The whole timeout at the container's full price."""
        return self.usd_per_hour * self.timeout_s / 3600

    @property
    def trainer_budget_usd(self) -> float:
        """train.budget_usd: --hours at usd_per_hour (the trainer's wall clock)."""
        return round(self.hours * self.usd_per_hour, 4)


def make_plan(a: argparse.Namespace) -> Plan:
    smoke = bool(a.smoke)
    gpu_type = a.gpu_type or ("L40S" if smoke else "B200")
    gpus = a.gpus if a.gpus is not None else (2 if smoke else 1)
    if not 1 <= gpus <= 8:
        raise ValueError(f"--gpus must be 1..8, got {gpus}")
    if gpu_type not in GPU_USD_PER_HOUR:
        raise ValueError(f"--gpu-type {gpu_type!r}: one of {sorted(GPU_USD_PER_HOUR)}")
    if smoke and a.decay_steps:
        raise ValueError("--decay-steps is a run-mode option")
    dispatch = a.dispatch or ("loop" if smoke else "grouped")
    if dispatch not in ("loop", "grouped", "padded"):
        raise ValueError(f"--dispatch {dispatch!r}: loop, grouped or padded")
    micro = a.micro_batch or (2 if smoke else 32)
    cpu, mem = resources(gpus)
    if smoke:
        p = Plan("smoke", gpu_type, gpus, SMOKE_TIMEOUT_S, hours=0.0, micro_batch=micro,
                 dispatch=dispatch, compile=bool(a.compile), steps1=a.steps1,
                 steps2=a.steps2, cpu=cpu, memory_mib=mem)
        if p.steps1 < 3 or p.steps2 < 2:
            raise ValueError("--steps1 >= 3 and --steps2 >= 2 (the warmup is 2 steps)")
        return p
    accum = BATCH_TOKENS // (micro * CONTEXT)
    if BATCH_TOKENS % (micro * CONTEXT) or accum % gpus:
        raise ValueError(f"micro_batch {micro} gives grad_accum {BATCH_TOKENS / (micro * CONTEXT):g}, "
                         f"which must be a whole multiple of --gpus {gpus}")
    if not 0 < a.hours <= 23:
        raise ValueError(f"--hours must be in (0, 23], got {a.hours}")
    timeout = int(a.hours * 3600 + TIMEOUT_MARGIN_S)
    p = Plan("decay" if a.decay_steps else "run", gpu_type, gpus, timeout, hours=a.hours,
             micro_batch=micro, dispatch=dispatch, compile=a.compile is not False,
             decay_steps=a.decay_steps or 0, cpu=cpu, memory_mib=mem)
    if dispatch == "grouped" and gpu_type not in ("B200", "H100"):
        p.notes.append("grouped dispatch uses torch._grouped_mm, which may need Hopper or "
                       "newer; use --dispatch loop on this GPU")
    return p


# ---- what runs in the container (plain functions; tests call them on the CPU) -----------


def rank_command(*, python: str, marker_dir: str | Path, config: str, run_id: str,
                 run_dir: str | Path, overrides: dict[str, Any], flags: list[str],
                 device: str = "cuda") -> list[str]:
    """One rank's command: quipu.rankrun (quipu.train plus the rank's record)."""
    cmd = [python, "-m", "quipu.rankrun", "--marker-dir", str(marker_dir), "--",
           "--config", config, "--run-id", run_id, "--run-dir", str(run_dir),
           "--device", device]
    for k, v in overrides.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        cmd += ["--override", f"{k}={v}"]
    return cmd + list(flags)


def train_command(*, python: str, nproc: int, **kw: Any) -> list[str]:
    """torchrun (as `python -m torch.distributed.run --standalone`) over rank_command."""
    return ([python, "-m", "torch.distributed.run", "--standalone",
             f"--nproc_per_node={nproc}"] + rank_command(python=python, **kw)[1:])


def run_processes(cmd: list[str], *, nproc: int, cwd: Path, env: dict[str, str],
                  log_path: Path, timeout_s: float, echo: Callable[[str], None] = print
                  ) -> int:
    """The local stand-in for torchrun (Windows, where this torch's torchrun cannot
    start its rendezvous store): nproc copies of a rank command with the env:// variables
    torchrun would set, each logging to <log>.rank<R>.log. Returns 0 when every rank
    exits 0, else the first non-zero exit (-9 for a timeout)."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    procs, files = [], []
    for r in range(nproc):
        e = {**env, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(r),
             "LOCAL_RANK": str(r), "WORLD_SIZE": str(nproc), "LOCAL_WORLD_SIZE": str(nproc)}
        f = open(log_path.with_name(f"{log_path.stem}.rank{r}.log"), "w", encoding="utf-8",
                 errors="replace")
        files.append(f)
        procs.append(subprocess.Popen(cmd, cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT))
    deadline = time.monotonic() + timeout_s
    failed_at = None
    try:
        # As torchrun does: once one rank fails, the others (likely waiting for it in a
        # collective) get FAIL_GRACE_S to finish, then are stopped.
        while any(p.poll() is None for p in procs):
            now = time.monotonic()
            if failed_at is None and any(p.poll() not in (None, 0) for p in procs):
                failed_at = now
            if now > deadline or (failed_at is not None and now - failed_at > FAIL_GRACE_S):
                break
            time.sleep(0.2)
        codes = [p.poll() if p.poll() is not None else -9 for p in procs]
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for f in files:
            f.close()
    echo(f"ranks exited {codes}")
    return next((c for c in codes if c != 0), 0)


def stream_command(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path,
                   timeout_s: float, echo: Callable[[str], None] = print) -> int:
    """Run cmd, echoing and logging every line of its merged output; kill it at the
    timeout (exit code -9 then). Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1)

        def pump() -> None:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                echo(line.rstrip("\n"))
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            echo(f"timeout after {timeout_s:.0f} s: stopping torchrun")
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            reader.join(timeout=10)
            return -9
        reader.join(timeout=30)
        return proc.returncode


def read_markers(marker_dir: str | Path) -> dict[int, dict[str, Any]]:
    out = {}
    for p in sorted(Path(marker_dir).glob("rank*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[int(rec["rank"])] = rec
        except (OSError, ValueError, KeyError) as exc:
            out[int(p.stem.removeprefix("rank"))] = {"unreadable": repr(exc)}
    return out


def tokens_per_second(rec: dict[str, Any], skip: int = 1) -> float | None:
    """batch_tokens / the median step time after the first `skip` steps (warm-up)."""
    secs = (rec.get("step_seconds") or [])[skip:] or (rec.get("step_seconds") or [])
    tokens = rec.get("tokens_per_step")
    if not secs or not tokens:
        return None
    return tokens / max(statistics.median(secs), 1e-9)


def _rank_failures(tag: str, markers: dict[int, dict[str, Any]], world: int, *,
                   start: int, end: int, steps: int, accum_per_rank: int,
                   ok_exits: tuple[int, ...] = (0,)) -> list[str]:
    f = []
    if sorted(markers) != list(range(world)):
        f.append(f"{tag}: rank records {sorted(markers)}, expected 0..{world - 1} "
                 "(a rank without a record died before it could write one)")
    for r, m in sorted(markers.items()):
        if "unreadable" in m:
            f.append(f"{tag}: rank {r} record unreadable ({m['unreadable']})")
            continue
        if m.get("exit") not in ok_exits:
            f.append(f"{tag}: rank {r} exit {m.get('exit')}, expected {ok_exits}")
        if m.get("world") != world:
            f.append(f"{tag}: rank {r} saw world size {m.get('world')}, expected {world}")
        if m.get("start_step") != start:
            f.append(f"{tag}: rank {r} started at step {m.get('start_step')}, expected {start}")
        if m.get("final_step") != end:
            f.append(f"{tag}: rank {r} stopped at step {m.get('final_step')}, expected {end}")
        pm = m.get("params_match") or []
        if not pm:
            f.append(f"{tag}: rank {r} never checked params_match (no checkpoint?)")
        elif not all(pm):
            f.append(f"{tag}: rank {r} params_match {pm}: the replicas diverged")
        if end not in (m.get("checkpoints") or []):
            f.append(f"{tag}: rank {r} took no checkpoint at step {end} "
                     f"(checkpoints {m.get('checkpoints')})")
        n = len(m.get("batches") or [])
        if n != steps * accum_per_rank:
            f.append(f"{tag}: rank {r} read {n} micro-batches, expected "
                     f"{steps} steps x {accum_per_rank}")
    ranks = [set(m.get("batches") or []) for m in markers.values() if "unreadable" not in m]
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] & ranks[j]:
                f.append(f"{tag}: ranks {i} and {j} read {len(ranks[i] & ranks[j])} "
                         "identical micro-batches (data not split over the ranks)")
    return f


def check_smoke(*, world: int, steps1: int, steps2: int, accum_per_rank: int,
                rc1: int, rc2: int, markers1: dict[int, dict[str, Any]],
                markers2: dict[int, dict[str, Any]], runlog: dict[str, Any] | None
                ) -> list[str]:
    """Every reason the two-launch smoke failed ([] = PASS)."""
    f = []
    for tag, rc in (("run 1", rc1), ("run 2 (resume)", rc2)):
        if rc != 0:
            f.append(f"{tag}: torchrun exit {rc}")
    f += _rank_failures("run 1", markers1, world, start=0, end=steps1, steps=steps1,
                        accum_per_rank=accum_per_rank)
    f += _rank_failures("run 2", markers2, world, start=steps1, end=steps1 + steps2,
                        steps=steps2, accum_per_rank=accum_per_rank)
    for r in range(world):
        a = set((markers1.get(r) or {}).get("batches") or [])
        b = set((markers2.get(r) or {}).get("batches") or [])
        if a & b:
            f.append(f"rank {r}: the resume re-read {len(a & b)} micro-batches the first "
                     "run had already trained on")
    if not runlog:
        return f + ["no run log from rank 0"]
    steps = [s for s in runlog.get("steps", [])]
    got = [s.get("step") for s in steps]
    want = list(range(1, steps1 + steps2 + 1))
    if got != want:
        f.append(f"run log steps {got}, expected {want}")
    losses = [s.get("train_loss") for s in steps]
    if not losses or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        f.append(f"non-finite or missing train loss: {losses}")
    elif len(losses) >= 4:
        tail = statistics.mean(losses[-3:])
        if not tail < losses[0]:
            f.append(f"loss not decreasing: first {losses[0]:.4f}, mean of the last three "
                     f"{tail:.4f}")
    if not runlog.get("resumes"):
        f.append("the run log records no resume")
    return f


def smoke_overrides(*, shard_dir: str, ckpt_dir: str, micro_batch: int, context: int,
                    world: int, steps: int, steps1: int, dispatch: str, compile: bool,
                    usd_per_hour: float, budget_usd: float) -> dict[str, Any]:
    """Two micro-batches per rank per step; checkpoint (and evaluate) at steps1."""
    batch = micro_batch * context * 2 * world
    return {"data.shard_dir": shard_dir, "train.ckpt_dir": ckpt_dir,
            "train.micro_batch": micro_batch, "train.batch_tokens": batch,
            "train.total_tokens": steps * batch, "train.warmup_steps": 2,
            "train.ckpt_every": steps1, "train.ckpt_keep": 1, "train.eval_every": steps1,
            "train.eval_batches": 2, "train.compile": compile,
            "model.moe_dispatch": dispatch, "train.usd_per_hour": round(usd_per_hour, 4),
            "train.budget_usd": round(budget_usd, 4)}


def _env(cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "win32":
        env["USE_LIBUV"] = "0"          # torchrun's TCPStore on Windows (local tests)
    return env


def run_smoke(work: str | Path, *, config: str, shard_dir: str, world: int, steps1: int,
              steps2: int, micro_batch: int, dispatch: str, compile: bool, device: str,
              usd_per_hour: float, budget_usd: float, cwd: Path, timeout_s: float,
              extra_overrides: dict[str, Any] | None = None, launcher: str = "torchrun",
              python: str = sys.executable, echo: Callable[[str], None] = print
              ) -> dict[str, Any]:
    """The two torchrun launches of the smoke and their checks. Returns the summary
    ({"passed", "failures", "tokens_per_s", "losses", ...}) and prints it. launcher
    "processes" starts the ranks without torchrun (run_processes; local CPU tests)."""
    from quipu.config import load_config

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cfg = load_config(Path(cwd) / config, _nest(extra_overrides or {}))
    context = cfg.model.context
    common = dict(shard_dir=shard_dir, ckpt_dir=str(work / "ckpt"), micro_batch=micro_batch,
                  context=context, world=world, steps1=steps1, dispatch=dispatch,
                  compile=compile, usd_per_hour=usd_per_hour, budget_usd=budget_usd)
    env = _env(Path(cwd))
    rcs, markers = [], []
    t0 = time.monotonic()
    for i, (steps, flags) in enumerate(((steps1, []), (steps1 + steps2, ["--resume"])), 1):
        over = {**(extra_overrides or {}), **smoke_overrides(steps=steps, **common)}
        kw = dict(python=python, marker_dir=work / f"markers{i}", config=config,
                  run_id="smoke", run_dir=work / "runs", overrides=over, flags=flags,
                  device=device)
        echo(f"=== smoke launch {i}: {steps} steps" + (" (resume)" if flags else ""))
        left = timeout_s - (time.monotonic() - t0)
        budget = max(60.0, left * (0.55 if i == 1 else 1.0))
        if launcher == "processes":
            rcs.append(run_processes(rank_command(**kw), nproc=world, cwd=Path(cwd), env=env,
                                     log_path=work / f"run{i}.log", timeout_s=budget,
                                     echo=echo))
        else:
            rcs.append(stream_command(train_command(nproc=world, **kw), cwd=Path(cwd),
                                      env=env, log_path=work / f"run{i}.log",
                                      timeout_s=budget, echo=echo))
        markers.append(read_markers(work / f"markers{i}"))
        if i == 1 and rcs[0] != 0:
            echo("launch 1 failed; the resume is skipped")
            rcs.append(-1)
            markers.append({})
            break
    try:
        runlog = json.loads((work / "runs" / "smoke.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        runlog = None
    failures = check_smoke(world=world, steps1=steps1, steps2=steps2, accum_per_rank=2,
                           rc1=rcs[0], rc2=rcs[1], markers1=markers[0], markers2=markers[1],
                           runlog=runlog)
    tok = [tokens_per_second(m.get(0, {})) for m in markers]
    summary = {"passed": not failures, "failures": failures, "world": world,
               "tokens_per_s": {"run1": tok[0], "run2": tok[1]},
               "losses": [s.get("train_loss") for s in (runlog or {}).get("steps", [])],
               "exit_codes": rcs, "wall_s": round(time.monotonic() - t0, 1),
               "micro_batch": micro_batch, "dispatch": dispatch, "compile": compile}
    echo(format_smoke(summary))
    return summary


def format_smoke(s: dict[str, Any]) -> str:
    lines = [f"===== SMOKE {'PASS' if s['passed'] else 'FAIL'} ({s['world']} ranks, "
             f"micro_batch {s['micro_batch']}, dispatch {s['dispatch']}, compile "
             f"{'on' if s['compile'] else 'off'}, {s['wall_s']} s)"]
    for k, v in s["tokens_per_s"].items():
        lines.append(f"  {k}: " + (f"{v:,.0f} tokens/s (all ranks, median step)" if v
                                   else "no step times"))
    if s["losses"]:
        lines.append("  losses: " + " ".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                                            for x in s["losses"]))
    lines += [f"  FAIL: {x}" for x in s["failures"]]
    return "\n".join(lines)


def _sha256(path: str, cwd: Path) -> str:
    import hashlib

    p = Path(path) if Path(path).is_absolute() else Path(cwd) / path
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in flat.items():
        section, key = k.split(".", 1)
        out.setdefault(section, {})[key] = v
    return out


# ---- run mode: what to start from --------------------------------------------------------


def _pointer_file(ckpt_dir: Path) -> Path | None:
    """The checkpoint latest.pt points at, or None without one."""
    latest = ckpt_dir / "latest.pt"
    if not latest.is_file():
        return None
    import torch

    pointer = torch.load(latest, map_location="cpu", weights_only=True)
    path = ckpt_dir / pointer["file"]
    if not path.is_file():
        raise FileNotFoundError(f"{latest} points at {path}, which is missing")
    return path


_STEP = re.compile(r"^step_(\d+)\.pt$")


def init_checkpoint(init_dir: Path) -> Path:
    """The pretrain checkpoint in /init: latest.pt's file, else the one step_*.pt."""
    ptr = _pointer_file(init_dir)
    if ptr is not None:
        return ptr
    found = sorted(p for p in init_dir.glob("step_*.pt") if _STEP.match(p.name))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{init_dir}: expected one step_*.pt (the pretrain's final checkpoint), found "
            f"{[p.name for p in found]}; upload it with `modal volume put quipu-ckpt ...`")
    return found[0]


@dataclass
class RunSetup:
    run_id: str
    ckpt_dir: Path
    flags: list[str]
    overrides: dict[str, Any]
    note: str


def plan_run(ckpt_root: Path, *, decay_steps: int = 0, batch_tokens: int = BATCH_TOKENS
             ) -> RunSetup:
    """From the state of the checkpoint Volume: resume the stable run, start it from
    /init (first month), or branch a decay from the stable checkpoint."""
    stable = ckpt_root / "stable"
    if decay_steps:
        src = _pointer_file(stable)
        if src is None:
            raise FileNotFoundError(f"no stable checkpoint in {stable} to branch a decay from")
        s = int(_STEP.match(src.name).group(1))
        ckpt = ckpt_root / "decay" / f"step_{s:06d}"
        over = {"train.decay_start_step": s, "train.decay_steps": decay_steps,
                "train.total_tokens": (s + decay_steps) * batch_tokens}
        run_id = f"{RUN_ID}-decay-{s:06d}"
        if (ckpt / "latest.pt").is_file():
            return RunSetup(run_id, ckpt, ["--resume"], over,
                            f"resume the decay branch from stable step {s}")
        return RunSetup(run_id, ckpt, ["--branch-from", str(src)], over,
                        f"branch a {decay_steps}-step decay from stable step {s} ({src.name})")
    if (stable / "latest.pt").is_file():
        return RunSetup(RUN_ID, stable, ["--resume"], {}, "resume the stable run")
    init = init_checkpoint(ckpt_root / "init")
    return RunSetup(RUN_ID, stable, ["--init-from", str(init), "--init-optimizers"], {},
                    f"first launch: weights and optimizer states from {init.name}, wsd "
                    "re-warmup from step 0")


def set_aside_stale_log(run_dir: Path, run_id: str, resuming: bool) -> Path | None:
    """A run log left by a launch that died before its first checkpoint would make a
    fresh start fail (the trainer refuses to overwrite a log): rename it aside."""
    log = run_dir / f"{run_id}.json"
    if resuming or not log.is_file():
        return None
    aside = log.with_name(f"{run_id}.stale-{int(time.time())}.json")
    log.rename(aside)
    return aside


def run_training(*, ckpt_root: Path, pool_root: Path, val_dir: Path, view_dir: Path,
                 world: int, micro_batch: int, dispatch: str, compile: bool, hours: float,
                 usd_per_hour: float, decay_steps: int, cwd: Path, timeout_s: float,
                 commit: Callable[[], None] | None = None, device: str = "cuda",
                 config: str = CONFIG, python: str = sys.executable,
                 extra_overrides: dict[str, Any] | None = None,
                 local_run_dir: Path = Path("/tmp/runs"), launcher: str = "torchrun",
                 echo: Callable[[str], None] = print) -> dict[str, Any]:
    """The monthly run (or a decay branch) in the container. The run log is written
    on local disk (the trainer rewrites it every step) and copied to the Volume's
    runs/ every COMMIT_EVERY_S and at the end."""
    from quipu.config import load_config
    from quipu.shard_view import make_shard_view

    base = {"data.shard_dir": str(view_dir), "train.micro_batch": micro_batch,
            "model.moe_dispatch": dispatch, "train.compile": compile,
            "train.usd_per_hour": round(usd_per_hour, 4),
            "train.budget_usd": round(hours * usd_per_hour, 4)}
    cfg = load_config(Path(cwd) / config, _nest({**base, **(extra_overrides or {})}))
    sha = _sha256(cfg.data.tokenizer, cwd)
    view = make_shard_view(view_dir, pool_root, val_dir, sha, val_tokenizer_sha256=sha)
    echo(f"shard view: {view['train']['shards']} train shards "
         f"({view['train']['tokens']:,} tokens), val {view['val']['tokens']:,} tokens")
    setup = plan_run(ckpt_root, decay_steps=decay_steps, batch_tokens=cfg.train.batch_tokens)
    echo(f"plan: {setup.note}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir_vol = ckpt_root / "runs"
    run_dir_vol.mkdir(parents=True, exist_ok=True)
    run_dir = local_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    resuming = "--resume" in setup.flags
    aside = set_aside_stale_log(run_dir_vol, setup.run_id, resuming)
    if aside:
        echo(f"set aside a run log without a checkpoint: {aside.name}")
    if resuming:
        src = run_dir_vol / f"{setup.run_id}.json"
        if not src.is_file():
            raise FileNotFoundError(f"resuming {setup.run_id} but its run log {src} is missing")
        shutil.copyfile(src, run_dir / src.name)
    logs = ckpt_root / "logs" / f"{setup.run_id}-{stamp}"
    n = 1
    while logs.exists():                        # two launches in one second
        n += 1
        logs = ckpt_root / "logs" / f"{setup.run_id}-{stamp}-{n}"
    over = {**base, "train.ckpt_dir": str(setup.ckpt_dir), **setup.overrides,
            **(extra_overrides or {})}
    kw = dict(python=python, marker_dir=logs / "markers", config=config,
              run_id=setup.run_id, run_dir=run_dir, overrides=over, flags=setup.flags,
              device=device)

    def copy_back() -> None:
        log = run_dir / f"{setup.run_id}.json"
        if log.is_file():
            shutil.copyfile(log, run_dir_vol / log.name)

    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                copy_back()
                if commit:
                    commit()
            except Exception as exc:          # a missed commit is retried next time
                echo(f"warning: volume commit failed: {exc!r}")
    t = threading.Thread(target=committer, daemon=True)
    t.start()
    t0 = time.monotonic()
    try:
        if launcher == "processes":
            rc = run_processes(rank_command(**kw), nproc=world, cwd=cwd, env=_env(cwd),
                               log_path=logs / "torchrun.log", timeout_s=timeout_s, echo=echo)
        else:
            rc = stream_command(train_command(nproc=world, **kw), cwd=cwd, env=_env(cwd),
                                log_path=logs / "torchrun.log", timeout_s=timeout_s,
                                echo=echo)
    finally:
        stop.set()
        copy_back()
        if commit:
            commit()
    markers = read_markers(logs / "markers")
    exits = {r: m.get("exit") for r, m in markers.items()}
    ok = (rc in (0, 4) and sorted(markers) == list(range(world))
          and len(set(exits.values())) == 1 and next(iter(exits.values())) in (0, 4))
    m0 = markers.get(0, {})
    out = {"ok": ok, "torchrun_exit": rc, "rank_exits": exits, "run_id": setup.run_id,
           "plan": setup.note, "start_step": m0.get("start_step"),
           "final_step": m0.get("final_step"), "checkpoints": m0.get("checkpoints"),
           "params_match": m0.get("params_match"), "tokens_per_s": tokens_per_second(m0, 5),
           "wall_h": round((time.monotonic() - t0) / 3600, 3), "logs": str(logs)}
    if m0.get("start_step") is not None and m0.get("final_step") is not None:
        out["tokens"] = (m0["final_step"] - m0["start_step"]) * (m0.get("tokens_per_step") or 0)
    echo("===== RUN " + ("OK" if ok else "FAILED") + "\n" + json.dumps(out, indent=2))
    return out


# ---- Modal -------------------------------------------------------------------------------

PIP = ["torch==2.11.0", "numpy==2.5.3", "tokenizers==0.23.2", "tiktoken==0.14.0",
       "safetensors==0.8.0", "torchao==0.18.0"]

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("build-essential")                  # torch.compile builds C++ wrappers
        .pip_install(*PIP)
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_dir(HERE / "quipu", "/src/quipu", ignore=["__pycache__", "memory"])
        .add_local_file(HERE / "configs" / "quipu-moe.toml", "/src/configs/quipu-moe.toml")
        .add_local_file(HERE / CONFIG, f"/src/{CONFIG}")
        .add_local_file(HERE / TOKENIZER, f"/src/{TOKENIZER}")
    )
    pool_vol = modal.Volume.from_name("quipu-pool")
    shards_vol = modal.Volume.from_name("quipu-shards")
    ckpt_vol = modal.Volume.from_name("quipu-ckpt", create_if_missing=True)
    app = modal.App("quipu-train", image=image)

    def _check_tokenizer_file() -> None:
        import hashlib
        have = hashlib.sha256((Path("/src") / TOKENIZER).read_bytes()).hexdigest()
        if have != TOKENIZER_SHA256:
            raise RuntimeError(f"{TOKENIZER} in the image is {have}, not {TOKENIZER_SHA256}")

    @app.function(gpu="L40S:2", cpu=10.0, memory=64 * 1024, timeout=SMOKE_TIMEOUT_S,
                  ephemeral_disk=64 * 1024,
                  volumes={POOL_MOUNT: pool_vol, SHARDS_MOUNT: shards_vol})
    def smoke(world: int, steps1: int, steps2: int, micro_batch: int, dispatch: str,
              compile: bool, usd_per_hour: float) -> dict:
        from quipu.shard_view import make_shard_view

        _check_tokenizer_file()
        view = make_shard_view(VIEW_DIR, Path(POOL_MOUNT), Path(SHARDS_MOUNT) / "val",
                               TOKENIZER_SHA256, val_tokenizer_sha256=TOKENIZER_SHA256)
        print(f"shard view: {view['train']['shards']} train shards, "
              f"{view['train']['tokens']:,} tokens", flush=True)
        # The trainer's own backstop: ~6 min per launch, inside the 15 min timeout.
        return run_smoke("/tmp/smoke", config=CONFIG, shard_dir=VIEW_DIR, world=world,
                         steps1=steps1, steps2=steps2, micro_batch=micro_batch,
                         dispatch=dispatch, compile=compile, device="cuda",
                         usd_per_hour=usd_per_hour, budget_usd=usd_per_hour * 0.1,
                         cwd=Path("/src"), timeout_s=SMOKE_TIMEOUT_S - 120)

    @app.function(gpu="B200", cpu=6.0, memory=40 * 1024, timeout=4 * 3600,
                  ephemeral_disk=64 * 1024,
                  volumes={POOL_MOUNT: pool_vol, SHARDS_MOUNT: shards_vol, CKPT_MOUNT: ckpt_vol})
    def train(world: int, micro_batch: int, dispatch: str, compile: bool, hours: float,
              usd_per_hour: float, decay_steps: int, timeout_s: int) -> dict:
        _check_tokenizer_file()
        ckpt_vol.reload()
        return run_training(ckpt_root=Path(CKPT_MOUNT), pool_root=Path(POOL_MOUNT),
                            val_dir=Path(SHARDS_MOUNT) / "val", view_dir=Path(VIEW_DIR),
                            world=world, micro_batch=micro_batch, dispatch=dispatch,
                            compile=compile, hours=hours, usd_per_hour=usd_per_hour,
                            decay_steps=decay_steps, cwd=Path("/src"),
                            timeout_s=timeout_s - 300, commit=ckpt_vol.commit)


# ---- local driver ------------------------------------------------------------------------


def describe(p: Plan) -> str:
    lines = [f"mode {p.mode}: {p.gpus} x {p.gpu_type} ({p.gpu_spec}), {p.cpu:g} CPU cores, "
             f"{p.memory_mib // 1024} GiB memory",
             f"price (list, estimate): ${p.usd_per_hour:.2f}/h = {p.gpus} x "
             f"${GPU_USD_PER_HOUR[p.gpu_type]:.2f} GPU + ${p.cpu * CPU_USD_PER_CORE_HOUR:.2f} CPU"
             f" + ${p.memory_mib / 1024 * MEM_USD_PER_GIB_HOUR:.2f} memory",
             f"function timeout {p.timeout_s / 60:.0f} min -> worst case ${p.worst_case_usd:.2f}"]
    if p.mode == "smoke":
        world = p.gpus
        batch = p.micro_batch * CONTEXT * 2 * world
        lines += [f"config {CONFIG}, random init, pool view at {VIEW_DIR}, micro_batch "
                  f"{p.micro_batch}, dispatch {p.dispatch}, compile {'on' if p.compile else 'off'}",
                  f"launch 1: {p.steps1} steps of {batch:,} tokens, checkpoint at {p.steps1}; "
                  f"launch 2: --resume to step {p.steps1 + p.steps2}"]
        for i, (steps, flags) in enumerate(((p.steps1, []), (p.steps1 + p.steps2, ["--resume"])), 1):
            over = smoke_overrides(shard_dir=VIEW_DIR, ckpt_dir="/tmp/smoke/ckpt",
                                   micro_batch=p.micro_batch, context=CONTEXT, world=world,
                                   steps=steps, steps1=p.steps1, dispatch=p.dispatch,
                                   compile=p.compile, usd_per_hour=p.usd_per_hour,
                                   budget_usd=p.usd_per_hour * 0.1)
            cmd = train_command(python="python", nproc=world,
                                marker_dir=f"/tmp/smoke/markers{i}", config=CONFIG,
                                run_id="smoke", run_dir="/tmp/smoke/runs", overrides=over,
                                flags=flags)
            lines.append(f"  $ {' '.join(cmd)}")
    else:
        lines += [f"trainer wall-clock budget {p.hours:g} h (train.budget_usd "
                  f"${p.trainer_budget_usd:.2f} at ${p.usd_per_hour:.2f}/h), micro_batch "
                  f"{p.micro_batch}, dispatch {p.dispatch}, compile {'on' if p.compile else 'off'}",
                  "state on Volume quipu-ckpt: " + ("decay branch of "
                  f"{p.decay_steps} steps from /stable's latest" if p.decay_steps else
                  "resume /stable, or start from /init on the first launch"),
                  f"  $ python -m torch.distributed.run --standalone --nproc_per_node={p.gpus} "
                  f"-m quipu.rankrun --marker-dir /ckpt/logs/<run>/markers -- --config {CONFIG} "
                  "--run-id <run> --run-dir /tmp/runs --device cuda --override "
                  f"data.shard_dir={VIEW_DIR} --override train.ckpt_dir=/ckpt/<stable|decay/step_S> "
                  f"--override train.micro_batch={p.micro_batch} --override "
                  f"model.moe_dispatch={p.dispatch} ... <--resume | --init-from /init/<ckpt> "
                  "--init-optimizers | --branch-from /ckpt/stable/step_S.pt>"]
    lines += [f"note: {n}" for n in p.notes]
    return "\n".join(lines)


def volume_state() -> str:
    """What quipu-ckpt holds (read-only listing), for the run-mode preflight."""
    try:
        vol = modal.Volume.from_name("quipu-ckpt")
        have = {e.path.strip("/") for e in vol.listdir("/", recursive=True)}
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {str(exc)[:120]})"
    stable = "stable/latest.pt" in have
    init = sorted(p for p in have if p.startswith("init/") and p.endswith(".pt"))
    return f"stable/latest.pt {'present' if stable else 'absent'}; init: {init or 'none'}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true", help="the multi-GPU smoke (default 2x L40S)")
    p.add_argument("--plan-only", action="store_true",
                   help="print what would run and its cost (and the guard's verdict); "
                        "launch nothing")
    p.add_argument("--gpu-type", default=None, help=f"one of {sorted(GPU_USD_PER_HOUR)}")
    p.add_argument("--gpus", type=int, default=None, help="1..8 (smoke default 2, run 1)")
    p.add_argument("--dispatch", default=None,
                   help="model.moe_dispatch: loop | grouped | padded (smoke default loop, "
                        "run default grouped)")
    p.add_argument("--micro-batch", type=int, default=None, help="smoke default 2, run 32")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile (smoke default off, run default on)")
    p.add_argument("--steps1", type=int, default=6, help="smoke: steps before the checkpoint")
    p.add_argument("--steps2", type=int, default=4, help="smoke: steps after the resume")
    p.add_argument("--hours", type=float, default=3.75,
                   help="run: the trainer's wall-clock budget (default 3 h 45 min)")
    p.add_argument("--decay-steps", type=int, default=0,
                   help="run: branch a decay of this many steps from the stable checkpoint")
    p.add_argument("--cap-usd", type=float, default=CAP_USD,
                   help=f"the guard's cap for the month (default {CAP_USD}, at most "
                        f"{HARD_LIMIT_USD})")
    h = p.add_argument_group("--here: the same smoke / run on this machine's GPUs, no Modal")
    h.add_argument("--here", action="store_true",
                   help="run on this machine (torchrun over its GPUs); no Modal, no billing "
                        "guard (the trainer's wall-clock backstop still applies)")
    h.add_argument("--pool", default=None, help="the pool: manifest.json and mixed/")
    h.add_argument("--val", default=None, help="the validation shards (val/shard_000.bin)")
    h.add_argument("--ckpt-root", default=None,
                   help="run: init/, stable/, decay/, runs/, logs/ as on the Volume")
    h.add_argument("--work", default="/tmp/quipu-train", help="scratch: the view, smoke "
                   "checkpoints, the live run log")
    h.add_argument("--usd-per-hour", type=float, default=1.0,
                   help="the trainer's backstop rate; at the default 1.0 its budget_usd is "
                        "just --hours")
    a = p.parse_args(argv)
    if not 0 < a.cap_usd <= HARD_LIMIT_USD:
        p.error(f"--cap-usd must be in (0, {HARD_LIMIT_USD}]")
    if a.here:
        need = ["pool", "val"] + ([] if a.smoke else ["ckpt_root"])
        missing = [f"--{n.replace('_', '-')}" for n in need if not getattr(a, n)]
        if missing:
            p.error(f"--here needs {', '.join(missing)}")
        if a.usd_per_hour <= 0:
            p.error("--usd-per-hour must be > 0")
    return a


def main_here(a: argparse.Namespace, plan: Plan) -> int:
    """--here: run_smoke / run_training on this machine, torchrun over its GPUs."""
    import hashlib

    from quipu.shard_view import make_shard_view

    have = hashlib.sha256((HERE / TOKENIZER).read_bytes()).hexdigest()
    if have != TOKENIZER_SHA256:
        print(f"error: {TOKENIZER} is {have}, not the model's {TOKENIZER_SHA256}",
              file=sys.stderr)
        return 2
    work = Path(a.work)
    view = work / "shards"
    print("not on Modal: no billing guard; the trainer's wall-clock backstop applies",
          flush=True)
    if plan.mode == "smoke":
        make_shard_view(view, Path(a.pool), Path(a.val), TOKENIZER_SHA256,
                        val_tokenizer_sha256=TOKENIZER_SHA256)
        res = run_smoke(work / "smoke", config=CONFIG, shard_dir=str(view), world=plan.gpus,
                        steps1=plan.steps1, steps2=plan.steps2, micro_batch=plan.micro_batch,
                        dispatch=plan.dispatch, compile=plan.compile, device="cuda",
                        usd_per_hour=a.usd_per_hour, budget_usd=a.usd_per_hour * 0.5,
                        cwd=HERE, timeout_s=2 * SMOKE_TIMEOUT_S)
        return 0 if res["passed"] else 1
    res = run_training(ckpt_root=Path(a.ckpt_root), pool_root=Path(a.pool),
                       val_dir=Path(a.val), view_dir=view, world=plan.gpus,
                       micro_batch=plan.micro_batch, dispatch=plan.dispatch,
                       compile=plan.compile, hours=plan.hours, usd_per_hour=a.usd_per_hour,
                       decay_steps=plan.decay_steps, cwd=HERE, timeout_s=plan.timeout_s,
                       local_run_dir=work / "runs")
    return 0 if res["ok"] else 1


def main(argv: list[str] | None = None, *, metered: Callable[[], float] = read_metered_cost
         ) -> int:
    a = parse_args(argv)
    try:
        plan = make_plan(a)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.here:
        if a.plan_only:
            print(f"--here: {plan.mode} on {plan.gpus} local GPU(s), micro_batch "
                  f"{plan.micro_batch}, dispatch {plan.dispatch}, compile "
                  f"{'on' if plan.compile else 'off'}; plan only: nothing launched", flush=True)
            return 0
        return main_here(a, plan)
    print(describe(plan), flush=True)
    try:
        spent = metered()
    except Exception as exc:
        print(f"budget guard: cannot read this month's metered cost ({exc}); refusing",
              file=sys.stderr)
        return 3
    ok, msg = check_budget(spent, plan.worst_case_usd, a.cap_usd)
    print(msg, flush=True)
    if plan.mode != "smoke" and modal is not None:
        print(f"quipu-ckpt: {volume_state()}", flush=True)
    if a.plan_only:
        print("plan only: nothing launched", flush=True)
        return 0
    if not ok:
        return 3
    if modal is None:
        print("error: the modal package is not installed in this Python", file=sys.stderr)
        return 2
    with modal.enable_output(), app.run():
        opts = dict(gpu=plan.gpu_spec, cpu=plan.cpu, memory=plan.memory_mib,
                    timeout=plan.timeout_s)
        if plan.mode == "smoke":
            res = smoke.with_options(**opts).remote(
                plan.gpus, plan.steps1, plan.steps2, plan.micro_batch, plan.dispatch,
                plan.compile, plan.usd_per_hour)
            print(format_smoke(res), flush=True)
            return 0 if res["passed"] else 1
        res = train.with_options(**opts).remote(
            plan.gpus, plan.micro_batch, plan.dispatch, plan.compile, plan.hours,
            plan.usd_per_hour, plan.decay_steps, plan.timeout_s)
        print(json.dumps(res, indent=2), flush=True)
        return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
