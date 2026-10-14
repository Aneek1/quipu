"""Training throughput of quipu-moe on Modal GPUs: scripts/measure_throughput.py with the
config of the real run (remote-runs/quipu-moe/results/moe/run_config.toml).

One train shard and the val shard are uploaded to the Volume "quipu-shards" once; each
GPU then times real training steps (compile, Muon, accumulation) at several micro
batches. The 5090 run measured 44,331 tokens/s at micro_batch 4.

    modal run scripts/remote/modal_bench.py --gpus "B200,H100" --micro-batches "8,16,32"
    modal run scripts/remote/modal_bench.py --gpus B200 --micro-batches 32 --dispatch grouped
"""
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/src")
RUN_CONFIG = "remote-runs/quipu-moe/results/moe/run_config.toml"
SHARDS = ["train/shard_000.bin", "val/shard_000.bin"]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential")                          # torch.compile builds C++ wrappers
    .pip_install("torch>=2.9.1", "numpy>=2.1", "tiktoken>=0.8", "safetensors>=0.8.0",
                 "tokenizers>=0.23.2", "torchao>=0.18.0", "datasets>=3.0", "tqdm>=4.66")
    .add_local_dir(HERE / "quipu", "/src/quipu", ignore=["__pycache__", "memory"])
    .add_local_file(HERE / "scripts" / "measure_throughput.py", "/src/scripts/measure_throughput.py")
    .add_local_file(HERE / RUN_CONFIG, "/src/run_config.toml")
)
vol = modal.Volume.from_name("quipu-shards", create_if_missing=True)
app = modal.App("quipu-bench", image=image)


def bench(micro_batches: str, dispatch: str) -> str:
    # The run config points at data/shards-moe; the Volume holds the same layout.
    cfg = Path("/src/run_config.toml").read_text()
    cfg = cfg.replace('shard_dir = "data/shards-moe"', 'shard_dir = "/shards"')
    assert "moe_dispatch" not in cfg and "[model]\n" in cfg
    cfg = cfg.replace("[model]\n", f'[model]\nmoe_dispatch = "{dispatch}"\n', 1)
    Path("/tmp/bench.toml").write_text(cfg)
    out = []
    for mb in micro_batches.split(","):
        r = subprocess.run(
            ["python", "scripts/measure_throughput.py", "--config", "/tmp/bench.toml",
             "--micro-batch", mb, "--warmup", "2", "--steps", "6", "--vram-budget-gib", "1000"],
            cwd="/src", capture_output=True, text=True, env={"PYTHONPATH": "/src", "PATH": "/usr/local/bin:/usr/bin:/bin"})
        log = r.stdout + r.stderr[-3000:]
        print(f"--- micro_batch {mb} (exit {r.returncode})\n{log}", flush=True)
        out.append(f"--- micro_batch {mb} (exit {r.returncode})\n" +
                   "\n".join(l for l in log.splitlines()
                             if "tokens/s" in l or "peak VRAM" in l or "Error" in l))
    return "\n".join(out)


kw = dict(volumes={"/shards": vol}, cpu=8.0, memory=65536, timeout=45 * 60)


@app.function(gpu="B200", **kw)
def bench_b200(micro_batches: str, dispatch: str) -> str:
    return bench(micro_batches, dispatch)


@app.function(gpu="H100", **kw)
def bench_h100(micro_batches: str, dispatch: str) -> str:
    return bench(micro_batches, dispatch)


@app.local_entrypoint()
def main(gpus: str = "B200,H100", micro_batches: str = "8,16,32", dispatch: str = "loop"):
    have = set()
    for d in ("train", "val"):
        try:
            have |= {e.path for e in vol.listdir(d)}
        except Exception:                                    # not uploaded yet
            pass
    with vol.batch_upload(force=True) as up:
        for s in SHARDS:
            if s not in have:
                print("uploading", s, flush=True)
                up.put_file(str(HERE / "data" / "shards-moe" / s), s)
    fns = {"B200": bench_b200, "H100": bench_h100}
    calls = {g: fns[g].spawn(micro_batches, dispatch) for g in gpus.split(",")}
    for g, c in calls.items():
        print(f"===== {g}\n{c.get()}", flush=True)
