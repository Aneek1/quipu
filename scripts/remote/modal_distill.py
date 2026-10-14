"""Distillation v2 on Modal: vLLM (bf16 teacher) + scripts/distill_generate_v2.py on one GPU.

Prompts and answers live on the Modal Volume "quipu-distill-v2" under /vol:
    /vol/prompts/part-NNNN.jsonl (+ manifest.json)   uploaded from the laptop
    /vol/generations/part-NNNN.modal-<gpu>.jsonl      written here, resumable

Upload a prompt part, then run for a fixed number of minutes (the generator is paused
cleanly when time is up; rerun to continue):
    modal volume put quipu-distill-v2 data/distill-v2/prompts/part-0019.jsonl /prompts/part-0019.jsonl
    modal run scripts/remote/modal_distill.py --parts 19:19 --minutes 12
Pull answers back:
    modal volume get quipu-distill-v2 /generations data/distill-v2/

Cost control: each call is capped by --minutes (plus vLLM start-up, about 2-4 min) and by
the function timeout; the Modal workspace budget limit is the hard cap."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

TEACHER = "Qwen/Qwen2.5-Coder-7B-Instruct"
REV = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
GPU = "H100"
TAG = "modal-h100-bf16"
# The repo root on the laptop; inside the container this module sits alone in /root.
HERE = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.31.0", "huggingface_hub")
    .env({"HF_HOME": "/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download as s; "
                  f"s('{TEACHER}', revision='{REV}')\"")
    .add_local_file(HERE / "scripts" / "distill_generate_v2.py", "/src/scripts/distill_generate_v2.py")
)
vol = modal.Volume.from_name("quipu-distill-v2", create_if_missing=True)
app = modal.App("quipu-distill-v2", image=image)


@app.function(gpu=GPU, cpu=8.0, memory=32768, volumes={"/vol": vol}, timeout=6 * 3600)
def generate(parts: str, minutes: float, concurrency: int = 256) -> dict:
    t0 = time.time()
    server = subprocess.Popen(
        ["vllm", "serve", TEACHER, "--revision", REV, "--dtype", "bfloat16", "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.92", "--served-model-name", "teacher", "--port", "8000",
         "--max-num-seqs", str(concurrency)],
        stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(120):
        if server.poll() is not None:
            raise RuntimeError("vLLM exited:\n" + open("/tmp/vllm.log").read()[-3000:])
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
            break
        except Exception:
            time.sleep(5)
    else:
        raise RuntimeError("vLLM not healthy:\n" + open("/tmp/vllm.log").read()[-3000:])
    ready = time.time() - t0
    print(f"vLLM ready in {ready:.0f}s", flush=True)

    # A private root per container (prompts and generations are the shared Volume dirs), so
    # each container's PAUSE file stops only that container when several run in parallel.
    run = Path("/tmp/run")
    run.mkdir(exist_ok=True)
    gens = Path("/vol/generations")
    gens.mkdir(exist_ok=True)
    for name, target in (("prompts", "/vol/prompts"), ("generations", "/vol/generations")):
        if not (run / name).exists():
            (run / name).symlink_to(target)
    pause = run / "PAUSE"
    pause.unlink(missing_ok=True)

    stop_commits = threading.Event()

    def commit_every(seconds: float = 300) -> None:   # a crash loses at most ~5 min of answers
        while not stop_commits.wait(seconds):
            try:
                vol.commit()
            except Exception as e:
                print("volume commit failed:", e, flush=True)

    threading.Thread(target=commit_every, daemon=True).start()

    def rows() -> int:
        return sum(sum(1 for _ in f.open()) for f in gens.glob(f"*.{TAG}.jsonl")) if gens.exists() else 0

    before = rows()
    timer = threading.Timer(minutes * 60, pause.touch)
    timer.start()
    g0 = time.time()
    rc = subprocess.run(
        [sys.executable, "/src/scripts/distill_generate_v2.py", "--root", str(run), "--parts", parts,
         "--direction", "down", "--endpoint", "http://127.0.0.1:8000", "--teacher-tag", TAG,
         "--revision", REV, "--concurrency", str(concurrency), "--max-new-tokens", "1024"]).returncode
    timer.cancel()
    stop_commits.set()
    gen_s = time.time() - g0
    pause.unlink(missing_ok=True)
    server.terminate()
    after = rows()
    tokens = 0
    import json
    for f in gens.glob(f"*.{TAG}.jsonl"):
        for line in f.open():
            try:
                tokens += json.loads(line).get("new_tokens", 0)
            except ValueError:
                pass
    vol.commit()
    out = {"gpu": GPU, "rc": rc, "vllm_ready_s": round(ready), "generate_s": round(gen_s),
           "new_rows": after - before, "rows_total": after, "tokens_total": tokens,
           "rows_per_s": round((after - before) / max(gen_s, 1), 2)}
    print(out, flush=True)
    return out


@app.local_entrypoint()
def main(parts: str = "19:19", minutes: float = 12.0, concurrency: int = 256, each: str = ""):
    """--each "18,17,16,15" runs one container per part, in parallel (one GPU each)."""
    if each:
        jobs = [f"{int(p)}:{int(p)}" for p in each.split(",")]
        for out in generate.starmap([(j, minutes, concurrency) for j in jobs]):
            print(out)
    else:
        print(generate.remote(parts, minutes, concurrency))
