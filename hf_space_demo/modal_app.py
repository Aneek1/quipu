"""Quipu chat demo on Modal: the same Gradio app, served from a CPU container that
scales to zero when idle.

    modal serve hf_space_demo/modal_app.py    # temporary URL, for testing
    modal deploy hf_space_demo/modal_app.py   # stable URL

The model weights are baked into the image at build time, and the container's memory is
snapshotted after the model has loaded (enable_memory_snapshot): a cold start restores
that snapshot instead of importing torch and reading 4 GB of weights again. One
container at most: Gradio needs sticky sessions.

Cloudflare Turnstile guards every path (gate.py). Its keys come from the Modal secret
"turnstile" (TURNSTILE_SITE_KEY, TURNSTILE_SECRET_KEY), which the owner creates."""
import os
import sys
from pathlib import Path

import modal

REPO = "AneekC/quipu-moe-1B-A149M-chat-distill"
HERE = Path(__file__).parent
MODEL_DIR = "/model"


def download() -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=MODEL_DIR)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gradio", "fastapi[standard]", "safetensors", "tokenizers", "huggingface_hub")
    .run_function(download)
    .env({"QUIPU_REPO": MODEL_DIR})
    .add_local_file(HERE / "app.py", "/root/app.py")
    .add_local_file(HERE / "kv_generate.py", "/root/kv_generate.py")
    .add_local_file(HERE / "gate.py", "/root/gate.py")
)

app = modal.App("quipu-chat", image=image)

with image.imports():
    # Runs at container start, before the snapshot is taken: torch, the weights and the
    # Gradio Blocks are all in the snapshot.
    sys.path.insert(0, "/root")
    import gradio as gr
    from fastapi import FastAPI

    import app as chat
    import gate


@app.function(cpu=4.0, memory=8192, max_containers=1, scaledown_window=600, timeout=600,
              enable_memory_snapshot=True, secrets=[modal.Secret.from_name("turnstile")])
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    chat.torch.set_num_threads(chat.usable_cpus())   # thread pools are not restored from a snapshot
    chat.demo.queue(default_concurrency_limit=1)
    web = FastAPI()
    gate.Gate(os.environ["TURNSTILE_SITE_KEY"], os.environ["TURNSTILE_SECRET_KEY"]).install(web)
    return gr.mount_gradio_app(web, chat.demo, path="/")
