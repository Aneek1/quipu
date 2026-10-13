"""Quipu chat demo on Modal: the same Gradio app, served from a CPU container that
scales to zero when idle.

    modal serve hf_space_demo/modal_app.py    # temporary URL, for testing
    modal deploy hf_space_demo/modal_app.py   # stable URL

The model weights are baked into the image at build time, so a cold start only loads
them from local disk. One container at most: Gradio needs sticky sessions."""
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
)

app = modal.App("quipu-chat", image=image)


@app.function(cpu=4.0, memory=8192, max_containers=1, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def ui():
    import sys

    sys.path.insert(0, "/root")
    import gradio as gr
    from fastapi import FastAPI

    import app as chat  # loads the model once per container

    chat.demo.queue(default_concurrency_limit=1)
    return gr.mount_gradio_app(FastAPI(), chat.demo, path="/")
