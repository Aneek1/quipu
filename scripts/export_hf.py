"""Build the Hugging Face upload folder for quipu-114m, and prove it loads.

    python -m uv run python scripts/export_hf.py [--out hf_export]

Writes config.json, model.safetensors (fp32, final step), milestones/*.safetensors
(bf16, as trained), modeling_quipu.py and README.md. lm_head.weight is left out of
every file because it is the embedding matrix (tied); safetensors refuses to store
two names for one storage anyway.

Before returning it reloads the final weights through the standalone
modeling_quipu.py and checks the logits match quipu.model.Quipu on real validation
tokens, so a broken export never reaches the upload step.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quipu.config import load_config  # noqa: E402
from quipu.data import read_shard  # noqa: E402
from quipu.model import Quipu  # noqa: E402

TIED = "lm_head.weight"


def _strip_tied(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {k: v.detach().to(dtype).contiguous() for k, v in state.items() if k != TIED}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quipu-114m.toml")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="hf_export")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    ckpt_dir = REPO_ROOT / args.ckpt_dir
    out = REPO_ROOT / args.out
    (out / "milestones").mkdir(parents=True, exist_ok=True)

    pointer = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=True)
    final = torch.load(ckpt_dir / pointer["file"], map_location="cpu", weights_only=False)
    final_step = int(final["step"])
    print(f"final checkpoint: {pointer['file']} (step {final_step})")

    config = dataclasses.asdict(cfg.model) | {
        "architecture": "quipu",
        "tokenizer": "tiktoken:gpt2",
        "tie_word_embeddings": True,
        "trained_steps": final_step,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    save_file(_strip_tied(final["model"], torch.float32), str(out / "model.safetensors"),
              metadata={"step": str(final_step), "dtype": "float32"})

    for p in sorted((ckpt_dir / "milestones").glob("step_*.pt")):
        state = torch.load(p, map_location="cpu", weights_only=True)
        save_file(_strip_tied(state, torch.bfloat16), str(out / "milestones" / f"{p.stem}.safetensors"),
                  metadata={"step": str(int(p.stem.split("_")[1])), "dtype": "bfloat16"})
        print(f"milestone {p.stem}")

    for name in ("modeling_quipu.py", "README.md"):
        shutil.copy2(REPO_ROOT / "hf" / name, out / name)

    # Parity: the standalone loader must reproduce the training model's logits.
    spec = importlib.util.spec_from_file_location("modeling_quipu", out / "modeling_quipu.py")
    mq = importlib.util.module_from_spec(spec)
    sys.modules["modeling_quipu"] = mq  # dataclasses look the module up while building the class
    spec.loader.exec_module(mq)
    exported = mq.load(str(out))
    reference = Quipu(cfg.model)
    reference.load_state_dict(final["model"])
    reference.eval()

    val = sorted((REPO_ROOT / cfg.data.shard_dir / "val").glob("*.bin"))[0]
    tokens = torch.from_numpy(read_shard(val)[: 2 * 256 + 1].astype("int64"))
    x = tokens[:-1].view(2, 256)
    y = tokens[1:].view(2, 256)
    with torch.no_grad():
        a, b = reference(x), exported(x)
    diff = (a - b).abs().max().item()
    loss = torch.nn.functional.cross_entropy(b.view(-1, b.shape[-1]), y.reshape(-1)).item()
    print(f"parity: max |logit diff| = {diff:.2e}; loss on 512 val tokens = {loss:.3f}")
    if diff > 1e-4:
        print("FAIL: exported model does not match the training model", file=sys.stderr)
        return 1
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
