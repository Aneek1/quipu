#!/usr/bin/env bash
# Distillation v2 on a rented box: vLLM (bf16 teacher) + generator for parts 0..BOX_LAST (default 11),
# stopping at the spend cap. Run in tmux from /workspace/quipu after setup.sh:
#   bash scripts/remote/distill_v2_box.sh <usd_per_hour> <teacher_revision> [budget_usd] [box_start_epoch] \
#     > /workspace/distill_v2.log 2>&1
# Everything (including GEN_EXIT=) goes to stdout; the caller redirects it to one log.
# Rerunnable: after GEN_EXIT=3 rerun the same command; after GEN_EXIT=4 kill vLLM first, then rerun.
# Does NOT stop vLLM or the spend ledger on exit: the controller does that after copy-back
# (python3 -m quipu.spend stop). Exit 2 = vLLM/GPU failure or prompt parts missing/altered;
# otherwise the script ends 0 and the generator's code is on the GEN_EXIT= line.
set -euo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/workspace/hf
# FlashInfer's sampler JIT-compiles on first use (needs ninja + nvcc, absent from the runtime
# image); decoding is greedy, so vLLM's own sampler is enough.
export VLLM_USE_FLASHINFER_SAMPLER=0
RATE=$1; REV=$2; BUDGET=${3:-3.00}; BOX_START=${4:-}
# Box takes parts 0..11 (600k prompts): ~370-token answers mean $3 of 3090 time needs more than 250k.
# The laptop works 19 down to 12, so the ranges cannot overlap.
export BOX_LAST=${BOX_LAST:-11}
# vLLM 0.31.0 = newest stable on PyPI (2026-10-12); supports Ampere (RTX 3090, sm_86) bf16
# and Qwen2.5 (Qwen2ForCausalLM).
VLLM_VERSION=0.31.0
VENV=/workspace/vllm-env
cd /workspace/quipu

# Verify ALL uploaded prompt parts 0..BOX_LAST against the manifest before spending anything.
python3 - <<'PY' || exit 2
import hashlib, json, os, sys
from pathlib import Path
d = Path("data/distill-v2/prompts")
want = {e["file"]: e["sha256"] for e in json.loads((d / "manifest.json").read_text())["parts"]}
bad = False
for i in range(int(os.environ["BOX_LAST"]) + 1):
    name = f"part-{i:04d}.jsonl"
    p = d / name
    if not p.exists():
        print(f"PROMPT MISSING {name}")
        bad = True
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if want.get(name) != h.hexdigest():
        print(f"PROMPT MISMATCH {name}")
        bad = True
    else:
        print(f"prompt OK {name}")
sys.exit(1 if bad else 0)
PY

if ! python3 -m quipu.spend show --usd-only >/dev/null 2>&1; then
  if [ -n "$BOX_START" ]; then
    python3 -m quipu.spend start --usd-per-hour "$RATE" --box-start "$BOX_START"
  else
    python3 -m quipu.spend start --usd-per-hour "$RATE"
  fi
fi

VLLM_PID=
if curl -sf localhost:8000/health >/dev/null; then
  echo "vLLM already serving on :8000, skipping launch"
else
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
  "$VENV/bin/python" -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "vllm torch cannot see the GPU (driver too old?)"; nvidia-smi || true; exit 2; }
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
  "$VENV/bin/vllm" serve Qwen/Qwen2.5-Coder-7B-Instruct --revision "$REV" \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.92 \
    --served-model-name teacher --port 8000 > /workspace/vllm.log 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && break
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 10
  done
  if ! curl -sf localhost:8000/health >/dev/null; then
    echo "vLLM did not start"; tail -50 /workspace/vllm.log
    kill "$VLLM_PID" 2>/dev/null || true
    exit 2
  fi
fi

rc=0
uv run --frozen python scripts/distill_generate_v2.py --parts "0:$BOX_LAST" --direction up \
  --endpoint http://127.0.0.1:8000 --teacher-tag bf16-vllm --revision "$REV" \
  --concurrency 128 --max-new-tokens 1024 --budget-usd "$BUDGET" || rc=$?
echo "GEN_EXIT=$rc"
