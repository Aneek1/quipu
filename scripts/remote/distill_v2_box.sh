#!/usr/bin/env bash
# Distillation v2 on a rented box: vLLM (bf16 teacher) + generator for parts 0..4,
# stopping at the spend cap. Run in tmux from /workspace/quipu after setup.sh:
#   bash scripts/remote/distill_v2_box.sh <usd_per_hour> <teacher_revision> [budget_usd]
# Does NOT stop vLLM or the spend ledger on exit: the controller does that after copy-back
# (python3 -m quipu.spend stop). Exit codes: 2 = vLLM failed / prompt parts missing or
# altered; otherwise the script ends 0 and the generator's code is on the GEN_EXIT= line.
set -euo pipefail
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
RATE=$1; REV=$2; BUDGET=${3:-3.00}
# vLLM 0.31.0 = newest stable on PyPI (2026-10-12); supports Ampere (RTX 3090, sm_86) bf16
# and Qwen2.5 (Qwen2ForCausalLM).
VLLM_VERSION=0.31.0
cd /workspace/quipu

# Verify uploaded prompt parts against the manifest before spending anything.
python3 - <<'PY' || exit 2
import hashlib, json, sys
from pathlib import Path
d = Path("data/distill-v2/prompts")
want = {e["file"]: e["sha256"] for e in json.loads((d / "manifest.json").read_text())["parts"]}
found = 0
for i in range(5):
    name = f"part-{i:04d}.jsonl"
    p = d / name
    if not p.exists():
        continue
    found += 1
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if want.get(name) != h.hexdigest():
        print(f"PROMPT MISMATCH {name}")
        sys.exit(1)
    print(f"prompt OK {name}")
if not found:
    print("no prompt parts 0..4 present")
    sys.exit(1)
PY

python3 -m quipu.spend start --usd-per-hour "$RATE"
python3 -m venv /workspace/vllm-env
/workspace/vllm-env/bin/pip install -q "vllm==$VLLM_VERSION"
/workspace/vllm-env/bin/vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --revision "$REV" \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.92 \
  --served-model-name teacher --port 8000 > /workspace/vllm.log 2>&1 &
VLLM_PID=$!
for _ in $(seq 1 120); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 "$VLLM_PID" 2>/dev/null || break
  sleep 10
done
if ! curl -sf localhost:8000/health >/dev/null; then
  echo "vLLM did not start"; tail -50 /workspace/vllm.log
  kill "$VLLM_PID" 2>/dev/null || true
  exit 2
fi
mkdir -p data/distill-v2
rc=0
uv run python scripts/distill_generate_v2.py --parts 0:4 --direction up \
  --endpoint http://127.0.0.1:8000 --teacher-tag bf16-vllm --revision "$REV" \
  --concurrency 128 --max-new-tokens 1024 --budget-usd "$BUDGET" || rc=$?
echo "GEN_EXIT=$rc"
echo "GEN_EXIT=$rc" >> /workspace/distill_v2_box.log
