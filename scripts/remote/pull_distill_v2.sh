#!/usr/bin/env bash
# Copy distill v2 generations back from the box: bash scripts/remote/pull_distill_v2.sh HOST PORT
# Copies every part-*.bf16-vllm.* file (finished and partial). Finished parts (.done present)
# are sha256-checked against the .done sidecar; prints OK <file>, PARTIAL <file> <rows>,
# or HASH MISMATCH <file>. Exit 1 on any mismatch, 2 if scp fails.
# A PARTIAL file may end in a torn (incomplete) last line; consumers must tolerate that.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=$1; PORT=$2
DEST=data/distill-v2/generations
mkdir -p "$DEST"
scp -P "$PORT" -o StrictHostKeyChecking=no \
  "root@$HOST:/workspace/quipu/data/distill-v2/generations/part-*.bf16-vllm.*" "$DEST/" \
  || { echo "scp failed (no generations yet, or connection)"; exit 2; }
bad=0
for f in "$DEST"/part-*.bf16-vllm.jsonl; do
  [ -e "$f" ] || continue
  d=${f%.jsonl}.done
  if [ -e "$d" ]; then
    want=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$d" | tr -d '\r')
    got=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$want" = "$got" ]; then
      echo "OK $(basename "$f")"
    else
      echo "HASH MISMATCH $(basename "$f")"; bad=1
    fi
  else
    rows=$(wc -l < "$f" | tr -d ' \r')
    echo "PARTIAL $(basename "$f") $rows"
  fi
done
exit "$bad"
