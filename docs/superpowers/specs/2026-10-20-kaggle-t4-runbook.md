# Kaggle 2x T4 runbook (quipu-moe continued pretraining)

Config: configs/quipu-moe-cpt-t4.toml. Spec: 2026-10-20-t4-low-memory-training-design.md.

## Once (owner)

1. Private Kaggle dataset `quipu-data-v3-part1`, laid out like the pool so that
   `quipu.shard_view.make_shard_view` can read it:

       manifest.json        the pool manifest, its "shards" list cut to the files uploaded
       mixed/shard_*.bin    about 1B tokens of data v3 chunk 0 (uint16 .bin, ~2 GB),
                            from the Modal volume quipu-pool
       val/shard_*.bin      the English validation shard, from the Modal volume
                            quipu-shards (val/)
       tokenizer.json       artifacts/tokenizer/tokenizer.json (not in git; sha256
                            9f3e33d074f3a0d7da234eec11a59de1e20aa756855d1ef552fc9706b9006943)

   Download with `modal volume get` (Git Bash, MSYS_NO_PATHCONV=1), upload in the
   Kaggle UI. make_shard_view refuses a manifest that lists shards which are not
   there, so cut the list before uploading:

   ```python
   import json, pathlib
   d = pathlib.Path("quipu-data-v3-part1")
   m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
   have = {p.name for p in (d / "mixed").glob("shard_*.bin")}
   m["shards"] = [s for s in m["shards"] if s["file"] in have]
   (d / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
   ```

2. Private Kaggle dataset `quipu-moe-step8129`:
   remote-runs/quipu-moe/checkpoints/quipu-moe/step_008129.pt (8.1 GB).
3. Notebook settings: Accelerator "GPU T4 x2", Internet on (clone from GitHub),
   Persistence: files.

## Every session

Paths below assume the dataset mount names above; check them with `ls /kaggle/input`.

```bash
cd /kaggle/working
git clone https://github.com/Aneek1/quipu.git && cd quipu && pip install -q -e .
mkdir -p artifacts/tokenizer
cp /kaggle/input/quipu-data-v3-part1/tokenizer.json artifacts/tokenizer/
nvidia-smi                                                  # two T4s

# The trainer's shard view (as scripts/remote/modal_train.py builds it): train/ and
# val/ as symlinks into the read-only input, manifest.json with the tokenizer hash.
python - <<'EOF'
import hashlib
from pathlib import Path
from quipu.shard_view import make_shard_view
data = Path("/kaggle/input/quipu-data-v3-part1")
sha = hashlib.sha256(Path("artifacts/tokenizer/tokenizer.json").read_bytes()).hexdigest()
v = make_shard_view("/kaggle/working/shards", data, data / "val", sha,
                    val_tokenizer_sha256=sha)
print(v["train"]["shards"], "train shards,", v["train"]["tokens"], "tokens")
EOF
```

First session only:

```bash
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-cpt \
  --run-dir /kaggle/working/runs --device cuda \
  --override data.shard_dir=/kaggle/working/shards \
  --init-from /kaggle/input/quipu-moe-step8129/step_008129.pt --init-optimizers
```

Later sessions: attach the previous version's output as an input, copy its `ckpt/`
and `runs/` back to /kaggle/working (the trainer resumes from ckpt/latest.pt and
appends to runs/kaggle-cpt.json), then resume:

```bash
PREV=/kaggle/input/<previous version output>
cp -r $PREV/ckpt $PREV/runs /kaggle/working/
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-cpt \
  --run-dir /kaggle/working/runs --device cuda \
  --override data.shard_dir=/kaggle/working/shards \
  --resume
```

Run it as "Save Version -> Save & Run All (commit)" so it continues in the background.
The config's budget (11.25 "dollars" at 1 per hour) stops it with a checkpoint before
Kaggle's 12-hour limit.

Disk: /kaggle/working holds 20 GB. A checkpoint is ~8.1 GB, and while one is being
saved the previous one still exists (ckpt_keep = 1 prunes it after the save), so
expect ~16.2 GB at peak. Keep nothing else large in /kaggle/working.

## First session: checks before a long run (about 1 hour)

- Memory at full depth, before the real run:

  ```bash
  torchrun --standalone --nproc_per_node=2 scripts/lowmem_probe.py memory \
    --shards /kaggle/working/shards/train --layers 16 --micro-batch 2 --t4 \
    --out /kaggle/working/lowmem_probe.json
  ```

  peak_gib per rank must be under 14.5. If it is close, lower micro_batch, or lower
  loss_chunk_tokens to 1024-2048 (each 4096-token chunk peaks at ~2.4 GB).
- First logged train loss within 1% of the bf16 run's logged loss at step 8129
  (remote-runs/quipu-moe/results logs); 50 steps with no non-finite loss (fp16
  overflows that halve the scale are fine; they show as loss_scale and overflows in
  the run log).
- Note step time and tokens/s; set micro_batch (largest that fits) and ckpt_every
  (~30-45 min) and commit the change to configs/quipu-moe-cpt-t4.toml.

If the loss check fails or non-finite steps repeat: stop, T4s go back to eval and
generation only (spec, "Testing").
