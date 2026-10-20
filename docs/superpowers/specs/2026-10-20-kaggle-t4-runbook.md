# Kaggle 2x T4 runbook (quipu-moe continued pretraining)

Config: configs/quipu-moe-cpt-t4.toml. Spec: 2026-10-20-t4-low-memory-training-design.md.

## Once (owner)

0. Push the branch with the feature: `git push -u origin distill-v2`. It is not on
   GitHub yet, and every session below clones it.
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

## Budget

~1B tokens is ~1,900 steps (524,288 tokens a step). At 25-60 s a step (estimate;
the first session measures it) a session's 11.25 h is ~700-1,600 steps, so the
part-1 data is about 2 sessions. After that the stream re-reads it: the `wraps`
counter (printed on the step line, and in the run log's steps) goes above 0. Watch
it; repeated passes over 1B tokens are not what this run is for.

More data: adding shards to the dataset between sessions changes the stream, so the
saved stream position no longer means anything and `--resume` would read the wrong
tokens. Add data only by starting a new run id from the latest checkpoint:
`--init-from /kaggle/working/ckpt/latest.pt --init-optimizers` with a new
`--run-id` (weights and optimizer state carry over; the step counter, stream and LR
schedule restart at 0, including the 250-step re-warmup).

## Every session: setup

Paths below assume the dataset mount names above; check them with `ls /kaggle/input`.

Do not `pip install -e .`: pyproject.toml pins Python >= 3.13 and torch >= 2.9.1,
and pip could replace Kaggle's CUDA build of torch. The package runs from the repo
root with `PYTHONPATH=.` instead. It needs Python >= 3.11 (config.py uses tomllib;
the quipu package parses under the 3.11 grammar), torch, numpy and tokenizers
(quipu.train imports quipu.bpe, which imports tokenizers at the top); Kaggle's image
has all of them. Install only what an import error names, e.g.
`pip install -q tokenizers`, never torch.

```bash
cd /kaggle/working
git clone -b distill-v2 https://github.com/Aneek1/quipu.git && cd quipu
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python -c "import numpy, tokenizers, quipu.train; print('imports ok')"
nvidia-smi                                                  # two T4s, ~15 GB each
mkdir -p artifacts/tokenizer
cp /kaggle/input/quipu-data-v3-part1/tokenizer.json artifacts/tokenizer/

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

(In a notebook, `export` in one `%%bash` cell does not carry to the next; keep each
session's commands in one cell, or prefix each command with the two variables.)

## First session (interactive, about 1.5 hours): checks, in this order

### (a) Environment

The setup block above: Python version, torch version and CUDA, two "Tesla T4"
devices, `imports ok`, the shard view's token count.

### (b) Memory at full depth

```bash
torchrun --standalone --nproc_per_node=2 scripts/lowmem_probe.py memory \
  --shards /kaggle/working/shards/train --layers 16 --micro-batch 2 --t4 \
  --out /kaggle/working/lowmem_probe.json
```

It trains until 2 optimizer steps are taken, so the optimizer state is allocated.
Pass: `valid` true, `peak_reserved_gib` under ~14 and `peak_allocated_gib` under
~13 (rank 0 reports; the ranks hold the same tensors). If it is close, lower
micro_batch to 1, or loss_chunk_tokens to 1024.

### (c) Short check run (own run id and directories, deleted afterwards)

Starts from step_008129 exactly as the real run will, for about 50 steps:

```bash
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-check \
  --run-dir /kaggle/working/check/runs --device cuda \
  --override data.shard_dir=/kaggle/working/shards \
  --override train.ckpt_dir=/kaggle/working/check/ckpt \
  --override train.ckpt_every=100000 --override train.eval_every=25 \
  --override train.budget_usd=1.0 \
  --init-from /kaggle/input/quipu-moe-step8129/step_008129.pt --init-optimizers
```

The 1-hour budget stops it with a checkpoint; interrupt it once step 50 is logged
to save quota. Then:

```bash
python - <<'EOF'
import json, math
r = json.load(open("/kaggle/working/check/runs/kaggle-check.json", encoding="utf-8"))
s = r["steps"]
print(len(s), "steps; non-finite:", sum(not math.isfinite(e["train_loss"]) for e in s))
print("last:", s[-1])          # loss_scale and overflows (fp16 skips) are logged here
print("evals:", r.get("evals"))
EOF
rm -rf /kaggle/working/check
```

Pass: at least 50 steps, no non-finite train loss. Overflows (the scale halving and
a skipped step) are fine; they show as `loss_scale` and `overflows`. Note the step
time and tokens/s from the step lines.

Then the val loss of the step_008129 weights under fp16. The original run logged
no eval at step 8129 (its evals are every 250 steps; the last is step 8000,
English val_loss 2.9788 in remote-runs/quipu-moe/results/moe/runs/quipu-moe.json,
bf16). So compare fp16 with fp32 on the same weights and the same val tokens, and
check both against step 8000's number:

```bash
python - <<'EOF'
import torch
from quipu.config import load_config
from quipu.eval import estimate_loss
from quipu.loader import TokenStream
from quipu.model_factory import build_model

cfg = load_config("configs/quipu-moe-cpt-t4.toml")
model = build_model(cfg.model)
ckpt = torch.load("/kaggle/input/quipu-moe-step8129/step_008129.pt",
                  map_location="cpu", weights_only=False)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
del ckpt
model = model.to("cuda")
# The original eval read eval_batches 20 x micro_batch 4 x 2048 tokens from the start
# of val; micro_batch 1 x 80 batches reads the same 80 rows in the same order.
val = TokenStream("/kaggle/working/shards/val", 1, cfg.model.context)
fp16 = estimate_loss(model, val, 80, "cuda", amp_dtype=torch.float16)
fp32 = estimate_loss(model, val, 80, "cuda", amp=False)
print(f"fp16 {fp16:.4f}  fp32 {fp32:.4f}  rel {abs(fp16 - fp32) / fp32:.2%}  "
      f"logged step 8000 (bf16) 2.9788")
EOF
```

Pass: fp16 within 1% of fp32, and both within ~1% of 2.9788 (step 8129 ended the
run at lr_min, so a little below step 8000's value is expected).

### (d) Only then: the long run, as a committed version

Set micro_batch (the largest that passed (b)) and ckpt_every (~30-45 min of steps
at the measured step time) in configs/quipu-moe-cpt-t4.toml, commit and push. Then
"Save Version -> Save & Run All (commit)": it runs in the background in a fresh
session, so the notebook holds only the setup block and this:

```bash
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-cpt \
  --run-dir /kaggle/working/runs --device cuda \
  --override data.shard_dir=/kaggle/working/shards \
  --init-from /kaggle/input/quipu-moe-step8129/step_008129.pt --init-optimizers
```

If a check fails (memory over, non-finite losses that repeat, fp16 val more than 1%
off): stop; T4s go back to eval and generation only (spec, "Testing").

## Later sessions

Attach the previous version's output as an input, copy its `ckpt/` and `runs/` back
to /kaggle/working (the trainer resumes from ckpt/latest.pt and appends to
runs/kaggle-cpt.json), then resume, again as a committed version:

```bash
PREV=/kaggle/input/<previous version output>
cp -r $PREV/ckpt $PREV/runs /kaggle/working/
torchrun --standalone --nproc_per_node=2 -m quipu.train \
  --config configs/quipu-moe-cpt-t4.toml --run-id kaggle-cpt \
  --run-dir /kaggle/working/runs --device cuda \
  --override data.shard_dir=/kaggle/working/shards \
  --resume
```

The config's budget (11.25 "dollars" at 1 per hour) stops it with a checkpoint before
Kaggle's 12-hour limit.

## Disk and RAM

Disk: /kaggle/working holds 20 GB. A checkpoint with bf16 momentum is ~6.2 GB (fp32
weights 4.0 + bf16 Muon momentum ~1.9 + fp32 AdamW moments ~0.3); while one is being
saved the previous one still exists (ckpt_keep = 1 prunes it after the save), so
expect ~12.4 GB at peak. Keep nothing else large in /kaggle/working (delete the
check run's directory, (c)).

RAM: `--init-from` loads the whole 8.1 GB step_008129.pt into CPU RAM on each rank,
~16 GB of the notebook's ~29 GB, while the model is being built; a resume loads the
~6.2 GB checkpoint per rank (~12.4 GB).
