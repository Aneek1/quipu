# Amber (DGX B200 trial node): day-1 checklist

Date: 14 Oct 2026. For an 8x B200 Linux node with SSH or container access. Goal of day 1:
prove the multi-GPU path on real GPUs (2 GPUs, then 8), measure tokens/s, then start the
continued-pretraining run. Everything below uses the same code as the Modal launcher
(`scripts/remote/modal_train.py --here`), with torchrun over the node's GPUs and no Modal.

**What has never been run** (be ready for it to break): data-parallel training under NCCL
(only CPU/gloo has been tested), more than one GPU at all, the grouped MoE dispatch inside
a real training run (it was only timed in `scripts/measure_throughput.py` on one B200),
`--init-optimizers` on the real 8.1 GB checkpoint (its structure was checked against the
config, not loaded end to end), and torch.compile under torchrun.

## 0. Before you start (laptop)

- The branch is local only (`distill-v2`; GitHub has `main`). Make a bundle:
  `git bundle create quipu-distill-v2.bundle distill-v2`
- Files the node needs, with sizes:

| What | Where it is | Size |
|---|---|---|
| Repo bundle | laptop | small |
| Pretrain checkpoint `step_008129.pt` (weights + Muon/AdamW state) | laptop `remote-runs/quipu-moe/checkpoints/quipu-moe/`; also Volume `quipu-ckpt:/init/` once uploaded | 8,138,512,198 B (8.1 GB); sha256 `b3b879f5...5daf61264e` |
| Pool (data v3, chunk 0) | Volume `quipu-pool`: `/mixed/shard_00000..00100.bin` + `/manifest.json` | 20,000,006,202 B (20.0 GB) + 79 KB |
| Validation shard (English) | Volume `quipu-shards:/val/shard_000.bin`, or laptop `data/shards-moe/val/` | 20 MB |
| Tokenizer | in the repo: `artifacts/tokenizer/tokenizer.json` | sha256 `9f3e33d0...b9006943` |

## 1. Environment (node)

```bash
nvidia-smi                      # 8x B200, driver >= 570 (the lock pins torch cu128)
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone quipu-distill-v2.bundle quipu && cd quipu && git checkout distill-v2
uv sync                         # Python 3.13, torch from the cu128 index (pyproject.toml)
uv run python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
uv run python -m pytest -q tests/test_dist.py tests/test_wsd_schedule.py tests/test_modal_train.py
```

If the node gives a container instead of a shell, any recent PyTorch image with Python
3.13 works; `pip install torch==2.11.0 numpy==2.5.3 tokenizers==0.23.2 tiktoken==0.14.0
safetensors==0.8.0 torchao==0.18.0` is what the Modal image uses.

## 2. Data and checkpoint onto the node

From Modal (needs `pip install modal` and `modal token new` on the node):

```bash
mkdir -p /data/pool /data/val /data/ckpt/init
modal volume get quipu-pool /manifest.json /data/pool/manifest.json
modal volume get quipu-pool /mixed /data/pool/            # 20 GB -> /data/pool/mixed/
modal volume get quipu-shards /val/shard_000.bin /data/val/shard_000.bin
modal volume get quipu-ckpt /init/step_008129.pt /data/ckpt/init/step_008129.pt   # only after the upload
sha256sum /data/ckpt/init/step_008129.pt                   # b3b879f54776...61264e
```

Cost: Modal egress is ~$0.04/GiB, so the pool is ~$0.80, the checkpoint ~$0.30. This month
was already at $25.04 metered on 14 Oct, so check `modal billing summary` first.

Without Modal: `scp` the checkpoint and `data/shards-moe/val/shard_000.bin` from the laptop
(~47 min for the checkpoint at the laptop's ~2.9 MB/s). The pool exists only on the Modal
Volume; there is no Hugging Face copy. The published model on Hugging Face has the weights
but no optimizer state: starting from it works, but drop `--init-optimizers` (edit
`plan_run` or start by hand with `--init-from` only).

## 3. Smoke: 2 GPUs, then 8

Same two-launch smoke as Modal's: random init, a few steps, checkpoint, resume, and checks
on every rank (exit codes, params_match, resume step, disjoint data per rank, loss goes
down). Prints `SMOKE PASS` or `SMOKE FAIL` with the reasons, and tokens/s.

```bash
uv run python scripts/remote/modal_train.py --smoke --here --gpus 2 \
    --pool /data/pool --val /data/val --work /data/work
uv run python scripts/remote/modal_train.py --smoke --here --gpus 8 \
    --pool /data/pool --val /data/val --work /data/work
# then the real dispatch and compile, at the real micro batch:
uv run python scripts/remote/modal_train.py --smoke --here --gpus 8 --micro-batch 32 \
    --dispatch grouped --compile --steps1 12 --steps2 8 \
    --pool /data/pool --val /data/val --work /data/work
```

Each smoke writes two 8.1 GB checkpoints under `--work`; keep ~40 GB free there. Logs:
`/data/work/smoke/run1.log`, `run2.log`, rank records in `markers1/`, `markers2/`.

## 4. Throughput check

- The last smoke above prints tokens/s for all 8 ranks (median step after the first).
  Reference points from Modal, ONE B200, 524,288-token steps: 112,860 tok/s at micro
  batch 32 with loop dispatch, 366,229 with grouped. Eight GPUs should give well over 8x
  the one-GPU loop number if the data-parallel path scales; if they give much less, the
  per-step all-reduce or the data reading (every rank reads every batch) is the suspect.
- One GPU, for a like-for-like comparison with the Modal bench (it has no `--override`, so
  link the view the smoke built to where the config looks; ignore its "projected full
  run" line, the cpt config has no fixed end):
  `ln -s /data/work/shards data/shards-cpt && uv run python scripts/measure_throughput.py --config configs/quipu-moe-cpt.toml --micro-batch 32 --warmup 2 --steps 6 --vram-budget-gib 1000`
  (loop dispatch: the config's default; the grouped number needs `moe_dispatch = "grouped"`
  in the config).

## 5. The real launch

```bash
uv run python scripts/remote/modal_train.py --here --gpus 8 --hours 20 \
    --pool /data/pool --val /data/val --ckpt-root /data/ckpt --work /data/work
```

- First launch: `/data/ckpt/stable/` is empty, so it starts from `/data/ckpt/init/` with the
  checkpoint's weights and optimizer states, wsd re-warmup from 6e-5 to 3e-4 over 250
  steps, then constant. Later launches resume `/data/ckpt/stable/`.
- It stops itself after `--hours` (exit 4 is the normal end) and checkpoints; Ctrl+C
  also checkpoints. Checkpoints every 500 steps, 2 kept, 8.1 GB each.
- At ~1.5-3M tokens/s on 8 GPUs (unmeasured), 20 h is roughly 100-200B tokens: far more
  than the pool's 10B. The stream wraps (re-reads the pool) and logs `wraps`. Build more
  pool chunks first (`scripts/remote/modal_pool.py --chunk-tokens 10e9`, ~$2.50 each), or
  keep the run short enough to stay under ~2 epochs.
- A release (decay branch): `... --here --gpus 8 --decay-steps N --ckpt-root /data/ckpt ...`
  branches from the stable checkpoint into `/data/ckpt/decay/step_S/`; the weights to
  publish are in its `milestones/`.
- Copy `/data/ckpt/stable/` (latest.pt and its step file) and `/data/ckpt/runs/` off the
  node before the trial ends (`modal volume put quipu-ckpt ...` puts it where the monthly
  Modal run expects it).

## Decisions left open

- The stable LR (3e-4 AdamW, 0.01 Muon: half the pretrain peak) is a judgement, not tuned.
- Whether grouped dispatch is trusted for the long run before a loss comparison with loop
  on the same steps.
- Pool size: 10B tokens is not enough for a multi-day 8-GPU run.
