# quipu-moe-a1b: a ~1B-active model for a DGX B200 trial (design)

Date: 14 Oct 2026. Status: draft for the owner. Config: `configs/quipu-moe-a1b.toml`
(test: `tests/test_config_a1b.py`).

## Goal

If the one-week free trial of an NVIDIA DGX B200 (8x B200, 178 GiB each) comes
through, train quipu's next model from scratch on it: ~1B active parameters, the
retrained tokenizer (with Arabic-script Urdu) and the data v3 pool
(`2026-10-14-data-v3-continued-pretraining-design.md`). If it does not, or if it is
a single B200, keep continuing the published 1B-A149M instead (see "Fallback").

## Constraints that set the size

- **Full replica per GPU.** `quipu/dist.py` is plain data parallel: every rank holds
  fp32 weights, fp32 grads and the optimizer state, with no sharding. So the
  model, its state and one micro-batch's activations must fit one 178 GiB B200, and
  it is the TOTAL parameter count (experts included) that this caps.
- **Bytes per parameter** (read from the code): parameters are fp32 `nn.Parameter`s
  under bf16 autocast, so a step holds fp32 weights (4) + fp32 grads (4) + a bf16 copy
  of each weight saved for backward (2; ExpertBank casts its banks once per forward,
  autocast caches the Linear casts). Muon (`quipu/optim/muon.py`) keeps one fp32
  momentum buffer (`zeros_like(grad)`, 4): **14 B/param** for every hidden matrix and
  expert bank. AdamW (embedding/tied head, router, norms) keeps two fp32 moments (8):
  18 B/param. Plus two 256 MiB all-reduce buckets.
- **No activation checkpointing** for the MoE layers (only the AttnRes mix has one),
  so activations scale with micro_batch x layers x width.

## Parameter counts

Counted with quipu's own factory (`build_model` on `torch.device("meta")`, no memory),
tied embedding once, routed experts at top_k / n_experts, embeddings in both (the
rule `quipu/model_card.count_params` uses). The rule reproduces the published model
exactly: **998.0M total / 148.7M active** for `configs/quipu-moe.toml`.

## Calibration (the current model, one B200)

Measured: grouped dispatch, micro_batch 32 (65,536 tokens per micro-step), 524,288-token
batches, Muon, torch.compile: **366,229 tok/s, 92.4 GiB peak**.

- Static: 959M Muon-side params x 14 B + 38.5M AdamW-side x 18 B = 13.2 GiB.
- LM head: bf16 logits plus the fp32 log-softmax cross-entropy saves, ~6 B x vocab per
  token = 18.0 GiB at 65,536 tokens (an assumption, not measured on its own).
- That leaves 61.2 GiB for 16 layers x 65,536 tokens: ~61 KB per token per layer.
- A component estimate of what each layer saves for backward (fp32 residual and norm
  inputs, q/k/v/o and the repeated K/V, SDPA output; the router's fp32 input, the
  k sorted copies of x, the expert gate/up/activation/scaled outputs, the shared
  expert), in bytes per token per layer:
  `42 d + 4 k d + 10 k h_e + 8 h_s`, gives 64.5 GiB for the current model, so a
  **calibration factor 0.95** is applied to it for the candidates.
- FLOPs per token = 6 x active params + 6 x n_layer x context x d_model (causal
  attention): 1.04 GFLOP now, so 366,229 tok/s is ~382 TFLOP/s, **~17% MFU** against
  the B200's ~2.25 PFLOP/s dense bf16.

Throughput for a candidate = 366,229 x (1.04 GFLOP / its FLOPs per token), i.e. **the
assumption is that MFU stays at ~17%**. Wider matmuls (d 2048 vs 768) usually raise
MFU, so this is meant to be the low end; it is unmeasured.

## Candidates

All: vocab 49,152, context 2048, head_dim 128, 4 KV heads, one shared expert,
SwiGLU, Muon, grouped dispatch. Memory is the per-GPU projection (static + activations
at that micro_batch); A = 1 B200 x 150 h; B = 8 B200 x 150 h at 87.5% scaling.

| | Shape | Total | Active | Static | micro_batch 8 | tok/s per B200 | A: tokens (per active param) | B: tokens (per active param) |
|---|---|---|---|---|---|---|---|---|
| C1 | d 1536, 24 layers, 128 experts x 384, top-8, shared 1536 | 5.84B | 0.741B | 76.9 GiB | 136 GiB | 78.0k | 42B (57) | 295B (398) |
| C2 | d 2048, 20 layers, 64 experts x 768, top-4, shared 2048 | 6.60B | 0.942B | 87.0 GiB | 140 GiB | 62.0k | 34B (36) | 235B (249) |
| **C6** | **d 2048, 22 layers, 64 experts x 512, top-6, shared 2048** | **5.04B** | **1.026B** | **66.7 GiB** | **129 GiB** | **56.9k** | **31B (30)** | **215B (210)** |
| C5 | d 2048, 24 layers, same experts as C6 | 5.49B | 1.111B | 72.5 GiB | 140 GiB | 52.6k | 28B (26) | 199B (179) |
| (rejected) | d 2048, 24 layers, 96 experts x 512, top-6 | 7.91B | 1.112B | 104 GiB | 172 GiB | 52.5k | | |

micro_batch 16 is over the card for every row (192-208 GiB); 12 would fit C6
(~161 GiB) but does not divide a power-of-two batch. A replica caps total
parameters at ~6-7B, so a 1B-active model here is ~5-6x sparse, less than quipu-moe's
6.7x; going sparser needs a sharded optimizer (ZeRO-1/FSDP), which is not built.

## Recommendation: C6 (`quipu-moe-5B-A1B`)

5,040.3M total / 1,026.4M active (926M non-embedding). Chosen over:
- **C2:** similar active size, but 1.6B more parameters to hold (11 GiB less margin),
  coarser experts (d/2.7, top-4) and two fewer layers. Finer experts at the same active
  width (DeepSeekMoE's granularity argument) are the better use of the budget.
- **C5:** 8% more active compute for two more layers, 11 GiB less margin, fewer tokens.
- **C1:** cheapest and gets the most tokens per parameter, but 0.74B active is below
  the target. It is the step-down if the first hour shows C6's MFU well below 17%.

### Memory at micro_batch 8 (per GPU)

| Item | GiB |
|---|---|
| fp32 weights | 18.8 |
| fp32 grads | 18.8 |
| Muon momentum (fp32, 4.94B params) | 18.4 |
| AdamW moments (103.6M params) | 0.8 |
| bf16 weight copies saved for backward | 9.4 |
| all-reduce buckets | 0.5 |
| activations, 22 layers x 16,384 tokens | 58.1 |
| LM head (logits + log-softmax) | 4.5 |
| **Projected peak** | **~129 of 178** |

The ~49 GiB margin (27%) covers the estimate's error (one calibration point), the
allocator's fragmentation under variable expert counts, and transients
(`params_match` makes a float64 copy of one tensor at a time, ~0.5 GB for an expert
bank; Muon's per-tensor Newton-Schulz workspace).

### Throughput and tokens

6.71 GFLOP per token, 6.4x the current model: ~57k tok/s per B200 at 17% MFU
(84k at 25%, 101k at 30%).
- **A (1 GPU, 150 h): ~31B tokens, ~30 per active parameter**: compute-optimal-ish
  (Chinchilla ~20), not the over-training a small model should get. Not worth a new
  model; see "Fallback".
- **B (8 GPUs, 150 h): ~215B tokens at 17% MFU (317B at 25%), ~210 per active
  parameter.** The config trains **160B** (76,293 steps): ~75% of the low projection,
  so the cosine schedule ends inside the week if scaling or MFU fall short. The
  trainer's schedule is cosine to a fixed `total_tokens`; a run cut short ends mid
  decay. Re-set `total_tokens` from the first hour's measured tok/s.

### Batch, learning rate, warmup, checkpoints

- **batch_tokens 2,097,152** (1,024 x 2048), grad_accum 128, 16 per rank on 8 GPUs.
  Every rank runs the whole Muon step (~74 TFLOP of Newton-Schulz across 22 layers'
  banks, not split over ranks) and one ~20 GB fp32 gradient all-reduce per step; at
  2M tokens a step is ~5.3 s on 8 GPUs and those fixed costs are ~5% of it (at 0.5M
  they would be ~20%). DeepSeek LLM's batch-size fit (B = 0.292 C^0.3271) gives ~2.2M
  tokens at this run's ~1.1e21 FLOPs; Muon is reported to tolerate large batches.
  The same file runs on 1 GPU (grad_accum 128), only slowly.
- **lr 4e-4** (AdamW groups only: embedding/tied head, router, norms), down from 6e-4:
  between sqrt(768/2048) = 0.61x width scaling and DeepSeek's compute fit (~7e-4 at
  this batch). lr_min = lr / 10 as before.
- **muon_lr 0.02 unchanged.** Muon's step is a spectral-norm step with a
  max(1, out/in)^0.5 shape scale, meant to carry across width; it was A/B-tested only
  at d 768, so the first-hour loss curve decides (step-down 0.015).
- **warmup_steps 1000** (~2.1B tokens, 1.3% of the run; quipu-moe had 500 x 0.5M).
- **ckpt_every 500** (~44 min on 8 GPUs), eval_every 250, ckpt_keep 3. On one B200 a
  step is ~37 s: override ckpt_every to ~75.
- micro_batch 8: the trainer refuses a different micro_batch once `plan.json`
  exists, so the first-hour peak check comes before the real launch.

### Vocabulary: keep 49,152

The tokenizer is retrained with `urd_Arab` in its sample (data v3 spec), which moves
merges toward Arabic script without growing the table; Urdu is a small share (the
Urdu slice of the 10% nine-language share), so it does not justify many rows. The
shards are uint16 (`quipu/data.py`), so anything above 65,536 needs a format change;
65,536 itself would add 34M parameters and ~3% FLOPs per token. If
`scripts/tokenizer_gate.py`'s new Arabic-script check fails at 49,152 (say urd_Arab
still under ~2.5 chars/token), 57,344 is the next step (a multiple of 128); only
`vocab_size` changes in the config.

## Risks

- **MFU at the larger width is unknown.** Everything above scales today's 17%. The
  grouped dispatch at d 2048, top-6, has not been run; it could be better (bigger
  GEMMs) or worse (more, smaller groups per expert). Measured in the first hour.
- **Replica memory.** One calibration point, and the LM-head share of the 92.4 GiB is
  assumed, not measured. If the real peak is >160 GiB, drop to micro_batch 4 (grad_accum
  256; slower per token) or to C1.
- **Data parallel scaling is unmeasured** (87.5% assumed). Rank 0 evaluates and
  writes checkpoints while 7 GPUs wait at the barrier.
- **Checkpoint size and save time.** ~41 GB per checkpoint (fp32 weights 20.2 GB +
  Muon momentum 19.7 GB + AdamW 0.8 GB), ~122 GB for three; milestones are ~10 GB
  bf16 each (~70 GB for seven). At 1-2 GB/s to local NVMe a save is ~20-40 s, ~1% of
  a 44-min interval. Copying checkpoints off the box at the end is 40+ GB per file:
  plan which ones leave.
- **Data readiness is likely the real bottleneck.** 160B tokens of shards-v3 is
  ~320 GB of uint16, built with tokenizer-v3, which does not exist yet. The data v3
  pool as specified is built for the CURRENT tokenizer in ~10B chunks on Modal CPUs;
  the laptop's ~2.9 MB/s link cannot feed the box. Before the trial: retrain and gate
  tokenizer-v3, and build (or stream from HF onto the DGX's own CPUs) at least the
  first ~40B tokens, building the rest on the box during the run. Thin sources will
  repeat: verified answers (0.2B at 1% of 160B = 8 epochs, cap it) and hin_Latn /
  urd_Latn (4-epoch cap in the data spec).
- **Fixed cosine.** A run stopped early is mid-decay; a warmup-stable-decay schedule
  would be safer for a trial of uncertain length but needs a trainer change (not in
  this spec).
- **New tokenizer, new model, new data at once.** No A/B at this size; the first hour
  is the only check before committing the week.

## First hour on the DGX (validation plan)

1. **2-GPU NCCL check**: `torchrun --nproc_per_node 2` on `configs/quipu-moe-smoke.toml`
   for a few steps; confirm the run log shows world_size 2 and grad_accum_per_rank,
   and that the first checkpoint passes `params_match` (it raises ReplicaMismatch
   otherwise). Then the same with 8.
2. **Memory**: C6 on one GPU, micro_batch 8, ~10 steps with compile; read the peak
   (`torch.cuda.max_memory_allocated`). Go if under ~160 GiB.
3. **Throughput**: 8 GPUs, ~50 steps after compile warm-up; tok/s total and per GPU,
   versus the 57k-per-GPU projection, and 8-GPU vs 1-GPU for the scaling factor. Set
   `total_tokens` to ~85% of what the measured rate gives in the hours left.
4. **Short loss curve**: ~300-500 steps (~30-45 min on 8 GPUs) at the config's lr; loss
   falling smoothly through warmup, no spikes from grad_clip, expert loads balanced
   (the trainer's moe health line). If unstable, muon_lr 0.015 and retry once.
5. Only then delete the trial's checkpoints and launch the real run from step 0
   (plan.json written with micro_batch 8).

## Fallback: one B200 (or no trial)

A 1B-active model on one B200 gets ~31B tokens in 150 h (~30 per active parameter):
a worse model per hour than continuing quipu-moe-1B-A149M, which at 366k tok/s takes
**~198B tokens in the same 150 h** (~1,300 per active parameter, on top of its 4.26B),
with the same tokenizer and the data v3 pool exactly as its spec describes. So with
one GPU: continue the current model (data v3 spec, monthly run launcher, micro_batch
32, grouped dispatch) and keep this config for when 8 GPUs are available.
