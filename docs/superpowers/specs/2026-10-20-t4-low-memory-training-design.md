# Training the 1B MoE on Kaggle's 2x T4

Status: design approved by the owner (20 Oct 2026), approach A.

## Goal

Continue pretraining quipu-moe-1B-A149M (998M total, 149M active) on Kaggle's free
2x T4 (16 GB each, ~30 GPU-hours a week), using the existing data-parallel trainer.
Everything is a config option; the defaults keep today's behaviour, so B200, Modal and
laptop runs are unchanged unless a config opts in.

## Why it doesn't fit today

Every rank holds the whole model (quipu/dist.py: replicated, hand-rolled bucketed
all-reduce). Per T4:

| Today, per rank | GB |
|---|---|
| fp32 weights (998M) | 4.0 |
| fp32 grads | 4.0 |
| Muon momentum fp32 (~960M matrix params) + AdamW m,v (embed/norms, ~38M) | ~4.1 |
| Activations at micro-batch 1 x 2,048 + full logits (2,048 x 49,152) | ~3-4 |
| CUDA context, all-reduce bucket (256 MB) | ~0.7 |
| **Total** | **~16, over a T4's ~15 usable** |

And: T4 (sm_75) has no bf16 tensor cores, and the trainer's only mixed precision is
bf16 autocast; Muon's Newton-Schulz runs in bf16 on any CUDA device; the `grouped`
expert dispatch (torch._grouped_mm) needs sm_90+.

## Target after the change, per T4

| | GB |
|---|---|
| fp32 weights | 4.0 |
| fp32 grads | 4.0 |
| Muon momentum bf16 + AdamW fp32 | ~2.2 |
| Activations with block checkpointing, chunked loss, micro-batch 2 | ~1-2 |
| CUDA context, bucket | ~0.7 |
| **Total** | **~12-13** |

Estimates; the laptop memory test and the first Kaggle session measure the real number.

## Components

### 1. `train.precision = "fp16"`

- Autocast dtype follows `precision` (bf16 or fp16) at every autocast site in
  train.py (the training step, eval, and the third site near line 1472). The router
  (moe.py, fp32 softmax) and the loss (section 4, fp32) are already/also fp32.
- **Loss scaler** (new, small, in quipu/train.py or quipu/lossscale.py), not
  torch.amp.GradScaler: the trainer has its own all-reduce point and non-finite skip,
  and has two optimizers (Muon + AdamW).
  - Each micro-batch's loss is multiplied by `scale` before backward.
  - After the all-reduce, gradients are divided by `scale`, then clipped as now.
  - Non-finite grads under fp16: halve `scale`, skip the step, log "overflow"; the
    step and lr stay put. This does **not** count toward MAX_NONFINITE_STREAK unless
    `scale` is already at the floor (1.0); a non-finite *loss* (forward) still counts
    as today.
  - After `growth_interval` (2,000) clean steps in a row, double `scale`.
  - Initial scale 2**16. State (`scale`, clean-step count) is saved in the checkpoint
    and restored on resume; a checkpoint without it (bf16 runs) starts at the initial
    scale.
- With `precision` bf16 or fp8 the scaler is off and the code path is today's.

### 2. `train.optimizer_state_dtype = "fp32" | "bf16"` (default fp32)

- Muon's `momentum_buffer` is allocated in this dtype. The update math casts the
  buffer to fp32, updates, and writes back (lerp in fp32, store bf16).
- AdamW state stays fp32 (small: embedding and norms).
- Checkpoints store the buffer in its dtype; loading a fp32-momentum checkpoint into a
  bf16-state run casts it down (and the other way up), so the published step_008129
  checkpoint resumes on Kaggle.
- **Newton-Schulz work dtype:** bf16 where `torch.cuda.is_bf16_supported(including_emulation=False)`
  (or the equivalent capability check, sm_80+), fp16 otherwise, fp32 on CPU as now.
  fp16 is safe here: X is normalised by its Frobenius norm first, so entries stay in
  [-1, 1] and the Gram matrices stay small.

### 3. `train.activation_checkpoint = "none" | "block"` (default none)

- `block`: each MoEBlock's forward runs under torch.utils.checkpoint (non-reentrant)
  in training mode only; eval and generation are unchanged.
- **MoE side effects.** The forward stores `last_stats` (expert counts, dropped) and
  `_last_scores` (router scores for Quantile Balancing). The recompute in backward
  runs the router again on the same input. The design requires the recompute to leave
  these exactly as the first forward left them: the recompute must not append, sum or
  stash twice. Implementation: the block detects recompute (a flag set by a wrapper
  around the checkpointed function, or torch's own recompute context) and skips the
  stash/stats writes when recomputing.
- Test: same seed and batch with `none` vs `block` gives the same loss, the same
  gradients (to fp32 tolerance), the same step_counts and the same balance-bias update.

### 4. `train.loss_chunk_tokens` (default 0 = off)

- When > 0, the training step asks the model for the final hidden states (after the
  last RMSNorm; a `return_hidden=True` path on QuipuMoE and Quipu), then computes
  `lm_head` + cross-entropy over chunks of that many tokens, each chunk in fp32 and
  under torch.utils.checkpoint, so the 2,048 x 49,152 logits never exist at once.
- Works for both pretrain (mean) and SFT (sum over assistant targets / N,
  IGNORE_INDEX); the sum over chunks equals today's loss exactly up to float order.
- Eval keeps the full-logits path.
- Test: chunked vs full gives equal loss and equal gradients (pretrain and SFT).

### 5. Guard rails

- At trainer start: `moe_dispatch = "grouped"` on a CUDA device below sm_90 stops
  with a clear message (use "loop" or "padded").
- `precision = "bf16"` on a device without bf16 support stops with a message pointing
  at fp16.
- `configs/kaggle-t4x2.toml`: an override set for the 1B run: precision fp16,
  optimizer_state_dtype bf16, activation_checkpoint block, loss_chunk_tokens 4096,
  moe_dispatch loop, micro_batch 2 (to be tuned in the first session), batch_tokens
  unchanged (524,288, so grad_accum = 128 total, 64 per rank), ckpt_every set from
  measured step time to about every 30 minutes, budget_usd 0.

### 6. Kaggle runbook (docs/superpowers/specs/, no code)

- Data: a private Kaggle dataset with ~1B tokens of data v3 chunk-0 shards (uint16,
  ~2 GB), taken from the Modal volume quipu-pool. Owner uploads it.
- Checkpoint in: the published step_008129.pt (8.1 GB) as a private Kaggle dataset;
  later sessions attach the previous session's output instead.
- Each session: background "Save & Run All" (12 h max), `torchrun --standalone
  --nproc_per_node=2 -m quipu.train --config ... --resume ...`, checkpoints to
  /kaggle/working, keep 1 (disk is 20 GB), stop with a margin before the 12 h limit.
- No HF token on Kaggle in this phase.

## Testing

Laptop (RTX 5060 8 GB, CPU for the rest):

1. Unit tests: loss scaler (overflow halves and skips, growth after the interval,
   floor counts toward the streak, state round-trips through a checkpoint); bf16
   momentum vs fp32 over 20 Muon steps on random matrices (relative error small);
   NS in fp16 vs fp32; chunked vs full loss and grads; checkpointed vs plain block
   (loss, grads, counts, balance update); grouped-on-old-GPU and bf16-on-T4 guards.
2. Precision parity on the laptop GPU: a reduced model with the 1B's width and expert
   layout (d_model 768, 64 experts, top-4, shared expert) but 4 layers, 200 steps,
   bf16 vs fp16 + all memory options, same seed and data: final loss within 1%, no
   inf/NaN steps other than scaler overflows, expert load spread unchanged.
3. Memory: peak allocated for the reduced model with and without the options, and
   per-block numbers used to project the 16-layer model on a T4.

Kaggle (first session, ~1 h of quota):

4. Resume step_008129.pt with the T4 config: the first logged training loss is within
   1% of the bf16 run's logged loss at that step; 50 steps with no non-finite loss.
5. Peak memory per rank, tokens per second, step time; pick micro_batch and
   ckpt_every from them.

If 2 or 4 fails (fp16 unstable for this MoE), stop: T4s stay for eval and generation,
and pretraining waits for B200 credits.

## Out of scope

Sharded optimizer state across ranks (approach B, only if the measured memory is too
tight), FSDP, 8-bit optimizers, CPU offload, torch.compile on T4, moving data or
checkpoints through Hugging Face.

## Changes during implementation

- The Kaggle config is `configs/quipu-moe-cpt-t4.toml` (inherits quipu-moe-cpt.toml),
  not `configs/kaggle-t4x2.toml`.
- Session clock: the spend guard at usd_per_hour 1.0, budget_usd 11.25 stops the run
  with a checkpoint after 11.25 h, instead of budget_usd 0.
- loss_chunk_tokens 2048, not 4096: at micro_batch 2 a 4096-token chunk is the whole
  micro-batch, so it saved nothing.
- scripts/lowmem_probe.py: both parity runs train until they have taken --steps
  optimizer steps (fp16 overflows skip steps; capped at 2x calls) and compare the
  loss tail over the same step indices; pass = within 1%, no non-finite loss, fp16
  took at least 90% of the steps; overflows reported separately. Memory mode trains
  until 2 steps were taken (optimizer state allocated) and reports peak allocated and
  reserved. `--force-fp16-ns` runs Newton-Schulz in fp16 to mimic a T4 on a newer GPU.
- Runbook order for the first session: environment check, probe memory run at 16
  layers, a short check run under its own run id (deleted afterwards), the fp16-vs-fp32
  val loss of step_008129 (no eval was logged at step 8129; step 8000's is the
  reference), and only then the long run as a committed version. No `pip install -e .`
  on Kaggle; PYTHONPATH=. instead.
