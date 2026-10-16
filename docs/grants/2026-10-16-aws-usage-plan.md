# Quipu: how the AWS credits would be used

Aneek Chattopadhyay, independent researcher and maintainer of Quipu
(https://github.com/Aneek1/quipu, https://quipulm.dev). 16 October 2026.

## The request

$9,650 for 12 months, from the AWS Pricing Calculator estimate
(https://calculator.aws/#/estimate?id=d52f0edcc86f964856b7f9e12f9af9e97b63d9f7):

| Item | Use | Per month | Cost / month |
|---|---|---|---|
| p6-b200.48xlarge (8x B200), $113.93/h | Pretraining | 6 h | $683.58 |
| g6e.xlarge (1x L40S), ~$1.86/h | Chat fine-tunes, teacher answers, evaluation | 60 h | ~$111.66 |
| S3, 400 GB | Training data, checkpoints, releases | 400 GB | ~$9.20 |
| **Total** | | | **$804.46 (x 12 = $9,653)** |

## Where things stand

- **quipu-moe-1B-A149M** (998M total / 149M active parameters per token), pretrained from
  scratch on 4.26B tokens for $18.75 on one rented RTX 5090. Weights are public
  under Apache-2.0.
- **Chat model and distillation:** a 7B open-weight teacher (Qwen2.5-Coder-7B-Instruct,
  Apache-2.0) generated 638,342 coding answers. Each one was run against its unit
  tests in a sandbox, and the 498,813 that passed were kept. MBPP went from 1.9% to 7.4%.
- **Data v3:** a 10B-token pretraining set from 42 open, licence-checked sources, built
  for $2.51. HumanEval and MBPP are filtered out of it.
- **Multi-GPU:** data-parallel training is verified on real GPUs (a 2-GPU run, with checkpoint and resume).
  The mixture-of-experts dispatch was rewritten with grouped matmuls: 113k to 366k
  tokens per second on one B200 (3.24x).
- **Current limit:** all of this has run on about $30 a month of free credit plus a
  laptop GPU. That is roughly 4 B200-hours a month, enough to continue the 1B model but not to
  train a bigger one.

## What the B200 hours would do (72 h over the year)

**Month 1: continue the published 1B model on data v3 (about 1 hour).**
One B200 runs the 1B model at a measured 366k tokens/s. Eight in data parallel, at an
assumed 87.5% scaling, gives ~2.56M tokens/s, so the 10B-token set takes ~65 minutes. The
same session checks the full AWS setup end to end: S3 to instance, checkpoint back
to S3, resume.

**Months 1-12: train quipu-moe-5B-A1B from scratch (~70 hours).**
The design is already written and its config is tested (`configs/quipu-moe-a1b.toml`):
5.04B total / 1.03B active parameters, 22 layers, 64 experts with 6 used per token.
Its memory was projected from the measured run (~129 of 178 GiB per GPU).

- **Throughput:** projected at ~57k tokens/s per B200, assuming the same efficiency
  as the measured 1B run (~17% of peak). Bigger matrices usually run more efficiently,
  so this should be the low end. On 8 GPUs at 87.5% scaling: ~400k tokens/s, about
  1.4B tokens per hour.
- **Tokens:** ~8.6B per 6-hour month, **~95-100B tokens over the year**: about 95
  tokens per active parameter, well past the compute-optimal point. That is where a small
  model should be trained.
- **Sessions:** each month's run resumes from the last checkpoint in S3. The learning
  rate uses warmup-stable-decay: every second month a short decay branch (~10% of that month's
  tokens) is cut off the stable run and released, while training continues from the
  stable checkpoint. That gives six public releases in the year.
- **Data:** the 42-source pool is rebuilt in 10B-token chunks (~$2.51 each, on CPUs;
  I cover this separately). Small language slices repeat at most 4 times.

**If p6-b200 capacity isn't available** in a given month, the same code runs on p5
(8x H100) for fewer tokens. The trainer is the same; only the micro-batch size changes.

## What the g6e.xlarge hours would do (60 h a month)

The L40S has 48 GB and supports bf16, which the free GPUs I use today (Kaggle T4s)
do not. Uses:

1. **Chat fine-tunes** of each released checkpoint on the 498k verified answers plus
   multi-turn data (below), in bf16 with no fp16 workaround.
2. **Teacher answers:** the open-weight teacher in vLLM generates multi-turn coding
   conversations where every turn is checked by unit tests. The current chat model
   loses track after one turn (shown on https://quipulm.dev/demo); this data is the fix.
3. **Evaluation** of every release: HumanEval, MBPP, bits per byte in the ten
   languages, and the multi-turn tests. Published with the release.

L40S throughput for these jobs has not been measured yet. The first month's hours will
measure it, and the split between the three uses will follow from that.

## Storage (400 GB of S3)

- Training data: ~100B tokens as uint16 shards, held about 2 chunks (~40 GB) at a time.
- Checkpoints: a 5B-A1B checkpoint with optimizer state is ~40 GB (fp32 weights + Muon momentum); the last 3 are kept,
  plus 1B-model checkpoints (~8 GB each).
- Distillation and multi-turn data, evaluation results, release staging.

## How spending is controlled

- Every training run has a budget guard that stops it before it passes its hour limit.
  The same pattern has kept this year's runs on Vast and Modal under their caps.
- AWS Budgets alerts at 50%, 80% and 100% of each month's $804.
- Instances are started for a run and terminated after the checkpoint is safely in S3.
  Nothing is left running idle.
- A spend ledger is kept per run and published with each release.

## What the community gets

- Six releases of quipu-moe-5B-A1B over the year, plus a continued 1B model, all
  Apache-2.0 on Hugging Face, with full model cards and evaluation results.
- The training code, data pipeline (42 sources, licence checks, benchmark filtering),
  verified distillation data and multi-turn data, all open on GitHub.
- An honest record of what worked and what didn't, including the A/B tests that
  lost, as on https://quipulm.dev/decisions.
- AWS credited in every model card and release note.
