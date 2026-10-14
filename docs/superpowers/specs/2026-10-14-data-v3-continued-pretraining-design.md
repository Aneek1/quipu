# Data v3 and monthly continued pretraining (design)

Date: 14 Oct 2026. Status: approved by the owner on 14 Oct (open questions answered below).

## Goal

Keep training quipu-moe (1.0B total / 149M active, 48k tokenizer, step 8129, 4.26B
tokens seen) on a growing pool of higher-quality data, a few billion tokens a month,
paid for by Modal's free $30/month. The pipeline is built to scale: when a compute
grant arrives (TRC, Lambda), the same pool and launcher take it to hundreds of
billions of tokens, and the same data later trains a 1B-active model.

"Trillions of tokens" is the direction, not this spec's promise. At 4 B200-hours a
month it is out of reach (see "Throughput and cost"); this spec makes every month
count and makes the jump cheap when bigger compute comes.

## Decisions (from the conversation, 14 Oct)

| Question | Decision |
|---|---|
| Monthly compute | Option A: **4 B200-hours a month on Modal**, inside the $30 free credit. The $30 hard limit stays. |
| Model | Continue the published 1B-A149M (same architecture and tokenizer), not a new model. |
| Data | Open, licence-checked corpora (below), Stack Exchange included, the 498k-ish verified coding answers as a high-quality slice. |
| Releases | A decayed checkpoint is published **every second month** (owner, 14 Oct). |
| Urdu | **Native-script Urdu (`urd_Arab`) added** alongside romanized Urdu (owner, 14 Oct). |
| Code access | The owner accepted the `bigcode/starcoderdata` terms; access checked from the laptop on 14 Oct (865 files listed, README downloaded). |
| Never | Claude-written data; datasets whose terms forbid training; synthetic data from models whose licences would bind our weights (Qwen/DeepSeek/Phi-4-derived NVIDIA sets). |

## Sources (licences checked 14 Oct 2026)

| Slice | Share | Source | Licence | Size | Notes |
|---|---|---|---|---|---|
| Code | 45% | `bigcode/starcoderdata` (permissive subset, 14 languages of the current mix) | per-file original licences, attribution | ~250B tokens | Gated: the owner accepts the terms on Hugging Face once. Content is on HF (no AWS). |
| Code (old) | 5% | `codeparrot/github-code-clean`, permissive-licence subsets only | per-file | 314 GB | The current pipeline's source; kept for continuity. |
| Code-related web | 3% | `OpenCoder-LLM/opc-fineweb-code-corpus` | MIT | 101M rows | Tutorials, docs, blog posts about code. |
| English | 20% | `HuggingFaceFW/fineweb-edu` (score ≥3 default) | ODC-By 1.0 | 1.3T | The current English source. |
| Q&A | 5% | Stack Exchange, **dumps up to April 2024 only** (RedPajama-1T `stackexchange` subset, or the archive.org dumps ≤ 2024-04) | CC BY-SA (attribution, share-alike) | ~20B | Dumps since June 2024 come with a click-through forbidding LLM training: not used. Keep author/URL fields for attribution. |
| Maths | 7% | `HuggingFaceTB/finemath` (finemath-4plus first, then 3plus) + `open-web-math/open-web-math` | ODC-By | 9.6B / 34B + 14.7B | |
| Reference | 4% | `wikimedia/wikipedia` (English + the nine languages), `allenai/peS2o` v2, `common-pile/arxiv_papers_filtered` | CC BY-SA / ODC-By / per-paper CC | ~6B / 42B / 29B | peS2o is a loading script: converted to parquet once. |
| Nine languages | 10% | `HuggingFaceFW/fineweb-2` (as now: ind, zsm, cmn split Hans/Hant, jpn, kor, tam, hin_Deva, hin_Latn, urd_Latn) | ODC-By | large for zh/ja/id/ko; thin for zsm/tam; **near zero for hin_Latn (0.2 GB) and urd_Latn (0.13 GB)** | Thin languages repeat at most 4 epochs, then their share goes to the others. Native-script Urdu (`urd_Arab`, 6.9 GB) is added at 1/3 of the Urdu share (see "Urdu script and the tokenizer"). |
| Verified answers | 1% | `data/distill-v2/kept` (Qwen2.5-Coder-7B-Instruct, Apache-2.0, every answer passed its unit tests) | Apache-2.0 teacher | ~0.2B | Mixed into pretraining (the planned step-2 experiment), and reused later for the chat fine-tune. |

Shares are of training tokens, set per monthly chunk. Not used (and why):
DCLM-baseline ("research only, not for production models"; revisit if needed),
the-stack-v2 / stack-edu (content only via Software Heritage S3 with an agreement;
revisit with grant compute), Nemotron-CC-v2 / Nemotron code / Nemotron-CC-Math
(synthetic or cleaned with Qwen/DeepSeek/Phi-4), Cosmopedia (synthetic; fine by
licence, left out to keep the pool non-synthetic).

The model card gains an attribution section (Stack Exchange, Wikipedia, starcoderdata
per-file licences, ODC-By sources) and notes the CC BY-SA share-alike question.

## Urdu script and the tokenizer

The 48k tokenizer was trained on romanized Urdu, not Arabic script. Measured on 14 Oct on the
same sentence: Urdu in Arabic script 1.34 characters per token, romanized Urdu 3.37, Hindi
(Devanagari) 2.64, English 4.81. So Arabic-script Urdu costs about 2.5x the tokens of the same
text romanized, and the model learns it from near-byte pieces.

- **Current model (this spec):** `urd_Arab` goes in as it is, at a small share (1/3 of the
  Urdu slice), so it is not starved of real text while the tokenizer stays fixed. Changing the
  vocabulary mid-run would mean new embedding rows and a re-tokenized pool.
- **Next model (the ~1B-active one):** the tokenizer is retrained with `urd_Arab` in its
  sample, and the gate in `scripts/tokenizer_gate.py` gains an Arabic-script check.

## Components

### 1. Pool builder (extend `scripts/build_shards.py`)
- New source kinds: Q&A (Stack Exchange posts rendered as "question, accepted answer,
  top answers", HTML stripped, attribution kept in the manifest), maths, reference, and
  parquet streaming for each repo above at a **pinned revision**.
- Same rules as now: 48k tokenizer, 100M-token shards of uint16, decontamination
  against HumanEval and MBPP (pinned) plus the distill-v2 prompts' benchmark check,
  exact dedup per source, manifest with every source's revision and token count.
- **Rolling pool:** built in chunks of ~10B tokens, written straight to a Modal
  Volume `quipu-pool` (not the laptop: its ~2.9 MB/s link makes 20 GB a 2-hour
  upload). The builder resumes where the last chunk ended, so the pool only ever
  runs ~1-2 months ahead of training.
- Runs on Modal CPU containers, one per source, in parallel.

### 2. Monthly training run (new `scripts/remote/modal_train.py`)
- One B200 container, resumes from the last checkpoint on a Volume
  (`quipu-ckpt`), trains for a fixed wall-clock budget (default 3 h 45 min) and saves
  a checkpoint well before Modal's hard limit could cut it.
- **Learning-rate schedule: warmup-stable-decay.** A short re-warmup from the
  pretrain's final LR to a stable LR, then constant across months (no fixed end
  date). Every second month (the release cadence), a short decay branch (~10% of that month's tokens) is
  trained from the stable checkpoint and published; the next month continues from
  the stable checkpoint, not the decayed one.
- Evaluation every month on the same milestone set as the pretrain (bits per byte
  per language, MBPP/HumanEval pass@1), appended to a progress table that the site
  and LinkedIn posts can quote.

### 3. Budget guard
- Before each launch: read `modal billing summary`, refuse if the month's spend plus
  the planned run exceeds **$27** (the $3 margin covers the demo and storage).
- The run's own timer is the second guard; the $30 hard limit is the third.

## Throughput and cost

Measured on Modal on 14 Oct (scripts/remote/modal_bench.py, run config of the real
pretrain: Muon, torch.compile, 524,288-token batches; 2 warm-up + 6 timed steps):

| GPU | micro_batch 8 | 16 | 32 | peak VRAM at 32 |
|---|---|---|---|---|
| B200 (178 GiB) | 34,719 tok/s | 58,078 | **112,860** | 92.9 GiB |
| H100 (79 GiB) | 21,801 | 37,284 | out of memory | — |
| RTX 5090 (the pretrain, micro_batch 4) | 44,331 | | | |
| **B200, grouped dispatch** (quipu.moe forward_grouped) | | | **366,229** (3.24x) | 92.4 GiB |

Throughput roughly doubles with each doubling of the micro batch: the step is
dominated by per-micro-step overhead (the MoE dispatch over 64 experts, recompiles
on empty experts), not by the GPU's maths. Two consequences:
- The monthly run uses the largest micro batch that fits: 32 (64 is worth one
  try: ~170 GiB projected, at the card's limit).
- Speeding up the MoE layer (grouped expert matmuls instead of a loop, no
  recompiles) is the cheapest way to get more tokens per month, on every GPU.
  A separate spec.

| Item | Cost / yield |
|---|---|
| B200, per hour incl. CPU/memory | ~$6.90 (Modal list price $6.25 + ~$0.65) |
| 4 B200-hours a month at 366,229 tok/s (grouped) | ~$27.50 → **~5.3B tokens/month** (1.6B with the old loop dispatch) |
| For comparison: Vast RTX 5090 at $0.53/h | ~8B tokens for the same $27.50, but paid in cash; Modal's $30 is free credit |
| Building a 10B-token chunk on Modal CPUs | measured on chunk 0 (14 Oct): **$2.51**, ~20 min, 2 rounds (42 + 36 containers), 101 shards |

At ~5.3B tokens a month the pool needs ~6B tokens per month built ahead.

At that rate, 100B tokens is years of free credit; the pool and launcher are what a
grant would plug into.

## Out of scope
- A 1B-active model (needs ~20B+ tokens and ~$800+ of compute: grant).
- The chat fine-tune on the verified answers (separate, ~$2-5, can run any month).
- New tokenizer.

## Answered (14 Oct)
1. `bigcode/starcoderdata` terms: accepted by the owner; access confirmed.
2. Native-script Urdu: yes (see above).
3. Releases: every second month.
