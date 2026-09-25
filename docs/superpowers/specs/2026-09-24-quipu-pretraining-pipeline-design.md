# Quipu — a from-scratch pretraining pipeline, validated at 114M

**Date:** 2026-09-24
**Status:** awaiting owner review
**Sub-project:** 1 of 5
**Hardware:** RTX 5060 Laptop (8 GB VRAM, Blackwell sm_120), i7-13650HX, 31.7 GB RAM, 367 GB free on C:

## 1. What Quipu is, eventually

A small language model, trained from scratch, that drives an agentic coding CLI, and matches
models 10–30x its size on retrieval-dominant coding tasks. Short context by design, with history
pushed into an external memory the model learns to address.

A *quipu* is the Incan recording device: knotted cords encoding records, read by running them
through your hands. Information mapped onto positions, retrieved by traversing them.

**This spec is not that.** It is the foundation the whole thing stands on, and nothing more.

## 2. What this sub-project is

A validated from-scratch pretraining pipeline that produces a **~114M-parameter** decoder-only
transformer in about a day on the 5060, with loss curves, resumable checkpoints, and a generated
results table.

### Why 124M-class before 350M

De-risking, and it is the only reason. Every failure mode of this pipeline — an OOM at 8 GB, a
shard-boundary bug, a checkpoint that will not resume, a causal mask that leaks, thermal
throttling, PyTorch not supporting Blackwell — surfaces at 114M in **one day** instead of at 350M
in **eight**. The 8-day run is paid once, after the recipe is proven.

This is standard practice; llm.c and nanoGPT both start here.

### Why the memory architecture is not in this spec

Phase 1 is byte-identical whether Quipu ends up with RETRO-style chunked cross-attention,
recurrent memory tokens, or retrieval at inference only. All three need the same thing first: a
working decoder-only transformer, a data pipeline, and a reproducible training loop. Deferring
costs nothing and the choice is better made after the training loop has been felt.

**The model in this spec has no retrieval and no memory. Deliberately boring.**

## 3. Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | **FineWeb-Edu only. No code data.** | The success criterion is a loss comparable to published 124M runs, and those baselines are on FineWeb. Mixing code moves the loss and destroys the comparison. Code enters at 350M. |
| 2 | **GPT-2 BPE (tiktoken, 50257), not our own tokenizer** | Same reason: the reference runs used it. A tokenizer trained here would confound the only external number we can check ourselves against. Training our own is a sub-project 2 option. |
| 3 | **Train on native Windows, not WSL** | WSL2 idle-death is a documented, already-costly failure on this machine (`vmIdleTimeout` kills long jobs). A 20-hour run cannot sit inside it. |
| 4 | **Tied input/output embeddings** | Saves 38.6M parameters at this scale, standard for small models, and keeps more of the budget in the layers. |
| 5 | **GQA with 4 KV heads, though MHA would be fine at this scale** | The recipe must be the one that scales to 350M and to short-context inference. Changing attention between phases would invalidate the validation. |
| 6 | **Success is "the pipeline works", not "the loss wins"** | This is a validation artifact. A 114M model is not useful on its own and the spec should not pretend otherwise. |

## 4. Architecture

Decoder-only transformer, modern recipe throughout: **RMSNorm** (pre-norm), **SwiGLU** feed-forward,
**RoPE** position encoding, **GQA**, no biases on linear layers, tied embeddings.

| Field | Value |
|---|---|
| vocab | 50,257 (GPT-2 BPE) |
| d_model | 768 |
| layers | 12 |
| attention heads | 12 (head_dim 64) |
| KV heads (GQA) | 4 |
| FFN hidden | 2,048 (SwiGLU, 3 matrices) |
| context | 1,024 |
| parameters | 114,114,048 with tied embeddings |

Parameter arithmetic, so the test can assert it rather than trust it:

```
embedding      50257 x 768                          = 38,597,376
per layer      attn   768x768x2 + 768x256x2         =  1,572,864
               ffn    768x2048x3                    =  4,718,592
               norms  768x2 (pre-attn, pre-ffn)     =      1,536
                                                      ----------
                                                       6,292,992
12 layers                                            = 75,515,904
final norm                                           =        768
                                                      ----------
total (tied lm_head)                                  114,114,048
```

The RMSNorm weights are 19,200 of that. They are small enough to round away in prose and
large enough to fail an exact-equality test, so the figure above is the one `test_model_shapes.py`
asserts.

## 5. Data

**FineWeb-Edu**, `sample-10BT` subset — ungated, CC-licensed, and the corpus the reference runs use.

- Budget: **1.5B training tokens** (~13 tokens/parameter, under Chinchilla-optimal). Measured
  throughput of ~18k tok/s would put the original 2.5B budget at ~39 hours; 1.5B is ~23 hours,
  which is acceptable for a pipeline-validation run.
- Held-out validation split: 10M tokens, never trained on
- Stored as flat `uint16` shards (vocab < 65536, so 2 bytes per token suffices)
- ~5 GB tokenized. Raw download is larger; shards are what persist.

Documents are concatenated with an end-of-text token between them and chunked at exactly 1,024
tokens. Sequences therefore cross document boundaries, which is what the reference runs do; the
alternative (padding to document length) wastes a large fraction of the budget.

## 6. Training

| Field | Value |
|---|---|
| precision | bf16 autocast, fp32 master weights |
| optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| peak LR | 6e-4, cosine decay to 6e-5 |
| warmup | 200 steps (7.0% of the run) |
| grad clip | 1.0 |
| total batch | ~0.5M tokens per step |
| micro-batch | 4 (grad_accum 128): 6.25 GiB allocated / 6.67 GiB reserved of the 8 GB card, the largest measured under the 7.0 GiB budget |
| steps | 2,861 (1,500,000,000 / 524,288) |
| estimated wall clock | 23.7–26.0 hours: `scripts/measure_throughput.py` measured 29.8–32.6 s/step (16.1k–17.6k tok/s) at micro_batch 4 on the real loop and data; eval and checkpoints add under 0.1 h |

The wall-clock estimate is measured, not derived: timed training steps of the real model on the
real shards (Task 12), plus the measured cost of one eval and one checkpoint save at their cadence.
Micro-batch 8 is not an option on this laptop: the Windows driver never raises CUDA OOM, it spills
VRAM into shared system RAM and the step becomes tens of times slower, so micro-batch is chosen by
peak reserved memory against a 7.0 GiB budget. Micro-batch 2 measured the same throughput within
noise (16.1k–16.8k tok/s), so it is the fallback if other GPU apps crowd the card.

### Resumability is a first-class requirement, not a nicety

The laptop will sleep, throttle, or be needed for other work inside a 20-hour window. A checkpoint
must restore: model weights, optimiser state, LR schedule position, step count, RNG states, and
the data loader's position in the shard stream. Anything less silently re-trains on the same
tokens or skips others, and the loss curve will not show it.

## 7. Components

Each is separately testable and has one job.

| File | Responsibility |
|---|---|
| `quipu/data.py` | Download FineWeb-Edu, tokenize, write `uint16` shards. Pure: text in, shards out. |
| `quipu/tokenizer.py` | Thin wrapper over tiktoken `gpt2`. Exists so sub-project 2 can swap it. |
| `quipu/model.py` | The transformer. No training code, no I/O. |
| `quipu/train.py` | Loop, accumulation, checkpointing, resume. |
| `quipu/eval.py` | Held-out loss, perplexity, generation probes. |
| `quipu/runlog.py` | Per-run JSON records. **Ported from the chatbot repo**, which already does this. |
| `quipu/results_table.py` | Generates RESULTS.md from run records. Also ported. |
| `configs/quipu-114m.toml` | One file that fully determines a run. |

`runlog.py` and `results_table.py` are lifted from `ai-chatbot-softrobotics/training/` rather than
rewritten. That repo's discipline — nothing in RESULTS.md typed by hand — is the single most
valuable thing to carry over.

## 8. Tests

Four that must exist, because each covers a failure that trains happily while being wrong:

1. **Shard round-trip** (`test_data_shards.py`) — tokens written equal tokens read back, count is
   exact, and the document separator appears at the expected positions. A shard bug is invisible
   in the loss curve.
2. **Parameter count and shapes** (`test_model_shapes.py`) — the model built from the config has
   exactly the parameter count computed in §4. Catches a silently mis-sized layer.
3. **The causal mask actually masks** (`test_model_shapes.py`) — perturbing token *t+1* must leave
   the logits at position *t* bit-identical. A leaking mask produces a beautiful loss curve and a
   worthless model. This is the most important test in the sub-project.
4. **Resume is exact** (`test_checkpoint_resume.py`) — save at step N, kill, resume, and the loss
   at step N+1 matches the uninterrupted run within float tolerance.

Every assertion gets mutation-tested: break the code, watch the test fail, restore. A test that
cannot fail is not evidence.

## 9. Success criteria

1. The 1.5B-token run **completes**, and survives at least one deliberate kill-and-resume.
2. Validation loss decreases smoothly and lands in the range published for 124M-class models at
   ~1.5B tokens. For calibration: llm.c reaches ~3.29 on FineWeb, but at **10B** tokens — nearly
   seven times this budget — so a materially higher number here is the expected result, not a failure.
   The target is a sane curve, not a specific figure.
3. The run reproduces from `configs/quipu-114m.toml` alone.
4. RESULTS.md is generated by script. Nothing typed by hand.
5. Generation probes produce English that is locally coherent. At this scale that is the whole
   bar — not factuality, not instruction following, not code.

## 10. Risks

| Risk | Handling |
|---|---|
| **PyTorch may not support Blackwell (sm_120) cleanly** | Torch is not installed on this machine. Task 1 installs it and proves a GPU matmul and a backward pass before anything else is built. If sm_120 needs a nightly or CUDA 12.8+, that is discovered on day one. |
| Thermal throttling below the 20 TFLOPS estimate | Task 1 measures real throughput on a short run and the token budget is re-decided from the measurement. |
| 8 GB OOM at the chosen batch size | Micro-batch is tuned empirically; activation checkpointing held in reserve. Gradient accumulation keeps the effective batch fixed regardless. |
| The laptop is unusable for ~a day | Accepted and stated. Full VRAM already stops other GPU apps on this machine. |
| `torch.compile` unreliable on Windows | Off by default. Enabled only if measured faster and stable. |

## 11. Not in scope

- Retrieval, memory tokens, cross-attention — sub-project 4.
- Code data, our own tokenizer, 350M — sub-project 2.
- OpenCode, tool calling, agentic traces — sub-project 3.
- Any claim about matching larger models — sub-project 5.
- Quantization, GGUF export, inference optimisation.

## 12. The road after this

| # | Sub-project | Rough size |
|---|---|---|
| 1 | **This spec** — pipeline validated at 114M | ~1 week |
| 2 | 350M base model, code data, own tokenizer | ~2 weeks |
| 3 | Agentic fine-tune against OpenCode's tool schema | ~2–3 weeks |
| 4 | Memory architecture and its ablation | ~3–4 weeks |
| 5 | The claim: repo-knowledge benchmark vs 3.5B–10B models | ~2 weeks |

Each gets its own spec, plan and build.

### Noted option for sub-project 3: scoring instead of generation

Emitting a tool call from nothing — name, valid JSON, argument values — is categorically harder
for a small model than choosing among candidates that are already enumerated. Reformulating the
agentic layer as **scoring a candidate set** rather than generating free-form text is therefore a
live design option, not a fallback.

Two pieces of evidence point at it:

- **Muose-50M-Decision** (2026) reached 79.94% across all 77 BANKING77 intents from a 50M model
  pretrained on 500M tokens on an RTX 3070 — by scoring enumerated options rather than generating.
  For calibration, a fine-tuned BERT-base baseline on the same benchmark is 93.66%
  (Casanueva et al.), so this is well short of an off-the-shelf encoder; the relevant point is
  that the *task reformulation* is what made a 50M model viable at all.
- **Aura's own measurements** already lean the same way: schema-mode constrained decoding lowered
  model-level false actions relative to free decoding, at a latency cost recorded in
  `auroraos/tests/results/`.

The cost is that the candidate set has to come from somewhere — retrieval, a registry, or a
generate-then-rerank pass — which is a design problem in its own right. Decided in sub-project 3,
on measurements, not now.
