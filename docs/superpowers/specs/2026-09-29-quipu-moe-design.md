# quipu-moe: a K3-core mixture-of-experts model trained on a rented GPU

**Status:** design, awaiting owner review · **Date:** 2026-09-29 · **Branch:** `pipeline-114m` (pushed to `main` on approval)

## 1. Goal

Train Quipu's second model from scratch: a ~0.98B-total / ~135M-active mixture-of-experts language model with a code-aware tokenizer, on ~10B tokens (60% code, 40% text), borrowing the ideas from Kimi K3 that transfer to small scale, on one rented RTX 5090 within the owner's remaining Vast credit.

This is "Track B" of the step-builder plan: the from-scratch model that will later be fine-tuned for chat and stepwise Flask + React building and compared with an open 1.5B model on the stepbuild benchmark. This spec covers pretraining only.

**Success:**
1. The tokenizer passes its gate (§4.3) before any GPU time is spent.
2. Three A/B runs (§6.1) produce matched, reported results; the full run uses the winners.
3. The full run completes (or stops cleanly at the budget cap) with no lost work, and its milestones, evaluation and model card are published.
4. Total spend on this spec stays within the owner's cap (default $20).

## 2. Sources and honest framing

Kimi K3 (Moonshot AI, weights and technical report released 2026-07-27; 2.8T total / 104B active) claims ~2.5× better scaling efficiency than Kimi K2 from a combination of architecture, optimizer, data and recipe changes. **The report gives no small-model ablations.** Whether each borrowed idea helps at ~135M active is exactly what §6.1 measures; results are published either way.

What is borrowed, and what is not:

| K3 component | Adopted? | Reason |
|---|---|---|
| Sparse MoE with shared + routed experts (DeepSeekMoE layout) | Yes | Core of this model |
| Quantile Balancing (auxiliary-loss-free router bias) | Yes | Stability of sparse routing |
| SiTU-GLU (soft-capped SwiGLU: β₁=4 gate, β₂=25 up) | Yes, A/B vs SwiGLU | Bounded activations for bf16 |
| Per-head Muon (Newton–Schulz per attention head) | Yes, A/B vs AdamW | Main optimizer gain in K2/K3 |
| Block Attention Residuals (learned pseudo-queries over block outputs) | Switchable, A/B | Reported to help across scales |
| RMSNorm before the routed up-projection | No (it belongs to LatentMoE) | We do not use a latent projection |
| LatentMoE | No | Saves communication across many GPUs; irrelevant on one GPU |
| Kimi Delta Attention hybrid | No, later spec | Large new kernel work; our retrieval memory covers long context for now |
| Gated MLA, vision tower, 1M context | No | Out of scope |

## 3. Model

| | quipu-114m (released) | quipu-moe |
|---|---|---|
| Parameters | 114M dense | ~0.98B total / ~135M active (~110M non-embedding active) |
| Layers × width | 12 × 768 | 16 × 768 |
| Attention | GQA 12 q / 4 kv heads, head dim 64, RoPE | same |
| Depth mixing | pre-norm residual | pre-norm residual, or Block AttnRes with 4 blocks (config switch) |
| Feed-forward | SwiGLU, 2,048 hidden | MoE per layer: 1 shared expert (hidden 768) + 64 routed experts (hidden 384 each), top-4 |
| Expert activation | — | SiTU-GLU or SwiGLU (config switch) |
| Routing | — | softmax router scores over 64 experts; Top-4 selection on score + per-expert bias b_j; weights renormalised over the selected 4 |
| Load balancing | — | Quantile Balancing: after each step, for each expert, set the bias so the expected fraction of tokens it would receive matches the target load (k/n); implemented from the batch's router scores with the Top-(k+1) cutoff rule in the report; no auxiliary loss term |
| Embeddings | tied, 50,257 vocab | tied, 32,768 vocab |
| Context | 1,024 | 2,048 |
| Precision | bf16 autocast | bf16 autocast; router and balancing math in fp32 |

**Parameter arithmetic** (per layer, d=768): attention 1.57M; shared expert (SiTU-GLU, three matrices, hidden 768) 1.77M; each routed expert (hidden 384) 0.885M × 64 = 56.6M; active per layer = 1.57 + 1.77 + 4 × 0.885 = 6.88M. Over 16 layers: 110M active non-embedding, ~959M total non-embedding, plus 25.2M tied embeddings.

**Implementation:** plain PyTorch. MoE dispatch groups tokens by expert (sort by expert id, run each expert on its slice, scatter back). No custom kernels. `torch.compile` is used when available on the Linux box and must match eager outputs in a test.

## 4. Tokenizer

### 4.1 Design
- Byte-level BPE, 32,768 tokens, trained with Hugging Face `tokenizers` on a ~2 GB sample of the §5 mix (same 60/40 ratio and language weights).
- Pre-tokenisation: split runs of spaces into their own pieces up to 16 characters, tabs likewise; newlines kept as their own tokens; digits split individually; otherwise a GPT-4-style regex for words and punctuation.
- Special tokens reserved from the start: `<|endoftext|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, and one token each for the step-builder's `=== FILE: ` and `=== END FILE ===` markers.
- Saved as a `tokenizer.json` shipped with the model.

### 4.2 Integration
`quipu/tokenizer.py` gains a backend that loads `tokenizer.json`; the GPT-2 path stays for quipu-114m. The shard builder and all evaluation scripts take the tokenizer from config.

### 4.3 Gate (must pass before any GPU session)
Measured on held-out code (the code validation set) and text (FineWeb-Edu validation), against GPT-2:

| Measure | GPT-2 | Required |
|---|---|---|
| Whitespace-only share of code tokens | 36.5% | < 15% |
| Characters per token, code | measured | better than GPT-2 |
| Characters per token, text | measured | within 5% of GPT-2 |
| Round trip `decode(encode(x)) == x` | — | exact on every validation document |

## 5. Pretraining data

- ~10.2B training tokens: 60% code, 40% text, interleaved deterministically as in the weekend-run builder.
- **Code (60%):** `codeparrot/github-code-clean`, permissive licences only (MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0, Unlicense), the existing filters (minified, vendored, `node_modules`, `dist`, oversized files dropped). Language weights within code: Python ~30%, JavaScript/JSX ~25%, TypeScript/TSX ~12%, HTML ~8% (hard cap), CSS ~5%, SQL ~5%, all others ~15%. If a language runs short, its remainder is redistributed proportionally and the achieved shares are reported.
- **Text (40%):** FineWeb-Edu `sample-10BT` (~10B tokens available; ~4.1B needed), as for quipu-114m.
- **Validation:** 10M text tokens and 5M code tokens held out, deduplicated by exact content hash against training.
- **Leakage guard:** any code file whose normalised content matches a stepbuild benchmark reference file (the existing `LeakageGuard`) is dropped and counted.
- **Where it is built:** on the rented box at the start of the session (download + tokenise with all CPU cores). ~20 GB of uint16 shards on the 100 GB disk. A manifest records sources, counts, achieved shares and the tokenizer hash.

## 6. Training

### 6.1 A/B runs (before the full run)
A scaled-down model (8 layers, same widths, same expert layout) on ~300M tokens per run, fixed seeds, differing in exactly one setting:

| Run | Compares | Kept if |
|---|---|---|
| 1 | Per-head Muon vs AdamW | Muon's final validation loss is lower |
| 2 | Block AttnRes on vs off | On is lower and costs ≤ 10% throughput |
| 3 | SiTU-GLU vs SwiGLU | Equal or lower loss and no more loss spikes |

The winner of each is re-run with a second seed to estimate noise; a difference smaller than that noise keeps the simpler option. All results (loss curves, throughput, spikes) go in the model card.

Muon details: Muon (Newton–Schulz, 5 iterations) for all 2-D weight matrices except embeddings; for Q, K and V projections the momentum is split per head and orthogonalised per head. AdamW for embeddings, norms, router, biases. Learning rates for both are set from short sweeps in the A/B scale-down (3 values each).

### 6.2 Throughput gate
A 15-minute full-size run measures tokens/second. The launcher prints projected hours and cost for 10B tokens. If projected cost exceeds the budget cap, the token target is reduced to fit, and the owner is shown the numbers before the long run starts.

### 6.3 Full run
~10B tokens, cosine schedule with warmup, bf16, batch ~0.5M tokens, milestones at ~2%, 5%, 10%, 20%, 40%, 70%, 100%. Resumable, so a later session can extend it (e.g. to 20B) without restarting.

### 6.4 Safety on a rented box
Reused from the weekend run: atomic checkpoints (~every 30 min), auto-resume on crash, non-finite guard, milestone snapshots, run log.
New:
- **Checkpoint sync to the laptop** every ~3 hours (rsync over SSH, latest checkpoint + milestones + logs).
- **Spend guard:** the launcher takes the instance's $/hr and a budget cap (default $20), logs running cost, and stops cleanly at the cap after a final checkpoint.
- **MoE health:** per-layer expert load histograms logged; alert (log line + summary flag) if any expert gets < 10% or > 300% of its target load for more than 500 steps.
- **End of session:** copy everything back, verify, then destroy the instance only after the owner's go-ahead.

## 7. Evaluation and release

- Validation loss (text, code) at each milestone, and **bits per byte**, which is comparable with quipu-114m despite the tokenizer change.
- The same fixed-prompt samples as quipu-114m at each milestone.
- The 1M-token needle evaluation with the retrieval memory (existing harness; tokenizer-aware).
- Expert specialisation: per-expert routing frequency on Python vs JS/JSX vs English validation text.
- Release on Hugging Face as `AneekC/quipu-moe` (weights, milestones, tokenizer, standalone loader, honest model card with the A/B results), plus site and post.
- **Not in this spec:** the stepbuild benchmark (needs the chat/step fine-tune), KDA, longer context.

## 8. Budget

| Item | Est. hours | Est. cost at ~$0.55/hr |
|---|---|---|
| Setup + data build | ~1.5–2 | ~$1 |
| A/B runs (4–5 short runs) | ~2–3 | ~$1.5 |
| Throughput gate | 0.25 | ~$0.15 |
| Full run, 10B tokens | ~31–35 | ~$17–19 |
| **Total** | | **~$20** (cap enforced) |

Estimates assume ~50–70 TFLOPS effective of the box's measured 176.7 TFLOPS; the throughput gate replaces them with measured numbers.

## 9. Testing (on the laptop, before renting)

- Tokenizer: gate metrics computed on fixtures; round trip; special tokens stable.
- MoE layer: output shapes; top-k selection; renormalised weights sum to 1; dispatch/scatter equals a naive per-token loop on random inputs; Quantile Balancing drives a deliberately skewed router toward target load in a toy loop.
- SiTU-GLU: matches SwiGLU near zero, bounded by β₁β₂ for large inputs.
- Per-head Muon: orthogonalised per-head blocks have near-unit singular values; the update equals full-matrix Muon when there is one head.
- Block AttnRes: with all pseudo-queries at zero the attention weights are uniform, so each layer's input equals the mean of the available block representations (checked exactly); gradients reach every block's parameters.
- Full model: parameter count within 1% of §3; a 50-step CPU/laptop-GPU smoke run decreases loss; checkpoint → resume is bit-identical in the next step.
- Launcher: spend guard stops at the cap (fake clock); sync command built correctly; MoE health alert fires on a fake skewed histogram.
- Data builder: achieved shares within ±1% on a small build; leakage guard drops a planted reference file.
