# Quipu

A small language model trained from scratch on one laptop GPU (RTX 5060 Laptop, 8 GB): own data pipeline, own training loop, own weights.

- **Weights, 7 training milestones and model card:** https://huggingface.co/AneekC/quipu-114m
- **Loss curve and samples:** https://quipu-lm.vercel.app

## quipu-114m

| | |
|---|---|
| Parameters | 114,114,048 (decoder-only: RMSNorm, RoPE, SwiGLU, grouped-query attention 12q/4kv, tied embeddings, 1,024-token context) |
| Data | 3B tokens: 80% FineWeb-Edu, 20% permissively licensed code (github-code-clean), minified/vendored/oversized files removed |
| Training | 45 h 22 min at ~18,400 tokens/s in bf16, zero restarts |
| Result | Text validation loss 6.38 → 3.27 |

It is a base model: fluent, on topic, and often wrong. Two measured findings from the run:

- GPT-2's tokenizer turns **36.5%** of code validation tokens into pure whitespace (2.0% for text), so the code loss (0.97) flatters the model. The next model gets a code-aware tokenizer.
- With a 1,024-token window and a small retrieval memory (BM25 over 256-token chunks, best chunk placed next to the question), a planted fact is answered correctly **75% (text) / 77% (code)** of the time in a 1M-token haystack, against 17% / 19% with no memory. Peak GPU memory ~1.2 GB.

## Layout

| Path | What |
|---|---|
| `quipu/` | Model, data pipeline, memory-mapped loader, trainer, evaluation, retrieval memory (`quipu/memory/`) |
| `scripts/` | Shard builder, GPU checks, pre-flight, the unattended weekend launcher, milestone evaluation, needle-in-a-haystack evaluation, Hugging Face export |
| `stepbuild/` | Work in progress: a harness and benchmark for building Flask + React apps one verified step at a time |
| `site/` | The project page |
| `docs/superpowers/` | Design specs and implementation plans for each piece |
| `tests/` | ~570 tests (CPU; GPU tests skip without CUDA) |

## Running the tests

```bash
python -m uv sync
CUDA_VISIBLE_DEVICES= python -m uv run python -m pytest -m "not gpu_gate" -q
```

## Licence

Code and weights: Apache-2.0. Training data keeps its own licences: FineWeb-Edu (ODC-By 1.0) and only MIT, Apache-2.0, BSD, ISC, CC0 and Unlicense code.
