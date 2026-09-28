---
license: apache-2.0
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- from-scratch
- small-language-model
- base-model
- grouped-query-attention
datasets:
- HuggingFaceFW/fineweb-edu
- codeparrot/github-code-clean
---

# quipu-114m

A 114,114,048-parameter language model trained from scratch on a single 8 GB laptop
GPU (RTX 5060 Laptop): own weights, own data pipeline, no fine-tuned base.

This is a **base model**. It continues text; it does not follow instructions or
answer questions. At this size it writes fluent, on-topic prose that is often
factually wrong. It is published as a baseline and as a record of what one
laptop can train in a weekend, not as something to rely on.

Project page: <https://quipu-lm.vercel.app> · Code: <https://github.com/Aneek1/quipu>

## Architecture

| | |
|---|---|
| Type | Decoder-only transformer, pre-norm |
| Parameters | 114,114,048 (input and output embeddings tied) |
| Layers / width | 12 layers, d_model 768, SwiGLU feed-forward 2,048 |
| Attention | Grouped-query: 12 query heads over 4 key/value heads, head dim 64 |
| Positions | Rotary embeddings (GPT-NeoX half-split), base 10,000 |
| Norm | RMSNorm, eps 1e-6 |
| Context | 1,024 tokens |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab 50,257) |

## Training

| | |
|---|---|
| Tokens | 2,999,975,936 (one pass, no data repeated) |
| Mix | 80% FineWeb-Edu (`sample-10BT`), 20% permissively licensed code from `github-code-clean` |
| Code filter | MIT, Apache-2.0, BSD, ISC, CC0, Unlicense only; minified, vendored and >16k-token files removed; HTML capped at 10% |
| Steps | 5,722 × 524,288 tokens |
| Optimiser | AdamW, cosine schedule, 200 warmup steps, bf16 autocast |
| Hardware | 1 × RTX 5060 Laptop GPU (8 GB), Windows |
| Wall clock | 45 h 22 min, ~18,400 tokens/s |
| Incidents | 0 resumes, 0 skipped (non-finite) steps |

## Results

Validation loss on held-out FineWeb-Edu text, and on held-out code files
deduplicated by exact hash against the training code.

| Step | Tokens | Text val loss | Code val loss* |
|---:|---:|---:|---:|
| 100 | 52M | 6.40 | 4.36 |
| 500 | 262M | 4.40 | 2.06 |
| 1,000 | 524M | 3.85 | 1.42 |
| 2,000 | 1.05B | 3.58 | 1.20 |
| 4,000 | 2.10B | 3.38 | 1.02 |
| 5,722 | 3.00B | 3.31 | 0.97 |

\* **The code loss is flattered by the tokenizer.** GPT-2's BPE splits indentation
into many whitespace tokens: 36.5% of the code validation tokens are pure
whitespace, against 2.0% for text. Those are easy to predict and pull the average
down. The same effect makes greedy code generation collapse into runs of spaces.
Do not compare this number with code models that use a code-aware tokenizer. The
next Quipu model uses one.

The text loss is not directly comparable with other small models either: the data
mix (20% code) and evaluation set differ.

### What it writes

Final model, greedy decoding:

> **Photosynthesis is the process by which** plants convert carbon dioxide into
> oxygen and release carbon dioxide into the atmosphere.

> **The history of the printing press** is a fascinating one. It was invented in
> 1848 by the French printer Pierre-Louis Leclerc.

Both are fluent and both are wrong, which is the honest summary of a 114M base
model. Greedy decoding also repeats itself. Sampling (temperature 0.8, top-k 50)
reads better. Samples from every milestone, for every prompt, are in the project
repository.

## Files

| File | |
|---|---|
| `model.safetensors` | Final weights, fp32 (step 5,722) |
| `milestones/step_*.safetensors` | bf16 snapshots at steps 100, 250, 500, 1,000, 2,000, 4,000 and 5,722, for studying how the model learns |
| `config.json` | Architecture |
| `modeling_quipu.py` | Standalone PyTorch implementation and loader |

`lm_head.weight` is not stored; it is the embedding matrix.

## Usage

```bash
pip install torch safetensors tiktoken huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys

path = hf_hub_download("AneekC/quipu-114m", "modeling_quipu.py")
spec = importlib.util.spec_from_file_location("modeling_quipu", path)
mq = importlib.util.module_from_spec(spec)
sys.modules["modeling_quipu"] = mq
spec.loader.exec_module(mq)

model = mq.load("AneekC/quipu-114m")                  # final model
early = mq.load("AneekC/quipu-114m", weights="milestones/step_001000.safetensors")

print(mq.generate(model, "Photosynthesis is the process by which",
                  max_new_tokens=60, temperature=0.8, seed=1337))
```

This is not a `transformers` model; there is no `AutoModel` class for it.

## Limitations

- Base model only: no instruction tuning, no safety tuning, no chat format.
- Frequently states false facts with confidence. Do not use its output as information.
- Code output is syntax-shaped but rarely correct, and greedy decoding degenerates
  into whitespace (see the tokenizer note above).
- English only. 1,024-token context.
- Trained on web text, which carries the biases of web text.

## Data and licence

Weights and code: Apache-2.0.

Training data: FineWeb-Edu (ODC-By 1.0, © Hugging Face) and the permissively
licensed subset of codeparrot/github-code-clean. Code files keep their original
licences; only MIT, Apache-2.0, BSD, ISC, CC0 and Unlicense files were used.
