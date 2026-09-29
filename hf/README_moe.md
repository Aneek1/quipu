---
license: apache-2.0
language:
- en
- id
- ms
- zh
- ja
- ko
- ta
- hi
- ur
library_name: pytorch
pipeline_tag: text-generation
{{base_model_yaml}}tags:
- from-scratch
- mixture-of-experts
- small-language-model
- multilingual
- code
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceFW/fineweb-2
- codeparrot/github-code-clean
{{datasets_yaml}}---

# {{model_name}}

A mixture-of-experts language model trained from scratch: **{{total_params}} parameters
in total, {{active_params}} active per token** ({{params}}). Own tokenizer, own data
pipeline, no fine-tuned base. Code and ten languages: English, Indonesian, Malay,
Simplified and Traditional Chinese, Japanese, Korean, Tamil, Hindi (Devanagari and
romanised) and romanised Urdu.

{{kind_line}}

At this size it is often wrong. It is published as an honest record of what a small
sparse model learns under a $20 compute cap, with every A/B result, including the
ones that lost. Fields marked _not yet measured_ had no result file when this card
was generated.

Project page: <https://quipu-lm.vercel.app> · Code: <https://github.com/Aneek1/quipu>

## Architecture

{{architecture}}

## Data

{{data}}

{{lid}}

### Benchmark decontamination

{{decontamination}}

{{sft_data}}

## Training

{{training}}

### A/B tests (run before the full run, results kept either way)

{{ab_results}}

### FP8

{{fp8}}

## Evaluation

### Bits per byte per language

{{bpb}}

### Code: HumanEval and MBPP

{{code_eval}}

### Expert usage

{{experts}}

### Samples

{{samples}}

## int4

{{int4}}

## Usage

```bash
pip install torch safetensors tokenizers huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys

path = hf_hub_download("{{repo_id}}", "modeling_quipu_moe.py")
spec = importlib.util.spec_from_file_location("modeling_quipu_moe", path)
mq = importlib.util.module_from_spec(spec)
sys.modules["modeling_quipu_moe"] = mq
spec.loader.exec_module(mq)

model = mq.load("{{repo_id}}")                                    # fp32
# model = mq.load("{{repo_id}}", weights="model-int4.safetensors")  # int4 file
tok = mq.load_tokenizer("{{repo_id}}")
{{usage_call}}
```

This is not a `transformers` model; there is no `AutoModel` class for it. The loader
runs the experts one after another without a KV cache: correct, not fast.

## Limitations

- Small: {{active_params}} parameters are active per token. It states false things
  with confidence; do not use its output as information.
{{mix_limitation}}
{{code_limitation}}
{{experts_limitation}}
- Trained on web text and public code, which carry their biases; no safety tuning.
{{chat_limitation}}

## Credits and licence

Weights and code: Apache-2.0. Language identification of the training text used
[{{lid_model}}](https://huggingface.co/{{lid_model}}) (MIT). Training data: FineWeb-Edu
and FineWeb-2 (ODC-By 1.0, © Hugging Face) and the permissively licensed subset of
codeparrot/github-code-clean (files keep their licences).{{sft_credits}} HumanEval
(MIT) and MBPP (CC-BY-4.0) were used for evaluation only and filtered (substring /
13-gram matching) from the training data.
