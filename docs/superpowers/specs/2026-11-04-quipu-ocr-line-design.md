# Quipu-OCR v1: reading text lines

Date: 4 Nov 2026. Status: design approved, not built.

## 1. Goal

Give Quipu (the 1B-total / 149M-active MoE, `configs/quipu-moe.toml`) the ability to
read a single line of text from an image and write it out. Printed and handwritten
English, one line per image.

Everything trains from scratch, the image encoder included. No pretrained vision
weights. This keeps the "built from scratch" story intact and costs data and compute;
the write-up says so.

Success for v1 means numbers, not a demo:

- character error rate (CER) and word error rate (WER) on held-out sets (section 6),
- reported next to Tesseract and TrOCR-small on the same lines,
- a hallucination check that shows the model reads rather than guesses.

Beating those baselines is not a requirement.

### Not in v1

Full pages, layout and reading order, multilingual text, formulas, tables, receipts and
forms. Full pages need many more image tokens than the 2,048-token context leaves room
for; they are a later spec.

## 2. Approach

Two stages, encoder first (option A of the 4 Nov discussion):

1. **Encoder stage (CTC).** A small ViT learns to read lines on its own, with a CTC
   head over characters. No language model. This is cheap, gives a standalone OCR
   baseline, and proves the encoder sees letters before it is joined to Quipu.
2. **Joint stage.** Drop the CTC head, add a connector, and train Quipu to write the
   line's text after the image tokens.

Rejected:

- **Everything end to end from step one.** A random encoder gives Quipu no signal for
  a long time, so Quipu learns to produce plausible text from language alone. That is
  the usual failure mode and it is hard to tell apart from slow progress.
- **A separate encoder-decoder OCR model (TrOCR style).** The cleanest OCR, but it is
  not Quipu reading. Not built; TrOCR-small is used as a baseline only.

## 3. Model

### 3.1 Input

- Grayscale line image, resized to 64 px high keeping the aspect ratio, width capped at
  1,024 px (wider lines are scaled down to fit), padded on the right to a multiple of 16.
- 16×16 patches: 4 rows × up to 64 columns = at most 256 patches per line.
- A padding mask marks the right-hand padding so attention ignores it.

### 3.2 Encoder (`quipu/vision.py`)

- ViT: 12 layers, width 384, 6 heads, MLP 1,536, pre-norm. About 22M parameters.
- Patch embedding: a 16×16 convolution, stride 16.
- Positions: learned row embedding (4) plus learned column embedding (64), added. A
  factorised table suits short, wide images and lets width vary.
- Output: one 384-wide vector per patch.

### 3.3 Encoder stage head (`quipu/ocr_ctc.py`)

- Mean over the 4 rows of each column, giving up to 64 column vectors, then a linear
  layer to the character set plus the CTC blank.
- Character set: printable ASCII (95) plus a few common symbols (°, £, €, ×, –, ', ').
  Fixed in a file so the encoder and evaluation agree.
- 64 columns give 64 CTC steps. CTC needs a blank between repeated letters, so a line
  needs up to about twice its length in steps in the worst case; the encoder stage keeps
  lines of at most 40 characters (and drops any line whose CTC path does not fit),
  filtered out, not truncated. The joint stage has no such limit and uses lines up to 60.

### 3.4 Connector and prompt (`quipu/ocr_lm.py`)

- Connector: a 2-layer MLP, 384 → 768 → 768, GELU between.
- Up to 256 image vectors go in front of the text. Sequence layout:

  `<img>` [image vectors] `</img>` `Transcribe:` [line text] `<eos>`

- `<img>` and `</img>` are two new tokens added to the 49,152 vocabulary. Their
  embeddings start as the mean of the existing embeddings.
- Loss only on the line text and `<eos>`.
- Image positions use Quipu's normal RoPE positions in order; nothing special.

## 4. Data

All data is generated or downloaded by scripts in `scripts/ocr/`, with a manifest that
records every source, its licence and a sha256.

### 4.1 Text to render

- Lines cut from Quipu's own data pool (the data v3 pool on Volume `quipu-pool`, or the
  local shards), split at sentence or phrase boundaries to 5–60 characters.
- Mixed in: random strings (digits, codes, mixed case, punctuation) at about 15%, so the
  model cannot lean on language to guess.
- **Never** text written by Claude or another model (standing rule). **Never** SP
  Manufacturing documents.
- Held-out text: a separate slice of the pool, removed from training text before
  rendering.

### 4.2 Synthetic printed lines (`scripts/ocr/render_lines.py`)

- Fonts: open-licensed fonts from the Google Fonts collection (SIL OFL / Apache). The
  renderer reads the licence file shipped with each font and refuses fonts without one.
  Serif, sans, monospace and display.
- Augmentation: size, weight, letter spacing; Gaussian blur; noise; JPEG quality 30–95;
  small rotation (±3°) and perspective; background textures (generated paper, gradients);
  ink bleed and erosion; random crop margins.
- Rendered on the fly in the data loader where CPU allows, otherwise pre-rendered to
  shards. Target pool: 10M distinct lines.

### 4.3 Synthetic handwriting (same script, `--style hand`)

- Handwriting-style fonts from the same collection, same licence check.
- Per-character jitter: baseline wobble, slant, size and spacing varied per letter;
  stroke thickness varied per line.
- This is a stand-in for real handwriting, not a substitute, and the write-up says so.

### 4.4 Real handwriting

- Candidate sets are checked for licence before use. IAM is non-commercial research only
  and needs registration, which does not fit public weights; it is evaluation only at
  most, and only if its terms allow that use. GNHK and others are checked one by one.
- A set is used for training only if its licence allows redistribution of models
  trained on it. Otherwise it is evaluation only, or not used.
- If no set can be used for training, v1 trains on synthetic handwriting only and the
  model card says so.

### 4.5 Mix

- Encoder stage: 70% printed, 30% synthetic handwriting (plus real handwriting if
  allowed).
- Joint stage: the same image mix, plus about 10% plain text samples from the pool (no
  image) so Quipu keeps its writing ability.

## 5. Training

### 5.1 Encoder stage

- CTC loss, AdamW, warmup then cosine, bf16 where the GPU has it (fp16 with loss scaling
  on T4, as in the T4 low-memory work).
- Hardware: Kaggle 2× T4 (free). At about 22M parameters and short sequences this fits
  easily.
- Stop rule: CER on the held-out synthetic printed set stops improving for two
  evaluations in a row.
- Checkpoint: `ocr-encoder/step_*.pt`, encoder weights only kept for the next stage.

### 5.2 Joint stage

- Start: Quipu from the latest continued-pretraining checkpoint (the Nov run, Volume
  `quipu-ckpt` `/stable/`) and the encoder from 5.1.
- Phase 1: encoder frozen, connector and Quipu train. Phase 2: everything trains, encoder
  LR at 0.1× the rest, for the last ~20% of steps.
- Hardware: 1–2 B200 hours on Modal (December credit, through `modal_train.py`'s budget
  guard pattern), or Kaggle T4s with the low-memory config if credit is short. The plan
  states the cost before any paid launch.

## 6. Evaluation (`scripts/ocr/eval_ocr.py`)

- Held-out sets, fixed once and hashed:
  - synthetic printed (held-out text, held-out fonts),
  - synthetic handwriting (held-out text, held-out fonts),
  - real handwriting, if a licence allows evaluation use,
  - nonsense lines: random characters and made-up words (the hallucination check).
- Metrics: CER and WER (Levenshtein), plus exact-line match.
- Baselines on the same images: Tesseract (Apache-2.0), TrOCR-small (MIT). The
  encoder-stage CTC model is a third baseline.
- Greedy decoding, no language-model rescoring.
- Hallucination check: on nonsense lines, the joint model's CER should not be much worse
  than the CTC baseline's. If it is, Quipu is guessing words, and that is reported.
- Output: a results table in the repo, the same format as other Quipu result tables.

## 7. Files

| File | Job |
|---|---|
| `quipu/vision.py` | the ViT encoder |
| `quipu/ocr_ctc.py` | CTC head, character set, greedy CTC decode |
| `quipu/ocr_lm.py` | connector, sequence building, the joint model wrapper |
| `quipu/ocr_data.py` | line dataset, mask, collation, the text-only mix |
| `scripts/ocr/build_text.py` | cut lines from the pool, make the held-out split |
| `scripts/ocr/render_lines.py` | printed and handwriting rendering, font licence check |
| `scripts/ocr/train_ctc.py` | encoder stage |
| `scripts/ocr/train_ocr_lm.py` | joint stage |
| `scripts/ocr/eval_ocr.py` | evaluation and baselines |
| `configs/ocr-ctc.toml`, `configs/ocr-lm.toml` | the two stages |

Each module gets unit tests: patch shapes and masks, CTC decode on known inputs, the
sequence layout and loss mask, the renderer's licence refusal.

## 8. Risks

- **A from-scratch encoder may be weak on handwriting.** Synthetic fonts are a poor match
  for real hands. The CTC baseline shows this early and cheaply.
- **Licences for real handwriting may rule out training.** Handled in 4.4.
- **Adding two tokens changes the vocabulary.** The tokenizer and the model's embedding
  size both change. Old checkpoints load with the two new rows added.
- **Credit.** The joint stage waits for December's Modal credit or runs slower on T4s.
