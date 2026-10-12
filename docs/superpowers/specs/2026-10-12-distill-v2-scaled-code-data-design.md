# Distillation v2: scaled, execution-verified code data (design)

Date: 12 Oct 2026. Status: approved in conversation, awaiting written review.

## Goal

Grow the CUDA arm's verified teacher data from 3,277 answers (v1) to as many
verified tokens as the budget allows, as pretraining data for the step-2
experiment (distilled data mixed into pretraining vs normal mix, equal compute).
Code only; multilingual data is a separate, later spec.

Why: v1 distillation (fine-tune only) tripled MBPP at the same point (old chat
model 3/130 vs distilled 11/130; distilled final 19/257 = 7.4%), but fine-tuning
mostly teaches format. Changing what the model knows needs teacher data in
pretraining, at hundreds of millions of tokens, not thousands.

## Decisions (from the conversation)

| Question | Decision |
|---|---|
| Scope | Code only. Multilingual (in-language explanations + checkable maths) next. |
| Teacher | `Qwen/Qwen2.5-Coder-7B-Instruct`, Apache-2.0, pinned revision. Never Claude. |
| Answer style | Explain-first, structured (see "Teacher system prompt"). Style is described in words only; no Claude-written example answers in the prompt, and the data/model are never labelled "Claude-style". |
| Prompt source | nvidia/OpenCodeInstruct only (v1 pass rate 74%); APPS dropped (31%). |
| Cloud budget | Existing Vast credit only ($3.57); hard cap **$3.00**, $0.57 left as reserve. No top-up. |
| Cloud GPU | One RTX 3090 (~$0.17/h, ~17 h), teacher in **bf16**, vLLM. No 3090-vs-5090 benchmark. |
| Laptop | Does the rest: RTX 5060 Laptop 8 GB, 4-bit teacher via llama.cpp server, pause/resume any time. |
| Verification | On the laptop CPU (free), resumable, per part file. |

Expected yield: cloud ~90-150M generated tokens (~65-105M kept, ~120-170k solved
problems at ~1,024-token answers); laptop ~15-25M tokens per overnight session.

## Components

### 1. Prompt builder (extend `scripts/build_distill_prompts.py`)
- New flags: `--all-shards` (read all 50 OpenCodeInstruct parquet files at the
  pinned revision, not two), `--no-apps`, `--part-size 50000`, `--out data/distill-v2/prompts`.
- Same rules as v1: `average_test_score == 1.0`, every assert calls one function
  (the entry point), `quipu.decontam` against HumanEval/MBPP at the pinned commits,
  exact dedup on whitespace-collapsed text. Reference solutions never written out.
- Output: `part-0000.jsonl` ... `part-NNNN.jsonl` (same row schema as v1, plus
  `part`), deterministic order (seed 1337), and `manifest.json` (sources, sha256,
  counts per drop reason, sha256 of every part).
- v1 `data/distill-prompts/` stays untouched; a v2 prompt that equals a v1 prompt
  keeps the same id so v1 answers are not regenerated.

### 2. Generator (`scripts/distill_generate_v2.py`, new)
- An HTTP client for an OpenAI-compatible `/v1/chat/completions` endpoint. The
  same script drives vLLM (box) and llama.cpp `llama-server` (laptop).
- Flags: `--endpoint`, `--direction up|down` (box takes parts from the first
  upward, laptop from the last downward), `--concurrency` (box ~128, laptop ~4-8),
  `--max-new-tokens 1024`, `--teacher-tag` (e.g. `bf16-vllm`, `q4km-llamacpp`).
- Decoding: greedy (temperature 0), the system prompt below, the problem as the
  user turn.
- **Teacher system prompt** (fixed, stored in the script and in every manifest):
  > Start with one or two sentences restating what the problem asks and the
  > approach you'll take. Then give a short numbered plan (2-4 steps). Then the
  > complete function in one ```python block, with clear names and a brief
  > docstring, and comments only where the logic isn't obvious. End with a short
  > "Edge cases" line covering empty input, boundaries, and time complexity. Be
  > direct and friendly, don't repeat yourself, and no filler like "Great question!".
  It describes the style only; it contains no example answers, so no
  Claude-written text can reach the training data.
- Output: `data/distill-v2/generations/part-NNNN.<teacher-tag>.jsonl`, one row per
  prompt: id, part, teacher, revision, teacher_tag, response, finish (eos|length),
  new_tokens. Each row appended and flushed when it completes.
- **Claiming without coordination:** a part is "taken" by a worker when its
  generations file exists. Box walks up, laptop walks down; each skips parts that
  already have a complete file (sidecar `part-NNNN.done` with row count + sha256)
  or a file from the other worker. They meet in the middle; at most one part is
  ever worked twice (if both start it at once), which verification dedups by id.
- **Resume:** on start, ids already in a part's file are skipped. Stopping at any
  point (Ctrl+C, window closed, lid shut, power) loses at most the in-flight
  requests.
- **Pause file:** if `data/distill-v2/PAUSE` exists, the generator finishes
  in-flight requests, flushes and exits 0. Delete it and rerun to resume.
- Stops itself (exit 0) when `quipu.spend` reports the cap reached (box only,
  `--budget-usd`).

### 3. Verifier (extend `scripts/distill_verify.py`)
- `--parts-dir data/distill-v2`: verifies every generations part that has no
  up-to-date `verified/part-NNNN.jsonl`, resumable per part; dedups by id (first
  answer wins, box bf16 preferred over laptop q4).
- Same rule as v1: kept = finished on eos, a fenced block defines the entry point,
  every assert passes in `quipu.sandbox`.
- `verify_summary.json` across all parts, plus per teacher_tag pass rates (shows
  whether the laptop's 4-bit teacher is meaningfully worse).
- The kept text (plan + code + edge-case line, as written) is what pretraining
  uses; turning it into shards belongs to the step-2 spec.

### 4. Box run (cloud)
- Rent one 3090, reliability >= 0.98, only after showing the exact offer and
  price and getting "yes".
- Setup: clone repo, install vLLM, download the teacher at the pinned revision,
  hash-check, start `vllm serve` (bf16, `--max-model-len 2048`, gpu memory 0.92).
- First 15 minutes: confirm vLLM runs and measure tokens/s; stop if broken.
- `quipu.spend start --usd-per-hour <offer>`; generator `--budget-usd 3.00`.
- Copy-back: each completed part (+ `.done`) scp'd to the laptop and sha256
  checked. The box keeps everything until the final copy, so a stopped laptop
  watcher loses nothing. After the final verified copy: `quipu.spend stop`,
  destroy the box, record spend.

### 5. Laptop run
- `llama-server` (Windows CUDA build) with a Q4_K_M GGUF of the same teacher
  (an official Qwen GGUF if one exists at a pinned revision, else converted from
  the pinned weights), `-np 4..8` parallel slots, ~5-6 GB VRAM.
- `scripts/distill_laptop.ps1 start|pause|status`: start launches the server and
  generator (`--direction down`), pause creates the PAUSE file and stops the
  server, status prints parts done, rows, tokens and tok/s.
- Optional overnight scheduled task (23:00-07:00), only if the user asks.
- Never run alongside a GPU benchmark run (they share 8 GB).

## Testing (before any spend)

Unit tests (`tests/test_distill_v2.py`):
- extract_code on plan + code + edge-case replies (and replies with two blocks);
- part claiming: up and down walkers over 6 parts never take the same part
  unless started together; complete parts are skipped;
- resume skips done ids; a PAUSE file stops cleanly with everything flushed;
- verifier dedup prefers bf16 over q4 for the same id.

Smoke run: build prompts with `--limit` to one small part, run 50 prompts on the
laptop through llama-server, verify, read 5 answers by eye, check pass rate is
in the v1 range (~65-75%) and that answers follow the style (restate, plan,
code, edge cases); adjust the system prompt wording if not. Only then rent the box.

## Out of scope
- Multilingual data (next spec).
- Building pretraining shards from the kept data and the step-2 experiment.
- Any Claude-generated training text.

## Risks
- vLLM on the rented box fails to start: the 15-minute check costs ~$0.05; fall
  back to the laptop only.
- Host dies mid-run (as on 9-12 Oct): lose at most the current part's unsynced
  rows; spend cap still holds.
- Laptop RAM pressure (seen on 12 Oct): generator itself is light; the server
  holds the model in VRAM; the pause file gives a clean stop.
- Prompt supply: ~5M OpenCodeInstruct rows before filtering; far more than this
  budget can use.
