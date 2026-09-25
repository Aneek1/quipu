# Quipu weekend run — 3B tokens, full-stack code, milestones, 1M-token retrieval memory

**Date:** 2026-09-25
**Status:** design approved in brainstorming, additions 1–4 and the HTML cap approved 2026-09-25; awaiting owner review of this document
**Builds on:** `2026-09-24-quipu-pretraining-pipeline-design.md` (sub-project 1, Tasks 1–13 done)
**Changes the run:** yes — this supersedes that spec's data mix and token budget for the weekend run

## 1. What changes and why

| # | Change | Why |
|---|---|---|
| 1 | Budget 1.5B → **3B tokens** (~50 h at the measured ~16.5k tok/s) | The weekend has ~60 h; 3B is past Chinchilla-optimal (~2.3B) for 114M. Owner's choice, accepting that one interruption pushes the run into Monday. |
| 2 | Mix **80% FineWeb-Edu / 20% full-stack code** | Owner wants a model that is useful for code, including full-stack work. |
| 3 | Loader **memory-maps** shards instead of loading them into RAM | 3B tokens = 6 GB; the concatenate-into-RAM loader would peak ~12 GB and does not scale. Data is read as training reaches it. |
| 4 | **Milestone snapshots**, never pruned | See what the model could do at each stage — grammar, then coherence, then facts, then code. |
| 5 | **Approach A: 1M-token retrieval memory + needle test**, run automatically after training | First measured step toward the owner's ~1M-token context goal on 8 GB. |
| 6 | **One launcher** that runs training then evaluation, and keeps Windows awake | The owner starts it and returns Monday. |

## 2. Honest expectations

- **Code:** ~0.6B code tokens across ~11 languages, at 114M parameters, with GPT-2's tokenizer (poor at indentation). Expect syntactically plausible JS/HTML/CSS/SQL/Python that is frequently wrong. Useful code *writing* needs sub-project 2 (350M, own tokenizer), instruction tuning, and the retrieval map. This run teaches the shape of each language.
- **Approach A** is retrieval-augmented generation: a fixed retriever finds the chunk, Quipu copies from it. It tests the memory plumbing and the model's copying, not a learned memory. The learned memory is approach B, next week.
- **Long-document perplexity** is deferred to approach B; A is not expected to improve it and measuring it now would show nothing.
- **Validation loss** stays on FineWeb-Edu only, so it remains comparable with published text-only runs; because 20% of training is code, expect it to be slightly higher than a pure-text run at the same token count.

## 3. Data

### 3.1 Sources

| Split | Source | Tokens |
|---|---|---|
| train, text | `HuggingFaceFW/fineweb-edu` sample-10BT | 2,400,000,000 |
| train, code | `codeparrot/github-code-clean` | 600,000,000 |
| val (trainer's loss) | FineWeb-Edu, taken first from the stream as today | 10,000,000 |
| val, code (post-run only) | `github-code-clean`, **held-out parquet files never used for train** | 5,000,000 |

`github-code-clean` is ungated and carries `language` and `license` per file. Keep only:

- **Languages (full stack):** Python, JavaScript, TypeScript, HTML, CSS, PHP, Java, GO, SQL, Shell, Dockerfile — in their natural proportions within that set, **except HTML, capped at 10% of code tokens** (measured natural share ≈24%, much of it generated boilerplate). Once HTML reaches 10% of the code tokens written so far, further HTML documents are skipped.
- **Licences (permissive):** mit, apache-2.0, bsd-2-clause, bsd-3-clause, isc, cc0-1.0, unlicense. GPL and others are dropped.

The held-out code files are a fixed range of parquet shard indices (e.g. the last 40 of 880) that the train builder never opens. The exact range is recorded in the manifest.

**The code val is deduplicated against train.** GitHub is full of forks and copied files, so held-out files will still contain code seen in training. Code train is built first; every code train document's content hash is kept; any code val document whose hash is in that set is dropped. The number dropped is recorded in the manifest. (Exact-content dedup only — near-duplicates remain; stated as a limit.)

### 3.2 Mixing

Documents from the two sources are interleaved **deterministically**: before each document, draw from whichever source is furthest below its target share of tokens written so far (text 0.8, code 0.2). No randomness, so a rebuild reproduces byte-identical shards. The mix is therefore uniform across the run, not "all text then all code".

`quipu.data.encode_document` stays the single rule for blank-skip + EOT. Code documents get no special markers in this run; per-language tags are a sub-project 2 decision alongside the tokenizer.

### 3.3 Manifest additions

Tokens and document counts **per source and per language**, the licence filter, the language filter, the held-out code shard range, the text/code target shares and the achieved shares.

### 3.4 Memory-mapped loading

`TokenStream` opens each shard with `np.memmap` (read-only, `<u2`) and indexes a logical concatenation without copying. Behaviour is unchanged: same batches for the same position, same wrap counter, same state dict, same validation of odd byte counts and of positions. Existing loader tests must pass unmodified, plus a test that peak RSS stays small when opening a large shard set.

## 4. Training changes

- `total_tokens = 3_000_000_000` → **5,722 steps** (3e9 // 524,288); warmup 200 (3.5%). `micro_batch = 4` as measured.
- **Milestones** at steps **100, 250, 500, 1000, 2000, 4000** and the final step: **bf16 model weights only** (no optimiser), written atomically to `checkpoints/milestones/step_NNNNNN.pt`, never pruned, ~230 MB each (~1.6 GB total). Milestones are separate from resumable checkpoints and are not used for resume.
- **Keep-awake:** while training runs on Windows, `train.py` calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` and clears it on exit (including on exceptions). No system setting is changed; it is the request a media player makes. It does **not** stop Windows Update restarts — the owner pauses updates before starting.

## 5. Approach A — retrieval memory over 1M tokens

### 5.1 Components (new package `quipu/memory/`)

| Unit | Job |
|---|---|
| `chunker.py` | Split a token sequence into fixed 256-token chunks with their offsets |
| `bm25.py` | Hand-written BM25 over chunk token ids (exact identifier matching) |
| `dense.py` | `intfloat/multilingual-e5-small` embeddings of decoded chunks; cosine top-k |
| `fusion.py` | Reciprocal-rank fusion of BM25 and dense rankings |
| `window.py` | Pack the top chunks, then the prompt, into a ≤1,024-token window (prompt last) |

### 5.2 The needle test (`scripts/needle_eval.py`)

- **Haystacks:** text (FineWeb val) and code (held-out code val), each at **1k, 32k, 128k and 1M tokens**.
- **Needles:** a fact with a random 5-digit value, planted at depths **0, 25, 50, 75, 100%**, **20 trials** per (haystack, size, depth), seeded.
  - text: `The access code for the {adjective} vault is {5 digits}.` → prompt `The access code for the {adjective} vault is`
  - code: `{ADJ}_VAULT_CODE = {5 digits}` → prompt `{ADJ}_VAULT_CODE =`
- **Conditions:** memory **off** (last 1,024 tokens only — the needle is visible only when it falls there); memory **on** with BM25, dense, and fused retrieval.
- **Scoring:** greedy-decode 8 tokens; pass if the exact 5-digit value appears. Report **retrieval hit rate** (was the needle's chunk in the window) separately from **copy accuracy given a hit**, so every failure is attributable.
- **Also recorded:** peak VRAM, retrieval latency, index build time, per condition.
- Runs on the final checkpoint; writes `results/needle/*.json` and a table.

### 5.3 Code loss at milestones (`scripts/milestone_eval.py`)

Loss on the FineWeb val and on the code val for every milestone snapshot, so the run records *when* code began to be learned.

**Samples at every milestone:** generate from a fixed set of prompts — a few text openings and a few code openings (Python function, JS function, HTML page, SQL query) — greedy, plus one sampled continuation with a fixed seed. Same prompts at every milestone, saved to `results/milestones/samples.md`, so the stages of learning can be read side by side.

## 6. The launcher (`scripts/weekend.py`)

0. **Start guards** — refuse to start, with a message saying exactly what to fix, if: other processes hold more than 1.5 GB of GPU memory (Task 12 measured 2.8 GB held by desktop apps); the laptop is on battery; or fewer than 40 GB are free on the drive. `--force` overrides, for the owner only.
1. Run `quipu.train` (resuming automatically if a run log for the run id exists).
   **Auto-resume:** if training exits with a failure, relaunch it with `--resume` after a 2-minute pause, at most 3 times. Two exits are never retried: the non-finite stop (it would fail identically) and a user interrupt (Ctrl+C means stop). `train.py` gives these distinct exit codes — **3** for the non-finite stop, **130** for an interrupt — so the launcher can tell them apart from other crashes. Every attempt and its exit code is recorded in the weekend summary.
2. If training completed: run `milestone_eval.py`, then `needle_eval.py` if it exists, then `results_table.py --out RESULTS.md`. Each step is a separate process; a failure is logged and the next step still runs. **No evaluation step can modify checkpoints.**
3. Write a one-page `results/weekend_summary.md` with what ran, what passed, and where the outputs are.

## 7. Order of work tonight (critical path first)

1. Memory-mapped loader (small).
2. Code-mix builder + config → **rebuild the data** (the long pole: FineWeb 2.4B + code 0.6B download and tokenise).
3. Milestones + keep-awake.
4. Pre-flight re-run on the new data. **The run starts only after it passes.**
5. Approach A + milestone eval + launcher. If not ready when the owner leaves, the launcher runs whatever evaluation scripts exist when training ends; A can be finished during the run using CPU only (tests with the GPU hidden), or on Monday.

## 8. Not in scope

Approach B (learned kNN memory), long-document perplexity, iterative "explore while generating" retrieval, instruction tuning, a code tokenizer, 350M.

### Roadmap requirement recorded 2026-09-25: MCP / full-stack app building

The owner requires Quipu to build full-stack apps through MCP plugins inside OpenCode. That has two parts, neither of which belongs in pretraining:

1. **Serving (engineering, next week):** an OpenAI-compatible chat-completions endpoint with tool calling, so OpenCode — already an MCP client — can offer Quipu every configured MCP server's tools. No training dependency.
2. **Skill (sub-project 3):** agentic fine-tuning on tool-call traces so the model emits valid tool calls step after step; the "score enumerated candidates instead of generating" option (see the pipeline spec) is the leading approach for a small model.

Expectation stated plainly: short tool-driven tasks after sub-project 3; end-to-end full-stack app generation is the long-term target that needs 350M, the retrieval map (repo, framework docs, API signatures) and agentic fine-tuning together. It is not claimed for the 114M model.

### Roadmap requirements recorded 2026-09-25: document upload and image generation

Neither is a pretraining concern; neither changes the weekend run.

- **Document upload** — ingestion (PDF/DOCX text extraction → chunks) into the same memory store approach A builds, so answers come from the uploaded document. Reuse the ingestion/retrieval/citation design from the owner's soft-robotics chatbot rather than rebuilding it.
- **Image generation** — a text model cannot emit pixels, and training an image generator from scratch is out of reach on 8 GB. Route: Quipu calls an image model **as a tool** through the same tool-calling/MCP path — a local model that fits in 8 GB (never concurrently with training), or an external API behind an explicit opt-in because the prompt leaves the machine.

### Roadmap requirements recorded 2026-09-25: Quipu-Vision and Quipu-Image

The owner wants a separate Quipu image model working together with Quipu, VLM-style. Two sub-projects, each with its own design pass, neither using the GPU during the weekend run:

- **Sub-project 5 — Quipu-Vision (image understanding, first):** LLaVA-style — vision encoder → trained projector → Quipu. Start with a small pretrained, permissively licensed encoder and train only the projector plus a Quipu fine-tune on image–caption pairs (fits 8 GB; days). A from-scratch tiny ViT encoder is a later option. Pairs with document upload (diagrams in uploaded PDFs).
- **Sub-project 6 — Quipu-Image (image generation):** a small from-scratch diffusion model (~30M parameters, ~256 px) that Quipu drives through tool calls. Expect recognisable but crude images, best in a narrow domain; broad high-quality text-to-image is out of reach at this compute.
- **Both:** every image dataset is licence- and provenance-checked before use.
