# Multi-turn, verified chat data: building small things step by step (design)

Date: 15 Oct 2026. Status: approved by the owner on 15 Oct (answers at the end).

## Goal

A chat model that builds small things over several turns: write a function or a small
page, then change it when asked ("now ignore case", "add a reset button"), keeping what
already worked. Measured by tests at every turn, not by eye.

## Why (observed on 15 Oct)

- The chat model's fine-tune was almost all single-turn: aya 39,487 and distill 3,100
  one-question conversations, oasst2 6,084 human chats (few about code), stepbuild 397
  (one step each, from 5 repos). It never practised editing its own earlier answer.
- In the demo it copies instead: asked for an NEA temperature page after the vowel
  example, it rewrote the vowel counter, comments included. In a fresh chat the same
  prompt gets an HTML attempt. The demo now sends only the current message by default
  (commit 06e11fb); this spec is what lets that switch go back on.
- Unfenced code and wrong `# Output:` comments are common in its replies; verified
  data with fenced code fixes the first, tests catch the second.

## Decisions

| Question | Decision |
|---|---|
| Teacher | `Qwen/Qwen2.5-Coder-7B-Instruct`, Apache-2.0, pinned revision (as distill v2). Never Claude, never a model whose terms forbid training on outputs. |
| Who writes the follow-up requests | The teacher, from a fixed instruction we write that describes the kinds of change (below). The instruction is not training data; the requests and answers are. |
| What counts as verified | Every turn passes its own new tests **and** all earlier turns' tests (no regressions). |
| Turns per chain | 2-3 (turn 1 plus one or two changes). |
| Fine-tune context | 4,096 tokens with the NTK-scaled RoPE base (20,452) the demo already uses, so fine-tuning also trains the stretched positions. |
| Demo | "Use earlier messages" goes back on by default when the turn-2 pass rate on the held-out set is at least half the turn-1 rate. |

## Data

### A. Verified Python chains (main source)
1. **Seeds:** problems from distill v2's kept set (`data/distill-v2/kept`, 498,813
   rows: OpenCodeInstruct prompts, CC-BY-4.0, each with an entry point and a teacher
   answer that passed the problem's unit tests; the tests are joined by `id` from
   `data/distill-v2/prompts`). Turn 1 is the existing prompt and answer.
2. **Follow-up request:** the teacher is shown the turn-1 problem and code and asked
   for one change of a listed kind, each **backward compatible**, so the old tests
   must still pass: add an optional parameter whose default keeps the old behaviour;
   handle a new input type or edge case the old tests don't cover; return extra
   information only behind a flag; add input validation that raises on invalid input
   only. One request per call, plain words, no code.
3. **New tests, written separately:** a second teacher call sees only the request and
   the turn-1 code (not any answer) and writes 3-6 asserts for the new behaviour.
   Asserts that the turn-1 code already passes are dropped, so every kept assert
   actually tests the change; a chain with fewer than 2 left is dropped.
4. **Answer:** the teacher gets the whole conversation so far and replies with the
   complete updated function in one fenced block (the distill v2 style prompt).
5. **Verify** in `quipu.sandbox`: the answer must pass the turn's new asserts and
   every earlier turn's asserts. Repeat 2-5 for turn 3 on chains that passed turn 2.
6. **Copy check:** a turn whose code is identical to the previous turn's is dropped
   (it can't pass new asserts anyway; logged separately because it is the failure we
   are fixing).

Known weakness: the new asserts are the teacher's, and can be wrong. Keeping only
answers that pass them throws away good answers when the tests are wrong, but cannot
keep an answer that breaks the old, human-checked tests. The held-out eval below
reports how often written tests disagreed with every answer (an estimate of bad tests).

### B. Small web pages (second, smaller source)
Same loop for single-file HTML pages with inline JS: "a page with a counter and a
+1 button", then "add a reset button", then "start from 10". Checks run with Node
(v24 is installed) and jsdom: the page parses, required element ids exist, scripts
have no syntax errors, and scripted clicks give the expected text. The teacher
writes the page request and the check list (ids, click sequence, expected text) in a
fixed JSON form; seeds are teacher-written page ideas from a list of ~30 page kinds
we name (counter, to-do list, unit converter, tip calculator, ...). Pages that fetch
live data (like the NEA example) are out of scope: they can't be checked offline.

### C. Real step-by-step history (stepbuild, existing)
`stepbuild` already mines permissively licensed Flask/React repos into one-commit
steps (the 300-repo run: 92 licensed repos, 42 with examples). Change: consecutive
qualifying commits of one repo become one multi-turn conversation (2-3 steps), instead
of single-step rows. Same licence filter, same held-out-by-repo split, same bench
leakage guard.

### Mix for the fine-tune
Single-turn distill v2 (the planned 498k fine-tune), plus A, B, C, plus the existing
aya and oasst2 rows so languages and plain chat are not forgotten. Shares are set
after the smoke run shows yields; a starting point is 60% single-turn code, 25% A,
5% B, 5% C, 5% aya+oasst2 (by assistant tokens).

## Evaluation

Held out by seed problem (no seed appears in both train and eval):
- **Turn-k pass rate** (k = 1, 2, 3) on 500 held-out chains, the model answering
  every turn itself (its own earlier code stays in the conversation, like the demo).
- **Regression rate:** turns that break an earlier turn's tests.
- **Copy rate:** turns that repeat the previous code unchanged.
- **Unfenced-code rate.**
- MBPP and HumanEval pass@1 (no regression from 7.4% MBPP), the stepbuild bench
  (10 apps), and bits per byte on the nine languages (aya/oasst2 kept them in).

## Compute and cost

| Step | Where | Cost |
|---|---|---|
| Smoke run: 200 chains through the whole loop | Laptop, 4-bit teacher via llama-server (as distill v2) | free |
| Generation (target ~20k verified Python chains, ~1k pages) | Kaggle 2x T4 with vLLM in fp16 (30 GPU-hours a week, free); laptop overnight as backup | free; the smoke run measures chains per hour, and the target is cut to fit two weeks of Kaggle quota |
| Verification | Laptop CPU | free |
| Fine-tune at 4,096 tokens | **Kaggle 2x T4** (owner, 15 Oct), fp16 with loss scaling; Modal only as a fallback if fp16 proves unstable | free (fallback ~$3-6 of Modal credit) |

**Budget clash for November (needs the owner's call):** November's ~$27 of Modal credit
was planned for the 3.5-hour B200 continued-pretraining run (~$24). This fine-tune
(~$3-6) and the long-context phase (~$7) don't both fit beside it. Options: (1) this
fine-tune in November, CPT in December; (2) CPT in November, this in December;
(3) run the fine-tune on Kaggle T4s instead (needs an fp16 path in the trainer and a
check that the MoE is stable in fp16).

## Components

- `scripts/chain_generate.py`: drives the loop (request, tests, answer) against an
  OpenAI-compatible endpoint (vLLM or llama-server), resumable per part, pause file,
  same conventions as `distill_generate_v2.py`.
- `scripts/chain_verify.py`: sandbox checks per turn, regression and copy checks,
  `verify_summary.json` with drop reasons.
- `scripts/page_check.mjs` + `scripts/page_generate.py`: the web-page loop and its
  jsdom checker.
- `stepbuild/dataset`: an option to chain consecutive commits.
- `scripts/build_chat_data.py`: new sources `chains`, `pages`; context 4,096;
  multi-turn loss mask unchanged (all assistant turns).
- `scripts/chain_eval.py`: the turn-k, regression, copy and fence metrics.

## Testing before any spend

Unit tests: request/test/answer parsing; dropping asserts that turn 1 already passes;
regression check; copy check; resume and pause; jsdom checker on a known-good and a
known-bad page. Then the 200-chain smoke run on the laptop, reading 10 chains by eye
(are requests sensible, are tests about the change, are answers complete) before any
Kaggle or Modal time.

## Out of scope

- Pages that call live APIs (NEA, weather): not checkable offline.
- Whole apps in one turn; multi-file projects beyond what stepbuild covers.
- Any Claude-written training text.

## Answered (15 Oct)

1. November budget: **option 3**, the fine-tune runs on Kaggle; Modal's November credit
   stays for the B200 continued-pretraining run. Before the real fine-tune, a short
   fp16 run on Kaggle must match a bf16 run of the same steps (loss within 1%, no
   inf/NaN, router balance unchanged); if not, the fine-tune falls back to Modal.
2. Kaggle: the owner's account is verified for GPUs.
3. Size (owner didn't pick; Claude's default): 200-chain laptop smoke run, then 5k
   chains, look at the results, then 20k.
