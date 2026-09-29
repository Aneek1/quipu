# Stepbuild core: step dataset, harness and benchmark

**Status:** design, awaiting owner review · **Date:** 2026-09-28 · **Branch:** `pipeline-114m` (new package in the same repo)

## 1. Why this exists

The next Quipu goal is a model that **builds Flask + React apps one small, verified step at a time**, rather than writing a whole app in one pass. A small model cannot design an app; it can fill in one well-scoped step when the system around it keeps the step small, shows it a similar example, and checks the result by running it.

The work runs on two tracks that share this core:

| Track | Model | Purpose |
|---|---|---|
| A (fast) | Qwen2.5-Coder-1.5B (Apache-2.0), LoRA fine-tune on Kaggle | A usable step builder in ~2 weeks; proves the dataset and harness |
| B (own) | Quipu MoE, ~1B total / ~250M active, trained from scratch | The from-scratch model, scored against Track A |

This spec covers only the **shared core**: the step dataset, the harness, and the benchmark. Fine-tuning, the tokenizer and MoE pretraining each get their own spec later.

**Success for this spec:** the harness, driven by a scripted fake model, passes a known-good app and fails a known-bad one at the right step; the dataset builder produces a held-out-by-repo JSONL; and the benchmark runs end to end and prints its scores. No trained model is required to finish this spec.

## 2. Scope

**In:** `stepbuild/` package (dataset builder, harness, benchmark), 10 benchmark app specs with hidden acceptance tests, tests for all of it.

**Out:** any model training; Next.js (a later stretch goal); UI/browser testing (the frontend is checked by "it builds", not by clicking); multi-agent planning; anything that sends data to a third-party LLM.

**Data rule:** no training data is generated with Claude, GPT or any other model whose terms restrict training on outputs. Examples come from real, permissively licensed git history only.

## 3. Components

New top-level package `stepbuild/` in the quipu repo, following the repo's conventions (explanatory docstrings, frozen dataclasses, strict validation, atomic writes via `quipu.fsio`).

### 3.1 Step dataset — `stepbuild/dataset/`

| Unit | Job |
|---|---|
| `discover.py` | Find candidate repos through the GitHub API using the `gh` CLI (already authenticated on this machine; no token handled by our code). Collects three tagged groups: `fullstack` (Flask in `requirements.txt`/`pyproject.toml` **and** `react` in `package.json`), `flask` only, and `react` only. Full-stack repos are preferred; single-side repos supply examples for their half of the plan, and each example carries its tag. |
| `licence.py` | Keep only repos whose GitHub-detected licence SPDX id is one of `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `0BSD`, `Unlicense`. Record repo, licence and commit range in `data/stepbuild/SOURCES.jsonl` for attribution. |
| `mine.py` | Full clone (history needed) into a cache dir; walk first-parent history oldest → newest; turn each qualifying commit into one example. |
| `filters.py` | Commit-level and file-level filters (below). Shared rules with Quipu's code filter where they overlap. |
| `format.py` | Render an example into the chat format (below) and write JSONL shards atomically. |
| `split.py` | Split **by repo**: 90% train, 5% validation, 5% test. A repo's commits never cross splits. |

**Commit filters (all must pass):**
- not a merge commit; not the root commit;
- touches 1–3 files and 1–200 changed lines (added + removed);
- every touched file is source we want: `.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.css`, `.html`, `.sql`, `requirements.txt`, `package.json`;
- no touched path under `node_modules/`, `dist/`, `build/`, `vendor/`, `.venv/`, `migrations/versions/`, and no `*.min.*`, `*.map`, lockfiles;
- commit message after stripping is ≥ 12 characters and not matching a low-information list (`wip`, `fix`, `fixes`, `update`, `updates`, `changes`, `minor`, `cleanup`, `typo`, `commit`, `test`, `.`), case-insensitive, and not just a merge/revert line.

**Example format (one JSON object per line):**
```json
{
  "repo": "owner/name", "licence": "MIT", "commit": "abc123", "split": "train",
  "messages": [
    {"role": "system", "content": "You build Flask + React apps one small step at a time. Reply with the complete new contents of each file you change, in the FILE block format."},
    {"role": "user", "content": "STEP: <commit message>\n\nCONTEXT FILES:\n<up to 3 related files, pre-commit contents, each in a FILE block>\n\nPROJECT TREE:\n<at most 60 files, backend/ and frontend/src/ first>"},
    {"role": "assistant", "content": "<one FILE block per changed file, post-commit contents>"}
  ]
}
```
`FILE` block: a line `=== FILE: <path> ===`, the full file contents, a line `=== END FILE ===`. **Full-file output, not diffs:** small models are unreliable at producing applicable patches, and full files for ≤200-line changes to modest files fit the context. Files over 400 lines are excluded (the commit is dropped) so outputs stay bounded.

**Context file selection:** the pre-commit versions of the changed files, plus up to 2 further files chosen by BM25 over the repo at that commit using the commit message as the query (reusing `quipu.memory.bm25`). Total user message capped at 6,000 tokens (Qwen tokenizer count via `tokenizers` if installed, else a 4-chars-per-token estimate); examples over the cap are dropped, never truncated mid-file.

**Project tree:** PROJECT TREE lists at most 60 files (backend/ and frontend/src/ first), shared by dataset and harness. Past the cap a final line reads `... (N more files not shown)`. The miner passes every file at any depth; no depth limit.

**Scale target:** 300–1,000 repos; the builder reports repos found / licensed / mined and examples kept vs dropped per filter. No minimum example count is required to finish this spec, but the report must be produced.

### 3.2 Harness — `stepbuild/harness/`

| Unit | Job |
|---|---|
| `model.py` | `StepModel` protocol: `complete(messages: list[dict]) -> str`. Implementations later (Qwen, Quipu); here only `ScriptedModel` (returns canned replies in order) for tests. |
| `plan.py` | Fixed Flask + React build order, instantiated per app spec into concrete steps: 1 data model, 2 API routes, 3 API tests, 4 React components, 5 wiring (API client + App), 6 run check. Each step has a title, the files it may create or edit, and which checks run after it. |
| `blocks.py` | Parse `FILE` blocks from model output; reject paths outside the project, absolute paths, `..`, or files not allowed for the step. |
| `sandbox.py` | Materialise the project in a temp dir from a fixed template (Flask app skeleton + Vite React skeleton, dependencies pre-installed once into a cache and linked/copied); run checks with timeouts; no network during checks (best-effort on Windows: checks run with proxy env vars pointed at an unreachable address; documented as best-effort). |
| `checks.py` | `pytest -q` (API), `python -m pyflakes` (lint, errors only), `npm run build` (frontend), each with a timeout; returns pass/fail plus the trimmed error output (last 60 lines). |
| `retrieve.py` | For each step, BM25 over a local example library (the train split of the step dataset) using step title + app spec as the query; include the top 2 examples in the prompt. |
| `runner.py` | Drive plan → step → parse → write → check → retry (max 3 retries per step, error output appended to the conversation) → next step. Stops the app at the first step that fails after retries. Writes a per-app trace JSON. |

### 3.3 Benchmark — `stepbuild/bench/`

- **10 app specs** in `stepbuild/bench/apps/<name>/spec.md`, one paragraph each: todo list, notes, bookmarks, recipe box, expense tracker, reading list, habit tracker, contact book, simple inventory, todo list with login (session auth).
- **Hidden acceptance tests** per app in `stepbuild/bench/apps/<name>/acceptance/` — pytest files that hit the Flask API (create/list/update/delete, validation errors, auth where specified) plus "`npm run build` succeeds". The model never sees these; they run after the harness finishes.
- **Reference solutions** per app (`reference/`), hand-written once, that pass their acceptance tests. They prove every spec is solvable in the template, and they are the "known-good" input for harness tests.
- `run.py` runs a `StepModel` over all 10 apps and writes `results/stepbuild/<model>/<app>.json` plus `summary.md`.

**Scores:** apps fully passing acceptance (%), steps passing on first try (%), mean retries per step, failing step histogram, wall time per app.

## 4. Error handling

- Every external call (GitHub API, git clone, subprocess checks) has a timeout; failures are logged per repo/app and the run continues with the next one.
- A step that still fails after 3 retries ends that app with status `failed_at_step_N`; the trace records every attempt's output and check errors.
- Model output with no parseable `FILE` block, a forbidden path, or a file not allowed for the step counts as a failed attempt (it consumes a retry) with a specific error message fed back.
- Dataset writes are atomic; a crash mid-build loses at most the repo in progress. A rerun skips repos already mined (recorded in a manifest).

## 5. Testing

TDD throughout. CPU-only, no network in unit tests (GitHub discovery is tested against recorded JSON fixtures; mining against a tiny git repo built in `tmp_path`).

- **filters/format/split:** each filter rule has a keep case and a drop case; repo-level split never leaks a repo across splits; low-information messages dropped.
- **blocks:** parses well-formed blocks; rejects traversal, absolute paths, disallowed files, unterminated blocks.
- **harness with ScriptedModel:** (1) replaying a reference solution step by step passes every check and the acceptance tests; (2) a scripted bad file makes the right step fail, retries are consumed, and the app ends `failed_at_step_N` with the error recorded; (3) a fix on the second attempt passes and records one retry.
- **bench:** every reference solution passes its own acceptance tests (this is a gate: a benchmark app without a passing reference is not allowed).
- **Mutation checks:** remove the by-repo split → leak test fails; skip the retry loop → retry test fails; allow `..` paths → traversal test fails.

## 6. Dependencies and environment

- Python: add `pyflakes` and `pytest` (already dev) to the project; the sandbox template uses Flask + pytest.
- Node 24 / npm 11 are installed on this machine; the React template uses Vite. `npm install` runs once into a cache; per-app sandboxes reuse it.
- GitHub access through `gh` (logged in as Aneek1). Respect rate limits: back off on 403/429, cache API responses on disk.

## 7. Risks

- **Commit messages are weak instructions.** Mitigated by the low-information filter; measured by the dropped-count report. If too few examples survive, a later spec may pair commits with the nearest issue/PR title.
- **Few repos with both Flask and React and a permissive licence.** Handled by design: single-side repos are collected too (§3.1 `discover.py`), and the report shows counts per tag.
- **Windows sandbox is not a security boundary.** The benchmark only runs code from our own models on our own specs; it is not safe for untrusted code, and the spec says so.
- **`npm run build` is slow** (~10–30 s). Accepted; it runs once per frontend step, not per token.
