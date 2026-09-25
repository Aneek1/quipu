# Quipu Weekend Run Implementation Plan

> **For agentic workers:** executed with superpowers:subagent-driven-development — one implementer per task, spec review then code-quality review, fixes re-reviewed.

**Goal:** Everything the 3B-token weekend run needs, built and pre-flighted before the owner starts it at ~16:00 on 2026-09-25; approach A built during the run.

**Spec:** `docs/superpowers/specs/2026-09-25-quipu-weekend-run-design.md`

**Written at requirement level, not full code.** The owner's 16:00 start leaves ~7 hours for build, review, a ~30-minute data rebuild and a pre-flight. The implementers on this project have handled requirement-level tasks well (Tasks 10–13); every task still carries its tests and mutation checks.

**Standing rules for every task**
- Branch `pipeline-114m`. Tests: `CUDA_VISIBLE_DEVICES= python -m uv run python -m pytest -m "not gpu_gate" -q` while the owner is working or while the GPU is training. The normal full run is allowed only when the GPU is idle and the owner has not asked for quiet.
- Never pip-install globally; add dependencies with `python -m uv add`, and commit `uv.lock`.
- Undo mutations from a scratch backup, never `git checkout -- <file>`.
- Commits: plain sentences, no trailers, never pushed. Never commit `data/`, `results/` or `checkpoints/`.
- One implementer at a time on this branch.

## Critical path — must be done before the run starts

### W1 — Memory-mapped loader  *(dispatched)*
`quipu/loader.py`: `np.memmap` per shard, logical concatenation, batches that cross shard boundaries assembled from slices. Behaviour and API unchanged; all existing loader tests pass unmodified. New tests: RSS stays flat over ~200 MB of shards; a batch spanning three shards. Mutation: first-shard-only reads → boundary tests fail.

### W2 — Code-mix data builder + config, then rebuild
- **Config** (`configs/quipu-114m.toml`, `quipu/config.py`):
  - `total_tokens = 3_000_000_000` (5,722 steps).
  - New data fields: `code_dataset = "codeparrot/github-code-clean"`, `code_share = 0.2`, `code_languages`, `code_licenses`, `html_cap = 0.1`, `code_val_tokens = 5_000_000`, `code_heldout_files` (last 40 of 880).
  - Validation and tests for the new fields: share in (0,1), non-empty lists, cap in (0,1].
- **Builder** (`scripts/build_shards.py`):
  - Read code parquet directly with `HfFileSystem` + pyarrow, columns `code, language, license, repo_name, path`. Don't use the dataset's loading script.
  - Keep only the configured languages and licences, with HTML skipped once it reaches `html_cap` of the code tokens written so far.
  - Deterministic interleave with FineWeb-Edu: before each document, draw from whichever source is furthest below its target share.
  - The FineWeb val is taken first, as today.
  - Code val comes from the held-out files, built **after** code train. It is deduplicated by exact content hash against every code-train document, and the number dropped is recorded.
  - The manifest records tokens and documents per source and per language, the filters, the held-out range, target vs achieved shares, and the dedup count.
  - Output: `data/shards/{train,val,code_val}`.
- **Tests** (fake in-memory sources, no network):
  - the achieved share lands within ±0.5% of target over a long build;
  - the HTML cap holds;
  - the licence and language filters drop what they should;
  - code val never contains a train hash;
  - the interleave is deterministic (identical output across two builds);
  - all existing builder tests still pass.
- **Mutations:** remove the dedup → the dedup test fails; remove the HTML cap → its test fails; use a random interleave → the determinism test fails.
- **Rebuild:** run the real build at **BELOW_NORMAL process priority** (the owner is in meetings). Verify exact totals: train 3,000,000,000; val 10,000,000; code_val ≤ 5,000,000 after dedup (report the number). Report achieved per-language shares.

### W3 — Trainer: milestones, exit codes, keep-awake
- `milestones = [100, 250, 500, 1000, 2000, 4000]` in the train config, plus the final step. At those steps the trainer saves **bf16 model state only**, atomically via `replace_with_retry`, to `<ckpt_dir>/milestones/step_NNNNNN.pt`.
  - These files are never pruned and never used for resume.
  - A resumed run does not rewrite milestones that already exist.
- Exit codes from `main()`: **3** for the non-finite stop, **130** for KeyboardInterrupt, **1** for any other crash, **0** on success.
- Keep-awake on Windows: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` for the duration of `run()`, always cleared in `finally`. It does nothing on other platforms.
- **Tests:**
  - milestones are written at the right steps, in bf16, and survive `ckpt_keep` pruning;
  - a resume doesn't duplicate or rewrite them;
  - each exit code is correct, checked via a subprocess or by calling `main` with a monkeypatched trainer;
  - the keep-awake call is made and cleared (monkeypatched ctypes).
- **Mutations:** prune milestones too → the survival test fails; drop the `finally` clear → the keep-awake test fails.

### W4 — Launcher `scripts/weekend.py`
- **Start guards:**
  - other processes' GPU memory > 1.5 GB (nvidia-smi `--query-compute-apps` and `--query-gpu=memory.used` minus our own);
  - running on battery (`GetSystemPowerStatus`);
  - less than 40 GB free.
  - Each prints the exact fix. `--force` overrides.
- **Training:** runs `python -m quipu.train` as a child process, adding `--resume` whenever a run log for the run id already exists.
- **Auto-resume:**
  - a failure exit other than 3 or 130 is retried up to 3 times with a 2-minute pause;
  - every attempt, its exit code and its time go into `results/weekend_summary.md`.
- **After a completed run:** `milestone_eval.py`, then `needle_eval.py` if present, then `results_table.py --out RESULTS.md`. Each is a separate process, and a failure is logged and the next step still runs.
- **Tests:** a fake child process with scripted exit codes covers retry on 1, no retry on 3 or 130, the retry cap, and the `--resume` flag on retries. The guards are tested with monkeypatched probes.
- **Mutation:** retry on exit 3 → its test fails.

### W5 — `scripts/milestone_eval.py`
- For every milestone plus the final checkpoint: text val loss and code val loss over fixed batches, and generations from fixed prompts.
  - Text openings: 3 prompts.
  - Code openings: Python def, JS function, HTML page, SQL query.
  - Decoding: greedy, plus one seeded sampled continuation.
- Writes `results/milestones/metrics.json` and `samples.md`. Read-only on the checkpoints.
- **Tests:** a tiny model with fake milestones gives the expected files and structure, and the checkpoints are untouched (mtime/hash).

### W6 — Pre-flight on the new data
- Extend `scripts/preflight.py` to assert that a milestone is written during its short run, and that the launcher guards pass or report.
- Run it once on the GPU **only after the owner confirms they're between meetings or finished**, since it takes about 4 minutes of full GPU. Every check must PASS before the owner starts the run.

## During the run (CPU only, GPU hidden)

### W7 — Approach A: `quipu/memory/` + `scripts/needle_eval.py`
Per spec §5. The e5-small embedder runs on the CPU during development; the real evaluation runs on the GPU after training, launched by the launcher. Tests use tiny models and fake haystacks.

## Owner checklist before starting at ~16:00
1. Close Teams, Edge, WhatsApp, Office and other GPU-heavy apps. The launcher refuses to start if other apps hold more than 1.5 GB.
2. Plug in to mains power.
3. Pause Windows Update for a week (Settings → Windows Update → Pause).
4. Run `python -m uv run python scripts/weekend.py`.
5. Leave the lid open, or set lid-close to "do nothing" on AC.
