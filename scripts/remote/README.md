# Running the heavy work on a rented GPU box

The laptop's RTX 5060 is enough for tests and small runs; mining the step dataset
and long training runs go faster on a rented Linux box with a 4090 or 5090. This
folder sets one up. Your GitHub login (`gh`) stays on the laptop: the box only
ever clones public repos over plain https and never gets any GitHub credential.

Building the quipu-moe shards needs no GPU: see
[build_shards_box.md](build_shards_box.md) (a cheap CPU-only box, `setup.sh --cpu-only`).

## The quipu-moe session

The order of a quipu-moe GPU session (the plan's "Session runbook"). The money is a
**non-refundable $24.06 credit**; the owner's allocation: **$20 for the whole GPU
session on one ledger**, of which the A/B step may use $3, with a $1.00 reserve
inside the $20 for the chat SFT, the evals and the copy-back. The ledger is
`quipu/spend.py` (`results/spend.json` on the box): everything the GPU box costs from
the moment it starts, setup included, plus the CPU shard box as an adjustment. Every
`--budget-usd` caps that ledger's **total**, not the tool's own spend. R is the box's
$/hour from the Vast instance card. **The ledger counts box hours only: Vast also
bills bandwidth** (per GB, on many hosts; see the offer's internet price), which no
tool here sees, so pick offers with cheap or free bandwidth and compare the Vast
console balance with the ledger at the points marked below.

The boxes clone **`main`** of `https://github.com/Aneek1/quipu.git`: the owner pushes
the local branch there first (`git push origin pipeline-114m:main`, plan step 0).
There is no `pipeline-114m` on GitHub.

0. **Shards** on a cheap CPU-only box: [build_shards_box.md](build_shards_box.md),
   steps 1-5 (its preflight, `scripts/remote/preflight.sh --cpu-box`, runs before the
   build). They are copied to the GPU box in step 3; destroy the CPU box after that.
1. **Rent the GPU box** (section 1 below: RTX 5090, CUDA >= 12.8, **>= 150 GB disk,
   >= 48 GB RAM**, cheap bandwidth) and, **before anything else**, start the spend
   ledger so the setup counts:

   ```bash
   ssh -p PORT root@HOST
   git clone --branch main https://github.com/Aneek1/quipu.git /workspace/quipu
   cd /workspace/quipu
   command -v python3 || { apt-get update -qq && apt-get install -y -qq python3; }
   python3 -m quipu.spend start --usd-per-hour R --box-start EPOCH
   ```

   (`quipu.spend` is standard library only, so the system python3 runs it before
   the venv exists. EPOCH is when the box was rented (`date -d '12:34' +%s`, or the
   instance card's start time), so the clone and, on an image without python3, its
   apt install are counted too. setup.sh installs python3 as well, so every later
   `python3 -m quipu.spend` works; if python3 ever is missing, `uv run python -m
   quipu.spend ...` is the same tool.)
2. **Setup**: `bash /workspace/quipu/scripts/remote/setup.sh` (section 2; the default
   branch is `main`). **Then check the ledger against the bill**: compare the Vast
   console's credit balance (and the instance's charges so far) with
   `python3 -m quipu.spend show`. They should agree to within a few cents; if the
   console shows more spent, record the difference (`python3 -m quipu.spend adjust
   --key console-diff --usd <amount>`) before going on.
3. **Tokenizer, shards and the stepbuild data**: copy the tokenizer from the laptop
   and check its hash (as in build_shards_box.md step 2), copy the shards from the
   CPU box and verify them (build_shards_box.md step 6: `sha256sum -c SHA256SUMS`,
   shard sizes against the manifest). The chat SFT (step 7) also needs the stepbuild
   train and test splits, which are only on the laptop. From the laptop (Git Bash):

   ```bash
   ssh -p PORT root@HOST mkdir -p /workspace/quipu/data/stepbuild
   scp -P PORT data/stepbuild/train-000.jsonl data/stepbuild/test-000.jsonl \
       root@HOST:/workspace/quipu/data/stepbuild/
   sha256sum data/stepbuild/train-000.jsonl data/stepbuild/test-000.jsonl
   ssh -p PORT root@HOST "cd /workspace/quipu && sha256sum data/stepbuild/*-000.jsonl"
   ```

   The two lists of hashes must be the same (train 212b09dc..., test 662e20c9... for
   the build of 2026-09-28). **Destroy the CPU box, then record what it cost** (its
   total from the Vast console's billing page, bandwidth included) on the same ledger:

   ```bash
   python3 -m quipu.spend adjust --key cpu-box --usd <amount>
   python3 -m quipu.spend show
   ```

   (The same key again replaces the amount, never adds it twice.)
3b. **Preflight the GPU box** (~30 minutes, ~$0.30 at $0.55/h), inside `tmux`:

   ```bash
   bash scripts/remote/preflight.sh --gpu-box --usd-per-hour R
   ```

   It checks the checkout (main, at origin), python3, disk, the tokenizer, that the
   spend ticker really ticks (on a scratch ledger), the POSIX-only tests the laptop
   never ran, the shards' sha256 and the manifest's tokenizer hash, the stepbuild
   files, Triton on sm_120 and the compile-parity tests, then runs 20-step probes of
   the real trainer (full model at micro_batch 8 and 4, compile on and off; the A/B
   model with AttnRes off and on, eager) and writes `results/preflight/probes.json`.
   It prints two decisions and a command; **settle both before step 4**:

   - **micro_batch** (configs/quipu-moe.toml says 4, grad_accum 64): 8 only if its
     probe fits (peak VRAM under 95% of the card) and is faster. If the probe says 8,
     every later command (run_moe.py, the SFT) gets `--override train.micro_batch=8`.
     **It must be settled before the gate**: once `results/moe/plan.json` exists, a
     different micro_batch (or compile setting) is a different run config and
     run_moe.py refuses it (exit 2).
   - **compile** for the full run (the A/B arms are always eager,
     configs/quipu-moe-ab.toml): keep `compile = true` only if the probe shows **at
     least 10% more tokens/s than eager AND no recompile_limit hit or recompile storm**
     in its log (the trainer logs the outcome and dynamo's counters: the run log's
     "compile" / "compile_stats", and "compile: ..." / "compile stats at step N: ..."
     lines). Otherwise pass `--override train.compile=false` to run_moe.py and the SFT.
     Why this can go either way: the loop dispatch breaks the graph at bincount /
     .tolist(), and each layer's frame recompiles on new expert sizes until dynamo's
     recompile_limit, after which those layers run eagerly anyway.
   - the **A/B plan at the measured rate**: the printed `ab_runs.py --dry-run
     --tokens-per-second <ab-c0's tokens/s>` line. Show it to the owner: if the plan
     does not fit $3, the levers are `--skip-sweeps` (the config's lr 6e-4 / muon_lr
     0.02, the usual literature values, instead of the two LR sweeps),
     `--shared-noise` (one seed re-run instead of three) and smaller `--arm-tokens` /
     `--sweep-tokens`; the owner chooses.

   A FAIL line says what to fix; fix it and run it again (`--probes NAME,...` repeats
   single probes, `--skip-probes` the rest only).
4. **A/B runs** (~3.5-4 h at the default plan; FP8 stays off, the owner's decision,
   so no `--with-fp8`), inside `tmux`, with the A/B budget of spec section 8 and the
   levers the owner chose in 3b:

   ```bash
   AB=$(python3 -m quipu.spend show --plus 3) || exit 1
   uv run python scripts/ab_runs.py --config configs/quipu-moe-ab.toml --out results/ab \
       --budget-usd "$AB" --usd-per-hour R
   ```

   `--budget-usd` caps the ledger total, so the A/B budget is "spent so far + $3": the
   A/B itself gets $3 whatever setup and the CPU box (steps 1-3) cost. `show --plus 3`
   prints spent + 3 and **fails closed**: if the ledger is missing, has no box
   session or is corrupt it prints nothing and exits non-zero, and `|| exit 1` stops
   there (it closes the shell or tmux window: go back to step 1) instead of handing
   the A/B a budget computed from $0. Run `--dry-run` first to see whether the run
   list fits (`--shared-noise` runs one seed re-run instead of three; off unless the
   owner chooses it). Every arm decides on the **mix-weighted validation loss** (code
   0.60 on code_val, English 0.28 on val, the nine languages 0.12 on val_lang/* by
   their text weights; the trainer logs each split's loss); summary.md says which
   metric it used, prints the per-split losses and each run's compile outcome. It
   writes `results/ab/summary.md` and `results/ab/winners.toml`; a budget stop (exit
   4) still writes both, with the missing arms keeping the simpler option.
5. **The full run**, inside `tmux` (plus the `--override` flags preflight asked for):

   ```bash
   uv run python scripts/remote/run_moe.py --config configs/quipu-moe.toml \
       --winners results/ab/winners.toml --budget-usd 20 --reserve-usd 1.00 --usd-per-hour R
   ```

   A 15-minute throughput gate (`--gate-minutes`) measures tokens/s on the real
   model, then `results/moe/plan.md` shows the tokens, hours and cost that fit the
   budget (spend so far + the run + the `--reserve-usd` 1.00 reserve for the chat SFT,
   the evals and the copy-back), and what was trimmed. The plan is costed at 95% of
   the measured rate (`--headroom 0.05`) plus each interval's eval (the mix-weighted
   evaluation, ~90 s) and checkpoint time, which a 15-minute gate barely sees; after
   the long run's first full interval the rate is checked once more and the run
   trimmed (never extended) if it is more than 2% slower. **Before the GO, check the
   Vast console balance against `python3 -m quipu.spend show` again** (bandwidth is
   not in the ledger: record a difference with `spend adjust --key console-diff`,
   and run the same command again so the plan re-fits). Show plan.md to the owner;
   on approval `touch results/moe/GO`. Waiting costs the box's rate (the launcher
   prints the per-minute cost while it waits). **The wait is bounded**: after
   `--go-max-wait-min` (90) minutes, or as soon as less than 80% of the approved
   tokens would still fit, the launcher writes plan.md and summary.md and exits 5. **If
   the owner cannot answer within that window**, **stop** (not destroy) the instance
   from the Vast console (the disk is kept; a stopped instance bills only a little
   for it): first `python3 -m quipu.spend stop` (so the stopped time is not
   counted), later start the instance, `python3 -m quipu.spend start --usd-per-hour
   R`, read plan.md again (an 80% stop re-fits it) and run the same command; it
   reuses `results/moe/plan.json` (no second gate). Restart `sync.sh` on the laptop
   too: it stops when it sees summary.md, which exit 5 writes.

   The long run then resumes from the gate's checkpoint (the gate's 15 minutes of
   training are kept), retries crashes from the last checkpoint, and stops cleanly
   (interrupt checkpoint, exit 4) if spend + the next eval interval + the reserve
   would pass the budget. **A crash in the gate is not retried**: the launcher exits
   (summary.md says why); fix the cause and run the same command again. The trainer
   runs in its own session: if the launcher is stopped or its terminal hangs up
   (SIGTERM/SIGHUP take the Ctrl+C path: the trainer checkpoints, exit 130). Should
   the launcher die outright (SIGKILL, OOM), the trainer's next output line hits the
   dead pipe: it checkpoints and exits 130 on its own (an orphan stop), appending
   the rest of its output to the attempt's log in `results/moe/logs/` (the launcher
   writes every line there as it arrives, so nothing before the death is lost
   either). Behind that, every trainer carries its own backstop (`train.budget_usd`
   = the budget left at its launch less the reserve, plus half the stop grace so the
   live launcher's guard always stops it first): it checkpoints and exits 4, and a
   signal arriving during that checkpoint is ignored. `python -m quipu.spend stop`
   refuses while the launcher or ab_runs is ticking the ledger (within the last 2
   minutes; `--force` overrides, and the launcher treats a session ended under it
   as a budget stop). Both release the ledger when they exit, so `spend stop` right
   after one has finished works without `--force`; only a tool that was SIGKILLed
   blocks it for the 2 minutes. At the end it runs milestone_eval and writes
   `results/moe/summary.md`, and plan.json records the run as completed, so the same
   command again does nothing. Run it again after any other stop: it reuses
   `results/moe/plan.json` and resumes (a larger `--budget-usd` continues a run the
   budget stopped). The trainer refuses shards whose manifest names another
   tokenizer than the config's (exit 2).
6. **On the laptop, during the long run** (Git Bash, from the repo):

   ```bash
   bash scripts/remote/sync.sh HOST PORT 3        # --dry-run prints the commands
   ```

   Every 3 hours it copies `checkpoints/quipu-moe/milestones/` (each ~2 GB snapshot
   once) and `results/` (no inductor caches) into `remote-runs/quipu-moe/`; the full
   `checkpoints/quipu-moe/latest.pt` and the checkpoint it points to (~12 GB: weights
   plus optimizer states) only on the first cycle, every 4th after it
   (`--ckpt-every N`) and the final one, which is the cycle that sees
   `results/moe/summary.md` on the box (then it stops). **Bandwidth**: each full
   checkpoint is ~12 GB out of the box, billed per GB on many Vast hosts and never in
   the ledger; `--ckpt-every 1` copies it every cycle, a larger N less often. It uses
   `~/.ssh/id_ed25519` (`SSH_KEY`), never agent forwarding, and `BatchMode=yes` (no
   password prompt: a key problem fails the cycle). Old `step_*.pt` copies stay
   unless you set `KEEP_LOCAL=N` (keep the N newest; the one `latest.pt` points to
   and `milestones/` are never deleted); `KEEP_LOCAL=N bash scripts/remote/sync.sh
   --prune-only` prunes once. rsync when installed (an interrupted file waits in
   `.rsync-partial/` and resumes), else scp (Git Bash).
7. **After pretraining**: its evaluation, then the chat SFT (M12) in the same session
   (paid from the reserve), then its evaluation. The post-run evaluation on the box:

   - milestone_eval (bits per byte per language, samples): run_moe.py runs it at
     the end (`results/moe/milestones/`);
   - the needle eval: `uv run python scripts/needle_eval.py --config
     results/moe/run_config.toml --ckpt latest --quick --device cuda` (a few
     minutes; `results/needle/quick`);
   - expert usage: `uv run python scripts/expert_usage.py --config
     results/moe/run_config.toml --device cuda` (`results/experts`);
   - HumanEval / MBPP (`scripts/code_eval.py`) and the Hugging Face export run on
     the laptop after the copy-back (no paid time; the tools memory-map the
     checkpoint, so the ~12 GB file needs no 12 GB of RAM). The export can also run
     on the box, inside the reserve, if the laptop is short of disk:
     `scripts/export_hf.py --config results/moe/run_config.toml --ckpt-dir
     checkpoints/quipu-moe --out hf_export_moe --int4`.

   **Before the SFT**: prune the pretraining checkpoints to the final one and the
   milestones (the SFT writes ~12 GB checkpoints to the same disk), and **check the
   Vast console balance against the ledger once more**:

   ```bash
   cd /workspace/quipu/checkpoints/quipu-moe && keep=$(grep -ao 'step_[0-9]*\.pt' latest.pt | head -n 1) \
       && [ -n "$keep" ] && ls step_*.pt | grep -vx "$keep" | xargs -r rm -f -- ; ls; cd /workspace/quipu
   ```

   (`milestones/` is a directory and `latest.pt` does not match `step_*.pt`: both
   stay.) configs/quipu-moe-sft.toml keeps one SFT checkpoint (`ckpt_keep = 1`).

   **The SFT runs under the same ledger guard**: launch it the way run_moe launches
   the trainer (spend guard + its own session), or at the least with the trainer's
   backstop set to what is left of the $20, less a copy-back margin (here $0.10),
   computed so that it fails closed. LEFT is computed **immediately before the
   trainer command**, after the data build: the build's ~10-15 minutes are box time
   too, and a LEFT taken before it would let the SFT spend them a second time.

   ```bash
   # the chat data (Aya + oasst2 + stepbuild train split; ~10-15 min of CPU, pinned
   # revisions from the config; writes data/chat-sft/ and its manifest.json):
   uv run python scripts/build_chat_data.py --config configs/quipu-moe-sft.toml || exit 1
   # what is left NOW, right before the SFT starts:
   SPENT=$(python3 -m quipu.spend show --usd-only) || exit 1
   LEFT=$(python3 -c "import sys; left = round(20 - $SPENT - 0.10, 4); \
   print(left) if left > 0 else sys.exit('nothing left of the budget')") || exit 1
   # the SFT, from the final pretraining checkpoint (fresh optimizer, step 0):
   uv run python -m quipu.train --config configs/quipu-moe-sft.toml --run-id quipu-moe-sft \
       --init-from checkpoints/quipu-moe --override train.budget_usd=$LEFT \
       --override train.usd_per_hour=R
   # then the chat card's samples: two fixed prompts per language, temperature 0,
   # the chat template, from the SFT checkpoint (+ its held-out loss per source):
   uv run python scripts/chat_eval.py --config configs/quipu-moe-sft.toml --device cuda
   ```

   The trainer's backstop clock starts at zero in every process: it counts only its
   own wall time, not what earlier attempts spent. So before any `--resume` (after a
   crash, a stop or an interrupt) run the SPENT and LEFT lines again and pass the new
   LEFT; reusing the first one would grant the whole amount a second time. The
   ledger's ticker has counted every attempt's box time, so the new LEFT is what is
   really left.

   `configs/quipu-moe-sft.toml` inherits `quipu-moe.toml` with `results/ab/winners.toml`
   merged on top (so the model, optimizer and precision are the pretrained run's; the
   LR is 0.1x its peak) and stops after 3 epochs of the chat data or 100M tokens,
   whichever is first. It also inherits micro_batch (4) and compile (true): **pass
   the same `--override train.micro_batch=N` / `train.compile=false` as the long run
   if preflight asked for them**. A crash or a stop: recompute LEFT (above), then the
   same command with that LEFT plus `--resume` continues the SFT's own checkpoint
   (`--init-from` is then ignored). A completed SFT writes its held-out chat loss per
   source (`data/chat-sft/val_by_source/*`, micro-batch 1) to
   `results/sft/val_by_source.json`, and chat_eval.py writes
   `results/sft/chat_samples.json` (the chat card shows only these, never the base
   model's samples). Its config has `budget_usd = 0.5` so a launch without the
   override is still bounded; never with `train.budget_usd 0` (no backstop at all).
   Try it: `uv run python scripts/chat.py --config configs/quipu-moe-sft.toml
   --temperature 0`.
8. **Copy everything back** from the laptop (Git Bash, from the repo), after the last
   sync.sh cycle:

   ```bash
   bash scripts/remote/copy_back.sh HOST PORT          # --dry-run prints the commands
   ```

   It copies exactly: the final pretraining checkpoint (`checkpoints/quipu-moe/
   latest.pt` + the step it names), its `milestones/`, the final SFT checkpoint
   (`checkpoints/quipu-moe-sft/`, same way, and its milestone), `results/` (with
   `results/moe/run_config.toml`, `results/ab/`, `results/sft/`, `results/preflight/`,
   `results/spend.json`), `data/shards-moe/{manifest.json,val,code_val,val_lang}` and
   `data/chat-sft/manifest.json` + `val_by_source/`, into `remote-runs/quipu-moe/`,
   then checks every file's sha256 against the box. It never deletes a local file,
   skips what already matches (a re-run resumes), and exits 1 if something is
   missing or does not verify: then **do not destroy the box**. Check on the laptop
   that the checkpoints load (e.g. `uv run python scripts/chat.py --config
   configs/quipu-moe-sft.toml --checkpoint remote-runs/quipu-moe/checkpoints/quipu-moe-sft/step_NNNNNN.pt
   --temperature 0`). Then `python3 -m quipu.spend stop` on the box and **stop (not
   destroy) the instance** in the Vast console while you wait for the owner's
   go-ahead (a stopped instance bills only its disk); destroy it (section 5) only
   once the owner has said so.

## 1. Rent a box (Vast.ai)

1. In the Vast console, **Account → Keys**, add your SSH **public** key
   (`~/.ssh/id_ed25519.pub`). Do it there, in the console; never paste keys
   (public or private) into a chat or a script.
2. **Search**: pick an RTX 4090 or RTX 5090 offer (quipu-moe: a 5090, 32 GB), and
   filter on **CUDA >= 12.8** (the project pins the cu128 torch wheels, which need
   NVIDIA driver 570 or newer; a 5090 needs cu128 anyway). For the quipu-moe session:
   **disk >= 150 GB** (shards ~20 GB, env ~8 GB, up to four ~12 GB training
   checkpoints at once plus ~2 GB milestones, the A/B's and the SFT's checkpoints)
   and **RAM >= 48 GB** (loading a 12 GB checkpoint for the SFT, the evals and the
   export). **Filter and sort by the bandwidth price too** (the offer's internet
   $/TB up and down): the checkpoints and shards cross the network, Vast bills that
   traffic per GB on many hosts, and the spend ledger never sees it.
3. Template: an Ubuntu 22.04/24.04 image with CUDA 12.8+ (for example the
   `pytorch` or `nvidia/cuda` templates; the setup installs its own torch, so the
   image's Python does not matter). Launch it with SSH access.
4. The instance card shows the SSH command, `ssh -p PORT root@HOST`.

## 2. Connect and set up

```bash
ssh -p PORT root@HOST
bash <(curl -fsSL https://raw.githubusercontent.com/Aneek1/quipu/main/scripts/remote/setup.sh)
```

or clone first and run it from the checkout:

```bash
git clone https://github.com/Aneek1/quipu.git /workspace/quipu
bash /workspace/quipu/scripts/remote/setup.sh
```

Options (flags or environment variables): `--repo-url` / `REPO_URL` (default
`https://github.com/Aneek1/quipu.git`), `--branch` / `BRANCH` (default `main`),
`--workdir` / `WORKDIR` (default `/workspace/quipu`), `--skip-gpu-bench` /
`SKIP_GPU_BENCH=1`. To try another branch that is on GitHub:
`bash setup.sh --branch NAME` (the quipu-moe session uses `main`).

The script is safe to re-run (it also updates the checkout with
`git pull --ff-only`). It:

- checks `nvidia-smi` and stops with a clear error if the driver is older than
  570 (too old for CUDA 12.8): destroy that instance and pick another offer;
- installs git, curl, ca-certificates and build-essential if missing, Node 24
  from NodeSource if `node` is missing or older than 22, and uv;
- clones or updates the repo and runs `uv sync --frozen` (Python 3.13 plus the
  locked torch cu128 wheel for Linux x86_64);
- prints the torch version, `torch.cuda.is_available()`, the device and runs one
  bf16 matmul on it;
- runs `tests/test_model_shapes.py` and `tests/stepbuild/test_blocks.py`, then
  `scripts/check_gpu.py` (about a minute and a half of sustained matmul; skip it
  with `--skip-gpu-bench`), then the npm-marked sandbox test once so the
  stepbuild npm cache (`~/.cache/quipu/stepbuild-npm-cache`) is warm.

Long jobs: run them inside `tmux` (`apt-get install -y tmux`, then `tmux new -s
work`; reattach with `tmux attach -t work`) so a dropped SSH connection does not
kill them.

## 3. The dataset in two phases

Discovery needs the GitHub API (code search, the full-stack checks, licences), so
it runs on the laptop, where `gh` is logged in. Mining needs only public https
clones and CPU, so it runs on the box.

On the laptop (reuses and fills the cache in `data/stepbuild/cache/`):

```bash
uv run python -m stepbuild.dataset.build --phase discover --limit 300 --out data/stepbuild
scp -P PORT data/stepbuild/candidates.json root@HOST:/workspace/candidates.json
```

On the box:

```bash
cd /workspace/quipu
uv run python -m stepbuild.dataset.build --phase mine \
    --candidates /workspace/candidates.json --out data/stepbuild
```

The mine phase makes no `gh` call. It writes the same shards, `SOURCES.jsonl`,
`manifest.json` and `report.md` as the one-command build (`--limit` without
`--phase` still does both in one go, as before). A rerun skips repos already
mined. Each repo keeps at most 50 examples after dedupe, spread over its history
(`--max-per-repo N` changes it, `0` turns the cap off); the cap is recomputed
from the mined data on every run, so changing it needs no re-mining. A candidate whose licence lookup failed on the laptop is reported as
failed; rerun discover on the laptop to retry it.

## 4. Bring results back

From the laptop (not the box: the box has no route to your laptop):

```bash
rsync -avz -e "ssh -p PORT" \
    --exclude repos/ --exclude cache/ \
    root@HOST:/workspace/quipu/data/stepbuild/ data/stepbuild-remote/
```

(`repos/` holds the bare clones and is large; leave it behind unless you want to
mine again elsewhere.) Checkpoints and results the same way, for example
`rsync -avz -e "ssh -p PORT" root@HOST:/workspace/quipu/checkpoints/ checkpoints/`.
Without rsync on Windows, `scp -P PORT -r root@HOST:/path local/` does the same.
For the quipu-moe run, `sync.sh` does this on a schedule and `copy_back.sh` copies the
final set once, with sha256 checks (see "The quipu-moe session", steps 6 and 8).

## 5. Stop the instance when done

You pay per hour while it runs. When the results are on the laptop, **stop** the
instance in the Vast console (stopped instances still bill a little for disk) or
**destroy** it (everything on it is gone). Do not leave a box idling overnight. For
the quipu-moe session: `python3 -m quipu.spend stop` first, then stop (not destroy)
while waiting for the owner's go-ahead; destroy only once copy_back.sh verified
everything and the owner said so.
