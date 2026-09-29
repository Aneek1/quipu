# Running the heavy work on a rented GPU box

The laptop's RTX 5060 is enough for tests and small runs; mining the step dataset
and long training runs go faster on a rented Linux box with a 4090 or 5090. This
folder sets one up. Your GitHub login (`gh`) stays on the laptop: the box only
ever clones public repos over plain https and never gets any GitHub credential.

Building the quipu-moe shards needs no GPU: see
[build_shards_box.md](build_shards_box.md) (a cheap CPU-only box, `setup.sh --cpu-only`).

## The quipu-moe session

The order of a quipu-moe GPU session (the plan's "Session runbook"). The budget
(`--budget-usd 20`) is the box ledger's, `quipu/spend.py` (`results/spend.json` on the
box): everything the GPU box costs from the moment it starts, setup included. R is
the box's $/hour from the Vast instance card.

0. **Shards** on a cheap CPU-only box: [build_shards_box.md](build_shards_box.md),
   steps 1-5. They are copied to the GPU box in step 3; destroy the CPU box after that.
1. **Rent the GPU box** (section 1 below: RTX 5090, CUDA >= 12.8, ~100 GB disk) and,
   **before anything else**, start the spend ledger so the setup counts:

   ```bash
   ssh -p PORT root@HOST
   git clone --branch pipeline-114m https://github.com/Aneek1/quipu.git /workspace/quipu
   cd /workspace/quipu && python3 -m quipu.spend start --usd-per-hour R
   ```

   (`quipu.spend` is standard library only, so the system python3 runs it before
   the venv exists; add `--box-start EPOCH` to count from when the box was rented.)
2. **Setup**: `bash /workspace/quipu/scripts/remote/setup.sh --branch pipeline-114m`
   (section 2).
3. **Tokenizer and shards**: copy the tokenizer from the laptop and check its hash
   (as in build_shards_box.md step 2), copy the shards from the CPU box and verify
   them (build_shards_box.md step 6: `sha256sum -c SHA256SUMS`, shard sizes against
   the manifest).
4. **A/B runs** (~3.5-4 h; FP8 stays off, the owner's decision, so no `--with-fp8`),
   inside `tmux`:

   ```bash
   uv run python scripts/ab_runs.py --config configs/quipu-moe-ab.toml --out results/ab \
       --budget-usd 20 --usd-per-hour R
   ```

   It writes `results/ab/summary.md` and `results/ab/winners.toml`.
5. **The full run**, inside `tmux`:

   ```bash
   uv run python scripts/remote/run_moe.py --config configs/quipu-moe.toml \
       --winners results/ab/winners.toml --budget-usd 20 --usd-per-hour R
   ```

   A 15-minute throughput gate (`--gate-minutes`) measures tokens/s on the real
   model, then `results/moe/plan.md` shows the tokens, hours and cost that fit the
   budget (spend so far + the run + a `--reserve-usd` 0.50 reserve for the chat SFT
   and final eval/export), and what was trimmed. Show it to the owner; on approval
   `touch results/moe/GO`. Waiting costs the box's rate (the launcher prints the
   per-minute cost while it waits). The long run then resumes from the gate's
   checkpoint (the gate's 15 minutes of training are kept), retries crashes from the
   last checkpoint, and stops cleanly (interrupt checkpoint, exit 4) if spend + the
   next eval interval + the reserve would pass the budget. At the end it runs
   milestone_eval and writes `results/moe/summary.md`. Run it again after any stop:
   it reuses `results/moe/plan.json` and resumes (a larger `--budget-usd` continues a
   run the budget stopped).
6. **On the laptop, during the long run** (Git Bash, from the repo):

   ```bash
   bash scripts/remote/sync.sh HOST PORT 3        # --dry-run prints the commands
   ```

   Every 3 hours it copies `checkpoints/quipu-moe/latest.pt` and the checkpoint it
   points to, `checkpoints/quipu-moe/milestones/` and `results/` (no inductor caches)
   into `remote-runs/quipu-moe/`, and stops after the cycle that sees
   `results/moe/summary.md` on the box. It uses `~/.ssh/id_ed25519` (`SSH_KEY`), never
   agent forwarding, and never deletes a local file: old `step_*.pt` copies (~12 GB
   each) stay until you remove them. rsync when installed, else scp (Git Bash).
7. **After pretraining**: its evaluation, then the chat SFT (M12) in the same session
   (paid from the reserve), then its evaluation.
8. **Copy everything back** (sync.sh's final cycle, plus the SFT outputs), verify on
   the laptop (the checkpoints load; sha256 against the box), get the owner's
   go-ahead, then destroy the box (section 5).

## 1. Rent a box (Vast.ai)

1. In the Vast console, **Account → Keys**, add your SSH **public** key
   (`~/.ssh/id_ed25519.pub`). Do it there, in the console; never paste keys
   (public or private) into a chat or a script.
2. **Search**: pick an RTX 4090 or RTX 5090 offer, and filter on **CUDA >= 12.8**
   (the project pins the cu128 torch wheels, which need NVIDIA driver 570 or
   newer; a 5090 needs cu128 anyway). 50+ GB of disk is comfortable.
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
`SKIP_GPU_BENCH=1`. To try a branch before it is merged:
`bash setup.sh --branch pipeline-114m`.

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
For the quipu-moe run, `sync.sh` does this on a schedule (see "The quipu-moe session").

## 5. Stop the instance when done

You pay per hour while it runs. When the results are on the laptop, **stop** the
instance in the Vast console (stopped instances still bill a little for disk) or
**destroy** it (everything on it is gone). Do not leave a box idling overnight.
