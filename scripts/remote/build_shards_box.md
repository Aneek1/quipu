# Building the quipu-moe shards on a cheap CPU box

The data v2 shards (`configs/quipu-moe.toml`, `scripts/build_shards.py`) need no GPU:
the work is downloading ~100-200 GB of parquet and tokenising it. Build them on a
cheap CPU-only Vast box, copy them to the GPU box, and destroy the CPU box. The build
runs unattended for hours; it resumes after a crash or preemption and stops early
(with a clear message) when the mix cannot be met, so box time is not wasted.

## 1. Rent the box

In the Vast console, search **CPU-only** offers (no GPU) with:

| what | minimum |
|---|---|
| CPU cores | 8 or more (the build uses at most 8 tokenising processes) |
| RAM | 32 GB |
| disk | 60 GB (80 GB leaves margin: ~11 GB code buckets, ~7 GB text, ~17 GB train shards, ~8 GB Python env, <= 7 downloaded files of ~0.36 GB) |
| inet_down | as fast as the price allows (>= 1 Gbps): the download dominates |
| reliability | >= 0.98 |

Ubuntu 22.04/24.04 image, SSH access. Sort by price; this box does nothing a GPU helps with.

## 2. Set it up

```bash
ssh -p PORT root@HOST
git clone --branch pipeline-114m https://github.com/Aneek1/quipu.git /workspace/quipu
bash /workspace/quipu/scripts/remote/setup.sh --cpu-only --branch pipeline-114m
```

`--cpu-only` skips every GPU, driver, CUDA and Node step, checks that fastText (the
LID filter) imports, and runs the shard builder's tests. Safe to re-run.

The trained tokenizer is not in git. Make its directory on the box, then copy it from
the laptop:

```bash
ssh -p PORT root@HOST mkdir -p /workspace/quipu/artifacts/tokenizer
scp -P PORT artifacts/tokenizer/tokenizer.json root@HOST:/workspace/quipu/artifacts/tokenizer/
```

and on the box check it is the one the gate passed
(`artifacts/tokenizer/sample_manifest.json` on the laptop has the same hash):

```bash
cd /workspace/quipu && sha256sum artifacts/tokenizer/tokenizer.json
# 9f3e33d074f3a0d7da234eec11a59de1e20aa756855d1ef552fc9706b9006943
```

## 3. Hugging Face token

Type it yourself, in your own SSH session, so it is in no file, no shell history and
no chat:

```bash
read -rs HF_TOKEN && export HF_TOKEN     # paste, Enter (nothing is echoed)
```

Everything started from this shell (including `nohup` below) inherits it. After a
reconnect, set it again before resuming.

In the same shell, turn off huggingface_hub's xet chunk cache: the build reads each
file once, so the cache would only fill the disk (nothing trims it as files are
deleted). Set it again after a reconnect, as the token:

```bash
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
```

## 4. Preflight (minutes, metadata only)

```bash
cd /workspace/quipu
uv run python scripts/build_shards.py --config configs/quipu-moe.toml \
    --train-tokens 8.3e9 --preflight
```

It reads only the language / licence / size / path columns of the 840 code train
files (~0.6 GB in all) and prints, per language: weight, quota (tokens), tokens
available in all 840 files, how many files it takes to fill the quota, and the share
the files can give; then how many files the build will read (and roughly how many GB
that is to download), which languages set that number, and a verdict. Exit 0 = OK,
exit 3 = a language of weight >= 5% would end more than 2 pp off target: do not start
the build; decide the weights first. The report is also saved to
`data/shards-moe/preflight.json`. Bytes become tokens at 3.79 characters per token
(`--chars-per-token`); treat the numbers as estimates.

## 5. The build (hours; resumable)

```bash
cd /workspace/quipu
nohup uv run python scripts/build_shards.py --config configs/quipu-moe.toml \
    --lid-filter --train-tokens 8.3e9 >> build.log 2>&1 &
tail -f build.log          # Ctrl-C stops the tail, not the build
```

Defaults: `--workers` = cores - 1, at most 8; `--download-workers 6` (whole code
files are downloaded ahead into `data/shards-moe/_work/dl`, each deleted once read);
`--lid-threshold 0.0` (any language-ID mismatch is dropped).

What it does, in order: loads the tokenizer (its vocabulary must be 49,152) and the
LID model (so a broken install fails in seconds), the English val split, code (a
checkpoint after every code file; from the 20th file on it projects the code mix at
the file cap and stops with an explanation if the projection misses at 5 file ends in
a row, with 1 pp more room before file 60; the code shares are checked again the
moment code is collected, before any text is read), English and the nine other
languages (a checkpoint after every text file), then train, code_val, val_lang and
`manifest.json`. On success `_work` is deleted. A download or read error that cannot
succeed on retry (404, no access, a full disk, a corrupt file) stops the build at
once (exit 1); network blips, 429 and 5xx are retried with backoff (12 attempts,
about 30 minutes).

**If the box is preempted or the build dies:** start the same command again. It reads
`data/shards-moe/_work/state.json`, truncates the collected files back to the last
checkpoint, deletes partial downloads and carries on (at most one code file or one
text file is redone). The result is byte-identical to an uninterrupted build. Running
it again after it has finished does nothing ("already built (use --fresh to
rebuild)", exit 0): a finished build is never rebuilt, or its manifest deleted, by
accident. Exit codes:

| exit | meaning | what to do |
|---|---|---|
| 0 | done (or already built) | copy the shards (below) |
| 1 | an unexpected error, including a download error that cannot succeed on retry (the last lines of `build.log` name the file and the error) | fix the cause (token, disk space, revision), then rerun the same command: it resumes |
| 2 | share error: the mix is off target (`collect_report.json` says why) | after code or text collection: decide the weights (or `--train-tokens`), then rerun with `--from-work`: it re-allocates from what was collected, without downloading it again. From the projection during code collection, the message gives two ways on: `--fresh` with changed weights (the code files read so far are downloaded again), or, if the projection is wrong, the same command with a larger `--projection-min-files` (resumes where it stopped) |
| 4 | the saved build cannot be resumed with these settings (the message lists the differences) | rerun with the original settings, or `--fresh` to start over (everything is downloaded again) |
| 5 | a tokenising worker died (out of memory?) | rerun the same command (fewer `--workers` if it was memory) |

Progress: `tail build.log`; the code checkpoint is `python -m json.tool
data/shards-moe/_work/state.json | grep next_file`.

## 6. Copy the shards to the GPU box and verify them

On the CPU box, checksum everything the trainer and evaluation use:

```bash
cd /workspace/quipu/data/shards-moe
find . -type f ! -path './_work/*' ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
wc -l SHA256SUMS; du -sh .
```

Then copy the directory, either way. Never forward the laptop's SSH agent (`ssh -A`)
to a rented box: anyone with root there could use it to log in wherever your keys
open.

- **Box to box with a throwaway key** (fastest; the data never passes through the
  laptop). On the CPU box, make a key that exists only for this copy:

  ```bash
  ssh-keygen -t ed25519 -N "" -C shards-copy -f /root/.ssh/shards_copy
  cat /root/.ssh/shards_copy.pub      # one line: copy it
  ```

  On the GPU box (your own SSH session from the laptop), allow that key:

  ```bash
  echo 'PASTE THE ONE LINE HERE' >> /root/.ssh/authorized_keys
  mkdir -p /workspace/quipu/data
  ```

  On the CPU box, copy (rsync resumes after a dropped connection; install it on both
  boxes with `apt-get install -y rsync`):

  ```bash
  rsync -a --partial --info=progress2 \
      -e "ssh -i /root/.ssh/shards_copy -o IdentitiesOnly=yes -p GPU_PORT" \
      /workspace/quipu/data/shards-moe/ root@GPU_HOST:/workspace/quipu/data/shards-moe/
  ```

  (without rsync: `scp -i /root/.ssh/shards_copy -P GPU_PORT -r /workspace/quipu/data/shards-moe root@GPU_HOST:/workspace/quipu/data/`)

  When the copy is done, remove the key from both boxes. On the GPU box:

  ```bash
  sed -i '/ shards-copy$/d' /root/.ssh/authorized_keys
  grep -c shards-copy /root/.ssh/authorized_keys    # 0
  ```

  and on the CPU box: `rm -f /root/.ssh/shards_copy /root/.ssh/shards_copy.pub`.

- **Through the laptop** (no key on either box; limited by the laptop's connection):

  ```bash
  scp -3 -r scp://root@CPU_HOST:CPU_PORT//workspace/quipu/data/shards-moe \
      scp://root@GPU_HOST:GPU_PORT//workspace/quipu/data/
  ```

On the GPU box, check every file and the manifest's token counts:

```bash
cd /workspace/quipu/data/shards-moe && sha256sum -c --quiet SHA256SUMS && echo ALL OK
cd /workspace/quipu && uv run python - <<'PY'
import json, pathlib
root = pathlib.Path("data/shards-moe"); m = json.loads((root / "manifest.json").read_text())
for split in ("val", "train", "code_val"):
    for s in m["splits"][split]["shards"]:
        assert (root / split / s["file"]).stat().st_size == 2 * s["tokens"], (split, s)
for bucket, v in m["splits"]["val_lang"].items():
    for s in v["shards"]:
        assert (root / "val_lang" / bucket / s["file"]).stat().st_size == 2 * s["tokens"], (
            bucket, s)
print("shard sizes match the manifest; train", m["splits"]["train"]["tokens"], "tokens;",
      len(m["splits"]["val_lang"]), "val_lang buckets")
PY
```

`sha256sum -c` prints nothing but `ALL OK` when every file matches. Then destroy the
CPU box in the Vast console: it bills per hour until you do.
