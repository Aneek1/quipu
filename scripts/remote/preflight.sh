#!/usr/bin/env bash
# Preflight a rented box for quipu-moe, after setup.sh and before any paid long job
# (scripts/remote/README.md, build_shards_box.md). Run it from the checkout:
#
#   bash scripts/remote/preflight.sh --cpu-box                       # ~5 min
#   bash scripts/remote/preflight.sh --gpu-box --usd-per-hour R \
#        [--probes full-mb4-c0,full-mb4-c1,ab-c0] [--probe-steps 20] [--skip-probes]
#
# Prints PASS / WARN / FAIL per check and exits 1 if anything FAILed (0 otherwise).
# It never touches results/spend.json or the run's checkpoints: the ticker test uses a
# scratch ledger and the probes a scratch checkpoint directory, both under
# $PREFLIGHT_DIR (default /workspace/preflight-scratch); probe results go to
# results/preflight/ (probes.json, logs/, runs/), which sync.sh and copy_back.sh bring
# back.
#
# Both boxes:
#   - the checkout is on $EXPECT_BRANCH (default main) at origin's commit, clean;
#   - python3 (for `python3 -m quipu.spend`), else uv's python (then use
#     `uv run python -m quipu.spend ...` in every runbook line);
#   - free disk (CPU box >= 70 GB; GPU box >= 110 GB with the shards already on it);
#   - the tokenizer's sha256 is the one the gate passed;
#   - the spend ticker really ticks: `quipu.spend start` on a scratch ledger, wait
#     75 s, last_seen must have moved (crontab or nohup loop), then `stop`;
#   - the POSIX-only tests that never run on the Windows laptop (signals, sessions,
#     SIGKILL / EPIPE orphan stops): test_train_exit_codes, test_run_moe,
#     test_ab_runs, test_spend (GPU hidden).
# CPU box (--cpu-box):
#   - one fastText prediction with the LID model at the pinned revision;
#   - Hugging Face reachability of every pinned dataset revision and the LID model
#     (metadata only, nothing downloaded);
#   - HF_TOKEN set and HF_XET_CHUNK_CACHE_SIZE_BYTES=0 in this shell.
# GPU box (--gpu-box):
#   - data/shards-moe: sha256sum -c SHA256SUMS, and the manifest's tokenizer sha256 is
#     the tokenizer file's;
#   - data/stepbuild/train-000.jsonl and test-000.jsonl present (the chat SFT needs them);
#   - torch sees the GPU, Triton imports, the card is sm_120 (a 5090; WARN otherwise);
#   - the compile-parity tests (tests/test_train_moe.py: CPU inductor and CUDA);
#   - the 20-step real-trainer probes (scripts/remote/compile_probe.py, ~25-30 min in
#     all, ~$0.30 at $0.55/h): full model at micro_batch 8 and 4 with compile on and
#     off, the A/B model with AttnRes off and on (compile off): peak VRAM, tokens/s,
#     recompiles -> results/preflight/probes.json, the micro_batch and compile
#     decisions, and the `ab_runs.py --dry-run --tokens-per-second <measured>`
#     command for the owner. --probes limits them (a comma list of names),
#     --skip-probes skips them.
set -uo pipefail

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null \
        || echo "usage: preflight.sh --cpu-box | --gpu-box --usd-per-hour R [--probes LIST] [--probe-steps N] [--skip-probes]"
}

MODE=""
RATE=""
PROBES=""
PROBE_STEPS=20
SKIP_PROBES=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu-box) MODE=cpu ;;
        --gpu-box) MODE=gpu ;;
        --usd-per-hour) [ $# -ge 2 ] || { usage >&2; exit 2; }; RATE=$2; shift ;;
        --probes) [ $# -ge 2 ] || { usage >&2; exit 2; }; PROBES=$2; shift ;;
        --probe-steps) [ $# -ge 2 ] || { usage >&2; exit 2; }; PROBE_STEPS=$2; shift ;;
        --skip-probes) SKIP_PROBES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "preflight.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if [ -z "$MODE" ]; then
    echo "preflight.sh: pass --cpu-box or --gpu-box" >&2
    exit 2
fi
if [ "$MODE" = gpu ] && [ "$SKIP_PROBES" -eq 0 ]; then
    case "$RATE" in
        ''|*[!0-9.]*|.*|*.*.*) echo "preflight.sh: --gpu-box needs --usd-per-hour R (the box's \$/h) for the probes" >&2; exit 2 ;;
    esac
fi
case "$PROBE_STEPS" in
    ''|*[!0-9]*) echo "preflight.sh: --probe-steps must be a whole number" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 2
export PATH="$HOME/.local/bin:$PATH"
EXPECT_BRANCH=${EXPECT_BRANCH:-main}
TOK_SHA=${TOKENIZER_SHA256:-9f3e33d074f3a0d7da234eec11a59de1e20aa756855d1ef552fc9706b9006943}
PF=${PREFLIGHT_DIR:-/workspace/preflight-scratch}
mkdir -p "$PF"
FAILS=0
WARNS=0
PY=(uv run --frozen python)

ok()   { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; WARNS=$((WARNS + 1)); }
bad()  { printf 'FAIL  %s\n' "$*"; FAILS=$((FAILS + 1)); }
# run NAME CMD...: PASS/FAIL on the exit code, output in $PF/NAME.log (tail on failure).
run() {
    local name=$1; shift
    if "$@" >"$PF/$name.log" 2>&1; then
        ok "$name"
    else
        bad "$name (exit $?; $PF/$name.log):"
        tail -n 15 "$PF/$name.log" | sed 's/^/        /'
    fi
}

echo "== quipu-moe preflight ($MODE box) in $REPO; logs in $PF"

# ---- both boxes ------------------------------------------------------------------------------
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
head=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
if [ "$branch" = "$EXPECT_BRANCH" ]; then ok "branch $branch at $head"; else bad "on branch '$branch', expected $EXPECT_BRANCH"; fi
if GIT_TERMINAL_PROMPT=0 git fetch -q origin "$EXPECT_BRANCH" 2>"$PF/fetch.log"; then
    remote_head=$(git rev-parse --short FETCH_HEAD)
    if [ "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)" ]; then
        ok "at origin/$EXPECT_BRANCH ($remote_head)"
    else
        bad "HEAD $head is not origin/$EXPECT_BRANCH ($remote_head): git pull --ff-only (or re-run setup.sh)"
    fi
else
    warn "could not fetch origin/$EXPECT_BRANCH to compare (see $PF/fetch.log)"
fi
if [ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    ok "no local changes to tracked files"
else
    warn "tracked files changed on the box: git status"
fi

if command -v python3 >/dev/null 2>&1; then
    SPEND=(python3 -m quipu.spend)
    ok "system $(python3 -V 2>&1) for quipu.spend"
elif "${PY[@]}" -c "pass" >/dev/null 2>&1; then
    SPEND=("${PY[@]}" -m quipu.spend)
    warn "no system python3: use 'uv run python -m quipu.spend ...' wherever the runbook says python3 -m quipu.spend"
else
    SPEND=()
    bad "neither python3 nor uv's python runs (apt-get install -y python3, or re-run setup.sh)"
fi

need=$([ "$MODE" = gpu ] && echo 110 || echo 70)
free_gb=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -n 1 | tr -dc 0-9)
if [ -n "$free_gb" ] && [ "$free_gb" -ge "$need" ]; then
    ok "disk: ${free_gb} GB free (>= $need)"
else
    bad "disk: ${free_gb:-?} GB free, want >= $need GB (rent more disk)"
fi

tok=artifacts/tokenizer/tokenizer.json
if [ -f "$tok" ] && [ "$(sha256sum "$tok" | cut -d' ' -f1)" = "$TOK_SHA" ]; then
    ok "tokenizer sha256 ${TOK_SHA:0:12}..."
else
    bad "tokenizer: $tok missing or not sha256 ${TOK_SHA:0:12}... (copy it from the laptop)"
fi

if [ "${#SPEND[@]}" -gt 0 ]; then
    L="$PF/spend-test.json"
    rm -f "$L" "$L".*
    last_seen() {
        "${SPEND[@]:0:${#SPEND[@]}-2}" -c "import json,sys; print(json.load(open(sys.argv[1]))['sessions'][-1]['last_seen'])" "$L" 2>/dev/null
    }
    if "${SPEND[@]}" --ledger "$L" start --usd-per-hour 0.01 >"$PF/ticker.log" 2>&1; then
        t0=$(last_seen)
        echo "        waiting 75 s for the idle ticker ($(grep -o 'crontab line\|tick loop\|ticker[^.]*' "$PF/ticker.log" | head -n 1))"
        sleep 75
        t1=$(last_seen)
        "${SPEND[@]}" --ledger "$L" stop --force >>"$PF/ticker.log" 2>&1
        if [ -n "$t0" ] && [ -n "$t1" ] && awk -v a="$t0" -v b="$t1" 'BEGIN { exit !(b > a) }'; then
            ok "spend ticker ticks on its own (scratch ledger)"
        else
            bad "spend ticker did not tick in 75 s ($PF/ticker.log): idle box time would not be counted"
        fi
    else
        bad "quipu.spend start failed on a scratch ledger ($PF/ticker.log)"
    fi
fi

run posix_tests env CUDA_VISIBLE_DEVICES= "${PY[@]}" -m pytest -q -o addopts="" -p no:cacheprovider \
    -m "not gpu_gate" tests/test_train_exit_codes.py tests/test_run_moe.py \
    tests/test_ab_runs.py tests/test_spend.py

# ---- CPU box -----------------------------------------------------------------------------------
if [ "$MODE" = cpu ]; then
    run lid_predict "${PY[@]}" - <<'PY'
from huggingface_hub import hf_hub_download
from quipu.config import load_config
from quipu import shard_mix as sm
d = load_config("configs/quipu-moe.toml").data
lid = sm.FastTextLid(hf_hub_download(d.lid_model, "lid-specialist-fasttext.ftz",
                                     revision=d.lid_revision))
cases = {"eng_Latn": "The committee will publish its annual report on the economy next week.",
         "zho_Hant": "我們今天在學校學習了很多關於歷史與地理的知識，老師講得非常清楚。",
         "zsm_Latn": "Kerajaan akan mengumumkan bajet baharu untuk rakyat Malaysia minggu hadapan."}
for want, text in cases.items():
    top, p, _ = lid.predict(text)
    print(want, "->", top, round(p, 3))
    assert top == want or want == "zsm_Latn", (want, top)   # zsm / ind confusion is expected
print("labels:", lid.labels())
PY
    run hf_access "${PY[@]}" - <<'PY'
from huggingface_hub import HfApi
from quipu.config import load_config
d = load_config("configs/quipu-moe.toml").data
api = HfApi()
for repo, rev in [(d.code_dataset, d.code_revision), (d.dataset, d.text_revision),
                  ("HuggingFaceFW/fineweb-2", d.fineweb2_revision),
                  ("openai/openai_humaneval", d.humaneval_revision),
                  ("google-research-datasets/mbpp", d.mbpp_revision)]:
    print(repo, rev[:12], api.dataset_info(repo, revision=rev).sha)
print(d.lid_model, d.lid_revision[:12], api.model_info(d.lid_model, revision=d.lid_revision).sha)
PY
    if [ -n "${HF_TOKEN:-}" ]; then ok "HF_TOKEN set in this shell"; else bad "HF_TOKEN not set (build_shards_box.md step 3)"; fi
    if [ "${HF_XET_CHUNK_CACHE_SIZE_BYTES:-}" = 0 ]; then
        ok "xet chunk cache off"
    else
        bad "export HF_XET_CHUNK_CACHE_SIZE_BYTES=0 in this shell (build_shards_box.md step 3)"
    fi
fi

# ---- GPU box -----------------------------------------------------------------------------------
if [ "$MODE" = gpu ]; then
    run shards_sha256 bash -c "cd data/shards-moe && sha256sum -c --quiet SHA256SUMS"
    run manifest_tokenizer "${PY[@]}" -c "
import hashlib, json
m = json.load(open('data/shards-moe/manifest.json'))
h = hashlib.sha256(open('$tok', 'rb').read()).hexdigest()
assert m['tokenizer']['sha256'] == h, 'the shards were built with another tokenizer'
print('train tokens', m['splits']['train']['tokens'])"
    if [ -s data/stepbuild/train-000.jsonl ] && [ -s data/stepbuild/test-000.jsonl ]; then
        ok "stepbuild train-000.jsonl and test-000.jsonl present (chat SFT)"
    else
        bad "data/stepbuild/train-000.jsonl / test-000.jsonl missing (README step 3: scp them from the laptop)"
    fi
    if "${PY[@]}" -c "
import sys, torch, triton
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'no CUDA device'
cap = torch.cuda.get_device_capability(0)
gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f'{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} {gib:.1f} GiB, triton {triton.__version__}')
sys.exit(0 if cap == (12, 0) else 3)" >"$PF/triton.log" 2>&1; then
        ok "triton + $(tail -n 1 "$PF/triton.log")"
    elif [ $? -eq 3 ]; then
        warn "not an sm_120 card (the plan assumes a 5090): $(tail -n 1 "$PF/triton.log")"
    else
        bad "torch/CUDA/Triton ($PF/triton.log):"
        tail -n 5 "$PF/triton.log" | sed 's/^/        /'
    fi
    run compile_parity "${PY[@]}" -m pytest -q -o addopts="" -p no:cacheprovider -rs \
        tests/test_train_moe.py -k "compiled_logits_match or compile"
    if grep -q "skipped" "$PF/compile_parity.log" 2>/dev/null; then
        warn "some compile tests were skipped: $(grep -o 'SKIPPED.*' "$PF/compile_parity.log" | head -n 2 | tr '\n' ' ')"
    fi
    if [ "$SKIP_PROBES" -eq 1 ]; then
        warn "probes skipped (--skip-probes): micro_batch and compile are not measured"
    else
        echo "        probes: ~25-30 min in all (each: model build, compile, $PROBE_STEPS steps, one eval, one checkpoint)"
        probe_args=(--usd-per-hour "$RATE" --steps "$PROBE_STEPS" --scratch "$PF/probes")
        [ -n "$PROBES" ] && probe_args+=(--only "$PROBES")
        if "${PY[@]}" scripts/remote/compile_probe.py "${probe_args[@]}" 2>&1 | tee "$PF/probes.log"; then
            ok "probes done: results/preflight/probes.json (decisions printed above)"
        else
            bad "no probe finished ($PF/probes.log, results/preflight/logs/)"
        fi
    fi
fi

echo "== $FAILS failure(s), $WARNS warning(s); logs in $PF"
[ "$FAILS" -eq 0 ]
