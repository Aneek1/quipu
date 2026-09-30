#!/usr/bin/env bash
# Set up quipu on a rented Linux GPU box (Vast.ai, Ubuntu image, NVIDIA driver
# present, root shell). Idempotent: every step checks first, so it is safe to
# re-run after a failure or to update the checkout.
#
#   bash setup.sh [--repo-url URL] [--branch NAME] [--workdir DIR] [--skip-gpu-bench]
#                 [--cpu-only]
#
# --cpu-only is for a CPU box that only builds shards (scripts/remote/
# build_shards_box.md): no nvidia-smi, driver, CUDA or GPU benchmark checks, no Node
# or npm cache; it checks fastText instead and runs the shard builder's tests.
# The same settings can come from the environment: REPO_URL, BRANCH, WORKDIR,
# SKIP_GPU_BENCH=1, CPU_ONLY=1. The repo is public and cloned over https with plain git:
# this box never needs, and must never be given, any GitHub credentials.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Aneek1/quipu.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace/quipu}"
SKIP_GPU_BENCH="${SKIP_GPU_BENCH:-0}"
CPU_ONLY="${CPU_ONLY:-0}"

NODE_MAJOR=24          # installed from NodeSource when node is missing or too old
MIN_NODE_MAJOR=22
MIN_DRIVER_MAJOR=570   # the lock pins torch cu128 wheels; CUDA 12.8 needs driver >= 570

usage() {
    echo "usage: setup.sh [--repo-url URL] [--branch NAME] [--workdir DIR] [--skip-gpu-bench] [--cpu-only]"
    echo "defaults: $REPO_URL, branch $BRANCH, into $WORKDIR"
}
while [ $# -gt 0 ]; do
    case "$1" in
        --repo-url|--branch|--workdir)
            [ $# -ge 2 ] || { usage >&2; exit 2; }
            case "$1" in
                --repo-url) REPO_URL="$2" ;;
                --branch) BRANCH="$2" ;;
                --workdir) WORKDIR="$2" ;;
            esac
            shift 2 ;;
        --skip-gpu-bench) SKIP_GPU_BENCH=1; shift ;;
        --cpu-only) CPU_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"   # where the uv installer puts uv

# ---------------------------------------------------------------- platform
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64|Linux-aarch64) ;;
    *) die "expected Linux on x86_64 (or aarch64), got $(uname -s) $(uname -m)" ;;
esac

if [ "$CPU_ONLY" = "1" ]; then
    log "CPU-only box (--cpu-only): no GPU, driver or CUDA checks"
    echo "$(nproc) cores, $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo) RAM, $(df -h --output=avail / | tail -n 1 | tr -d ' ') free on /"
else
    log "GPU and driver"
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "nvidia-smi not found: this instance has no NVIDIA driver visible. Rent a GPU instance with an NVIDIA CUDA image (or pass --cpu-only on a box that only builds shards)."
    nvidia-smi
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
    driver_major="${driver%%.*}"
    case "$driver_major" in
        ''|*[!0-9]*) die "could not read the NVIDIA driver version (got '$driver')" ;;
    esac
    if [ "$driver_major" -lt "$MIN_DRIVER_MAJOR" ]; then
        die "NVIDIA driver $driver is too old for the cu128 torch wheels this project pins (CUDA 12.8 needs driver >= $MIN_DRIVER_MAJOR). Destroy this instance and rent one whose image/host offers CUDA 12.8 or newer (in the Vast search, filter on 'CUDA >= 12.8')."
    fi
    echo "driver $driver: OK for CUDA 12.8"
fi

# ---------------------------------------------------------------- system packages
missing=()
# python3: the system python runs `python3 -m quipu.spend` (stdlib only) outside the venv.
for pkg in git curl ca-certificates build-essential python3; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if [ "${#missing[@]}" -gt 0 ]; then
    log "Installing ${missing[*]}"
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends "${missing[@]}"
else
    log "git, curl, ca-certificates, build-essential, python3 already installed"
fi

# ---------------------------------------------------------------- Node
node_major() {
    if command -v node >/dev/null 2>&1; then
        node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0
    else
        echo 0
    fi
}
if [ "$CPU_ONLY" = "1" ]; then
    log "Node: not needed on a shard-building box (--cpu-only)"
elif [ "$(node_major)" -lt "$MIN_NODE_MAJOR" ]; then
    log "Installing Node $NODE_MAJOR from NodeSource"
    tmp="$(mktemp)"
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" -o "$tmp"
    $SUDO bash "$tmp"
    rm -f "$tmp"
    $SUDO apt-get install -y nodejs
    [ "$(node_major)" -ge "$MIN_NODE_MAJOR" ] || die "node is still older than $MIN_NODE_MAJOR after the NodeSource install"
fi
if [ "$CPU_ONLY" != "1" ]; then
    log "node $(node --version), npm $(npm --version)"
fi

# ---------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH (expected in \$HOME/.local/bin)"
fi
log "$(uv --version)"

# ---------------------------------------------------------------- checkout
export GIT_TERMINAL_PROMPT=0   # a wrong URL fails instead of asking for a password
if [ -d "$WORKDIR/.git" ]; then
    log "Updating $WORKDIR ($BRANCH)"
    git -C "$WORKDIR" fetch origin "$BRANCH"
    if [ "$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
        git -C "$WORKDIR" checkout "$BRANCH"
    fi
    git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
elif [ -e "$WORKDIR" ] && [ -n "$(ls -A "$WORKDIR" 2>/dev/null)" ]; then
    die "$WORKDIR exists, is not empty and is not a git checkout; move it away or pass --workdir"
else
    log "Cloning $REPO_URL ($BRANCH) into $WORKDIR"
    mkdir -p "$(dirname "$WORKDIR")"
    git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR" \
        || die "git clone failed. The repo must be public (this box gets no GitHub login); check REPO_URL and BRANCH."
fi
cd "$WORKDIR"
echo "at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

# ---------------------------------------------------------------- Python env
log "uv sync (Python 3.13 and the locked cu128 torch)"
uv sync --frozen

if [ "$CPU_ONLY" = "1" ]; then
    log "fastText (the LID filter; built for Linux by uv sync)"
    uv run --frozen python -c "import fasttext; print('fasttext OK:', fasttext.__file__)" \
        || die "fasttext does not import: the --lid-filter build cannot run on this box"

    log "Shard builder tests (no network)"
    uv run --frozen python -m pytest tests/test_build_shards_v2.py tests/test_build_shards.py -q -o addopts=""

    log "Done (CPU-only). The checkout is $WORKDIR; next: scripts/remote/build_shards_box.md"
    echo "Destroy the instance in the Vast console once the shards are copied off: it bills per hour."
    exit 0
fi

log "torch and CUDA"
uv run --frozen python - <<'PY'
import sys

import torch

print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"torch.cuda.is_available(): {ok}")
if not ok:
    sys.exit("CUDA is not usable from torch: check the nvidia-smi output above; "
             "a driver older than 570 cannot run the cu128 wheels.")
name = torch.cuda.get_device_name(0)
cap = "".join(map(str, torch.cuda.get_device_capability(0)))
vram = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"device: {name}, sm_{cap}, {vram:.1f} GiB")
# No arch-list check: an sm_89 card (4090) runs the wheel's sm_86 kernels without
# being listed. A real kernel launch is the test ("no kernel image" fails here).
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print(f"bf16 matmul on the GPU: OK ({float((x @ x).float().abs().mean()):.3f})")
PY

# ---------------------------------------------------------------- sanity tests
log "Sanity tests"
uv run --frozen python -m pytest tests/test_model_shapes.py tests/stepbuild/test_blocks.py -q

if [ -f scripts/check_gpu.py ] && [ "$SKIP_GPU_BENCH" != "1" ]; then
    log "GPU throughput (scripts/check_gpu.py, about a minute and a half; SKIP_GPU_BENCH=1 skips it)"
    uv run --frozen python scripts/check_gpu.py
fi

log "Pre-warming the stepbuild npm cache (one npm ci, cached for later runs)"
uv run --frozen python -m pytest -m npm tests/stepbuild/test_sandbox.py -q

log "Done. The checkout is $WORKDIR; run things with: cd $WORKDIR && uv run ..."
echo "Stop (or destroy) the instance in the Vast console when you are finished: it bills per hour."
