#!/usr/bin/env bash
# Pull the quipu-moe run from the rented GPU box to the laptop, every few hours,
# until the run is over (spec 6.4, plan Task M9). Runs on the laptop (Git Bash).
#
#   bash scripts/remote/sync.sh [--dry-run] HOST PORT [INTERVAL_HOURS]
#
# Each cycle copies, from root@HOST:$REMOTE_DIR (default /workspace/quipu), into
# $LOCAL_DIR (default <repo>/remote-runs/quipu-moe):
#   - checkpoints/quipu-moe/latest.pt and the step_NNNNNN.pt it points to (the
#     pointer is fetched first and put in place only after its checkpoint arrived,
#     so the local pointer never names a file that is not there);
#   - checkpoints/quipu-moe/milestones/ (bf16 snapshots);
#   - results/ (run logs, plan, summary, spend ledger), without the inductor caches,
#     *.pt and *.tmp files.
# It stops after the cycle that sees results/moe/summary.md on the box (run_moe.py
# writes it only when the run is over). It never mirrors deletions (no rsync delete
# option). Old step_*.pt copies (~12 GB each for quipu-moe) stay unless KEEP_LOCAL=N
# is set: then, after each checkpoint sync, only the N newest local step_*.pt are
# kept; the one the local latest.pt points to and everything in milestones/ are never
# deleted (default: unset, keep all). `--prune-only` applies KEEP_LOCAL once and exits.
# A failed cycle (network, box busy) is reported and retried at the next one.
#
# rsync is used when it is installed; otherwise (Git Bash has no rsync) scp for the
# checkpoint files and ssh + tar for results/, so the excludes still apply. Force
# one with SYNC_TOOL=rsync or SYNC_TOOL=scp. rsync keeps an interrupted transfer in
# .rsync-partial/ (--partial-dir) and resumes it next cycle; a half file never sits
# under the checkpoint's own name.
#
# SSH: your own key only (SSH_KEY, default ~/.ssh/id_ed25519), no agent forwarding
# (ForwardAgent=no, IdentitiesOnly=yes), new host keys accepted on first use, and
# BatchMode=yes: a missing key fails the cycle instead of waiting for a password.
# --dry-run prints the commands of one cycle and runs nothing.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: bash scripts/remote/sync.sh [--dry-run] HOST PORT [INTERVAL_HOURS]
       KEEP_LOCAL=N bash scripts/remote/sync.sh --prune-only
  HOST PORT        the GPU box's SSH address (ssh -p PORT root@HOST)
  INTERVAL_HOURS   hours between syncs (default 3; decimals allowed)
env: REMOTE_USER (root), REMOTE_DIR (/workspace/quipu), LOCAL_DIR (<repo>/remote-runs/quipu-moe),
     SSH_KEY (~/.ssh/id_ed25519), SYNC_TOOL (auto | rsync | scp),
     KEEP_LOCAL (unset = keep every local step_*.pt; N = keep the N newest, never the
     one latest.pt points to, never milestones/)
EOF
}

DRY_RUN=0
PRUNE_ONLY=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --prune-only) PRUNE_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "sync.sh: unknown option $arg" >&2; usage >&2; exit 2 ;;
        *) ARGS+=("$arg") ;;
    esac
done
KEEP_LOCAL=${KEEP_LOCAL:-}
case "$KEEP_LOCAL" in
    '') ;;
    *[!0-9]*|0) echo "sync.sh: KEEP_LOCAL must be a whole number >= 1, got '$KEEP_LOCAL'" >&2; exit 2 ;;
esac
if [ "$PRUNE_ONLY" -eq 1 ]; then
    if [ "${#ARGS[@]}" -ne 0 ] || [ -z "$KEEP_LOCAL" ]; then
        echo "sync.sh: --prune-only takes no HOST/PORT and needs KEEP_LOCAL=N" >&2
        exit 2
    fi
    ARGS=(none 0)
fi
if [ "${#ARGS[@]}" -lt 2 ] || [ "${#ARGS[@]}" -gt 3 ]; then
    usage >&2
    exit 2
fi
HOST=${ARGS[0]}
PORT=${ARGS[1]}
INTERVAL_HOURS=${ARGS[2]:-3}
case "$PORT" in
    ''|*[!0-9]*) echo "sync.sh: PORT must be a number, got '$PORT'" >&2; exit 2 ;;
esac
if ! INTERVAL_S=$(awk -v h="$INTERVAL_HOURS" \
        'BEGIN { if (h !~ /^[0-9]+([.][0-9]+)?$/ || h + 0 <= 0) exit 1; printf "%d", h * 3600 }'); then
    echo "sync.sh: INTERVAL_HOURS must be a number > 0, got '$INTERVAL_HOURS'" >&2
    exit 2
fi

REMOTE_USER=${REMOTE_USER:-root}
REMOTE_DIR=${REMOTE_DIR:-/workspace/quipu}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
if [ -z "${LOCAL_DIR:-}" ]; then
    if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/remote-runs/quipu-moe"
    else
        LOCAL_DIR="$PWD/remote-runs/quipu-moe"
    fi
fi
# Git Bash: C:/... would read as host "C" to scp and rsync; use the /c/... form.
if command -v cygpath >/dev/null 2>&1; then
    LOCAL_DIR=$(cygpath -u "$LOCAL_DIR")
fi
TOOL=${SYNC_TOOL:-auto}
if [ "$TOOL" = auto ]; then
    if command -v rsync >/dev/null 2>&1; then TOOL=rsync; else TOOL=scp; fi
fi
case "$TOOL" in
    rsync|scp) ;;
    *) echo "sync.sh: SYNC_TOOL must be auto, rsync or scp, got '$TOOL'" >&2; exit 2 ;;
esac
if [ "$DRY_RUN" -eq 0 ] && [ "$PRUNE_ONLY" -eq 0 ] && [ ! -f "$SSH_KEY" ]; then
    echo "sync.sh: no SSH key at $SSH_KEY (set SSH_KEY)" >&2
    exit 2
fi

REMOTE="$REMOTE_USER@$HOST"
CKPT=checkpoints/quipu-moe
SUMMARY="$REMOTE_DIR/results/moe/summary.md"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o ForwardAgent=no -o BatchMode=yes
          -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=5)
SSH=(ssh -p "$PORT" "${SSH_OPTS[@]}")
SCP=(scp -P "$PORT" "${SSH_OPTS[@]}")
RSYNC_SSH="ssh -p $PORT ${SSH_OPTS[*]}"
RSYNC=(rsync -av --partial-dir=.rsync-partial -e "$RSYNC_SSH")

say() { printf '[sync %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Print a command as one line (arguments with spaces in double quotes).
show() {
    local out="" a
    for a in "$@"; do
        case "$a" in *" "*) out+="\"$a\" " ;; *) out+="$a " ;; esac
    done
    printf '%s\n' "${out% }"
}

run() {
    if [ "$DRY_RUN" -eq 1 ]; then show "$@"; else "$@"; fi
}

remote_done() {
    if [ "$DRY_RUN" -eq 1 ]; then
        show "${SSH[@]}" "$REMOTE" "test -f $SUMMARY"
        return 1
    fi
    "${SSH[@]}" "$REMOTE" "test -f $SUMMARY"
}

# The checkpoint file a local copy of latest.pt points to. torch.save stores the
# pointer {"file": "step_NNNNNN.pt"} uncompressed, so the name is in the bytes.
pointer_target() {
    if [ "$DRY_RUN" -eq 1 ]; then echo step_NNNNNN.pt; return 0; fi
    grep -ao 'step_[0-9]*\.pt' "$1" | head -n 1
}

remote_milestones() {
    "${SSH[@]}" "$REMOTE" "ls -1 $REMOTE_DIR/$CKPT/milestones 2>/dev/null || true"
}

sync_checkpoint() {
    local incoming="$LOCAL_DIR/$CKPT/latest.pt.incoming" step
    if [ "$TOOL" = rsync ]; then
        run "${RSYNC[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/latest.pt" "$incoming" || return 1
    else
        run "${SCP[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/latest.pt" "$incoming" || return 1
    fi
    step=$(pointer_target "$incoming") || true
    if [ -z "$step" ]; then
        say "could not read the checkpoint name from latest.pt; skipping the checkpoint"
        return 1
    fi
    if [ "$TOOL" = rsync ]; then
        run "${RSYNC[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/$step" "$LOCAL_DIR/$CKPT/" || return 1
    elif [ "$DRY_RUN" -eq 1 ] || [ ! -f "$LOCAL_DIR/$CKPT/$step" ]; then
        # scp has no resume: fetch to a temporary name, rename when complete.
        run "${SCP[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/$step" "$LOCAL_DIR/$CKPT/$step.part" || return 1
        run mv "$LOCAL_DIR/$CKPT/$step.part" "$LOCAL_DIR/$CKPT/$step"
    fi
    run mv "$incoming" "$LOCAL_DIR/$CKPT/latest.pt"
}

sync_milestones() {
    if [ "$TOOL" = rsync ]; then
        run "${RSYNC[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/milestones/" "$LOCAL_DIR/$CKPT/milestones/"
        return
    fi
    local names name
    if [ "$DRY_RUN" -eq 1 ]; then
        show "${SSH[@]}" "$REMOTE" "ls -1 $REMOTE_DIR/$CKPT/milestones"
        names=step_NNNNNN.pt
    else
        names=$(remote_milestones) || return 1
    fi
    for name in $names; do
        case "$name" in step_*.pt) ;; *) continue ;; esac
        [ "$DRY_RUN" -eq 0 ] && [ -f "$LOCAL_DIR/$CKPT/milestones/$name" ] && continue
        run "${SCP[@]}" "$REMOTE:$REMOTE_DIR/$CKPT/milestones/$name" \
            "$LOCAL_DIR/$CKPT/milestones/$name.part" || return 1
        run mv "$LOCAL_DIR/$CKPT/milestones/$name.part" "$LOCAL_DIR/$CKPT/milestones/$name"
    done
}

sync_results() {
    if [ "$TOOL" = rsync ]; then
        run "${RSYNC[@]}" --exclude inductor-cache/ --exclude '*.pt' --exclude '*.tmp' \
            "$REMOTE:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"
        return
    fi
    local pack="tar -C $REMOTE_DIR -cf - --exclude=inductor-cache --exclude='*.pt' --exclude='*.tmp' results"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s | %s\n' "$(show "${SSH[@]}" "$REMOTE" "$pack")" "tar -xf - -C $LOCAL_DIR"
        return 0
    fi
    "${SSH[@]}" "$REMOTE" "$pack" | tar -xf - -C "$LOCAL_DIR"
}

# KEEP_LOCAL=N: keep the N newest local step_NNNNNN.pt (zero-padded, so a name sort is
# a step sort), never the one the local latest.pt points to (it is always kept, even
# beyond N), never anything under milestones/ (not matched: only this directory's own
# step_*.pt files are candidates). Without a readable latest.pt nothing is deleted.
prune_local() {
    [ -n "$KEEP_LOCAL" ] || return 0
    local dir="$LOCAL_DIR/$CKPT" keep n=0 f
    [ -d "$dir" ] || return 0
    if [ ! -f "$dir/latest.pt" ]; then
        say "KEEP_LOCAL: no local latest.pt yet; nothing pruned"
        return 0
    fi
    keep=$(grep -ao 'step_[0-9]*\.pt' "$dir/latest.pt" | head -n 1) || true
    if [ -z "$keep" ]; then
        say "KEEP_LOCAL: could not read the checkpoint name from latest.pt; nothing pruned"
        return 0
    fi
    for f in $(ls -1 "$dir" | grep -E '^step_[0-9]+\.pt$' | sort -r); do
        n=$((n + 1))
        if [ "$n" -le "$KEEP_LOCAL" ] || [ "$f" = "$keep" ]; then
            continue
        fi
        [ -f "$dir/$f" ] || continue
        say "KEEP_LOCAL=$KEEP_LOCAL: removing the older local copy $f"
        run rm -f "$dir/$f"
    done
}

cycle() {
    local ok=0
    run mkdir -p "$LOCAL_DIR/$CKPT/milestones" "$LOCAL_DIR/results"
    if sync_checkpoint; then prune_local; else say "checkpoint sync failed"; ok=1; fi
    sync_milestones || { say "milestone sync failed"; ok=1; }
    sync_results || { say "results sync failed"; ok=1; }
    return $ok
}

if [ "$PRUNE_ONLY" -eq 1 ]; then
    prune_local
    exit 0
fi

say "syncing $REMOTE:$REMOTE_DIR -> $LOCAL_DIR with $TOOL every $INTERVAL_HOURS h" \
    "(${INTERVAL_S} s)$([ "$DRY_RUN" -eq 1 ] && echo '; dry run: commands only')"
while true; do
    finished=0
    if remote_done; then finished=1; fi
    if cycle; then say "cycle done"; else say "cycle had failures; trying again next time"; fi
    if [ "$finished" -eq 1 ]; then
        say "the run's summary is on the box: final sync done, stopping"
        exit 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        exit 0
    fi
    sleep "$INTERVAL_S"
done
