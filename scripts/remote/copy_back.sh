#!/usr/bin/env bash
# Copy exactly what the laptop needs from the quipu-moe GPU box, then verify every
# file's sha256 against the box. Runs on the laptop (Git Bash), after the chat SFT and
# its evaluation, before the box is destroyed (scripts/remote/README.md step 8).
#
#   bash scripts/remote/copy_back.sh [--dry-run] HOST PORT
#
# The set (paths relative to $REMOTE_DIR, default /workspace/quipu; copied to the same
# relative paths under $LOCAL_DIR, default <repo>/remote-runs/quipu-moe, sync.sh's):
#   checkpoints/quipu-moe/latest.pt + the step_NNNNNN.pt it points to   (pretrain final)
#   checkpoints/quipu-moe/milestones/step_*.pt                          (milestones)
#   checkpoints/quipu-moe-sft/latest.pt + the step_NNNNNN.pt it points to (SFT final)
#   checkpoints/quipu-moe-sft/milestones/step_*.pt                      (its final step)
#   results/ (everything but inductor caches, *.pt, *.tmp; includes
#     results/moe/run_config.toml, results/ab, results/sft, results/spend.json)
#   data/shards-moe/{manifest.json, val/, code_val/, val_lang/}       (evaluation splits)
#   data/chat-sft/manifest.json and data/chat-sft/val_by_source/
# Not the training shards, the chat training data or any older step_*.pt.
#
# How: one ssh call lists the set on the box and hashes it (sha256sum; a few minutes
# for ~40 GB) into $LOCAL_DIR/copy_back.sha256; each file whose local copy is missing
# or differs is fetched with scp to NAME.part and renamed when complete; then
# `sha256sum -c copy_back.sha256` checks every file on the laptop. An expected item the
# box does not have is reported as MISSING (exit 1 at the end, after copying the
# rest). It never deletes a local file (no rm, no mirror): a local copy that differs
# from the box's is replaced by the box's version, nothing else is touched. Re-running
# skips what already matches, so an interrupted copy resumes file by file.
#
# SSH as sync.sh: your key (SSH_KEY, default ~/.ssh/id_ed25519), IdentitiesOnly,
# ForwardAgent=no, BatchMode=yes. --dry-run prints the commands and runs nothing.
# Exit: 0 all copied and verified, 1 something missing / failed / did not verify,
# 2 usage.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: bash scripts/remote/copy_back.sh [--dry-run] HOST PORT
env: REMOTE_USER (root), REMOTE_DIR (/workspace/quipu), LOCAL_DIR (<repo>/remote-runs/quipu-moe),
     SSH_KEY (~/.ssh/id_ed25519)
EOF
}

DRY_RUN=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "copy_back.sh: unknown option $arg" >&2; usage >&2; exit 2 ;;
        *) ARGS+=("$arg") ;;
    esac
done
if [ "${#ARGS[@]}" -ne 2 ]; then
    usage >&2
    exit 2
fi
HOST=${ARGS[0]}
PORT=${ARGS[1]}
case "$PORT" in
    ''|*[!0-9]*) echo "copy_back.sh: PORT must be a number, got '$PORT'" >&2; exit 2 ;;
esac
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
if command -v cygpath >/dev/null 2>&1; then
    LOCAL_DIR=$(cygpath -u "$LOCAL_DIR")
fi
if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$SSH_KEY" ]; then
    echo "copy_back.sh: no SSH key at $SSH_KEY (set SSH_KEY)" >&2
    exit 2
fi

REMOTE="$REMOTE_USER@$HOST"
SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o ForwardAgent=no -o BatchMode=yes
          -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=5)
SSH=(ssh -p "$PORT" "${SSH_OPTS[@]}")
SCP=(scp -P "$PORT" "${SSH_OPTS[@]}")
SUMS="$LOCAL_DIR/copy_back.sha256"

say() { printf '[copy-back %s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

show() {
    local out="" a
    for a in "$@"; do
        case "$a" in *" "*) out+="\"$a\" " ;; *) out+="$a " ;; esac
    done
    printf '%s\n' "${out% }"
}

# Runs on the box (sh): print "MISSING <what>" for an expected item that is not there,
# then the sha256sum lines of every file of the set (relative paths).
read -r -d '' LIST_SCRIPT <<EOF || true
cd $REMOTE_DIR || { echo "MISSING $REMOTE_DIR"; exit 0; }
list() {
  for d in checkpoints/quipu-moe checkpoints/quipu-moe-sft; do
    if [ -f \$d/latest.pt ]; then
      echo \$d/latest.pt
      f=\$(grep -ao 'step_[0-9]*[.]pt' \$d/latest.pt | head -n 1)
      if [ -n "\$f" ] && [ -f \$d/\$f ]; then echo \$d/\$f; else echo "MISSING \$d/<the step latest.pt names>" >&2; fi
    else
      echo "MISSING \$d/latest.pt" >&2
    fi
    [ -d \$d/milestones ] && find \$d/milestones -maxdepth 1 -type f -name 'step_*.pt'
  done
  [ -d checkpoints/quipu-moe/milestones ] || echo "MISSING checkpoints/quipu-moe/milestones/" >&2
  if [ -d results ]; then
    find results -type f ! -path '*/inductor-cache/*' ! -name '*.pt' ! -name '*.tmp'
  else
    echo "MISSING results/" >&2
  fi
  [ -f results/moe/run_config.toml ] || echo "MISSING results/moe/run_config.toml" >&2
  for f in data/shards-moe/manifest.json data/chat-sft/manifest.json; do
    if [ -f \$f ]; then echo \$f; else echo "MISSING \$f" >&2; fi
  done
  for d in data/shards-moe/val data/shards-moe/code_val data/shards-moe/val_lang data/chat-sft/val_by_source; do
    if [ -d \$d ]; then find \$d -type f; else echo "MISSING \$d/" >&2; fi
  done
}
list 2>&1 >/dev/null | grep '^MISSING ' || true
list 2>/dev/null | sort -u | while IFS= read -r f; do sha256sum "\$f"; done
EOF

missing=0
failed=0

if [ "$DRY_RUN" -eq 1 ]; then
    say "dry run: commands only; the box's list and hashes:"
    show "${SSH[@]}" "$REMOTE" "sh -s" "<<LIST_SCRIPT"
    printf '%s\n' "$LIST_SCRIPT"
    say "then, for each listed file F that is missing or differs locally:"
    show "${SCP[@]}" "$REMOTE:$REMOTE_DIR/F" "$LOCAL_DIR/F.part"
    show mv "$LOCAL_DIR/F.part" "$LOCAL_DIR/F"
    say "then the check:"
    show cd "$LOCAL_DIR" "&&" sha256sum -c --quiet copy_back.sha256
    exit 0
fi

mkdir -p "$LOCAL_DIR"
say "listing and hashing the copy-back set on $REMOTE:$REMOTE_DIR (minutes for ~40 GB)"
listing=$("${SSH[@]}" "$REMOTE" "sh -s" <<<"$LIST_SCRIPT") || {
    echo "copy_back.sh: the listing on the box failed (ssh)" >&2; exit 1; }
: > "$SUMS.new"
while IFS= read -r line; do
    case "$line" in
        "MISSING "*) say "$line"; missing=$((missing + 1)) ;;
        ?*) printf '%s\n' "$line" >> "$SUMS.new" ;;
    esac
done <<<"$listing"
mv "$SUMS.new" "$SUMS"
total=$(wc -l < "$SUMS" | tr -d ' ')
say "$total files in the set; hashes in $SUMS"

n=0
while IFS= read -r line; do
    want=${line%% *}
    rel=${line#* }
    rel=${rel#\*}
    rel=${rel# }
    n=$((n + 1))
    dest="$LOCAL_DIR/$rel"
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    say "[$n/$total] $rel"
    # </dev/null: scp's ssh must not eat the rest of the list this loop reads.
    if "${SCP[@]}" "$REMOTE:$REMOTE_DIR/$rel" "$dest.part" </dev/null; then
        mv -f "$dest.part" "$dest"
    else
        say "FAILED to copy $rel (a .part file may be left; re-run to retry)"
        failed=$((failed + 1))
    fi
done < "$SUMS"

say "verifying every file against the box's sha256"
if (cd "$LOCAL_DIR" && sha256sum -c --quiet copy_back.sha256); then
    say "all $total files match the box"
else
    say "some files do NOT match the box (listed above); re-run to fetch them again"
    failed=$((failed + 1))
fi
if [ "$missing" -gt 0 ] || [ "$failed" -gt 0 ]; then
    say "$missing expected item(s) missing on the box, $failed failure(s): do not destroy the box yet"
    exit 1
fi
say "done: everything needed is on the laptop in $LOCAL_DIR"
