#!/usr/bin/env bash
set -u

hosts=(
  "17|/localhome/mirzaei/fb/GraphVAE-REQ"
  "18|/local-scratch2/mirzaei/fb/GraphVAE-REQ"
  "16|/localhome/mirzaei/fb/GraphVAE-REQ"
  "19|/local-scratch2/mirzaei/fb/GraphVAE-REQ"
  "13|/localhome/mirzaei/fb/GraphVAE-REQ"
)

for spec in "${hosts[@]}"; do
    IFS='|' read -r host repo_dir <<< "$spec"
    echo
    echo "cs-cl-$host"
    ssh -o BatchMode=yes -o ConnectTimeout=5 \
        "mirzaei@cs-cl-$host.cmpt.sfu.ca" bash -s -- "$repo_dir" <<'REMOTE' 2>/dev/null || {
repo_dir=$1
status_dir="$repo_dir/run_status/multihop_smoothed_3seed"
log_dir="$repo_dir/run_logs/multihop_smoothed_3seed"

printf '%-28s %-14s %s\n' RUN STATE PROGRESS
for status_file in "$status_dir"/gv_*.status; do
    [[ -f "$status_file" ]] || continue
    run_id=$(basename "$status_file" .status)
    state=$(awk '{print $1}' "$status_file")
    log_file="$log_dir/$run_id.log"
    progress=''
    if [[ "$state" != "QUEUED" ]]; then
        progress=$(grep 'Epoch:' "$log_file" 2>/dev/null | tail -n 1 | sed 's/^[[:space:]]*//')
        if [[ -z "$progress" ]]; then
            progress=$(grep -E 'Data-driven smoothed pruning|loading:|CUDA OOM|Traceback' \
                "$log_file" 2>/dev/null | tail -n 1 | sed 's/^[[:space:]]*//')
        fi
    fi
    [[ -n "$progress" ]] || progress='-'
    printf '%-28s %-14s %.180s\n' "$run_id" "$state" "$progress"
done
REMOTE
        echo "UNREACHABLE"
    }
done
