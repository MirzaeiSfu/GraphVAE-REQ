#!/usr/bin/env bash
set -u

runs=(
  "cs-cl-19|/local-scratch2/mirzaei/fb/GraphVAE-REQ|gv_mutag_total_s0"
  "cs-cl-19|/local-scratch2/mirzaei/fb/GraphVAE-REQ|gv_mutag_total_s1"
  "cs-cl-18|/local-scratch2/mirzaei/fb/GraphVAE-REQ|gv_mutag_full_s0"
  "cs-cl-18|/local-scratch2/mirzaei/fb/GraphVAE-REQ|gv_mutag_full_s1"
  "cs-cl-13|/localhome/mirzaei/fb/GraphVAE-REQ|gv_ptc_total_s0"
  "cs-cl-19|/local-scratch2/mirzaei/fb/GraphVAE-REQ|gv_ptc_total_s1"
  "cs-cl-17|/localhome/mirzaei/fb/GraphVAE-REQ|gv_ptc_full_s0"
  "cs-cl-17|/localhome/mirzaei/fb/GraphVAE-REQ|gv_ptc_full_s1"
)

printf '%-9s %-22s %-10s %s\n' HOST RUN STATE PROGRESS
for spec in "${runs[@]}"; do
    IFS='|' read -r host repo_dir run_id <<< "$spec"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "mirzaei@$host.cmpt.sfu.ca" bash -s -- \
        "$repo_dir" "$run_id" <<'REMOTE' 2>/dev/null || {
repo_dir=$1
run_id=$2
status_file="$repo_dir/run_status/multihop_smoothed/$run_id.status"
log_file="$repo_dir/run_logs/multihop_smoothed/$run_id.log"
state=MISSING
[[ -f "$status_file" ]] && state=$(awk '{print $1}' "$status_file")
progress=$(grep 'Epoch:' "$log_file" 2>/dev/null | tail -n 1 | sed 's/^[[:space:]]*//')
[[ -n "$progress" ]] || progress=$(tail -n 1 "$log_file" 2>/dev/null)
printf '%-9s %-22s %-10s %s\n' "$(hostname)" "$run_id" "$state" "$progress"
REMOTE
        printf '%-9s %-22s %-10s %s\n' "$host" "$run_id" UNREACHABLE '-'
    }
done
