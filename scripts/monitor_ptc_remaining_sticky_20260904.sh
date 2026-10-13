#!/usr/bin/env bash
set -uo pipefail

REPO=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
ROOT=/project/cs-schulte-lab/ali/ptc_aids_motif_true_long_20260904
RUNNER="$REPO/scripts/run_one_ptc_aids_motif_long_20260904.sh"
MONITOR_LOG="$ROOT/monitor_ptc_remaining_sticky_20260904.log"
LOCK_FILE="$ROOT/status/monitor_ptc_remaining_sticky_20260904.lock"
SLEEP_SECONDS=1800

mkdir -p "$ROOT/alloc_logs" "$ROOT/status"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another monitor is already running" >> "$MONITOR_LOG"
  exit 0
fi

log() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "$MONITOR_LOG"
}

submit_run() {
  local run_id=$1
  local dataset=$2
  local mode=$3
  local seed=$4
  local node=$5
  local mem=$6
  local prune_threshold=$7
  local prune_cap=$8
  local log_file="$ROOT/alloc_logs/${run_id}.%j.sticky_${node}.out"

  local submit_output
  if submit_output=$(
    cd "$REPO" && sbatch --parsable \
      --partition=cs-gpu-research \
      --account=cs-schulte \
      --time=7-00:00:00 \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=4 \
      --mem="$mem" \
      --gres=gpu:1 \
      --nodelist="$node" \
      --requeue \
      --job-name="$run_id" \
      --output="$log_file" \
      "$RUNNER" "$run_id" "$dataset" "$mode" "$seed" 0.1 "$prune_threshold" "$prune_cap"
  ); then
    log "SUBMITTED run_id=$run_id dataset=$dataset mode=$mode seed=$seed sticky_node=$node mem=$mem job_id=$submit_output"
  else
    log "SUBMIT_FAILED run_id=$run_id dataset=$dataset mode=$mode seed=$seed sticky_node=$node mem=$mem"
  fi
}

check_or_submit() {
  local run_id=$1
  local dataset=$2
  local mode=$3
  local seed=$4
  local node=$5
  local mem=$6
  local prune_threshold=$7
  local prune_cap=$8
  local status_file="$ROOT/status/${run_id}.status"

  if [[ -f "$status_file" ]] && grep -q '^COMPLETE ' "$status_file"; then
    log "COMPLETE run_id=$run_id sticky_node=$node"
    return 0
  fi

  local queue_state
  queue_state=$(squeue -h -u "$USER" -n "$run_id" -o "%i|%T|%N|%R" | head -n 1)
  if [[ -n "$queue_state" ]]; then
    log "IN_QUEUE run_id=$run_id sticky_node=$node state=$queue_state"
    return 0
  fi

  submit_run "$run_id" "$dataset" "$mode" "$seed" "$node" "$mem" "$prune_threshold" "$prune_cap"
}

log "START monitor sleep_seconds=$SLEEP_SECONDS"
while true; do
  log "CHECK_BEGIN"
  check_or_submit long_ptc_m0p1_total_count_s1 PTC total_count 1 cs-venus-13 16G 0.0 256
  check_or_submit long_ptc_m0p1_total_count_s2 PTC total_count 2 cs-venus-06 16G 0.0 256
  check_or_submit long_ptc_m0p1_full_matrix_s1 PTC full_matrix 1 cs-venus-13 16G 0.0 256
  check_or_submit long_ptc_m0p1_full_matrix_s2 PTC full_matrix 2 cs-venus-13 16G 0.0 256
  check_or_submit long_aids_m0p1_total_count_s2 AIDS total_count 2 cs-venus-20 128G 5.0 64
  check_or_submit long_aids_m0p1_full_matrix_s1 AIDS full_matrix 1 cs-venus-12 128G 5.0 64
  check_or_submit long_aids_m0p1_full_matrix_s2 AIDS full_matrix 2 cs-venus-20 128G 5.0 64
  log "CHECK_END"
  sleep "$SLEEP_SECONDS"
done
