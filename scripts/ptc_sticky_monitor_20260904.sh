#!/usr/bin/env bash
set -u

REPO=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
ROOT=/project/cs-schulte-lab/ali/ptc_aids_motif_true_long_20260904
RUNNER="$REPO/scripts/run_one_ptc_aids_motif_long_20260904.sh"
PARTITION=cs-gpu-research
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
MOTIF_WEIGHT=0.1
CPUS=4
TIME_LIMIT=7-00:00:00

LOG_DIR="$ROOT/monitor_logs"
ALLOC_LOG_DIR="$ROOT/alloc_logs"
STATUS_DIR="$ROOT/status"
mkdir -p "$LOG_DIR" "$ALLOC_LOG_DIR" "$STATUS_DIR"

MONITOR_LOG="$LOG_DIR/ptc_sticky_monitor_20260904.log"
LOCK_FILE="$STATUS_DIR/ptc_sticky_monitor_20260904.lock"
ASSIGNMENT_FILE="$STATUS_DIR/ptc_sticky_assignments_20260904.tsv"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date --iso-8601=seconds) another monitor is already running" >> "$MONITOR_LOG"
  exit 0
fi

cat > "$ASSIGNMENT_FILE" <<'ASSIGNMENTS'
run_id	dataset	mode	seed	node	mem	prune_threshold	prune_cap
long_ptc_m0p1_total_count_s1	PTC	total_count	1	cs-venus-06	16G	0.0	256
long_ptc_m0p1_total_count_s2	PTC	total_count	2	cs-venus-06	16G	0.0	256
long_ptc_m0p1_full_matrix_s1	PTC	full_matrix	1	cs-venus-13	16G	0.0	256
long_ptc_m0p1_full_matrix_s2	PTC	full_matrix	2	cs-venus-09	16G	0.0	256
ASSIGNMENTS

submit_run() {
  local run_id=$1
  local dataset=$2
  local mode=$3
  local seed=$4
  local node=$5
  local mem=$6
  local prune_threshold=$7
  local prune_cap=$8
  local out_file="$ALLOC_LOG_DIR/${run_id}.%j.sticky_${node}.out"

  sbatch --parsable \
    --partition="$PARTITION" \
    --nodelist="$node" \
    --job-name="$run_id" \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS" \
    --mem="$mem" \
    --time="$TIME_LIMIT" \
    --output="$out_file" \
    "$RUNNER" "$run_id" "$dataset" "$mode" "$seed" \
    "$MOTIF_WEIGHT" "$prune_threshold" "$prune_cap"
}

check_once() {
  local now
  now=$(date --iso-8601=seconds)
  echo "$now check starting" >> "$MONITOR_LOG"

  tail -n +2 "$ASSIGNMENT_FILE" | while IFS=$'\t' read -r run_id dataset mode seed node mem prune_threshold prune_cap; do
    local status_file="$STATUS_DIR/${run_id}.status"
    if [[ -f "$status_file" ]] && grep -q '^COMPLETE ' "$status_file"; then
      echo "$now $run_id complete; no action" >> "$MONITOR_LOG"
      continue
    fi

    local active
    active=$(squeue -h -u "$USER" -n "$run_id" -o "%i|%t|%N|%R" | head -n 1)
    if [[ -n "$active" ]]; then
      local job_id state nodes reason
      IFS='|' read -r job_id state nodes reason <<< "$active"
      if [[ "$state" == "R" ]]; then
        echo "$now $run_id running job=$job_id node=$nodes fixed_node=$node" >> "$MONITOR_LOG"
      else
        echo "$now $run_id queued state=$state job=$job_id fixed_node=$node reason=$reason" >> "$MONITOR_LOG"
      fi
      continue
    fi

    local submitted
    submitted=$(submit_run "$run_id" "$dataset" "$mode" "$seed" "$node" "$mem" "$prune_threshold" "$prune_cap" 2>>"$MONITOR_LOG")
    local rc=$?
    if [[ $rc -eq 0 ]]; then
      echo "$now $run_id submitted job=$submitted fixed_node=$node mode=$mode seed=$seed" >> "$MONITOR_LOG"
    else
      echo "$now $run_id submit_failed rc=$rc fixed_node=$node mode=$mode seed=$seed" >> "$MONITOR_LOG"
    fi
  done
}

echo "$(date --iso-8601=seconds) monitor starting interval=${INTERVAL_SECONDS}s" >> "$MONITOR_LOG"
while true; do
  check_once
  sleep "$INTERVAL_SECONDS"
done
