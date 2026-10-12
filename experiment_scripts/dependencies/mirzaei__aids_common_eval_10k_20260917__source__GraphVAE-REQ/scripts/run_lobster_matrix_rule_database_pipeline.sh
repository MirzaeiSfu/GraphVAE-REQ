#!/usr/bin/env bash
set -euo pipefail

# Monitor the four already-launched matrix-motif rule-database runs, collect
# their artifacts, verify all periodic checkpoints, and perform leakage-free
# normalized_table2_table3 checkpoint selection on validation before test.

DATE_PREFIX="${DATE_PREFIX:-20260719}"
SCHEDULE="${SCHEDULE:-CLUSTER_GPU_CONFIGS_LOBSTER_MATRIX_RULE_DATABASE_SWEEP.txt}"
REPO_PATHS="${REPO_PATHS:-CLUSTER_REPO_PATHS_LOBSTER_MATRIX_RULE_DATABASE_SWEEP.txt}"
RUN_ROOT="${RUN_ROOT:-runs/$DATE_PREFIX/lobster_matrix_motif_rule_database_sweep}"
COLLECT_ROOT="${COLLECT_ROOT:-collected_runs}"
POSTHOC_OUTPUT_DIR="${POSTHOC_OUTPUT_DIR:-$COLLECT_ROOT/$DATE_PREFIX/lobster_matrix_motif_rule_database_normalized_table2_table3_selection}"
SELECTOR="${SELECTOR:-scripts/select_lobster_normalized_table2_table3.py}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
EXPECTED_RUNS=4
EXPECTED_CHECKPOINTS_PER_RUN=5
EXPECTED_EPOCHS=20000

RUN_ROOT="${RUN_ROOT%/}"
RUN_ROOT_NAME="$(basename "$RUN_ROOT")"
COLLECTED_RUNS_ROOT="$COLLECT_ROOT/$DATE_PREFIX/$RUN_ROOT_NAME"
CANDIDATE_JSON="$POSTHOC_OUTPUT_DIR/validation_candidates.json"
SSH_OPTS=(-o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" -o StrictHostKeyChecking=accept-new)

declare -A REPO_BY_HOST

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._-]*//; s/[._-]*$//'
}

job_dir_for() {
  local host="$1"
  local gpu="$2"
  local config="$3"
  printf '%s__%s_gpu%s' \
    "$(sanitize "$(basename "$config" .yaml)")" \
    "$(sanitize "$host")" \
    "$(sanitize "$gpu")"
}

load_repo_paths() {
  local raw_line line host repo_path extra
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line_is_blank "$line" && continue
    read -r host repo_path extra <<<"$line"
    if [[ -z "${host:-}" || -z "${repo_path:-}" || -n "${extra:-}" ]]; then
      echo "[preflight] invalid repo-path row: $raw_line" >&2
      exit 2
    fi
    REPO_BY_HOST["$host"]="$repo_path"
  done < "$REPO_PATHS"
}

preflight() {
  local row_count
  for required in "$SCHEDULE" "$REPO_PATHS" "$SELECTOR"; do
    if [[ ! -f "$required" ]]; then
      echo "[preflight] missing required file: $required" >&2
      exit 2
    fi
  done
  if [[ ! -x "$SELECTOR_PYTHON" ]]; then
    echo "[preflight] selector Python is unavailable: $SELECTOR_PYTHON" >&2
    exit 2
  fi
  if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[preflight] POLL_SECONDS must be a positive integer" >&2
    exit 2
  fi
  load_repo_paths
  row_count="$(awk '!/^([[:space:]]*#|[[:space:]]*$)/ {count++} END {print count+0}' "$SCHEDULE")"
  if [[ "$row_count" -ne "$EXPECTED_RUNS" ]]; then
    echo "[preflight] expected $EXPECTED_RUNS schedule rows; found $row_count" >&2
    exit 2
  fi
  echo "[preflight] monitoring $EXPECTED_RUNS launched runs"
}

probe_job() {
  local host="$1"
  local session="$2"
  local log_path="$3"
  ssh -n "${SSH_OPTS[@]}" "$host" \
    "session='$session'; log_path='$log_path'; epoch=0; if [[ -f \"\$log_path\" ]]; then found_epoch=\$(sed -n 's/.*Epoch: *\\([0-9][0-9]*\\).*/\\1/p' \"\$log_path\" | tail -n 1); [[ -n \"\$found_epoch\" ]] && epoch=\$((10#\$found_epoch)); fi; if tmux has-session -t \"\$session\" 2>/dev/null; then state=active; elif [[ -f \"\$log_path\" ]] && grep -q 'trainning time:' \"\$log_path\"; then state=complete; elif [[ -f \"\$log_path\" ]]; then state=failed; else state=missing; fi; printf '%s\\t%s\\n' \"\$state\" \"\$epoch\""
}

wait_for_training() {
  local raw_line line host gpu config extra repo_path job_dir session log_path
  local result state epoch complete active failed unreachable
  while true; do
    complete=0
    active=0
    failed=0
    unreachable=0
    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
      line="${raw_line%%#*}"
      line_is_blank "$line" && continue
      read -r host gpu config extra <<<"$line"
      repo_path="${REPO_BY_HOST[$host]:-}"
      if [[ -z "$repo_path" ]]; then
        echo "[monitor] no repo path for $host" >&2
        return 1
      fi
      job_dir="$(job_dir_for "$host" "$gpu" "$config")"
      session="$(sanitize "${DATE_PREFIX}_${job_dir}")"
      log_path="${repo_path%/}/$RUN_ROOT/$job_dir/stdout.log"
      if result="$(probe_job "$host" "$session" "$log_path")"; then
        IFS=$'\t' read -r state epoch <<<"$result"
      else
        state=unreachable
        epoch=?
      fi
      case "$state" in
        complete) complete=$((complete + 1)) ;;
        active) active=$((active + 1)) ;;
        failed|missing) failed=$((failed + 1)) ;;
        unreachable) unreachable=$((unreachable + 1)) ;;
        *) failed=$((failed + 1)) ;;
      esac
      printf '[monitor] %-8s gpu%-2s %-11s epoch=%s/%s %s\n' \
        "$host" "$gpu" "$state" "$epoch" "$EXPECTED_EPOCHS" "$(basename "$config")"
    done < "$SCHEDULE"
    echo "[monitor] summary complete=$complete active=$active unreachable=$unreachable failed=$failed"
    if ((failed > 0)); then
      echo "[monitor] stopping because a run failed or disappeared" >&2
      return 1
    fi
    if ((complete == EXPECTED_RUNS)); then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

collect_results() {
  scripts/cluster_collect_results.sh \
    --repo-paths "$REPO_PATHS" \
    --remote-run-root "$RUN_ROOT" \
    --collect-root "$COLLECT_ROOT" \
    --date-prefix "$DATE_PREFIX" \
    --ssh-connect-timeout "$SSH_CONNECT_TIMEOUT"
}

verify_and_build_run_args() {
  local -n output_args=$1
  local raw_line line host gpu config extra job_dir seed_dir checkpoint_count
  local verified=0
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line_is_blank "$line" && continue
    read -r host gpu config extra <<<"$line"
    job_dir="$(job_dir_for "$host" "$gpu" "$config")"
    seed_dir="$COLLECTED_RUNS_ROOT/$job_dir/seed_0"
    for required in validationGraphs_adj_.npy heldoutTestGraphs_adj_.npy; do
      if [[ ! -f "$seed_dir/$required" ]]; then
        echo "[verify] missing $seed_dir/$required" >&2
        return 1
      fi
    done
    checkpoint_count="$(find "$seed_dir" -maxdepth 1 -type f -name 'periodic_epoch_*.pt' | wc -l)"
    if [[ "$checkpoint_count" -ne "$EXPECTED_CHECKPOINTS_PER_RUN" ]]; then
      echo "[verify] expected $EXPECTED_CHECKPOINTS_PER_RUN checkpoints in $seed_dir; found $checkpoint_count" >&2
      return 1
    fi
    for epoch in 04000 08000 12000 16000 20000; do
      if [[ ! -f "$seed_dir/periodic_epoch_${epoch}.pt" ]]; then
        echo "[verify] missing $seed_dir/periodic_epoch_${epoch}.pt" >&2
        return 1
      fi
    done
    output_args+=(--run-dir "$seed_dir")
    verified=$((verified + 1))
    echo "[verify] $job_dir: $checkpoint_count checkpoints"
  done < "$SCHEDULE"
  if [[ "$verified" -ne "$EXPECTED_RUNS" ]]; then
    echo "[verify] expected $EXPECTED_RUNS runs; verified $verified" >&2
    return 1
  fi
}

run_posthoc_selection() {
  local -a run_args=()
  verify_and_build_run_args run_args
  mkdir -p "$POSTHOC_OUTPUT_DIR"
  "$SELECTOR_PYTHON" "$SELECTOR" evaluate \
    "${run_args[@]}" \
    --output-json "$CANDIDATE_JSON" \
    --validation-rollouts 10 \
    --validation-seed 20260714 \
    --gin-runs 10 \
    --gin-seed 0 \
    --device cpu \
    --torch-threads 2 \
    --expected-checkpoints-per-run "$EXPECTED_CHECKPOINTS_PER_RUN"
  "$SELECTOR_PYTHON" "$SELECTOR" finalize \
    --candidate-json "$CANDIDATE_JSON" \
    --output-dir "$POSTHOC_OUTPUT_DIR" \
    --expected-candidates 20 \
    --expected-runs "$EXPECTED_RUNS" \
    --test-generation-seed 21260714 \
    --test-gin-runs 10 \
    --test-gin-seed 0 \
    --device cpu \
    --torch-threads 2
}

main() {
  preflight
  wait_for_training
  echo "[pipeline] all training jobs completed"
  collect_results
  echo "[pipeline] collected artifacts under $COLLECTED_RUNS_ROOT"
  run_posthoc_selection
  echo "[pipeline] complete: $POSTHOC_OUTPUT_DIR/report.md"
}

main "$@"
