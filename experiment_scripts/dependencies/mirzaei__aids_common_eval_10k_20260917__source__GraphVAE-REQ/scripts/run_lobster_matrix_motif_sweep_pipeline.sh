#!/usr/bin/env bash
set -euo pipefail

# Monitor the already-launched Lobster matrix-motif sweep, collect all worker
# artifacts, verify the complete checkpoint inventory, and run leakage-free
# per-run posthoc selection. This controller never launches or relaunches jobs.

DATE_PREFIX="${DATE_PREFIX:-20260718}"
SCHEDULE="${SCHEDULE:-CLUSTER_GPU_CONFIGS_LOBSTER_MATRIX_MOTIF_SWEEP.txt}"
REPO_PATHS="${REPO_PATHS:-CLUSTER_REPO_PATHS_LOBSTER_MATRIX_MOTIF_SWEEP.txt}"
MANIFEST="${MANIFEST:-configs/matrix_motif/matrix_weight_sweep_manifest.csv}"
RUN_ROOT="${RUN_ROOT:-runs/$DATE_PREFIX/lobster_matrix_motif_weight_sweep}"
COLLECT_ROOT="${COLLECT_ROOT:-collected_runs}"
POSTHOC_OUTPUT_DIR="${POSTHOC_OUTPUT_DIR:-$COLLECT_ROOT/$DATE_PREFIX/lobster_matrix_motif_posthoc_selection}"
SELECTOR="${SELECTOR:-scripts/select_lobster_checkpoints_per_run.py}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"
EXPECTED_CHECKPOINTS_PER_RUN="${EXPECTED_CHECKPOINTS_PER_RUN:-5}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-20000}"
VALIDATION_ROLLOUTS="${VALIDATION_ROLLOUTS:-10}"
VALIDATION_SEED="${VALIDATION_SEED:-20260714}"
THIRD_PARTY_REPEATS="${THIRD_PARTY_REPEATS:-10}"
THIRD_PARTY_MAX_GRAPHS="${THIRD_PARTY_MAX_GRAPHS:-1000}"
THIRD_PARTY_SEED="${THIRD_PARTY_SEED:-0}"
THIRD_PARTY_DEVICE="${THIRD_PARTY_DEVICE:-auto}"

RUN_ROOT_CLEAN="${RUN_ROOT%/}"
RUN_ROOT_NAME="$(basename "$RUN_ROOT_CLEAN")"
COLLECTED_RUNS_ROOT="$COLLECT_ROOT/$DATE_PREFIX/$RUN_ROOT_NAME"
SSH_OPTS=(
  -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT"
  -o StrictHostKeyChecking=accept-new
)

declare -A REPO_BY_HOST
declare -A EXPECTED_JOB_DIRS
declare -A SCHEDULED_CONFIGS
declare -A SCHEDULED_SLOTS

quote_words() {
  local quoted=()
  local word
  local word_q
  for word in "$@"; do
    printf -v word_q '%q' "$word"
    quoted+=("$word_q")
  done
  printf '%s' "${quoted[*]}"
}

quote_one() {
  printf '%q' "$1"
}

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._-]*//; s/[._-]*$//'
}

line_is_blank() {
  [[ -z "${1//[[:space:]]/}" ]]
}

experiment_name_for() {
  basename "$1" .yaml
}

job_dir_for() {
  local host="$1"
  local gpu="$2"
  local config="$3"
  local experiment_name
  local host_token
  local gpu_token
  experiment_name="$(sanitize "$(experiment_name_for "$config")")"
  host_token="$(sanitize "$host")"
  gpu_token="$(sanitize "$gpu")"
  printf '%s__%s_gpu%s' "$experiment_name" "$host_token" "$gpu_token"
}

session_for() {
  local host="$1"
  local gpu="$2"
  local config="$3"
  sanitize "${DATE_PREFIX}_$(job_dir_for "$host" "$gpu" "$config")"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer; received: $value" >&2
    exit 2
  fi
}

load_repo_paths() {
  local raw_line
  local line
  local host
  local repo_path
  local extra

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

validate_schedule() {
  local raw_line
  local line
  local host
  local gpu
  local config
  local extra
  local repo_path
  local slot
  local job_dir
  local row_count=0
  local -a manifest_configs=()
  local -a schedule_configs=()
  local index

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line_is_blank "$line" && continue
    read -r host gpu config extra <<<"$line"
    if [[ -z "${host:-}" || -z "${gpu:-}" || -z "${config:-}" || -n "${extra:-}" ]]; then
      echo "[preflight] invalid schedule row: $raw_line" >&2
      exit 2
    fi
    repo_path="${REPO_BY_HOST[$host]:-}"
    if [[ -z "$repo_path" ]]; then
      echo "[preflight] no worker repo path for $host" >&2
      exit 2
    fi
    if [[ ! -f "$config" ]]; then
      echo "[preflight] missing local config: $config" >&2
      exit 2
    fi
    slot="$host:$gpu"
    if [[ -n "${SCHEDULED_SLOTS[$slot]:-}" ]]; then
      echo "[preflight] duplicate worker/GPU slot: $slot" >&2
      exit 2
    fi
    if [[ -n "${SCHEDULED_CONFIGS[$config]:-}" ]]; then
      echo "[preflight] duplicate scheduled config: $config" >&2
      exit 2
    fi
    job_dir="$(job_dir_for "$host" "$gpu" "$config")"
    if [[ -n "${EXPECTED_JOB_DIRS[$job_dir]:-}" ]]; then
      echo "[preflight] duplicate output job directory: $job_dir" >&2
      exit 2
    fi
    SCHEDULED_SLOTS["$slot"]=1
    SCHEDULED_CONFIGS["$config"]=1
    EXPECTED_JOB_DIRS["$job_dir"]=1
    schedule_configs+=("$config")
    row_count=$((row_count + 1))
  done < "$SCHEDULE"

  if ((row_count != EXPECTED_RUNS)); then
    echo "[preflight] expected $EXPECTED_RUNS schedule rows, found $row_count" >&2
    exit 2
  fi

  mapfile -t manifest_configs < <(
    awk -F, 'NR > 1 {sub(/\r$/, "", $1); print $1}' "$MANIFEST" | sort
  )
  mapfile -t schedule_configs < <(printf '%s\n' "${schedule_configs[@]}" | sort)
  if ((${#manifest_configs[@]} != EXPECTED_RUNS)); then
    echo "[preflight] expected $EXPECTED_RUNS configs in $MANIFEST, found ${#manifest_configs[@]}" >&2
    exit 2
  fi
  for ((index = 0; index < EXPECTED_RUNS; index++)); do
    if [[ "${manifest_configs[$index]}" != "${schedule_configs[$index]}" ]]; then
      echo "[preflight] schedule does not match the matrix sweep manifest" >&2
      echo "[preflight] manifest: ${manifest_configs[$index]}" >&2
      echo "[preflight] schedule: ${schedule_configs[$index]}" >&2
      exit 2
    fi
  done
}

preflight() {
  require_positive_integer POLL_SECONDS "$POLL_SECONDS"
  require_positive_integer SSH_CONNECT_TIMEOUT "$SSH_CONNECT_TIMEOUT"
  require_positive_integer EXPECTED_RUNS "$EXPECTED_RUNS"
  require_positive_integer EXPECTED_CHECKPOINTS_PER_RUN "$EXPECTED_CHECKPOINTS_PER_RUN"
  require_positive_integer EXPECTED_EPOCHS "$EXPECTED_EPOCHS"
  require_positive_integer VALIDATION_ROLLOUTS "$VALIDATION_ROLLOUTS"
  require_positive_integer THIRD_PARTY_REPEATS "$THIRD_PARTY_REPEATS"
  require_positive_integer THIRD_PARTY_MAX_GRAPHS "$THIRD_PARTY_MAX_GRAPHS"

  for required_file in "$SCHEDULE" "$REPO_PATHS" "$MANIFEST" "$SELECTOR"; do
    if [[ ! -f "$required_file" ]]; then
      echo "[preflight] missing required file: $required_file" >&2
      exit 2
    fi
  done
  if [[ ! -x scripts/cluster_collect_results.sh ]]; then
    echo "[preflight] collector is missing or not executable: scripts/cluster_collect_results.sh" >&2
    exit 2
  fi
  if [[ "$SELECTOR_PYTHON" == */* ]]; then
    if [[ ! -x "$SELECTOR_PYTHON" ]]; then
      echo "[preflight] Python is missing or not executable: $SELECTOR_PYTHON" >&2
      exit 2
    fi
  elif ! command -v "$SELECTOR_PYTHON" >/dev/null 2>&1; then
    echo "[preflight] Python command not found: $SELECTOR_PYTHON" >&2
    exit 2
  fi
  if [[ "$THIRD_PARTY_DEVICE" != auto && "$THIRD_PARTY_DEVICE" != cpu && "$THIRD_PARTY_DEVICE" != cuda ]]; then
    echo "[preflight] THIRD_PARTY_DEVICE must be auto, cpu, or cuda" >&2
    exit 2
  fi

  load_repo_paths
  validate_schedule
  echo "[preflight] exact $EXPECTED_RUNS-run matrix sweep inventory verified"
  echo "[preflight] remote run root: $RUN_ROOT_CLEAN"
  echo "[preflight] collected run root: $COLLECTED_RUNS_ROOT"
  echo "[preflight] posthoc output: $POSTHOC_OUTPUT_DIR"
}

probe_remote_job() {
  local host="$1"
  local session="$2"
  local primary_log="$3"
  local fallback_log="$4"
  local probe_script
  local remote_command

  probe_script=$(cat <<'EOF'
session=$1
primary_log=$2
fallback_log=$3
log_path=$primary_log
if [[ ! -f "$log_path" && -f "$fallback_log" ]]; then
  log_path=$fallback_log
fi

epoch=0
if [[ -f "$log_path" ]]; then
  found_epoch=$(sed -n 's/.*Epoch: *\([0-9][0-9]*\).*/\1/p' "$log_path" | tail -n 1)
  if [[ -n "$found_epoch" ]]; then
    epoch=$((10#$found_epoch))
  fi
fi

if tmux has-session -t "$session" 2>/dev/null; then
  state=active
elif [[ -f "$log_path" ]] && grep -q 'trainning time:' "$log_path"; then
  state=complete
elif [[ -f "$log_path" ]]; then
  state=failed
else
  state=missing
fi
printf '%s\t%s\t%s\n' "$state" "$epoch" "$log_path"
EOF
)
  remote_command="$(quote_words bash -c "$probe_script" _ "$session" "$primary_log" "$fallback_log")"
  ssh -n "${SSH_OPTS[@]}" "$host" "$remote_command"
}

show_remote_log_tail() {
  local host="$1"
  local log_path="$2"
  local remote_command
  remote_command="$(quote_words tail -n 20 "$log_path")"
  echo "[failure] last 20 log lines from $host:$log_path" >&2
  ssh -n "${SSH_OPTS[@]}" "$host" "$remote_command" >&2 || true
}

wait_for_training() {
  local raw_line
  local line
  local host
  local gpu
  local config
  local extra
  local repo_path
  local job_dir
  local session
  local remote_run_dir
  local result
  local state
  local epoch
  local log_path
  local complete
  local active
  local failed
  local unreachable

  echo "[monitor] polling already-launched jobs; this script never relaunches them"
  while true; do
    complete=0
    active=0
    failed=0
    unreachable=0

    while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
      line="${raw_line%%#*}"
      line_is_blank "$line" && continue
      read -r host gpu config extra <<<"$line"
      repo_path="${REPO_BY_HOST[$host]}"
      job_dir="$(job_dir_for "$host" "$gpu" "$config")"
      session="$(session_for "$host" "$gpu" "$config")"
      remote_run_dir="${repo_path%/}/$RUN_ROOT_CLEAN/$job_dir"

      if result="$(probe_remote_job \
        "$host" \
        "$session" \
        "$remote_run_dir/stdout.log" \
        "$remote_run_dir/seed_0/stdout.log")"; then
        IFS=$'\t' read -r state epoch log_path <<<"$result"
      else
        state=unreachable
        epoch=?
        log_path="$remote_run_dir/stdout.log"
      fi

      case "$state" in
        complete)
          complete=$((complete + 1))
          ;;
        active)
          active=$((active + 1))
          ;;
        failed|missing)
          failed=$((failed + 1))
          ;;
        unreachable)
          unreachable=$((unreachable + 1))
          ;;
        *)
          echo "[monitor] invalid status from $host GPU $gpu: $result" >&2
          failed=$((failed + 1))
          state=invalid
          ;;
      esac

      printf '[monitor] %-8s gpu%-2s %-8s epoch=%s/%s %s\n' \
        "$host" "$gpu" "$state" "$epoch" "$EXPECTED_EPOCHS" "$(basename "$config")"
      if [[ "$state" == failed ]]; then
        echo "[failure] tmux session vanished without completion marker: $session" >&2
        echo "[failure] log: $host:$log_path" >&2
        show_remote_log_tail "$host" "$log_path"
      elif [[ "$state" == missing ]]; then
        echo "[failure] neither tmux session nor training log exists: $session" >&2
        echo "[failure] expected log: $host:$log_path" >&2
      fi
    done < "$SCHEDULE"

    echo "[monitor] summary complete=$complete active=$active unreachable=$unreachable failed=$failed"
    if ((failed > 0)); then
      echo "[monitor] stopping because at least one launched job failed or disappeared" >&2
      return 1
    fi
    if ((complete == EXPECTED_RUNS)); then
      echo "[monitor] all $EXPECTED_RUNS jobs completed successfully"
      return 0
    fi
    if ((complete + active + unreachable != EXPECTED_RUNS)); then
      echo "[monitor] status accounting did not cover all expected jobs" >&2
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

collect_results() {
  echo "[collect] collecting completed worker runs"
  scripts/cluster_collect_results.sh \
    --repo-paths "$REPO_PATHS" \
    --remote-run-root "$RUN_ROOT_CLEAN" \
    --collect-root "$COLLECT_ROOT" \
    --date-prefix "$DATE_PREFIX" \
    --ssh-connect-timeout "$SSH_CONNECT_TIMEOUT"
  echo "[collect] artifacts collected under $COLLECTED_RUNS_ROOT"
}

verify_collected_results() {
  local raw_line
  local line
  local host
  local gpu
  local config
  local extra
  local job_dir
  local seed_dir
  local epoch_token
  local -a checkpoints
  local -a validation_files
  local verified=0
  local -a expected_checkpoint_epochs=(04000 08000 12000 16000 20000)

  if [[ ! -d "$COLLECTED_RUNS_ROOT" ]]; then
    echo "[verify] collected run root does not exist: $COLLECTED_RUNS_ROOT" >&2
    return 1
  fi

  mapfile -t validation_files < <(
    find "$COLLECTED_RUNS_ROOT" -type f -name validationGraphs_adj_.npy -print | sort
  )
  if ((${#validation_files[@]} != EXPECTED_RUNS)); then
    echo "[verify] expected $EXPECTED_RUNS collected runs, found ${#validation_files[@]} validation sets" >&2
    return 1
  fi

  shopt -s nullglob
  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%%#*}"
    line_is_blank "$line" && continue
    read -r host gpu config extra <<<"$line"
    job_dir="$(job_dir_for "$host" "$gpu" "$config")"
    seed_dir="$COLLECTED_RUNS_ROOT/$job_dir/seed_0"
    if [[ ! -d "$seed_dir" ]]; then
      echo "[verify] missing collected run directory: $seed_dir" >&2
      return 1
    fi
    if [[ ! -f "$seed_dir/validationGraphs_adj_.npy" ]]; then
      echo "[verify] missing validation graphs: $seed_dir/validationGraphs_adj_.npy" >&2
      return 1
    fi
    if [[ ! -f "$seed_dir/heldoutTestGraphs_adj_.npy" ]]; then
      echo "[verify] missing held-out graphs: $seed_dir/heldoutTestGraphs_adj_.npy" >&2
      return 1
    fi
    checkpoints=("$seed_dir"/periodic_epoch_*.pt)
    if ((${#checkpoints[@]} != EXPECTED_CHECKPOINTS_PER_RUN)); then
      echo "[verify] expected $EXPECTED_CHECKPOINTS_PER_RUN checkpoints in $seed_dir, found ${#checkpoints[@]}" >&2
      return 1
    fi
    if ((EXPECTED_CHECKPOINTS_PER_RUN == ${#expected_checkpoint_epochs[@]})); then
      for epoch_token in "${expected_checkpoint_epochs[@]}"; do
        if [[ ! -f "$seed_dir/periodic_epoch_${epoch_token}.pt" ]]; then
          echo "[verify] missing checkpoint: $seed_dir/periodic_epoch_${epoch_token}.pt" >&2
          return 1
        fi
      done
    fi
    verified=$((verified + 1))
    echo "[verify] $job_dir: ${#checkpoints[@]} checkpoints"
  done < "$SCHEDULE"
  shopt -u nullglob

  if ((verified != EXPECTED_RUNS)); then
    echo "[verify] expected to verify $EXPECTED_RUNS runs, verified $verified" >&2
    return 1
  fi
  echo "[verify] verified $verified runs with $EXPECTED_CHECKPOINTS_PER_RUN checkpoints each"
}

run_posthoc_selection() {
  echo "[posthoc] selecting and materializing one validation-best checkpoint per run"
  echo "[posthoc] Random-GIN uses structural features; device=$THIRD_PARTY_DEVICE"
  "$SELECTOR_PYTHON" "$SELECTOR" \
    --runs-root "$COLLECTED_RUNS_ROOT" \
    --output-dir "$POSTHOC_OUTPUT_DIR" \
    --expected-runs "$EXPECTED_RUNS" \
    --expected-checkpoints-per-run "$EXPECTED_CHECKPOINTS_PER_RUN" \
    --validation-rollouts "$VALIDATION_ROLLOUTS" \
    --seed "$VALIDATION_SEED" \
    --device cpu \
    --run-third-party-eval \
    --third-party-repeats "$THIRD_PARTY_REPEATS" \
    --third-party-max-graphs "$THIRD_PARTY_MAX_GRAPHS" \
    --third-party-seed "$THIRD_PARTY_SEED" \
    --third-party-device "$THIRD_PARTY_DEVICE"
  echo "[posthoc] selection, materialization, and Random-GIN evaluation complete"
  echo "[posthoc] report: $POSTHOC_OUTPUT_DIR/report.md"
}

main() {
  preflight
  wait_for_training
  collect_results
  verify_collected_results
  run_posthoc_selection
  echo "[pipeline] complete"
}

main "$@"
