#!/usr/bin/env bash
set -euo pipefail

# Complete the three Lobster sweep waves, collect them, and run leakage-free
# post-training checkpoint selection. Wave 1 may already be active.

DATE_PREFIX="${DATE_PREFIX:-20260714}"
REPO_PATHS="${REPO_PATHS:-CLUSTER_REPO_PATHS_LOBSTER_POSTHOC.txt}"
PYTHON_PATHS="${PYTHON_PATHS:-CLUSTER_MICRO_PYTHON_PATHS.txt}"
SELECTOR_PYTHON="${SELECTOR_PYTHON:-/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
COLLECT_ROOT="${COLLECT_ROOT:-collected_runs}"

repo_for_host() {
  awk -v host="$1" '$1 == host { print $2; exit }' "$REPO_PATHS"
}

session_for_row() {
  local host="$1" gpu="$2" config="$3"
  local name
  name="$(basename "$config" .yaml)"
  printf '%s_%s__%s_gpu%s' "$DATE_PREFIX" "$name" "$host" "$gpu"
}

wait_for_wave() {
  local wave="$1" schedule="$2" run_root="$3"
  while true; do
    local active=0 failures=0
    while read -r host gpu config extra; do
      [[ -z "${host:-}" || "$host" == \#* ]] && continue
      local repo name session run_dir result
      repo="$(repo_for_host "$host")"
      name="$(basename "$config" .yaml)"
      session="$(session_for_row "$host" "$gpu" "$config")"
      run_dir="$repo/$run_root/${name}__${host}_gpu${gpu}"
      result="$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$host" \
        "if tmux has-session -t '$session' 2>/dev/null; then echo active; \
         elif grep -q 'trainning time:' '$run_dir/seed_0/stdout.log' 2>/dev/null || grep -q 'trainning time:' '$run_dir/stdout.log' 2>/dev/null; then echo complete; \
         elif [[ -e '$run_dir/stdout.log' ]]; then echo failed; \
         else echo pending; fi")"
      case "$result" in
        active) active=$((active + 1)) ;;
        pending) active=$((active + 1)) ;;
        failed)
          echo "[$wave] missing active session or completion marker: $host gpu$gpu $config" >&2
          failures=$((failures + 1))
          ;;
      esac
    done < "$schedule"
    echo "[$wave] active=$active failures=$failures"
    ((failures == 0)) || return 1
    ((active == 0)) && return 0
    sleep "$POLL_SECONDS"
  done
}

collect_wave() {
  local wave="$1" run_root="$2" repo_paths="${3:-$REPO_PATHS}"
  scripts/cluster_collect_results.sh \
    --repo-paths "$repo_paths" \
    --remote-run-root "$run_root" \
    --collect-root "$COLLECT_ROOT" \
    --date-prefix "$DATE_PREFIX"
  echo "[$wave] collection complete"
}

launch_wave() {
  local wave="$1" schedule="$2" run_root="$3"
  while read -r host gpu config extra; do
    [[ -z "${host:-}" || "$host" == \#* ]] && continue
    local repo name session run_dir state one_row
    repo="$(repo_for_host "$host")"
    name="$(basename "$config" .yaml)"
    session="$(session_for_row "$host" "$gpu" "$config")"
    run_dir="$repo/$run_root/${name}__${host}_gpu${gpu}"
    state="$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$host" \
      "if tmux has-session -t '$session' 2>/dev/null; then echo active; \
       elif grep -q 'trainning time:' '$run_dir/stdout.log' 2>/dev/null; then echo complete; \
       else echo pending; fi")"
    if [[ "$state" != pending ]]; then
      echo "[$wave] skip $state row: $host gpu$gpu $config"
      continue
    fi
    one_row="$(mktemp)"
    printf '%s %s %s\n' "$host" "$gpu" "$config" > "$one_row"
    scripts/cluster_run_schedule.sh \
      --repo-paths "$REPO_PATHS" \
      --schedule "$one_row" \
      --date-prefix "$DATE_PREFIX" \
      --run-root "$run_root" \
      --python-paths "$PYTHON_PATHS"
    rm -f "$one_row"
  done < "$schedule"
  echo "[$wave] launch complete"
}

W1_ROOT="runs/$DATE_PREFIX/lobster_posthoc_sweep_wave1"
W2_ROOT="runs/$DATE_PREFIX/lobster_posthoc_sweep_wave2"
W3_ROOT="runs/$DATE_PREFIX/lobster_posthoc_sweep_wave3"

wait_for_wave wave1 CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_WAVE1.txt "$W1_ROOT"
collect_wave wave1 "$W1_ROOT"
wait_for_wave wave2 CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_WAVE2.txt "$W2_ROOT"
collect_wave wave2 "$W2_ROOT"
wait_for_wave wave3 CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_WAVE3.txt "$W3_ROOT"
collect_wave wave3 "$W3_ROOT" CLUSTER_REPO_PATHS_LOBSTER_POSTHOC_WAVE3.txt

"$SELECTOR_PYTHON" scripts/select_lobster_checkpoints.py \
  --runs-root "$COLLECT_ROOT/$DATE_PREFIX/lobster_posthoc_sweep_wave1" \
  --runs-root "$COLLECT_ROOT/$DATE_PREFIX/lobster_posthoc_sweep_wave2" \
  --runs-root "$COLLECT_ROOT/$DATE_PREFIX/lobster_posthoc_sweep_wave3" \
  --output-dir "$COLLECT_ROOT/$DATE_PREFIX/lobster_posthoc_selection" \
  --validation-rollouts 10 \
  --test-rollouts 50 \
  --device cpu

echo "[pipeline] complete"
