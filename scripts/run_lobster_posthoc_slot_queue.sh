#!/usr/bin/env bash
set -euo pipefail

# Advance one host/GPU independently through the three Lobster sweep schedules.
# The wave-level pipeline remains responsible for collection and selection.

if (($# != 2)); then
  echo "Usage: $0 HOST GPU" >&2
  exit 2
fi

HOST="$1"
GPU="$2"
DATE_PREFIX="${DATE_PREFIX:-20260714}"
REPO_PATHS="${REPO_PATHS:-CLUSTER_REPO_PATHS_LOBSTER_POSTHOC.txt}"
PYTHON_PATHS="${PYTHON_PATHS:-CLUSTER_MICRO_PYTHON_PATHS.txt}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REPO="$(awk -v host="$HOST" '$1 == host { print $2; exit }' "$REPO_PATHS")"

[[ -n "$REPO" ]] || { echo "No repo path for $HOST" >&2; exit 2; }

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9_.-]/_/g; s/^[._-]*//; s/[._-]*$//'
}

for wave in 1 2 3; do
  schedule="CLUSTER_GPU_CONFIGS_LOBSTER_POSTHOC_WAVE${wave}.txt"
  row="$(awk -v host="$HOST" -v gpu="$GPU" '$1 == host && $2 == gpu { print; exit }' "$schedule")"
  [[ -n "$row" ]] || continue
  read -r host gpu config <<< "$row"
  name="$(basename "$config" .yaml)"
  run_root="runs/$DATE_PREFIX/lobster_posthoc_sweep_wave${wave}"
  run_dir="$REPO/$run_root/${name}__${host}_gpu${gpu}"
  session="$(sanitize "${DATE_PREFIX}_${name}__${host}_gpu${gpu}")"

  while true; do
    state="$(ssh -n -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$HOST" \
      "if tmux has-session -t '$session' 2>/dev/null; then echo active; \
       elif grep -q 'trainning time:' '$run_dir/stdout.log' 2>/dev/null; then echo complete; \
       elif [[ -e '$run_dir/stdout.log' ]]; then echo failed; \
       else echo pending; fi")"
    case "$state" in
      complete)
        echo "[slot $HOST gpu$GPU] wave$wave complete"
        break
        ;;
      active)
        echo "[slot $HOST gpu$GPU] wave$wave active"
        sleep "$POLL_SECONDS"
        ;;
      pending)
        one_row="$(mktemp)"
        printf '%s\n' "$row" > "$one_row"
        scripts/cluster_run_schedule.sh \
          --repo-paths "$REPO_PATHS" \
          --schedule "$one_row" \
          --date-prefix "$DATE_PREFIX" \
          --run-root "$run_root" \
          --python-paths "$PYTHON_PATHS"
        rm -f "$one_row"
        echo "[slot $HOST gpu$GPU] wave$wave launched"
        sleep "$POLL_SECONDS"
        ;;
      *)
        echo "[slot $HOST gpu$GPU] wave$wave failed or unknown state: $state" >&2
        exit 1
        ;;
    esac
  done
done

echo "[slot $HOST gpu$GPU] queue complete"
