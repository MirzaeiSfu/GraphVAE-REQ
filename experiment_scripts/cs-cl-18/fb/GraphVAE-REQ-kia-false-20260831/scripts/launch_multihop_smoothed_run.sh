#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 REPO_DIR PYTHON CONFIG SEED GPU RUN_ID" >&2
    exit 2
fi

repo_dir=$1
python_bin=$2
config_file=$3
seed=$4
gpu=$5
run_id=$6

mkdir -p "$repo_dir/run_logs/multihop_smoothed" "$repo_dir/run_status/multihop_smoothed"
log_file="$repo_dir/run_logs/multihop_smoothed/$run_id.log"
status_file="$repo_dir/run_status/multihop_smoothed/$run_id.status"

printf 'RUNNING host=%s gpu=%s seed=%s started=%s\n' \
    "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status_file"

cd "$repo_dir" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1

"$python_bin" main.py --config "$config_file" --seed "$seed" 2>&1 | tee "$log_file"
exit_code=${PIPESTATUS[0]}

if [[ $exit_code -eq 0 ]]; then
    state=COMPLETE
else
    state=FAILED
fi

printf '%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
    "$state" "$(hostname)" "$gpu" "$seed" "$exit_code" \
    "$(date --iso-8601=seconds)" > "$status_file"
echo "[$run_id] $state (exit code $exit_code)"

# Keep the tmux pane available so logs and the final state can still be viewed.
exec bash
