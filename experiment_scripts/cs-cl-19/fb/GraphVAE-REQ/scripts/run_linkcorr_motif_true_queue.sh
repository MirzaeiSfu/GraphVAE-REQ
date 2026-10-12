#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 REPO_DIR PYTHON GPU QUEUE_ID MANIFEST OOM_PRUNE_CAP" >&2
    exit 2
fi

repo_dir=$1
python_bin=$2
gpu=$3
queue_id=$4
manifest=$5
oom_prune_cap=$6

campaign=linkcorr_motif_true_3seed
log_dir="$repo_dir/run_logs/$campaign"
status_dir="$repo_dir/run_status/$campaign"
queue_status="$status_dir/${queue_id}.queue.status"

mkdir -p "$log_dir" "$status_dir"

while IFS=$'\t' read -r run_id config_file seed; do
    [[ -n "${run_id:-}" ]] || continue
    status_file="$status_dir/${run_id}.status"
    if [[ ! -f "$status_file" ]] || ! grep -q '^COMPLETE ' "$status_file"; then
        printf 'QUEUED host=%s gpu=%s seed=%s queue=%s queued=%s\n' \
            "$(hostname)" "$gpu" "$seed" "$queue_id" \
            "$(date --iso-8601=seconds)" > "$status_file"
    fi
done < "$manifest"

printf 'RUNNING host=%s gpu=%s started=%s\n' \
    "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" > "$queue_status"

cd "$repo_dir" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1

failures=0
while IFS=$'\t' read -r run_id config_file seed; do
    [[ -n "${run_id:-}" ]] || continue
    log_file="$log_dir/${run_id}.log"
    status_file="$status_dir/${run_id}.status"

    if [[ -f "$status_file" ]] && grep -q '^COMPLETE ' "$status_file"; then
        echo "[$run_id] already complete; skipping"
        continue
    fi

    printf 'RUNNING host=%s gpu=%s seed=%s queue=%s attempt=1 started=%s\n' \
        "$(hostname)" "$gpu" "$seed" "$queue_id" \
        "$(date --iso-8601=seconds)" > "$status_file"

    "$python_bin" main.py --config "$config_file" --seed "$seed" \
        2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    attempt=1

    if [[ $exit_code -ne 0 ]] && \
       grep -Eqi 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "$log_file"; then
        attempt=2
        printf 'RETRYING_OOM host=%s gpu=%s seed=%s queue=%s attempt=2 prune_cap=%s motif_batch_size=1 started=%s\n' \
            "$(hostname)" "$gpu" "$seed" "$queue_id" "$oom_prune_cap" \
            "$(date --iso-8601=seconds)" > "$status_file"
        printf '\n[%s] CUDA OOM: retrying with motif_prune_max_values_per_rule=%s and motif_batch_size=1\n' \
            "$run_id" "$oom_prune_cap" | tee -a "$log_file"
        "$python_bin" main.py --config "$config_file" --seed "$seed" \
            --motif_prune_max_values_per_rule "$oom_prune_cap" \
            --motif_batch_size 1 2>&1 | tee -a "$log_file"
        exit_code=${PIPESTATUS[0]}
    fi

    if [[ $exit_code -eq 0 ]]; then
        state=COMPLETE
    else
        state=FAILED
        failures=$((failures + 1))
    fi

    printf '%s host=%s gpu=%s seed=%s queue=%s attempt=%s exit_code=%s finished=%s\n' \
        "$state" "$(hostname)" "$gpu" "$seed" "$queue_id" "$attempt" \
        "$exit_code" "$(date --iso-8601=seconds)" > "$status_file"
    echo "[$run_id] $state (attempt $attempt, exit code $exit_code)"
done < "$manifest"

if [[ $failures -eq 0 ]]; then
    queue_state=COMPLETE
else
    queue_state=COMPLETE_WITH_FAILURES
fi
printf '%s host=%s gpu=%s failures=%s finished=%s\n' \
    "$queue_state" "$(hostname)" "$gpu" "$failures" \
    "$(date --iso-8601=seconds)" > "$queue_status"

exec bash
