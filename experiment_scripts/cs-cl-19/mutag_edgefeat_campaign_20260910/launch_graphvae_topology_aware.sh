#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:-1}
root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
repo=$root/source/GraphVAE-REQ
python_bin=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
campaign=/local-scratch2/mirzaei/mutag_topology_aware_20260912
mkdir -p "$campaign/logs" "$campaign/status"
log=$campaign/logs/seed_${seed}.log
status=$campaign/status/seed_${seed}.status

printf 'RUNNING host=%s gpu=%s seed=%s started=%s\n' \
  "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
"$python_bin" -u main.py \
  --config configs/mutag_edgefeat_campaign/full_matrix_topology_aware.yaml \
  --seed "$seed" \
  --device cuda:0 \
  --graph_save_path "$campaign/full_matrix" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}
if (( exit_code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
  "$state" "$(hostname)" "$gpu" "$seed" "$exit_code" \
  "$(date --iso-8601=seconds)" > "$status"
exit "$exit_code"
