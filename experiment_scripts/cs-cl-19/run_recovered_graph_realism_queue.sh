#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 || ($# - 2) % 2 != 0 )); then
  echo "usage: $0 REPO GPU_ID LABEL RUN_DIR [LABEL RUN_DIR ...]" >&2
  exit 2
fi

repo=$1
gpu_id=$2
shift 2
source /local-scratch2/mirzaei/miniconda3/etc/profile.d/conda.sh
conda activate micro
cd "$repo"

while (( $# )); do
  label=$1
  run_dir=$2
  shift 2
  seed=${label##*_s}
  echo "[$(date --iso-8601=seconds)] START $label"
  CUDA_VISIBLE_DEVICES="$gpu_id" python -u scripts/evaluate_graph_realism_batch_fixed.py \
    --run-dir "$run_dir" \
    --json-filename graph_realism_random_gin_recovered.json \
    --summary-csv "$run_dir/graph_realism_recovered_summary.csv" \
    --repeats 10 \
    --max-graphs 1000 \
    --seed "$seed" \
    --device cuda \
    2>&1 | tee "$run_dir/evaluation_recovery_20260902.log"
  echo "[$(date --iso-8601=seconds)] DONE $label"
done
