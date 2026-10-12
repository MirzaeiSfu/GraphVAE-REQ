#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || ($# - 4) % 2 != 0 )); then
  echo "usage: $0 REPO CACHE_DIR OUTPUT_ROOT GPU_ID LABEL RUN_DIR [LABEL RUN_DIR ...]" >&2
  exit 2
fi

repo=$1
cache_dir=$2
output_root=$3
gpu_id=$4
shift 4

source /local-scratch2/mirzaei/miniconda3/etc/profile.d/conda.sh
conda activate micro
cd "$repo"
mkdir -p "$output_root"

while (( $# )); do
  label=$1
  run_dir=$2
  shift 2
  echo "[$(date --iso-8601=seconds)] START $label $run_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" python \
    scripts/evaluate_attributed_graph_realism_checkpoints_fixed.py \
    --run-dir "$run_dir" \
    --dataset-cache-dir "$cache_dir" \
    --modes topology_control decoded_node \
    --repeats 10 \
    --max-graphs 1000 \
    --generation-batch-size 16 \
    --generation-seed 12345 \
    --evaluator-seed 0 \
    --device auto \
    --output-dir "$output_root/$label"
  echo "[$(date --iso-8601=seconds)] DONE $label"
done
