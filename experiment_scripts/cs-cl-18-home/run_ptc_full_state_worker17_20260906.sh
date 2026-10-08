#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 GPU SETTING:SEED [...]" >&2
  exit 2
fi
gpu=$1
shift
root=/localhome/mirzaei/ali/ptc_full_state_worker_20260906
python_bin=/localhome/mirzaei/miniconda3/envs/micro/bin/python
dataset_cache="$root/cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl"
selection_config="$root/configs/ptc_full_state_correlation.yaml"
export CUDA_VISIBLE_DEVICES="$gpu"
cd "$root/code"

for item in "$@"; do
  setting=${item%%:*}
  seed=${item##*:}
  output="$root/results/${setting}_seed${seed}.json"
  log="$root/logs/${setting}_seed${seed}.log"
  if [[ -s "$output" ]]; then
    echo "[skip] $setting seed $seed"
    continue
  fi
  if [[ "$setting" == false ]]; then
    model_config="$root/inputs/false/seed_${seed}/run_config_used.yaml"
    checkpoint="$root/inputs/false/seed_${seed}/best_validation_mmd_model"
  else
    model_config="$root/configs/ptc_${setting}_effective.yaml"
    checkpoint="$root/inputs/true/${setting}/seed_${seed}/best_validation_mmd_model"
  fi
  echo "[start] $setting seed $seed $(date -Is)"
  "$python_bin" -u scripts/evaluate_motif_count_distance.py \
    --config "$model_config" \
    --motif-selection-config "$selection_config" \
    --checkpoint "$checkpoint" \
    --dataset-cache "$dataset_cache" \
    --motif-cache-dir "$root/cache" \
    --output "$output" --seed "$seed" --device cuda:0 \
    --generation-batch-size 32 --count-batch-size 32 \
    --adj-threshold 0.5 --setting "${setting}_full_state" \
    --dataset-label PTC >"$log" 2>&1
  echo "[done] $setting seed $seed $(date -Is)"
done

echo "QUEUE_COMPLETE gpu=$gpu $(date -Is)"
