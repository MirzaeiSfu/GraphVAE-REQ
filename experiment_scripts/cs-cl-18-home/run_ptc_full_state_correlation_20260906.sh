#!/usr/bin/env bash
set -euo pipefail

repo=/local-scratch2/mirzaei/count_distance_manifest_fix/GraphVAE-REQ
python_bin=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
root=/local-scratch2/mirzaei/ptc_full_state_correlation_20260906
metric_root=/local-scratch2/mirzaei/ptc_rule_metrics_20260906
true_root=/local-scratch2/mirzaei/motif_true_clean_20260906/ptc
false_root=/local-scratch2/new/gather/datasets/ptc/setting_01
dataset_cache="$metric_root/cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl"
motif_cache=/local-scratch2/mirzaei/fb/GraphVAE-REQ/cache_motifs/multihop_smoothed
selection_config="$metric_root/configs/ptc_full_state_correlation.yaml"
evaluator="$repo/scripts/evaluate_motif_count_distance.py"

mkdir -p "$root/results" "$root/logs"
cd "$repo"

run_one() {
  local setting=$1 seed=$2 model_config=$3 checkpoint=$4
  local output="$root/results/${setting}_seed${seed}.json"
  local log="$root/logs/${setting}_seed${seed}.log"
  if [[ -s "$output" ]]; then
    echo "[skip] $setting seed $seed"
    return
  fi
  echo "[start] $setting seed $seed $(date -Is)"
  "$python_bin" -u "$evaluator" \
    --config "$model_config" \
    --motif-selection-config "$selection_config" \
    --checkpoint "$checkpoint" \
    --dataset-cache "$dataset_cache" \
    --motif-cache-dir "$motif_cache" \
    --output "$output" \
    --seed "$seed" --device cuda:0 \
    --generation-batch-size 4 --count-batch-size 4 \
    --adj-threshold 0.5 --setting "${setting}_full_state" \
    --dataset-label PTC >"$log" 2>&1
  echo "[done] $setting seed $seed $(date -Is)"
}

for seed in 0 1 2; do
  run_one false "$seed" \
    "$false_root/seed_${seed}/run_config_used.yaml" \
    "$false_root/seed_${seed}/best_validation_mmd_model"
done
for mode in total_count full_matrix; do
  for seed in 0 1 2; do
    run_one "$mode" "$seed" \
      "$metric_root/configs/ptc_${mode}_effective.yaml" \
      "$true_root/$mode/seed_${seed}/best_validation_mmd_model"
  done
done
echo "ALL_COMPLETE $(date -Is)"
