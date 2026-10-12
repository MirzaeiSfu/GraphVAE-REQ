#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/ptc_full_state_correlation_20260906
metric_root=/local-scratch2/mirzaei/ptc_rule_metrics_20260906
true_root=/local-scratch2/mirzaei/motif_true_clean_20260906/ptc
false_root=/local-scratch2/new/gather/datasets/ptc/setting_01
repo=/local-scratch2/mirzaei/count_distance_manifest_fix/GraphVAE-REQ
python_bin=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
dataset_cache="$metric_root/cache/PTC_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-gin-node-label-v2.pkl"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$repo"

for item in "$@"; do
  setting=${item%%:*}; seed=${item##*:}
  output="$root/results/${setting}_seed${seed}.json"
  log="$root/logs/${setting}_seed${seed}.log"
  [[ -s "$output" ]] && { echo "[skip] $setting seed $seed"; continue; }
  if [[ "$setting" == false ]]; then
    config="$false_root/seed_${seed}/run_config_used.yaml"
    checkpoint="$false_root/seed_${seed}/best_validation_mmd_model"
  else
    config="$metric_root/configs/ptc_${setting}_effective.yaml"
    checkpoint="$true_root/$setting/seed_${seed}/best_validation_mmd_model"
  fi
  echo "[start] $setting seed $seed $(date -Is)"
  "$python_bin" -u scripts/evaluate_motif_count_distance.py \
    --config "$config" \
    --motif-selection-config "$metric_root/configs/ptc_full_state_correlation.yaml" \
    --checkpoint "$checkpoint" --dataset-cache "$dataset_cache" \
    --motif-cache-dir /local-scratch2/mirzaei/fb/GraphVAE-REQ/cache_motifs/multihop_smoothed \
    --output "$output" --seed "$seed" --device cuda:0 \
    --generation-batch-size 32 --count-batch-size 32 \
    --adj-threshold 0.5 --setting "${setting}_full_state" \
    --dataset-label PTC >"$log" 2>&1
  echo "[done] $setting seed $seed $(date -Is)"
done
