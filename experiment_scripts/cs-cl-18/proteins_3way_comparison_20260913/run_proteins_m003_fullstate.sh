#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/proteins_motif003_fullstate_trainref_20260913
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
script=/local-scratch2/mirzaei/eval-count-v5-20260831/scripts/evaluate_motif_count_distance_correlation.py
shared=/local-scratch2/mirzaei/normalized_rule_correlation_assets/shared
base=/local-scratch2/mirzaei/motif_corr_new_20260912/proteins

for seed in 0 1 2; do
  "$py" -u "$script" \
    --config "$base/seed_$seed/run_config_used.yaml" \
    --motif-selection-config "$shared/proteins_selection.yaml" \
    --checkpoint "$base/seed_$seed/best_validation_mmd_model" \
    --dataset-cache "$base/seed_$seed/dataset.pkl" \
    --motif-cache-dir "$base/motif_cache" \
    --output "$root/seed_$seed.json" \
    --seed "$seed" \
    --device cpu \
    --generation-batch-size 32 \
    --count-batch-size 128 \
    > "$root/seed_$seed.log" 2>&1 &
done

wait
date -Is > "$root/COMPLETE"
