#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/proteins_false_fullstate_trainref_20260913
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
script=/local-scratch2/mirzaei/eval-count-v5-20260831/scripts/evaluate_motif_count_distance_correlation.py
shared=/local-scratch2/mirzaei/normalized_rule_correlation_assets/shared
models=/local-scratch2/mirzaei/normalized_rule_correlation_assets/proteins/false
current=/local-scratch2/mirzaei/motif_corr_new_20260912/proteins

mkdir -p "$root"
for seed in 0 1 2; do
  "$py" -u "$script" \
    --config "$shared/proteins_false.yaml" \
    --motif-selection-config "$shared/proteins_selection.yaml" \
    --checkpoint "$models/seed_$seed.model" \
    --dataset-cache "$current/seed_0/dataset.pkl" \
    --motif-cache-dir "$shared" \
    --output "$root/seed_$seed.json" \
    --seed "$seed" \
    --device cpu \
    --generation-batch-size 32 \
    --count-batch-size 128 \
    > "$root/seed_$seed.log" 2>&1 &
done

wait
date -Is > "$root/COMPLETE"
