#!/usr/bin/env bash
# Count every available collection (skip existing counts and missing inputs) for the synthetic fix.
set -uo pipefail
R=/local-scratch2/mirzaei/rule_mmd_20260923; P=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
export CUDA_VISIBLE_DEVICES=${1:-0}
for DS in GRID TRIANGULAR_GRID LOBSTER; do
  D=$R/datasets_fix/$DS
  for n in train test; do [ -s $D/counts/$n.npz ] || $P $R/rule_mmd.py count --dataset-dir $D --device cuda --name $n --output $D/counts/$n.npz; done
  while IFS=$'\t' read -r name path; do
    [ -s $D/counts/$name.npz ] || $P $R/rule_mmd.py count --dataset-dir $D --device cuda --name $name --generated $path --output $D/counts/$name.npz
  done < <($P $R/synthetic_fix/list_items.py $D)
done
echo DONE $(date)
