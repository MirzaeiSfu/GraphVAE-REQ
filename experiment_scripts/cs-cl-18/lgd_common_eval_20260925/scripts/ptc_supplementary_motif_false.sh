#!/usr/bin/env bash
# Supplementary (not part of the stored corrected tables): PTC GraphVAE motif=False structural MMD
# re-scored against the common 70-graph reference with the same evaluator used for DeFoG/LGD.
# Inputs are the existing motif=False collections staged on 2026-09-13 (read only, byte copies).
set -euo pipefail
gpu=${1:-0}
B=/local-scratch2/mirzaei/lgd_common_eval_20260925
PY=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
root=$B/PTC/supplementary_motif_false_common_reference
mkdir -p "$root/stage/ptc" "$root/results"
exec > >(tee -a "$root/pipeline.log") 2>&1
cp -p /local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts/ptc/real_test_graphs.pt "$root/stage/ptc/"
for seed in 0 1 2; do
  mkdir -p "$root/stage/ptc/generated/seed_$seed"
  cp -p /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913/generated/graphvae_motif_false_seed${seed}.pt "$root/stage/ptc/generated/seed_$seed/generated_graphs.pt"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" /localhome/mirzaei/evaluate_defog_synthetic_20260906.py \
    --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$root/stage" \
    --dataset ptc --seed "$seed" --output "$root/results/motif_false_seed_${seed}.json" --device cuda
done
echo COMPLETE > "$root/status"
