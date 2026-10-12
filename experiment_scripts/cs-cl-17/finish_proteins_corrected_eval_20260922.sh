#!/usr/bin/env bash
set -euo pipefail

generation_root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910/corrected_generation_20260922
eval_root=/local-scratch2/mirzaei/proteins_defog_corrected_eval_20260922
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
repo=/local-scratch2/mirzaei/fb/GraphVAE-REQ
convert=/local-scratch2/mirzaei/edge_feature_random_gin_20260917/convert_pyg_collection_to_dgl.py
evaluate=/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py
reference=/local-scratch2/mirzaei/archive_completion_20260922/PROTEINS/true_full_003/seed_0/reference_attributed_graphs.bin

mkdir -p "$eval_root"
echo WAITING >"$eval_root/status.txt"
for seed in 0 1 2; do
  until grep -q '^COMPLETE' "$generation_root/seed_$seed/status.txt" 2>/dev/null; do sleep 60; done
done

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for seed in 0 1 2; do
  out="$eval_root/seed_$seed"
  mkdir -p "$out"
  echo "EVALUATING seed=$seed" >"$eval_root/status.txt"
  "$py" "$convert" \
    --input "$generation_root/seed_$seed/generated_graphs.pt" \
    --output "$out/generated.bin"
  "$py" "$evaluate" \
    --repo "$repo" \
    --generated "$out/generated.bin" \
    --reference "$reference" \
    --output "$out/common_metrics.json" \
    --label proteins_defog_corrected \
    --seed "$seed" --device cuda --repeats 10
  cp "$generation_root/seed_$seed/checkpoint.sha256" "$generation_root/seed_$seed/generated_graphs.sha256" "$out/"
  echo COMPLETE >"$out/COMPLETE"
done

rsync -a -e 'ssh -o BatchMode=yes' "$generation_root/" \
  cs-cl-19:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PROTEINS/defog/corrected_generation_20260922/
rsync -a -e 'ssh -o BatchMode=yes' "$eval_root/" \
  cs-cl-19:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PROTEINS/evaluations/defog_corrected_20260922/
echo COMPLETE >"$eval_root/status.txt"
