#!/usr/bin/env bash
set -euo pipefail

remote_root=/local-scratch2/mirzaei/aids_defog_independent_20260922
local_root=/local-scratch2/mirzaei/aids_defog_independent_20260922
eval_root=/local-scratch2/mirzaei/aids_defog_replacement_eval_20260922
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
repo=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ
convert=/local-scratch2/mirzaei/edge_feature_random_gin_20260917/convert_pyg_collection_to_dgl.py
evaluate=/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py
reference=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin

mkdir -p "$local_root" "$eval_root"
echo WAITING >"$eval_root/status.txt"
for seed in 4 5; do
  until ssh -o BatchMode=yes cs-cl-09 "grep -q '^COMPLETE' '$remote_root/seed_$seed/status.txt'"; do sleep 120; done
done
rsync -a -e 'ssh -o BatchMode=yes' cs-cl-09:/local-scratch2/mirzaei/aids_defog_independent_20260919/seed_3/ "$local_root/seed_3/"
for seed in 4 5; do
  rsync -a -e 'ssh -o BatchMode=yes' "cs-cl-09:$remote_root/seed_$seed/" "$local_root/seed_$seed/"
done

export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for seed in 3 4 5; do
  out="$eval_root/seed_$seed"
  mkdir -p "$out"
  echo "EVALUATING seed=$seed" >"$eval_root/status.txt"
  "$py" "$convert" \
    --input "$local_root/seed_$seed/generation/generated_graphs.pt" \
    --output "$out/generated.bin"
  "$py" "$evaluate" \
    --repo "$repo" \
    --generated "$out/generated.bin" \
    --reference "$reference" \
    --output "$out/common_metrics.json" \
    --label aids_defog_independent \
    --seed "$seed" --device cuda --repeats 10
  cp "$local_root/seed_$seed/checkpoint.sha256" "$local_root/seed_$seed/generated_graphs.sha256" "$out/"
  echo COMPLETE >"$out/COMPLETE"
done

rsync -a -e 'ssh -o BatchMode=yes' "$local_root/" \
  cs-cl-19:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/experiments/defog_independent_20260922/
rsync -a -e 'ssh -o BatchMode=yes' "$eval_root/" \
  cs-cl-19:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/evaluations/defog_independent_20260922/
echo COMPLETE >"$eval_root/status.txt"
