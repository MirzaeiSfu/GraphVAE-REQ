#!/usr/bin/env bash
set -euo pipefail
gpu=${1:-0}
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PROTEINS
root=/local-scratch2/mirzaei/common_eval_fix_20260923/PROTEINS
repo=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
common=/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py
mkdir -p "$root/stage" "$root/results"
exec >"$root/pipeline.log" 2>&1
trap 'echo FAILED >"$root/status"' ERR
echo STAGING >"$root/status"
remote=mirzaei@cs-cl-19.cmpt.sfu.ca
rsync -a "$remote:$archive/evaluations/native_features_20260922/false/seed_0/reference_attributed_graphs.bin" "$root/stage/reference.bin"
for method in false true_full_003; do
  for seed in 0 1 2; do
    mkdir -p "$root/stage/$method/seed_$seed" "$root/results/$method/seed_$seed"
    rsync -a "$remote:$archive/evaluations/native_features_20260922/$method/seed_$seed/generated_attributed_graphs.bin" "$root/stage/$method/seed_$seed/generated.bin"
    echo RUNNING_${method}_seed_${seed} >"$root/status"
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$common" --repo "$repo" --generated "$root/stage/$method/seed_$seed/generated.bin" --reference "$root/stage/reference.bin" --output "$root/results/$method/seed_$seed/common_metrics.json" --label proteins_$method --seed "$seed" --device cuda --repeats 10
  done
done
for seed in 0 1 2; do
  mkdir -p "$root/stage/defog/seed_$seed" "$root/results/defog/seed_$seed"
  rsync -a "$remote:$archive/evaluations/defog_corrected_20260922/seed_$seed/generated.bin" "$root/stage/defog/seed_$seed/generated.bin"
  echo RUNNING_defog_seed_${seed} >"$root/status"
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$common" --repo "$repo" --generated "$root/stage/defog/seed_$seed/generated.bin" --reference "$root/stage/reference.bin" --output "$root/results/defog/seed_$seed/common_metrics.json" --label proteins_defog --seed "$seed" --device cuda --repeats 10
done
sha256sum "$root/stage/reference.bin" "$root/stage"/*/seed_*/generated.bin >"$root/SHA256SUMS"
echo COMPLETE >"$root/status"
