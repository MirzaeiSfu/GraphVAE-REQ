#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/edge_feature_random_gin_20260917
source_root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=$source_root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
status=$root/status/aids.status
mkdir -p "$root/status" "$root/results/aids"
trap 'printf "FAILED exit_code=%s finished=%s\n" "$?" "$(date --iso-8601=seconds)" > "$status"' ERR
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$status"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$repo:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
reference=$source_root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin
for setting in true_full false; do
  for seed in 0 1 2; do
    input=$source_root/evaluations/graphvae/$setting/seed_$seed/generated_attributed_graphs.bin
    "$python" "$root/evaluate_dgl_all_modes.py" --repo "$repo" --generated "$input" \
      --reference "$reference" --output "$root/results/aids/$setting/seed_$seed/random_gin_all_modes.json" \
      --label "aids_$setting" --seed "$seed" --device cuda --repeats 10
  done
done
for seed in 1 2; do
  input=$source_root/evaluations/defog/seed_$seed/generated_attributed_graphs.bin
  "$python" "$root/evaluate_dgl_all_modes.py" --repo "$repo" --generated "$input" \
    --reference "$reference" --output "$root/results/aids/defog/seed_$seed/random_gin_all_modes.json" \
    --label aids_defog --seed "$seed" --device cuda --repeats 10
done
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
trap - ERR
