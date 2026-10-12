#!/usr/bin/env bash
set -euo pipefail
repo=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
py=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
root=/local-scratch2/mirzaei/grid_matched_reference_eval_20260913
ref=/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/artifacts/grid/real_test_graphs.pt
mkdir -p "$root/generated" "$root/random_gin"
cd "$repo"
export PYTHONPATH="$repo/graph_evaluation/src:$repo"
for setting in total_count full_matrix; do
  for seed in 0 1 2; do
    src=/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/graphvae_linkcorr/grid/$setting/seed_$seed/Single_comp_generatedGraphs_adj_final_eval.npy
    gen=$root/generated/graphvae_${setting}_seed${seed}.pt
    "$py" /local-scratch2/mirzaei/prepare_grid_matched_graphvae.py --input "$src" --output "$gen" --method "GraphVAE-REQ-$setting" --seed "$seed"
    "$py" -m ggm_eval.worker legacy-evaluate --generated "$gen" --reference "$ref" --legacy-repo "$repo" --output "$root/random_gin/graphvae_${setting}_seed${seed}.json" --modes topology_control --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda
  done
done
for seed in 0 1 2; do
  gen=$repo/runs/defog/frozen_eval/grid/generated/seed_$seed/generated_graphs.pt
  "$py" -m ggm_eval.worker legacy-evaluate --generated "$gen" --reference "$ref" --legacy-repo "$repo" --output "$root/random_gin/defog_seed${seed}.json" --modes topology_control --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda
done
date -Is > "$root/COMPLETE"
