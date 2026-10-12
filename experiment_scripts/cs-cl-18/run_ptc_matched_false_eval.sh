#!/usr/bin/env bash
set -euo pipefail
repo=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
py=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
root=/local-scratch2/mirzaei/ptc_matched_reference_eval_20260913
ref=/local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts/ptc/real_test_graphs.pt
mkdir -p "$root/generated" "$root/random_gin"
cd "$repo"
export PYTHONPATH="$repo/graph_evaluation/src:$repo"
for seed in 0 1 2; do
  src=/local-scratch2/new/gather/datasets/ptc/setting_01/seed_$seed/Single_comp_generatedGraphs_adj_final_eval.npy
  gen=$root/generated/graphvae_motif_false_seed${seed}.pt
  "$py" /local-scratch2/mirzaei/prepare_grid_matched_graphvae.py --dataset PTC --feature-dim 19 --feature-schema 'gin-node-label-v2|export=decoded_node' --input "$src" --output "$gen" --method GraphVAE-motif-false --seed "$seed"
  "$py" -m ggm_eval.worker legacy-evaluate --generated "$gen" --reference "$ref" --legacy-repo "$repo" --output "$root/random_gin/motif_false_seed${seed}.json" --modes topology_control --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda
done
date -Is > "$root/COMPLETE"
