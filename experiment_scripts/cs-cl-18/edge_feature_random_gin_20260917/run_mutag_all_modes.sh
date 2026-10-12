#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/edge_feature_random_gin_20260917
repo=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
status=$root/status/mutag.status
mkdir -p "$root/status" "$root/results/mutag" "$root/work/mutag_true02"
trap 'printf "FAILED exit_code=%s finished=%s\n" "$?" "$(date --iso-8601=seconds)" > "$status"' ERR
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$status"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$repo:$repo/vendor/python:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

# Regenerate the three motif=True, alpha_motif=0.2 checkpoints once so all four
# modes use the identical generated/reference collections.
args=()
for seed in 0 1 2; do args+=(--run-dir "/local-scratch2/mirzaei/mutag_threshold_calibration_20260912/checkpoints/seed_$seed"); done
cd "$repo"
"$python" scripts/evaluate_attributed_graph_realism_checkpoints.py "${args[@]}" \
  --config /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/configs/mutag_edgefeat_campaign/full_matrix.yaml \
  --dataset-cache-dir /local-scratch2/mirzaei/mutag_threshold_calibration_20260912/dataset_cache \
  --split test --modes topology_control decoded_node decoded_edge decoded_node_edge \
  --max-graphs 39 --generation-batch-size 8 --generation-seed 12345 \
  --evaluator-seed 0 --repeats 10 --device cuda \
  --output-dir "$root/results/mutag/true_full_motif02" --save-dgl

# Motif=False has three saved edge-aware feature-bearing collections.
common_ref=$root/work/mutag_reference.bin
for seed in 0 1 2; do
  src=/local-scratch2/mirzaei/mutag_new_results_20260911/seed_$seed/graphvae_motif_false
  mkdir -p "$root/work/mutag_false/seed_$seed" "$root/results/mutag/false/seed_$seed"
  "$python" "$root/convert_pyg_collection_to_dgl.py" --input "$src/generated_graphs.pt" \
    --output "$root/work/mutag_false/seed_$seed/generated.bin"
  if [[ ! -s "$common_ref" ]]; then
    "$python" "$root/convert_pyg_collection_to_dgl.py" --input "$src/real_test_graphs.pt" --output "$common_ref"
  fi
  "$python" "$root/evaluate_dgl_all_modes.py" --repo "$repo" \
    --generated "$root/work/mutag_false/seed_$seed/generated.bin" --reference "$common_ref" \
    --output "$root/results/mutag/false/seed_$seed/random_gin_all_modes.json" \
    --label mutag_false --seed "$seed" --device cuda --repeats 10
done

# The currently available edge-aware DeFoG collection.
mkdir -p "$root/work/mutag_defog/seed_0" "$root/results/mutag/defog/seed_0"
"$python" "$root/convert_pyg_collection_to_dgl.py" \
  --input /local-scratch2/mirzaei/mutag_defog_edgefeat_seeds_20260913/evaluation_seed0/generated_graphs_normalized.pt \
  --output "$root/work/mutag_defog/seed_0/generated.bin"
"$python" "$root/evaluate_dgl_all_modes.py" --repo "$repo" \
  --generated "$root/work/mutag_defog/seed_0/generated.bin" --reference "$common_ref" \
  --output "$root/results/mutag/defog/seed_0/random_gin_all_modes.json" \
  --label mutag_defog --seed 0 --device cuda --repeats 10
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
trap - ERR
