#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/archive_completion_20260922/PROTEINS
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
repo=/local-scratch2/mirzaei/fb/GraphVAE-REQ
helpers=/local-scratch2/mirzaei/edge_feature_random_gin_20260917
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
trap 'echo FAILED > "$root/baselines.status"' ERR
echo WAITING_FOR_TRUE_EVALUATION > "$root/baselines.status"
while [ ! -f "$root/COMPLETE" ]; do sleep 30; done
echo RUNNING > "$root/baselines.status"
for seed in 0 1 2; do
 "$py" /local-scratch/localhome/mirzaei/proteins_feature_evaluator_20260922.py --run-dir "/local-scratch2/new/gather/datasets/proteins/setting_01/seed_$seed" --dataset-cache-dir /local-scratch2/mirzaei/node_feature_eval_cache_v4 --split test --modes topology_control decoded_node --max-graphs 209 --generation-batch-size 8 --generation-seed 12345 --evaluator-seed 0 --repeats 10 --device cuda --output-dir "$root/false/seed_$seed" --save-dgl
 mkdir -p "$root/defog/seed_$seed"
 if [ "$seed" = 0 ]; then src=/local-scratch2/mirzaei/proteins_defog_3seed_20260910/final_eval/seed_0; else src=/local-scratch2/mirzaei/proteins_defog_3seed_20260910/collected_final_eval/seed_$seed; fi
 "$py" "$helpers/convert_pyg_collection_to_dgl.py" --input "$src/generated_graphs.pt" --output "$root/defog/seed_$seed/generated.bin"
 "$py" /local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py --repo "$repo" --generated "$root/defog/seed_$seed/generated.bin" --reference "$root/true_full_003/seed_0/reference_attributed_graphs.bin" --output "$root/defog/seed_$seed/common_metrics.json" --label proteins_defog --seed "$seed" --device cuda --repeats 10
done
rsync -a -e 'ssh -o BatchMode=yes' "$root/" mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PROTEINS/evaluations/native_features_20260922/
echo COMPLETE > "$root/baselines.status"
