#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/archive_completion_20260922/PROTEINS
mkdir -p "$root"
trap 'echo FAILED > "$root/FAILED"' ERR
repo=/local-scratch2/mirzaei/fb/GraphVAE-REQ
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for seed in 0 1 2; do
 "$py" /local-scratch/localhome/mirzaei/proteins_feature_evaluator_20260922.py --run-dir "/local-scratch2/mirzaei/motif_corr_new_20260912/proteins/seed_$seed" --dataset-cache-dir /local-scratch2/mirzaei/node_feature_eval_cache_v4 --split test --modes topology_control decoded_node --max-graphs 209 --generation-batch-size 8 --generation-seed 12345 --evaluator-seed 0 --repeats 10 --device cuda --output-dir "$root/true_full_003/seed_$seed" --save-dgl
done
rsync -a -e 'ssh -o BatchMode=yes' "$root/" mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/PROTEINS/evaluations/native_features_20260922/
echo COMPLETE > "$root/COMPLETE"
