#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?GRID or TRIANGULAR_GRID}
gpu=${2:-0}
case "$dataset" in
  GRID) defog_seeds="0 4 5"; slug=grid ;;
  TRIANGULAR_GRID) defog_seeds="0 1 3"; slug=triangular_grid ;;
  *) echo "unsupported dataset: $dataset" >&2; exit 2 ;;
esac
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921
root=/local-scratch2/mirzaei/common_eval_fix_20260923/$dataset
repo=/local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd
evaluator=/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
mkdir -p "$root/results"
exec >"$root/pipeline.log" 2>&1
trap 'echo FAILED >"$root/status"' ERR
echo STAGING >"$root/status"
remote=mirzaei@cs-cl-19.cmpt.sfu.ca
for method in motif_false motif_true_full_matrix; do
  mkdir -p "$root/stage/$method/$slug"
  rsync -a "$remote:$archive/$dataset/experiments/graphvae/motif_true_full_matrix/seed_0/real_test_graphs.pt" "$root/stage/$method/$slug/real_test_graphs.pt"
  for seed in 0 1 2; do
    mkdir -p "$root/stage/$method/$slug/generated/seed_$seed"
    rsync -a "$remote:$archive/$dataset/experiments/graphvae/$method/seed_$seed/generated_graphs.pt" "$root/stage/$method/$slug/generated/seed_$seed/generated_graphs.pt"
    echo RUNNING_${method}_seed_${seed} >"$root/status"
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$evaluator" --repo "$repo" --artifact-root "$root/stage/$method" --dataset "$slug" --seed "$seed" --output "$root/results/${method}_seed_${seed}.json" --device cuda
  done
done
mkdir -p "$root/stage/defog/$slug"
rsync -a "$remote:$archive/$dataset/experiments/graphvae/motif_true_full_matrix/seed_0/real_test_graphs.pt" "$root/stage/defog/$slug/real_test_graphs.pt"
for seed in $defog_seeds; do
  mkdir -p "$root/stage/defog/$slug/generated/seed_$seed"
  rsync -a "$remote:$archive/$dataset/experiments/defog/seed_$seed/generation/generated_graphs.pt" "$root/stage/defog/$slug/generated/seed_$seed/generated_graphs.pt"
  echo RUNNING_defog_seed_${seed} >"$root/status"
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$evaluator" --repo "$repo" --artifact-root "$root/stage/defog" --dataset "$slug" --seed "$seed" --output "$root/results/defog_seed_${seed}.json" --device cuda
done
sha256sum "$root/stage"/*/"$slug"/real_test_graphs.pt "$root/stage"/*/"$slug"/generated/seed_*/generated_graphs.pt >"$root/SHA256SUMS"
echo COMPLETE >"$root/status"
