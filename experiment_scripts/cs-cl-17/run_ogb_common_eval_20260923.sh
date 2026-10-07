#!/usr/bin/env bash
set -euo pipefail
gpu=${1:-0}
root=/local-scratch2/mirzaei/common_eval_fix_20260923/OGB
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/OGB
repo=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ
chainrepo=/local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
common=/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py
topology=/localhome/mirzaei/evaluate_topology_gin_20260923.py
prepare=/localhome/mirzaei/prepare_ogb_common_topology_20260923.py
remote=mirzaei@cs-cl-19.cmpt.sfu.ca
mkdir -p "$root/raw" "$root/results"
exec >"$root/pipeline.log" 2>&1
trap 'echo FAILED >"$root/status"' ERR
echo STAGING >"$root/status"
cp -p /local-scratch2/mirzaei/archive_completion_20260922/OGB/reference.bin "$root/raw/reference.bin"
for spec in false:setting_01 true_full:setting_03; do
  method=${spec%%:*}; setting=${spec##*:}
  for seed in 0 1 2; do
    src=$archive/sources/gather/$setting/seed_$seed/terminal_motif_mmd
    mkdir -p "$root/raw/$method/seed_$seed"
    rsync -a "$remote:$src/terminal_chain_binary_matrices.npz" "$root/raw/$method/seed_$seed/"
    rsync -a "$remote:$src/best_chain_random_gin_metrics.json" "$root/raw/$method/seed_$seed/"
  done
done
for seed in 0 1 2; do
  mkdir -p "$root/raw/defog/seed_$seed"
  rsync -a "$remote:$archive/evaluations/defog_seed_$seed/generated.bin" "$root/raw/defog/seed_$seed/generated.bin"
done
"$python" "$prepare" "$root" "$chainrepo"
for method in false true_full defog; do
  for seed in 0 1 2; do
    generated="$root/aligned/$method/seed_$seed/generated.bin"
    out="$root/results/$method/seed_$seed"
    mkdir -p "$out"
    echo RUNNING_${method}_seed_${seed} >"$root/status"
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$common" --repo "$repo" --generated "$generated" --reference "$root/aligned/reference.bin" --output "$out/structural.json" --label ogb_$method --seed "$seed" --device cuda --repeats 10 --skip-random-gin
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$topology" --repo "$repo" --generated "$generated" --reference "$root/aligned/reference.bin" --output "$out/random_gin_topology.json" --label ogb_$method --seed "$seed" --device cuda
  done
done
sha256sum "$root/aligned/reference.bin" "$root/aligned"/*/seed_*/generated.bin >"$root/SHA256SUMS"
echo COMPLETE >"$root/status"
