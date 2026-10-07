#!/usr/bin/env bash
# Evaluate staged LGD collections with the unchanged corrected evaluators.
# Usage: run_lgd_eval.sh DATASET GPU [SEEDS...]   (default seeds 0 1 2)
# Requires raw_pickles/<DATASET>_seed<k>/epoch_1999_graphs.pkl (fetch_lgd.sh).
set -euo pipefail
dataset=${1:?dataset}; gpu=${2:-0}; shift 2 || true
seeds=${*:-0 1 2}
B=${LGD_EVAL_BASE:-/local-scratch2/mirzaei/lgd_common_eval_20260925}
S=$B/scripts
PY=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
root=$B/$dataset
mkdir -p "$root/results"
exec > >(tee -a "$root/pipeline.log") 2>&1
trap 'echo FAILED >"$root/status"' ERR
echo "START $(date -Is) dataset=$dataset gpu=$gpu seeds=$seeds host=$(hostname)"
echo VERIFYING >"$root/status"
"$PY" "$S/verify_reference.py" "$dataset" $seeds
echo STAGING >"$root/status"
"$PY" "$S/stage_lgd.py" "$dataset" $seeds
for seed in $seeds; do
  echo "RUNNING_seed_$seed" >"$root/status"
  case "$dataset" in
    TRIANGULAR_GRID|GRID|LOBSTER)
      slug=$(echo "$dataset" | tr A-Z a-z)
      # identical to run_synthetic_common_eval_20260923.sh (evaluator, repo, env, flags)
      CUDA_VISIBLE_DEVICES=$gpu "$PY" /local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py \
        --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$root/stage/lgd" \
        --dataset "$slug" --seed "$seed" --output "$root/results/lgd_seed_${seed}.json" --device cuda ;;
    PROTEINS)
      # identical to run_proteins_common_eval_20260923.sh
      mkdir -p "$root/results/lgd/seed_$seed"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" /local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py \
        --repo /local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ \
        --generated "$root/stage/lgd/seed_$seed/generated.bin" \
        --reference /local-scratch2/mirzaei/common_eval_fix_20260923/PROTEINS/stage/reference.bin \
        --output "$root/results/lgd/seed_$seed/common_metrics.json" --label proteins_lgd --seed "$seed" --device cuda --repeats 10 ;;
    QM9)
      # identical to qm9_common_eval_20260917/run_common_structural.sh DeFoG call (structural + Random-GIN)
      mkdir -p "$root/results/lgd/seed_$seed"
      Q=/local-scratch2/mirzaei/qm9_common_eval_20260917
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$Q/evaluate_dgl_common_metrics.py" --repo "$Q/source/GraphVAE-REQ" \
        --generated "$root/stage/lgd/seed_$seed/generated.bin" \
        --reference "$Q/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin" \
        --output "$root/results/lgd/seed_$seed/common_metrics.json" --label lgd --seed "$seed" --device cuda --repeats 10 ;;
    PTC)
      # structural MMD + structural-feature Random-GIN: same evaluator as DeFoG PTC
      # (defog_ptc_full_metrics_20260907/structural); Random-GIN native/topology modes:
      # same ggm_eval legacy-evaluate call as defog_ptc_frozen_20260906/run_random_gin.py.
      mkdir -p "$root/results/lgd/seed_$seed"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" /localhome/mirzaei/evaluate_defog_synthetic_20260906.py \
        --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$root/stage/lgd" \
        --dataset ptc --seed "$seed" --output "$root/results/lgd/seed_$seed/structural.json" --device cuda
      R=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ; VPY=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
      (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$R/graph_evaluation/src:$R" "$VPY" -m ggm_eval.worker legacy-evaluate \
        --generated "$root/stage/lgd/ptc/generated/seed_$seed/generated_graphs.pt" \
        --reference "$root/stage/lgd/ptc/real_test_graphs.pt" --legacy-repo "$R" \
        --output "$root/results/lgd/seed_$seed/random_gin.json" --modes decoded_node topology_control \
        --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda) ;;
    *) echo "unsupported dataset $dataset" >&2; exit 2 ;;
  esac
done
find "$root/stage" -type f \( -name '*.pt' -o -name '*.bin' \) -exec sha256sum {} + | sort -k2 >"$root/SHA256SUMS"
"$PY" "$S/build_results.py" "$dataset" || echo "build_results failed (tables not refreshed)"
echo COMPLETE >"$root/status"
echo "DONE $(date -Is)"
