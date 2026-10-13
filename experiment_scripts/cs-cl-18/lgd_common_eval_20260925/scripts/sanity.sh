#!/usr/bin/env bash
# Sanity gate: re-run the unchanged evaluators on existing method/seed collections and
# compare every numeric leaf with the stored result.  Writes <BASE>/<DATASET>/sanity/
# SANITY_PASS or SANITY_FAIL.  Usage: sanity.sh DATASET GPU
set -euo pipefail
dataset=${1:?dataset}; gpu=${2:-0}
B=${LGD_EVAL_BASE:-/local-scratch2/mirzaei/lgd_common_eval_20260925}
PY=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
CMP="$PY $B/scripts/compare_sanity.py"
CE=/local-scratch2/mirzaei/common_eval_fix_20260923
Q=/local-scratch2/mirzaei/qm9_common_eval_20260917
out=$B/$dataset/sanity; mkdir -p "$out"; rm -f "$out/SANITY_PASS" "$out/SANITY_FAIL"
exec > >(tee -a "$out/sanity.log") 2>&1
echo "START $(date -Is) $dataset gpu=$gpu"
pass=1
check() { if ! $CMP "$@" | tee -a "$out/comparisons.txt" | grep -q '"PASS": true'; then pass=0; fi; }
case "$dataset" in
  TRIANGULAR_GRID|GRID)
    slug=$(echo "$dataset" | tr A-Z a-z)
    CUDA_VISIBLE_DEVICES=$gpu "$PY" /local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/evaluate_defog_synthetic_20260906.py \
      --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$CE/$dataset/stage/defog" \
      --dataset "$slug" --seed 0 --output "$out/defog_seed_0.json" --device cuda
    check "$CE/$dataset/results/defog_seed_0.json" "$out/defog_seed_0.json" --out "$out/COMPARISON.json" ;;
  LOBSTER)
    # LOBSTER DeFoG was evaluated by the synthetic campaign (not common_eval_fix); reproduce its seed 0
    A=/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906
    CUDA_VISIBLE_DEVICES=$gpu "$PY" $A/evaluate_defog_synthetic_20260906.py \
      --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$A/artifacts" \
      --dataset lobster --seed 0 --output "$out/defog_seed_0.json" --device cuda
    check "$A/metrics/lobster_seed0.json" "$out/defog_seed_0.json" --out "$out/COMPARISON.json" ;;
  PROTEINS)
    CUDA_VISIBLE_DEVICES=$gpu "$PY" /local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py \
      --repo /local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ \
      --generated "$CE/PROTEINS/stage/defog/seed_0/generated.bin" --reference "$CE/PROTEINS/stage/reference.bin" \
      --output "$out/defog_seed_0_common_metrics.json" --label proteins_defog --seed 0 --device cuda --repeats 10
    check "$CE/PROTEINS/results/defog/seed_0/common_metrics.json" "$out/defog_seed_0_common_metrics.json" --out "$out/COMPARISON.json" ;;
  QM9)
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$Q/evaluate_dgl_common_metrics.py" --repo "$Q/source/GraphVAE-REQ" \
      --generated "$Q/evaluations/defog/seed_0/generated_attributed_graphs.bin" \
      --reference "$Q/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin" \
      --output "$out/defog_seed_0_common_metrics.json" --label defog --seed 0 --device cuda --repeats 10
    check "$Q/evaluations/defog/seed_0/common_metrics.json" "$out/defog_seed_0_common_metrics.json" --out "$out/COMPARISON.json" ;;
  PTC)
    D=/local-scratch2/mirzaei/defog_ptc_frozen_20260906/artifacts
    CUDA_VISIBLE_DEVICES=$gpu "$PY" /localhome/mirzaei/evaluate_defog_synthetic_20260906.py \
      --repo /local-scratch2/new/GraphVAE-REQ-terminal-motif-mmd --artifact-root "$D" \
      --dataset ptc --seed 0 --output "$out/defog_seed_0_structural.json" --device cuda
    R=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ; VPY=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
    (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$R/graph_evaluation/src:$R" "$VPY" -m ggm_eval.worker legacy-evaluate \
      --generated "$D/ptc/generated/seed_0/generated_graphs.pt" --reference "$D/ptc/real_test_graphs.pt" --legacy-repo "$R" \
      --output "$out/defog_seed_0_random_gin.json" --modes decoded_node topology_control --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda)
    (cd "$R" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$R/graph_evaluation/src:$R" "$VPY" -m ggm_eval.worker legacy-evaluate \
      --generated /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913/generated/graphvae_motif_false_seed0.pt --reference "$D/ptc/real_test_graphs.pt" --legacy-repo "$R" \
      --output "$out/motif_false_seed_0_random_gin.json" --modes topology_control --repeats 10 --evaluator-seed 0 --nearest-k 5 --device cuda)
    check /local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/structural/seed_0.json "$out/defog_seed_0_structural.json" --out "$out/COMPARISON_structural.json"
    check /local-scratch2/mirzaei/defog_ptc_frozen_20260906/random_gin/ptc/seed_0/evaluation.json "$out/defog_seed_0_random_gin.json" --subtree evaluation --out "$out/COMPARISON_random_gin.json"
    check /local-scratch2/mirzaei/ptc_matched_reference_eval_20260913/random_gin/motif_false_seed0.json "$out/motif_false_seed_0_random_gin.json" --subtree evaluation --out "$out/COMPARISON_motif_false_random_gin.json" ;;
  *) echo "unsupported dataset $dataset" >&2; exit 2 ;;
esac
if [ "$pass" = 1 ]; then date -Is > "$out/SANITY_PASS"; echo SANITY_PASS; else date -Is > "$out/SANITY_FAIL"; echo SANITY_FAIL; exit 1; fi
