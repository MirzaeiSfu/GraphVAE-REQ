#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four matched 40/40 LOBSTER methods with independently trained
# GraphCL-GIN topology/node/edge/node+edge encoder families.

repo_root="${1:-/local-scratch2/graphvae-req-work/GraphVAE-REQ}"
python_bin="${PYTHON_BIN:-/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python}"
upstream_repo="${UPSTREAM_REPO:-/local-scratch2/graphvae-req-work/Self-Supervised-Models-for-GGM-Evaluation}"
ggm_src="${GGM_EVAL_SRC:-${repo_root}/graph_evaluation/src}"
input_root="${repo_root}/runs/20260726/lobster_attributed_graphcl_inputs"
encoder_root="${repo_root}/runs/20260726/lobster_attributed_graphcl_encoders"
output_root="${repo_root}/runs/20260726/lobster_attributed_graphcl_evaluations"
reference="${input_root}/real_heldout_test_graphs.pt"

export PYTHONPATH="${repo_root}/.graphcl_deps:${ggm_src}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_root}"

conditions=(
  lobster_graphvae_mm_fixed_split_native40_legacy
  lobster_kiarash_parity_kia40_2000_feature40_legacy
  lobster_semantic_hybrid_r001_legacy
  lobster_semantic_hybrid_r001_edgecount01_legacy
)

run_mode() {
  local mode="$1"
  local gpu="$2"
  local checkpoint_args=()
  local seed
  local condition

  for seed in 0 1 2; do
    checkpoint_args+=(
      --checkpoint "${encoder_root}/${mode}/seed_${seed}/checkpoint.pt"
    )
  done
  for condition in "${conditions[@]}"; do
    for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m ggm_eval.cli evaluate \
        --generated "${input_root}/${condition}/seed_${seed}/generated_attributed_graphs.pt" \
        --reference "${reference}" \
        "${checkpoint_args[@]}" \
        --output-dir "${output_root}/${mode}/${condition}/seed_${seed}" \
        --device cuda \
        --nearest-k 5 \
        --max-graphs 0 \
        --upstream-repo "${upstream_repo}" \
        --python "${python_bin}"
    done
  done
}

run_mode topology_control 0 >"${output_root}/topology_control.log" 2>&1 &
pid_topology=$!
run_mode decoded_node 0 >"${output_root}/decoded_node.log" 2>&1 &
pid_node=$!
run_mode decoded_edge 1 >"${output_root}/decoded_edge.log" 2>&1 &
pid_edge=$!
run_mode decoded_node_edge 1 >"${output_root}/decoded_node_edge.log" 2>&1 &
pid_full=$!

status=0
for pid in "${pid_topology}" "${pid_node}" "${pid_edge}" "${pid_full}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
