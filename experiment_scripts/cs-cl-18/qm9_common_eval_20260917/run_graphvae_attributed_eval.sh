#!/usr/bin/env bash
# Generate one frozen 512-graph test collection per checkpoint and evaluate
# Random-GIN both without intrinsic node features and with decoded node features.
set -euo pipefail

setting=${1:?setting: true_full or false}
gpu=${2:?physical GPU index}
root=/local-scratch2/mirzaei/qm9_common_eval_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python

while ! grep -q '^COMPLETE' "$root/status/source.status" 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo/vendor/python:${PYTHONPATH:-}"
status="$root/status/graphvae_${setting}.status"
printf 'RUNNING setting=%s gpu=%s started=%s\n' "$setting" "$gpu" "$(date --iso-8601=seconds)" > "$status"

args=()
for seed in 0 1 2; do
  args+=(--run-dir "$root/graphvae/$setting/seed_$seed")
done

cd "$repo"
set +e
"$python" -u scripts/evaluate_attributed_graph_realism_checkpoints.py \
  "${args[@]}" \
  --dataset-cache-dir "$root/cache/dataset" \
  --split test \
  --modes topology_control decoded_node \
  --max-graphs 512 \
  --generation-batch-size 4 \
  --generation-seed 12345 \
  --evaluator-seed 0 \
  --repeats 10 \
  --device cuda \
  --output-dir "$root/evaluations/graphvae/$setting" \
  --save-samples \
  --save-dgl
code=$?
set -e

if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s setting=%s gpu=%s exit_code=%s finished=%s\n' \
  "$state" "$setting" "$gpu" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
