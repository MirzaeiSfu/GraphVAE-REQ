#!/usr/bin/env bash
set -euo pipefail

setting=${1:?true_full or false}
gpu=${2:?GPU index}
root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
status=$root/status/graphvae_${setting}.status

while ! grep -q '^COMPLETE' "$root/status/prepare.status" 2>/dev/null; do sleep 30; done
printf 'RUNNING setting=%s gpu=%s started=%s\n' "$setting" "$gpu" "$(date --iso-8601=seconds)" > "$status"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1
export PYTHONPATH="$repo/vendor/python:${PYTHONPATH:-}"

args=()
for seed in 0 1 2; do args+=(--run-dir "$root/graphvae/$setting/seed_$seed"); done
cd "$repo"
set +e
"$python" -u scripts/evaluate_attributed_graph_realism_checkpoints.py \
  "${args[@]}" \
  --dataset-cache-dir "$root/cache/dataset" \
  --split test --modes topology_control decoded_node \
  --max-graphs 400 --generation-batch-size 4 --generation-seed 12345 \
  --evaluator-seed 0 --repeats 10 --device cuda \
  --output-dir "$root/evaluations/graphvae/$setting" --save-samples --save-dgl
code=$?
set -e
state=FAILED; ((code == 0)) && state=COMPLETE
printf '%s setting=%s exit_code=%s finished=%s\n' "$state" "$setting" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
