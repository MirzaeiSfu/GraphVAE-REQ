#!/usr/bin/env bash
set -euo pipefail

setting=${1:?true_full or false}
gpu=${2:?GPU index}
defog_seed=${3:?DeFoG seed}
root=/local-scratch2/mirzaei/qm9_common_eval_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python

while ! grep -q '^COMPLETE' "$root/status/graphvae_${setting}.status" 2>/dev/null; do sleep 30; done
while ! grep -q '^COMPLETE' "$root/status/graphvae_true_full.status" 2>/dev/null; do sleep 30; done
reference=$root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin
export CUDA_VISIBLE_DEVICES="$gpu"

for seed in 0 1 2; do
  directory=$root/evaluations/graphvae/$setting/seed_$seed
  "$python" "$root/evaluate_dgl_common_metrics.py" \
    --repo "$repo" \
    --generated "$directory/generated_attributed_graphs.bin" \
    --reference "$reference" \
    --output "$directory/common_structural.json" \
    --label "graphvae_$setting" --seed "$seed" --device cpu --skip-random-gin
done

while ! grep -q '^COMPLETE' "$root/status/defog_export.status" 2>/dev/null; do sleep 30; done
directory=$root/evaluations/defog/seed_$defog_seed
"$python" "$root/evaluate_dgl_common_metrics.py" \
  --repo "$repo" \
  --generated "$directory/generated_attributed_graphs.bin" \
  --reference "$reference" \
  --output "$directory/common_metrics.json" \
  --label defog --seed "$defog_seed" --device cuda --repeats 10

printf 'COMPLETE setting=%s defog_seed=%s finished=%s\n' \
  "$setting" "$defog_seed" "$(date --iso-8601=seconds)" \
  > "$root/status/common_${setting}.status"
