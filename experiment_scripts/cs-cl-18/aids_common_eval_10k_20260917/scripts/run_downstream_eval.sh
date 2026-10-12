#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
status=$root/status/downstream.status

for required in graphvae_true_full graphvae_false defog; do
  while ! grep -q '^COMPLETE' "$root/status/$required.status" 2>/dev/null; do sleep 30; done
done
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$status"
reference=$root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin

# Structural metrics for GraphVAE; RandomGIN is already produced by its evaluator.
for setting in true_full false; do
  for seed in 0 1 2; do
    out=$root/evaluations/graphvae/$setting/seed_$seed
    "$python" "$root/scripts/evaluate_dgl_common_metrics.py" --repo "$repo" \
      --generated "$out/generated_attributed_graphs.bin" --reference "$reference" \
      --output "$out/common_structural.json" --label "graphvae_$setting" \
      --seed "$seed" --device cuda --repeats 10 --skip-random-gin
  done
done

# Both structural and the two RandomGIN modes for the two completed DeFoG seeds.
for seed in 1 2; do
  out=$root/evaluations/defog/seed_$seed
  "$python" "$root/scripts/evaluate_dgl_common_metrics.py" --repo "$repo" \
    --generated "$out/generated_attributed_graphs.bin" --reference "$reference" \
    --output "$out/common_metrics.json" --label defog --seed "$seed" \
    --device cuda --repeats 10
done

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
