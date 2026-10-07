#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
status=$root/status/motif_tv.status
for required in graphvae_true_full graphvae_false defog; do
  while ! grep -q '^COMPLETE' "$root/status/$required.status" 2>/dev/null; do sleep 30; done
done
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$status"
reference=$root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin
dataset_cache=$(find "$root/cache/dataset" -maxdepth 1 -name 'AIDS_*.pkl' -print -quit)
config=$root/graphvae/true_full/seed_0/run_config_used.yaml
args=()
for setting in true_full false; do
  for seed in 0 1 2; do
    out=$root/evaluations/graphvae/$setting/seed_$seed
    args+=(--item "graphvae_$setting" "$seed" "$out/generated_attributed_graphs.bin" "$out/motif_tv.json")
  done
done
for seed in 1 2; do
  out=$root/evaluations/defog/seed_$seed
  args+=(--item defog "$seed" "$out/generated_attributed_graphs.bin" "$out/motif_tv.json")
done
export PYTHONPATH="$repo:$repo/scripts:${PYTHONPATH:-}"
"$python" -u "$root/scripts/evaluate_aids_motif_tv_batch.py" \
  --repo "$repo" --config "$config" --dataset-cache "$dataset_cache" \
  --motif-cache-dir "$root/cache/motif" --reference "$reference" \
  --device cpu --batch-size 16 "${args[@]}"
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
