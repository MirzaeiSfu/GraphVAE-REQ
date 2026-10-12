#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/qm9_common_eval_20260917
repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
for status in graphvae_true_full graphvae_false defog_export; do
  while ! grep -q '^COMPLETE' "$root/status/$status.status" 2>/dev/null; do sleep 30; done
done

reference=$root/evaluations/graphvae/true_full/seed_0/reference_attributed_graphs.bin
config=$root/graphvae/true_full/seed_0/run_config_used.yaml
dataset_cache=$root/cache/dataset/QM9_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-123_bfs-legacy_first_component_features-default.pkl
args=()
for setting in true_full false; do
  for seed in 0 1 2; do
    directory=$root/evaluations/graphvae/$setting/seed_$seed
    args+=(--item "graphvae_$setting" "$seed" "$directory/generated_attributed_graphs.bin" "$directory/motif_tv.json")
  done
done
for seed in 1 2; do
  directory=$root/evaluations/defog/seed_$seed
  args+=(--item defog "$seed" "$directory/generated_attributed_graphs.bin" "$directory/motif_tv.json")
done

"$python" "$root/evaluate_qm9_motif_tv_batch.py" \
  --repo "$repo" --config "$config" \
  --dataset-cache "$dataset_cache" \
  --motif-cache-dir "$root/cache/motif" \
  --reference "$reference" \
  --device cpu --batch-size 16 "${args[@]}"

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$root/status/motif_tv.status"
