#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
false_root=/local-scratch2/mirzaei/aids_false_defog_3seed_20260912
solar='ssh -p 24 mirzaei@solar.cs.sfu.ca'

mkdir -p "$root"/{status,logs,cache/dataset,cache/motif,source,graphvae/true_full,graphvae/false,defog/checkpoints}
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$root/status/prepare.status"

# Freeze the exact code and caches used by the completed AIDS campaigns.
rsync -a --delete \
  /local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ/ \
  "$root/source/GraphVAE-REQ/"
cp -a "$false_root/dataset_cache/shared/"*.pkl "$root/cache/dataset/"
cp -a /local-scratch2/mirzaei/fb/GraphVAE-REQ/cache_motifs/multihop_smoothed/aids_undir_feat_multi.pkl "$root/cache/motif/"

# Motif=True full: evaluate the explicit epoch-10,000 checkpoints, not a later
# or best-validation model.
for seed in 0 1 2; do
  out="$root/graphvae/true_full/seed_$seed"
  mkdir -p "$out"
  rsync -a -e 'ssh -p 24' \
    "mirzaei@solar.cs.sfu.ca:/home/mirzaei/aids_full_matrix_priority_20260910/runs/seed_$seed/periodic_epoch_10000.pt" \
    "$out/best_validation_mmd_model"
  rsync -a -e 'ssh -p 24' \
    "mirzaei@solar.cs.sfu.ca:/home/mirzaei/aids_full_matrix_priority_20260910/runs/seed_$seed/run_config_used.yaml" \
    "$out/run_config_used.yaml"
done

# Motif=False seeds 0 and 2 are local; seed 1 was trained on cs-cl-19.
for seed in 0 2; do
  mkdir -p "$root/graphvae/false/seed_$seed"
  cp -a "$false_root/runs/graphvae_false/seed_$seed/best_validation_mmd_model" \
    "$false_root/runs/graphvae_false/seed_$seed/run_config_used.yaml" \
    "$root/graphvae/false/seed_$seed/"
done
mkdir -p "$root/graphvae/false/seed_1"
rsync -a "mirzaei@cs-cl-19.cmpt.sfu.ca:$false_root/runs/graphvae_false/seed_1/best_validation_mmd_model" \
  "mirzaei@cs-cl-19.cmpt.sfu.ca:$false_root/runs/graphvae_false/seed_1/run_config_used.yaml" \
  "$root/graphvae/false/seed_1/"

# Completed DeFoG seeds. Seed 0 did not complete and is deliberately excluded.
cp -a "$false_root/runs/defog/seed_2/checkpoints/aids_defog_seed_2/last.ckpt" \
  "$root/defog/checkpoints/seed_2.ckpt"
rsync -a \
  "mirzaei@cs-cl-19.cmpt.sfu.ca:$false_root/runs/defog/seed_1/checkpoints/aids_defog_seed_1/last.ckpt" \
  "$root/defog/checkpoints/seed_1.ckpt"

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$root/status/prepare.status"
