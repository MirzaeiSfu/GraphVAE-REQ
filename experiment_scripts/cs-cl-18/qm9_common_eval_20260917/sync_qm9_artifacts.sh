#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/qm9_common_eval_20260917
solar=mirzaei@solar.cs.sfu.ca
ssh_cmd='ssh -p 24'

mkdir -p "$root/cache/dataset" "$root/cache/motif" "$root/graphvae/true_full" "$root/graphvae/false" "$root/logs" "$root/status" "$root/provenance"
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$root/status/sync.status"

rsync -a --partial -e "$ssh_cmd" \
  "$solar:/home/mirzaei/qm9_full_matrix_cp_smoothed_top10_20260914/dataset_cache/" \
  "$root/cache/dataset/"
rsync -a --partial -e "$ssh_cmd" \
  "$solar:/home/mirzaei/qm9_full_matrix_cp_smoothed_top10_20260914/motif_cache/" \
  "$root/cache/motif/"

for seed in 0 1 2; do
  mkdir -p "$root/graphvae/true_full/seed_$seed" "$root/graphvae/false/seed_$seed"
  rsync -a -e "$ssh_cmd" \
    "$solar:/home/mirzaei/qm9_full_fast250_cp_smoothed_top10_20260914/runs/full_matrix/seed_$seed/best_validation_mmd_model" \
    "$solar:/home/mirzaei/qm9_full_fast250_cp_smoothed_top10_20260914/runs/full_matrix/seed_$seed/best_validation_mmd.json" \
    "$solar:/home/mirzaei/qm9_full_fast250_cp_smoothed_top10_20260914/runs/full_matrix/seed_$seed/run_config_used.yaml" \
    "$root/graphvae/true_full/seed_$seed/"
  rsync -a -e "$ssh_cmd" \
    "$solar:/home/mirzaei/qm9_motif_false_fast250_20260915/runs/seed_$seed/best_validation_mmd_model" \
    "$solar:/home/mirzaei/qm9_motif_false_fast250_20260915/runs/seed_$seed/best_validation_mmd.json" \
    "$solar:/home/mirzaei/qm9_motif_false_fast250_20260915/runs/seed_$seed/run_config_used.yaml" \
    "$root/graphvae/false/seed_$seed/"
done

rsync -a -e "$ssh_cmd" \
  "$solar:/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/data.py" \
  "$solar:/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/motif_counting/motif_counter.py" \
  "$root/provenance/"

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$root/status/sync.status"
