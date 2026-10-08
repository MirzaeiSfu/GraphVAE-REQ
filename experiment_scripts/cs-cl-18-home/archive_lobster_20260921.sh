#!/usr/bin/env bash
set -u

ROOT=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/LOBSTER
LOG="$ROOT/manifests/transfer.log"
mkdir -p "$ROOT"/{experiments/graphvae/motif_false,experiments/graphvae/motif_true_total_count,experiments/graphvae/motif_true_full_matrix,experiments/defog,experiments/lgd,metrics,reports,manifests}
date -Is > "$ROOT/manifests/transfer_started.txt"

run_copy() {
  echo "[$(date -Is)] START $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] END rc=$rc $*" | tee -a "$LOG"
  return "$rc"
}

for seed in 0 1 2; do
  run_copy rsync -a --info=progress2 "mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster/seed_${seed}/" "$ROOT/experiments/graphvae/motif_false/seed_${seed}/"
  run_copy rsync -a --info=progress2 "mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster/total_count/seed_${seed}/" "$ROOT/experiments/graphvae/motif_true_total_count/seed_${seed}/"
  run_copy rsync -a --info=progress2 "mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster/full_matrix/seed_${seed}/" "$ROOT/experiments/graphvae/motif_true_full_matrix/seed_${seed}/"
  run_copy rsync -a --info=progress2 "mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/lobster/seed_${seed}/" "$ROOT/experiments/defog/seed_${seed}/"
done

run_copy rsync -a "mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920/results/GraphVAEReq-encoder-LOBSTER_seed2/" "$ROOT/experiments/lgd/encoder_seed_2/"
run_copy rsync -a "mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920/results/GraphVAEReq-diffusion-LOBSTER_seed2/" "$ROOT/experiments/lgd/diffusion_seed_2/"
run_copy rsync -a "mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/metrics/lobster_seed"'*.json' "$ROOT/metrics/"
run_copy rsync -a "mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/lobster_defog_historical_tv_20260913/" "$ROOT/metrics/defog_historical_tv/"

date -Is > "$ROOT/manifests/transfer_finished.txt"
find "$ROOT" -type f -printf '%P\t%s\n' | sort > "$ROOT/manifests/file_manifest.tsv"
du -sh "$ROOT" > "$ROOT/manifests/archive_size.txt"
echo "Archive transfer complete. Safe to exit this tmux session."
exec bash
