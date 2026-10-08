#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/LOBSTER
mkdir -p "$root"/{graphvae_motif_false,graphvae_motif_true_linkcorr,graphvae_motif_true_linkcorr_off,defog,evaluations/historical_training_reference_tv,reproducibility,reports}

ssh mirzaei@cs-cl-16.cmpt.sfu.ca 'tar -C /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed -cf - lobster' \
  | tar -C "$root/graphvae_motif_false" -xf -
ssh mirzaei@cs-cl-16.cmpt.sfu.ca 'tar -C /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed -cf - lobster' \
  | tar -C "$root/graphvae_motif_true_linkcorr" -xf -
ssh mirzaei@cs-cl-16.cmpt.sfu.ca 'tar -C /local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed -cf - lobster' \
  | tar -C "$root/graphvae_motif_true_linkcorr_off" -xf -
ssh mirzaei@cs-cl-16.cmpt.sfu.ca 'tar -C /local-scratch/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval -cf - jobs/lobster lobster' \
  | tar -C "$root/defog" -xf -
ssh mirzaei@cs-cl-18.cmpt.sfu.ca 'tar -C /local-scratch2/mirzaei/lobster_defog_historical_tv_20260913 -cf - .' \
  | tar -C "$root/evaluations/historical_training_reference_tv" -xf -
ssh mirzaei@cs-cl-18.cmpt.sfu.ca 'tar -C /local-scratch2/mirzaei/lobster_historical_tv_inputs -cf - .' \
  | tar -C "$root/reproducibility" -xf -
date -Is > "$root/ARCHIVE_COMPLETE"
du -sh "$root" > "$root/ARCHIVE_SIZE.txt"
