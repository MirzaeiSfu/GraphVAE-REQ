#!/usr/bin/env bash
set -euo pipefail
stage=/local-scratch2/mirzaei/LGD_AIDS_SOLAR_CHECKPOINTS_20260921
campaign=/local-scratch/mirzaei/LGD_3SEED_CAMPAIGN_20260920
mkdir -p "$stage/results"
exec >> "$stage/transfer.log" 2>&1
for seed in 0 1 2; do
  for stage_name in encoder diffusion; do
    name=GraphVAEReq-${stage_name}-AIDS_seed${seed}
    rsync -a --partial -e 'ssh -p 24' "mirzaei@solar.cs.sfu.ca:/home/mirzaei/LGD_3SEED_CAMPAIGN_20260920/results/$name/" "$stage/results/$name/"
    rsync -a --partial "$stage/results/$name/" "mirzaei@cs-cl-26.cmpt.sfu.ca:$campaign/results/$name/"
    ssh mirzaei@cs-cl-19.cmpt.sfu.ca "mkdir -p /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/experiments/lgd/$name"
    rsync -a "$stage/results/$name/" "mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/experiments/lgd/$name/"
  done
  gpu=$((seed%2))
  ssh mirzaei@cs-cl-26.cmpt.sfu.ca "tmux new-session -d -s lgd_eval_aids_s$seed 'flock /tmp/mirzaei_lgd_eval_gpu$gpu bash /local-scratch/localhome/mirzaei/run_lgd_multi_eval.sh $campaign AIDS $seed $gpu /local-scratch/localhome/mirzaei'"
done
date -Is > "$stage/COMPLETE"
