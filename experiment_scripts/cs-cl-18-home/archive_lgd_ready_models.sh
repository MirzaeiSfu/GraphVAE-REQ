#!/usr/bin/env bash
set -u
root=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921
exec >> "$root/lgd_model_transfer.log" 2>&1
for host in 16 18 26; do
  base=/local-scratch/mirzaei/LGD_3SEED_CAMPAIGN_20260920
  [ "$host" = 18 ] && base=/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920
  for ds in MUTAG PTC LOBSTER TRIANGULAR_GRID; do
    for seed in 0 1 2; do
      if ssh "mirzaei@cs-cl-$host.cmpt.sfu.ca" "test -f $base/results/GraphVAEReq-diffusion-${ds}_seed$seed/$seed/COMPLETE"; then
        mkdir -p "$root/$ds/experiments/lgd"
        for stage in encoder diffusion; do
          name=GraphVAEReq-${stage}-${ds}_seed$seed
          rsync -a --partial "mirzaei@cs-cl-$host.cmpt.sfu.ca:$base/results/$name/" "$root/$ds/experiments/lgd/$name/" || echo "FAILED $host $name"
        done
        rsync -a --exclude=__pycache__ --exclude=.git --exclude=logs --exclude=datasets "mirzaei@cs-cl-$host.cmpt.sfu.ca:$base/repo/" "$root/$ds/experiments/lgd/source_cs-cl-$host/"
        rsync -a "mirzaei@cs-cl-$host.cmpt.sfu.ca:$base/repo/datasets/graphvae_req/$ds/" "$root/$ds/experiments/lgd/frozen_dataset/"
      fi
    done
  done
done
date -Is
