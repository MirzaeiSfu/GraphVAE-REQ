#!/usr/bin/env bash
#SBATCH --job-name=lgd_heavy
#SBATCH --partition=schulte-lab-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/mirzaei/LGD_3SEED_CAMPAIGN_20260920/slurm/%A_%a.out
#SBATCH --error=/home/mirzaei/LGD_3SEED_CAMPAIGN_20260920/slurm/%A_%a.err

set -euo pipefail
campaign=/home/mirzaei/LGD_3SEED_CAMPAIGN_20260920
mkdir -p "$campaign/slurm" "$campaign/logs"

# Deployment is unpacked once on the shared filesystem. Jobs may be scheduled
# while that one-time setup is still finishing, so wait rather than fail.
for _ in $(seq 1 120); do
  [[ -f "$campaign/setup.ok" ]] && break
  sleep 30
done
[[ -f "$campaign/setup.ok" ]]

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  dataset=QM9
  seed="$SLURM_ARRAY_TASK_ID"
else
  dataset=AIDS
  seed="$((SLURM_ARRAY_TASK_ID - 3))"
fi

export LGD_CAMPAIGN_ROOT="$campaign"
export LGD_ENV_DIR="$campaign/env"
exec "$campaign/repo/run_lgd_seed.sh" "$dataset" "$seed" 0 \
  > "$campaign/logs/${dataset}_seed${seed}.log" 2>&1
