#!/usr/bin/env bash
# Corrected QM9 DeFoG seed campaign.  All seeds share the same frozen
# train/validation/test split; train.seed changes model/training randomness.
# Batch 128 is the largest stable batch on the lab TITAN RTX 24-GiB GPUs.
set -euo pipefail

seed=${1:?training seed required}
gpu=${2:?GPU index required}
source_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/source
data_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/data/seed_0
python=/localhome/mirzaei/qm9_defog_lab24_20260914/env/defog-benchmark/bin/python
root=/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
run_dir=$root/runs/seed_$seed
log=$root/logs/seed_$seed.log
status=$root/status/seed_$seed.status

mkdir -p "$run_dir" "$root/logs" "$root/status"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

printf 'RUNNING corrected_seed_fix=true seed=%s gpu=%s host=%s epochs=250 batch=128 started=%s\n' \
  "$seed" "$gpu" "$(hostname)" "$(date --iso-8601=seconds)" | tee "$status"

set +e
"$python" -u src/main.py \
  +experiment=qm9_with_h \
  dataset.datadir="$data_root" \
  train.seed="$seed" \
  train.n_epochs=250 \
  train.batch_size=128 \
  train.save_model=true \
  general.name="qm9_seedfixed250_seed_$seed" \
  general.wandb=disabled \
  general.gpus=1 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e

if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s corrected_seed_fix=true seed=%s gpu=%s host=%s exit_code=%s finished=%s\n' \
  "$state" "$seed" "$gpu" "$(hostname)" "$code" "$(date --iso-8601=seconds)" | tee "$status"
exit "$code"
