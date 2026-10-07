#!/usr/bin/env bash
# Three-seed QM9 DeFoG campaign on the same frozen train/val/test SMILES split.
# Epochs and training seeds match the GraphVAE fast250 benchmark. Batch 128 is
# required on the lab's 24-GiB TITAN RTX GPUs: batch 512 failed with CUDA OOM.
# The difference in batch/update budget must be disclosed in any comparison.
# DeFoG's best validation checkpoint is retained for later common evaluation.
set -euo pipefail

seed=${1:?training seed required}
gpu=${2:?GPU index required}
source_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/source
data_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/data/seed_0
python=/localhome/mirzaei/qm9_defog_lab24_20260914/env/defog-benchmark/bin/python
root=/localhome/mirzaei/qm9_defog_matched250_batch128_20260915
run_dir=$root/runs/seed_$seed
log=$root/logs/seed_$seed.log
status=$root/status/seed_$seed.status

mkdir -p "$run_dir" "$root/logs" "$root/status"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
printf 'RUNNING seed=%s gpu=%s host=%s epochs=250 batch=128 started=%s\n' \
  "$seed" "$gpu" "$(hostname)" "$(date --iso-8601=seconds)" | tee "$status"
set +e
"$python" -u src/main.py \
  +experiment=qm9_with_h \
  dataset.datadir="$data_root" \
  train.seed="$seed" \
  train.n_epochs=250 \
  train.batch_size=128 \
  train.save_model=true \
  general.name="qm9_frozen250_seed_$seed" \
  general.wandb=disabled \
  general.gpus=1 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e
if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s gpu=%s host=%s exit_code=%s finished=%s\n' \
  "$state" "$seed" "$gpu" "$(hostname)" "$code" "$(date --iso-8601=seconds)" | tee "$status"
exit "$code"
