#!/usr/bin/env bash
set -euo pipefail

seed=${1:?seed required}
gpu=${2:?GPU index required}
root=/localhome/mirzaei/qm9_defog_lab24_20260914
source_root=$root/work/source
python=$root/env/defog-benchmark/bin/python
run_dir=$root/work/runs/seed_$seed
log=$root/work/logs/seed_$seed.log
status=$root/work/status/seed_$seed.status

mkdir -p "$run_dir" "$root/work/logs" "$root/work/status"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

printf 'RUNNING seed=%s gpu=%s host=%s started=%s\n' \
  "$seed" "$gpu" "$(hostname)" "$(date --iso-8601=seconds)" | tee "$status"
set +e
"$python" -u src/main.py \
  +experiment=qm9_with_h \
  dataset.datadir="$root/work/data/seed_$seed" \
  train.seed="$seed" \
  train.n_epochs=250 \
  train.batch_size=512 \
  train.save_model=true \
  general.name="qm9_matched_split_seed_$seed" \
  general.wandb=disabled \
  general.gpus=1 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e
if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s gpu=%s host=%s exit_code=%s finished=%s\n' \
  "$state" "$seed" "$gpu" "$(hostname)" "$code" "$(date --iso-8601=seconds)" | tee "$status"
exit "$code"
