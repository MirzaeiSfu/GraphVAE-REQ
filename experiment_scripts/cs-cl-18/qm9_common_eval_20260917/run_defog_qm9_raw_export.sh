#!/usr/bin/env bash
set -euo pipefail

seed=${1:?seed required}
gpu=${2:?GPU required}
source_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/source
data_root=/localhome/mirzaei/qm9_defog_lab24_20260914/work/data/seed_0
python=/localhome/mirzaei/qm9_defog_lab24_20260914/env/defog-benchmark/bin/python
campaign=/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
original_checkpoint=$campaign/runs/seed_$seed/checkpoints/qm9_seedfixed250_seed_$seed/epoch=239.ckpt
checkpoint=$campaign/common_eval_raw/seed_${seed}_selected_epoch239.ckpt
run_dir=$campaign/common_eval_raw/seed_$seed
log=$campaign/common_eval_raw/seed_$seed.log
status=$campaign/common_eval_raw/seed_$seed.status

mkdir -p "$run_dir" "$campaign/common_eval_raw"
ln -sfn "$original_checkpoint" "$checkpoint"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root/src:$source_root:${PYTHONPATH:-}"
printf 'RUNNING seed=%s checkpoint=%s started=%s\n' "$seed" "$checkpoint" "$(date --iso-8601=seconds)" > "$status"

set +e
"$python" -u src/main.py \
  +experiment=qm9_with_h \
  dataset.datadir="$data_root" \
  train.seed="$seed" \
  train.batch_size=128 \
  general.test_only="$checkpoint" \
  general.generation_seed="$seed" \
  general.final_model_samples_to_generate=512 \
  general.final_model_samples_to_save=512 \
  general.final_model_chains_to_save=0 \
  general.num_sample_fold=1 \
  general.evaluate_all_checkpoints=false \
  general.name="qm9_seedfixed250_seed_${seed}_common_eval" \
  general.wandb=disabled \
  general.gpus=1 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e

if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s exit_code=%s finished=%s\n' "$state" "$seed" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
