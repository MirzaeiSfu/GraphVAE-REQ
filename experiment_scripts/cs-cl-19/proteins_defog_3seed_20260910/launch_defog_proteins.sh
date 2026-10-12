#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
source_root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
run_dir=$root/runs/seed_$seed
log=$root/logs/proteins_seed_$seed.log
status=$root/status/proteins_seed_$seed.status
mkdir -p "$run_dir" "$root/logs" "$root/status"

printf 'RUNNING host=%s gpu=%s seed=%s started=%s\n' \
  "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
cd "$source_root" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:$source_root/src"
export WANDB_MODE=disabled
export DGL_DOWNLOAD_DIR="$source_root/.dgl_cache"

"$python_bin" src/main.py \
  dataset=proteins \
  +experiment=proteins \
  dataset.datadir=data/proteins_3seed_20260910/ \
  train.seed="$seed" \
  train.n_epochs=1307 \
  train.batch_size=16 \
  train.save_model=true \
  general.name="proteins_3seed_20260910_seed_$seed" \
  general.wandb=disabled \
  general.check_val_every_n_epochs=22 \
  general.sample_every_val=1 \
  general.samples_to_generate=40 \
  general.final_model_samples_to_generate=210 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
  "$state" "$(hostname)" "$gpu" "$seed" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
