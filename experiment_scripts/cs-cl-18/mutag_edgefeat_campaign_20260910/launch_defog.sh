#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
repo=$root/source/DeFoG
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
run_dir=$root/runs/defog/seed_${seed}
log_dir=$root/logs
status_dir=$root/status
mkdir -p "$run_dir" "$log_dir" "$status_dir"
log=$log_dir/defog_edge_seed_${seed}.log
status=$status_dir/defog_edge_seed_${seed}.status

printf 'RUNNING host=%s gpu=%s seed=%s started=%s\n' "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo:$repo/src"
export WANDB_MODE=disabled
"$python_bin" src/main.py \
  dataset=mutag \
  +experiment=mutag \
  dataset.datadir=data/mutag_edgefeat/ \
  train.seed="$seed" \
  train.n_epochs=20000 \
  train.batch_size=131 \
  train.save_model=true \
  general.name="mutag_edgefeat_seed_${seed}" \
  general.wandb=disabled \
  general.check_val_every_n_epochs=1000 \
  general.sample_every_val=1 \
  general.samples_to_generate=40 \
  general.final_model_samples_to_generate=39 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$log"
exit_code=${PIPESTATUS[0]}
if (( exit_code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' "$state" "$(hostname)" "$gpu" "$seed" "$exit_code" "$(date --iso-8601=seconds)" > "$status"
exit "$exit_code"
