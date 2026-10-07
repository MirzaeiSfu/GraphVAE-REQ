#!/usr/bin/env bash
set -uo pipefail

gpu=${1:-0}
train_seed=3
generation_seed=1003
campaign=/local-scratch2/mirzaei/mutag_defog_replacement_seed3_20260912
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
train_dir=$campaign/train_seed_3
sample_dir=$campaign/final_samples_generation_seed_1003
status=$campaign/status.txt

mkdir -p "$train_dir" "$sample_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTHONPATH="$repo:$repo/src"

printf 'TRAINING host=%s gpu=%s train_seed=%s generation_seed=%s started=%s\n' \
  "$(hostname)" "$gpu" "$train_seed" "$generation_seed" "$(date --iso-8601=seconds)" > "$status"
cd "$repo" || exit 2
"$python" src/main.py \
  dataset=mutag \
  +experiment=mutag \
  dataset.datadir=data/mutag_edgefeat/ \
  train.seed="$train_seed" \
  train.n_epochs=20000 \
  train.batch_size=131 \
  train.save_model=true \
  general.name=mutag_edgefeat_defog_seed_3 \
  general.wandb=disabled \
  general.check_val_every_n_epochs=1000 \
  general.sample_every_val=1 \
  general.samples_to_generate=40 \
  general.final_model_samples_to_generate=39 \
  hydra.run.dir="$train_dir" \
  2>&1 | tee "$campaign/train.log"
code=${PIPESTATUS[0]}
if (( code != 0 )); then
  printf 'FAILED stage=train exit_code=%s finished=%s\n' "$code" "$(date --iso-8601=seconds)" > "$status"
  exit "$code"
fi

checkpoint=$(find "$train_dir/checkpoints" -type f -name 'last.ckpt' -print -quit)
if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
  printf 'FAILED stage=checkpoint_lookup finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
  exit 3
fi
printf 'SAMPLING checkpoint=%s train_seed=%s generation_seed=%s started=%s\n' \
  "$checkpoint" "$train_seed" "$generation_seed" "$(date --iso-8601=seconds)" > "$status"
"$python" src/main.py \
  dataset=mutag \
  +experiment=mutag \
  dataset.datadir=data/mutag_edgefeat/ \
  train.seed="$generation_seed" \
  train.batch_size=131 \
  general.test_only="$checkpoint" \
  general.name=mutag_edgefeat_defog_seed_3_final_sampling_seed_1003 \
  general.wandb=disabled \
  general.save_samples=true \
  general.final_model_samples_to_generate=39 \
  hydra.run.dir="$sample_dir" \
  2>&1 | tee "$campaign/final_sampling.log"
code=${PIPESTATUS[0]}
state=FAILED
(( code == 0 )) && state=COMPLETE
printf '%s host=%s gpu=%s train_seed=%s generation_seed=%s checkpoint=%s exit_code=%s finished=%s\n' \
  "$state" "$(hostname)" "$gpu" "$train_seed" "$generation_seed" "$checkpoint" "$code" \
  "$(date --iso-8601=seconds)" > "$status"
exit "$code"
