#!/usr/bin/env bash
set -euo pipefail

seed=${1:?training seed}
gpu=${2:?GPU index}
generation_seed=$((9200 + seed))
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
root=/local-scratch2/mirzaei/mutag_defog_independent_20260919
run=$root/seed_$seed

mkdir -p "$run"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$repo:${PYTHONPATH:-}"
cd "$repo"

"$python" -u src/main.py dataset=mutag +experiment=mutag \
  dataset.datadir=data/mutag_edgefeat/ \
  train.seed="$seed" train.n_epochs=20000 train.batch_size=131 \
  train.save_model=true \
  general.generation_seed="$generation_seed" \
  general.name="mutag_edgefeat_independent_seed_${seed}" \
  general.wandb=disabled general.gpus=1 \
  general.check_val_every_n_epochs=1000 general.sample_every_val=1 \
  general.samples_to_generate=40 \
  general.final_model_samples_to_generate=39 \
  general.final_model_samples_to_save=39 \
  general.final_model_chains_to_save=0 \
  hydra.run.dir="$run" \
  2>&1 | tee "$root/seed_${seed}.log"

checkpoint="$run/checkpoints/mutag_edgefeat_independent_seed_${seed}/last.ckpt"
generation_run="$run/generation"
mkdir -p "$generation_run"
"$python" -u src/main.py dataset=mutag +experiment=mutag \
  dataset.datadir=data/mutag_edgefeat/ \
  train.seed="$seed" train.batch_size=131 \
  general.test_only="$checkpoint" \
  general.generation_seed="$generation_seed" \
  general.final_model_samples_to_generate=39 \
  general.final_model_samples_to_save=39 \
  general.final_model_chains_to_save=0 \
  general.evaluate_all_checkpoints=false \
  general.name="mutag_edgefeat_independent_seed_${seed}_generation" \
  general.wandb=disabled general.gpus=1 \
  hydra.run.dir="$generation_run" \
  2>&1 | tee "$root/seed_${seed}_generation.log"

sha256sum "$run"/checkpoints/*/*.ckpt > "$run/checkpoint.sha256"
sha256sum "$generation_run/generated_graphs.pt" > "$run/generated_graphs.sha256"
printf 'COMPLETE training_seed=%s generation_seed=%s finished=%s\n' \
  "$seed" "$generation_seed" "$(date --iso-8601=seconds)" > "$run/status.txt"
