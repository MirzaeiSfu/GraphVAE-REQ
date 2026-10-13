#!/usr/bin/env bash
set -euo pipefail

repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
root=/local-scratch2/mirzaei/aids_defog_independent_20260919
run=$root/seed_3

mkdir -p "$run"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$repo:${PYTHONPATH:-}"
cd "$repo"

"$python" -u src/main.py dataset=aids +experiment=aids \
  dataset.datadir=data/aids_3seed_20260912/ \
  train.seed=3 train.n_epochs=1592 train.batch_size=16 \
  train.save_model=true \
  general.generation_seed=9303 \
  general.name=aids_defog_independent_seed_3 \
  general.wandb=disabled general.gpus=1 \
  general.final_model_samples_to_generate=400 \
  general.final_model_samples_to_save=400 \
  general.final_model_chains_to_save=0 \
  hydra.run.dir="$run" \
  2>&1 | tee "$root/seed_3.log"

checkpoint="$run/checkpoints/aids_defog_independent_seed_3/last.ckpt"
generation_run="$run/generation"
mkdir -p "$generation_run"
"$python" -u src/main.py dataset=aids +experiment=aids \
  dataset.datadir=data/aids_3seed_20260912/ \
  train.seed=3 train.batch_size=16 \
  general.test_only="$checkpoint" \
  general.generation_seed=9303 \
  general.final_model_samples_to_generate=400 \
  general.final_model_samples_to_save=400 \
  general.final_model_chains_to_save=0 \
  general.evaluate_all_checkpoints=false \
  general.name=aids_defog_independent_seed_3_generation \
  general.wandb=disabled general.gpus=1 \
  hydra.run.dir="$generation_run" \
  2>&1 | tee "$root/seed_3_generation.log"

sha256sum "$run"/checkpoints/*/*.ckpt > "$run/checkpoint.sha256"
sha256sum "$generation_run/generated_graphs.pt" > "$run/generated_graphs.sha256"
printf 'COMPLETE training_seed=3 generation_seed=9303 finished=%s\n' \
  "$(date --iso-8601=seconds)" > "$run/status.txt"
