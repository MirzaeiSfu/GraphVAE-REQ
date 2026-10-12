#!/usr/bin/env bash
set -euo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
root=/local-scratch2/mirzaei/aids_defog_independent_20260922
run=$root/seed_$seed
generation_seed=$((9300 + seed))

mkdir -p "$run"
if (( $(df --output=avail -B1 /local-scratch2 | tail -1) < 16106127360 )); then
  echo "FAILED insufficient disk space" >"$run/status.txt"
  exit 75
fi

export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$repo:${PYTHONPATH:-}"
cd "$repo"
echo "TRAINING seed=$seed generation_seed=$generation_seed host=$(hostname)" >"$run/status.txt"

"$python_bin" -u src/main.py dataset=aids +experiment=aids \
  dataset.datadir=data/aids_3seed_20260912/ \
  train.seed="$seed" train.n_epochs=1592 train.batch_size=16 \
  train.save_model=true \
  general.generation_seed="$generation_seed" \
  general.name="aids_defog_independent_seed_$seed" \
  general.wandb=disabled general.gpus=1 \
  general.final_model_samples_to_generate=400 \
  general.final_model_samples_to_save=400 \
  general.final_model_chains_to_save=0 \
  hydra.run.dir="$run" \
  2>&1 | tee "$root/seed_${seed}.log"

checkpoint="$run/checkpoints/aids_defog_independent_seed_$seed/last.ckpt"
generation_run="$run/generation"
mkdir -p "$generation_run"
echo "GENERATING seed=$seed generation_seed=$generation_seed host=$(hostname)" >"$run/status.txt"
"$python_bin" -u src/main.py dataset=aids +experiment=aids \
  dataset.datadir=data/aids_3seed_20260912/ \
  train.seed="$seed" train.batch_size=16 \
  general.test_only="$checkpoint" \
  general.generation_seed="$generation_seed" \
  general.final_model_samples_to_generate=400 \
  general.final_model_samples_to_save=400 \
  general.final_model_chains_to_save=0 \
  general.evaluate_all_checkpoints=false \
  general.name="aids_defog_independent_seed_${seed}_generation" \
  general.wandb=disabled general.gpus=1 \
  hydra.run.dir="$generation_run" \
  2>&1 | tee "$root/seed_${seed}_generation.log"

sha256sum "$run"/checkpoints/*/*.ckpt >"$run/checkpoint.sha256"
sha256sum "$generation_run/generated_graphs.pt" >"$run/generated_graphs.sha256"
printf 'COMPLETE training_seed=%s generation_seed=%s finished=%s\n' \
  "$seed" "$generation_seed" "$(date --iso-8601=seconds)" >"$run/status.txt"
