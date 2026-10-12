#!/usr/bin/env bash
set -euo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
source_root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
campaign_root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910
checkpoint="$campaign_root/runs/seed_$seed/checkpoints/proteins_3seed_20260910_seed_$seed/last.ckpt"
output_dir="$campaign_root/corrected_generation_20260922/seed_$seed"
status="$output_dir/status.txt"
generation_seed=$((9400 + seed))

mkdir -p "$output_dir"
test -s "$checkpoint"
if (( $(df --output=avail -B1 /local-scratch2 | tail -1) < 16106127360 )); then
  echo "FAILED insufficient disk space" >"$status"
  exit 75
fi

echo "GENERATING training_seed=$seed generation_seed=$generation_seed host=$(hostname)" >"$status"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$source_root:$source_root/src"
export DGL_DOWNLOAD_DIR="$source_root/.dgl_cache"

"$python_bin" -u src/main.py \
  dataset=proteins +experiment=proteins \
  dataset.datadir=data/proteins_3seed_20260910/ \
  train.seed="$seed" \
  general.generation_seed="$generation_seed" \
  general.name="proteins_seed_${seed}_corrected_generation_20260922" \
  general.wandb=disabled \
  general.test_only="$checkpoint" \
  general.final_model_samples_to_generate=210 \
  general.final_model_samples_to_save=210 \
  general.final_model_chains_to_save=0 \
  general.evaluate_all_checkpoints=false \
  hydra.run.dir="$output_dir" \
  >"$output_dir/generation.log" 2>&1

test -s "$output_dir/generated_graphs.pt"
sha256sum "$checkpoint" >"$output_dir/checkpoint.sha256"
sha256sum "$output_dir/generated_graphs.pt" >"$output_dir/generated_graphs.sha256"
printf 'COMPLETE training_seed=%s generation_seed=%s finished=%s\n' \
  "$seed" "$generation_seed" "$(date --iso-8601=seconds)" >"$status"
