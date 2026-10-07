#!/usr/bin/env bash
set -uo pipefail

source_root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
campaign_root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910
checkpoint="$campaign_root/runs/seed_0/checkpoints/proteins_3seed_20260910_seed_0/last.ckpt"
output_dir="$campaign_root/final_eval/seed_0"
log="$campaign_root/logs/proteins_seed_0_final_eval.log"

mkdir -p "$output_dir"
cd "$source_root" || exit 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:$source_root/src"
export WANDB_MODE=disabled
export DGL_DOWNLOAD_DIR="$source_root/.dgl_cache"

"$python_bin" src/main.py \
  dataset=proteins \
  +experiment=proteins \
  dataset.datadir=data/proteins_3seed_20260910/ \
  train.seed=0 \
  general.name=proteins_3seed_20260910_seed_0_final_eval \
  general.wandb=disabled \
  general.test_only="$checkpoint" \
  general.final_model_samples_to_generate=210 \
  general.final_model_samples_to_save=30 \
  hydra.run.dir="$output_dir" 2>&1 | tee "$log"
