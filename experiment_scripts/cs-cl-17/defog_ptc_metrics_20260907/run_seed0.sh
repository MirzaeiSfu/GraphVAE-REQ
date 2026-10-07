#!/usr/bin/env bash
set -euo pipefail

seed=0
gpu=0
root=/localhome/mirzaei/defog_ptc_metrics_20260907
source_root="$root/repo/source"
repo_root="$root/repo"
python_bin="$root/venv/bin/python"
out_dir="$root/generated_240/seed_0"

mkdir -p "$out_dir"
cd "$source_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo_root/graph_evaluation/src:$source_root"

exec "$python_bin" src/main.py \
  +experiment=ptc \
  dataset=frozen_graphvae \
  dataset.identity=PTC \
  dataset.root="$root/artifacts/ptc" \
  dataset.train_sha256=0750c12d4a0b7886a7175610aa780c699fd2c077e2d692d0d358d23780522dbf \
  dataset.validation_sha256=4814254636b5a1e473c49d801712e41bbf79505377bf1f0d399d8407bba8a4a6 \
  dataset.reference_sha256=aa0fa76e2a90491b7343aee12284e7127813317ef38ab52c8d849d74162558b9 \
  dataset.feature_mode=decoded_node \
  'dataset.feature_schema="gin-node-label-v2|export=decoded_node"' \
  train.seed="$seed" \
  train.batch_size=12 \
  general.wandb=disabled \
  general.validation_selection=loss \
  general.generation_seed=12345 \
  general.strict_generation=true \
  general.defog_commit=c631697b9cd5a2474d22ba12de33943c6b49e53e \
  general.name=ptc_seed0_generate240 \
  general.test_only="$root/checkpoints/seed_0.ckpt" \
  general.final_model_samples_to_generate=240 \
  general.final_model_samples_to_save=0 \
  general.final_model_chains_to_save=0 \
  hydra.run.dir="$out_dir"
