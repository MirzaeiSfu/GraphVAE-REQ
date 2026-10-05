#!/usr/bin/env bash
set -eo pipefail

dataset="${1:?usage: run_lgd_seed.sh DATASET SEED GPU_ID}"
seed="${2:?usage: run_lgd_seed.sh DATASET SEED GPU_ID}"
gpu_id="${3:?usage: run_lgd_seed.sh DATASET SEED GPU_ID}"

repo_dir="$(cd "$(dirname "$0")" && pwd)"
campaign_root="${LGD_CAMPAIGN_ROOT:-$(dirname "$repo_dir")}" 
env_dir="${LGD_ENV_DIR:-$campaign_root/env}"
result_root="$campaign_root/results"
dataset_dir="$repo_dir/datasets/graphvae_req"
mkdir -p "$result_root" "$campaign_root/logs"

if [[ -x "$env_dir/bin/python" ]]; then
  export CONDA_PREFIX="$env_dir"
  export PATH="$env_dir/bin:$PATH"
  export LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}"
else
  echo "LGD environment not found at $env_dir" >&2
  exit 1
fi
set -u

cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export WANDB_MODE=disabled

encoder_batch=32
diffusion_batch=32
encoder_epochs=200
diffusion_epochs=2000
if [[ "$dataset" == GRID || "$dataset" == TRIANGULAR_GRID ]]; then
  # These graphs are densified to N x N edge tensors; batch 32 exceeds the
  # 11--12 GiB lab GPUs even though the dataset itself is small.
  encoder_batch=4
  diffusion_batch=4
elif [[ "$dataset" == QM9 ]]; then
  # The generic batch size would make QM9 need millions of optimizer steps.
  # These are the batch sizes used by LGD's own unconditional QM9 configs.
  encoder_batch=256
  diffusion_batch=512
fi

encoder_tag="${dataset}_seed${seed}"
encoder_dir="$result_root/GraphVAEReq-encoder-${encoder_tag}/$seed"
diffusion_tag="${dataset}_seed${seed}"
diffusion_dir="$result_root/GraphVAEReq-diffusion-${diffusion_tag}/$seed"

echo "[$(date -Is)] dataset=$dataset seed=$seed gpu=$gpu_id encoder_start"
if [[ ! -f "$encoder_dir/COMPLETE" ]]; then
  python pretrain.py --cfg cfg/GraphVAEReq-encoder.yaml --repeat 1 \
    out_dir "$result_root" name_tag "$encoder_tag" \
    dataset.name "$dataset" dataset.dir "$dataset_dir" \
    seed "$seed" accelerator cuda:0 wandb.use False \
    train.batch_size "$encoder_batch" optim.max_epoch "$encoder_epochs"
  touch "$encoder_dir/COMPLETE"
fi

encoder_ckpt="$(python select_best_encoder_ckpt.py "$encoder_dir")"
printf '%s\n' "$encoder_ckpt" > "$encoder_dir/SELECTED_CHECKPOINT.txt"
echo "[$(date -Is)] dataset=$dataset seed=$seed encoder_done ckpt=$encoder_ckpt diffusion_start"

if [[ ! -f "$diffusion_dir/COMPLETE" ]]; then
  python train_diffusion.py --cfg cfg/GraphVAEReq-diffusion.yaml --repeat 1 \
    out_dir "$result_root" name_tag "$diffusion_tag" \
    dataset.name "$dataset" dataset.dir "$dataset_dir" \
    diffusion.first_stage_config "$encoder_ckpt" \
    seed "$seed" accelerator cuda:0 wandb.use False \
    train.batch_size "$diffusion_batch" optim.max_epoch "$diffusion_epochs"
  touch "$diffusion_dir/COMPLETE"
fi

echo "[$(date -Is)] dataset=$dataset seed=$seed complete"
