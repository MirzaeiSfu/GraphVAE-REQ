#!/usr/bin/env bash
# Corrected LGD pipeline (2026-09-24). Differences from run_lgd_seed.sh:
#  1. encoder checkpoint chosen by lowest validation loss_recon (never epoch 0);
#     the old selector used the untrained graph-head `loss` and always chose epoch 0.
#  2. optional reuse of an already-trained encoder run (REUSE_ENCODER_DIR).
#  3. after diffusion: sample with sample_tu.py and write GATE.json
#     (edgeless fraction, edge ratio vs reference).
# usage: run_lgd_fixed.sh DATASET SEED GPU_ID
# env:   LGD_CAMPAIGN_ROOT, LGD_ENV_DIR, REUSE_ENCODER_DIR, DIFFUSION_EPOCHS, ENCODER_EPOCHS
set -eo pipefail

dataset="${1:?usage: run_lgd_fixed.sh DATASET SEED GPU_ID}"
seed="${2:?usage: run_lgd_fixed.sh DATASET SEED GPU_ID}"
gpu_id="${3:?usage: run_lgd_fixed.sh DATASET SEED GPU_ID}"

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
encoder_epochs="${ENCODER_EPOCHS:-200}"
diffusion_epochs="${DIFFUSION_EPOCHS:-2000}"
if [[ "$dataset" == GRID || "$dataset" == TRIANGULAR_GRID ]]; then
  encoder_batch=4
  diffusion_batch=4
elif [[ "$dataset" == QM9 ]]; then
  encoder_batch=256
  diffusion_batch=512
  diffusion_epochs="${DIFFUSION_EPOCHS:-300}"
fi

tag="${dataset}_seed${seed}"
encoder_dir="$result_root/GraphVAEReq-encoder-${tag}/$seed"
diffusion_dir="$result_root/GraphVAEReq-diffusion-${tag}/$seed"

echo "[$(date -Is)] dataset=$dataset seed=$seed gpu=$gpu_id host=$(hostname) encoder_start"
if [[ ! -f "$encoder_dir/COMPLETE" ]]; then
  if [[ -n "${REUSE_ENCODER_DIR:-}" && -f "$REUSE_ENCODER_DIR/COMPLETE" ]]; then
    mkdir -p "$(dirname "$encoder_dir")"
    cp -a "$REUSE_ENCODER_DIR" "$encoder_dir"
    rm -f "$encoder_dir/SELECTED_CHECKPOINT.txt"
    echo "reused encoder from $REUSE_ENCODER_DIR" > "$encoder_dir/REUSED_FROM.txt"
    echo "[$(date -Is)] reused trained encoder $REUSE_ENCODER_DIR"
  else
    python pretrain.py --cfg cfg/GraphVAEReq-encoder.yaml --repeat 1 \
      out_dir "$result_root" name_tag "$tag" \
      dataset.name "$dataset" dataset.dir "$dataset_dir" \
      seed "$seed" accelerator cuda:0 wandb.use False \
      train.batch_size "$encoder_batch" optim.max_epoch "$encoder_epochs"
    touch "$encoder_dir/COMPLETE"
  fi
fi

encoder_ckpt="$(python select_best_encoder_ckpt.py "$encoder_dir")"
printf '%s\n' "$encoder_ckpt" > "$encoder_dir/SELECTED_CHECKPOINT.txt"
echo "[$(date -Is)] encoder selected by val loss_recon: $encoder_ckpt"

if [[ ! -f "$diffusion_dir/COMPLETE" ]]; then
  echo "[$(date -Is)] diffusion_start epochs=$diffusion_epochs batch=$diffusion_batch"
  python train_diffusion.py --cfg cfg/GraphVAEReq-diffusion.yaml --repeat 1 \
    out_dir "$result_root" name_tag "$tag" \
    dataset.name "$dataset" dataset.dir "$dataset_dir" \
    diffusion.first_stage_config "$encoder_ckpt" \
    seed "$seed" accelerator cuda:0 wandb.use False \
    train.batch_size "$diffusion_batch" optim.max_epoch "$diffusion_epochs"
  touch "$diffusion_dir/COMPLETE"
fi

if [[ ! -f "$diffusion_dir/sampled/GATE.json" ]]; then
  echo "[$(date -Is)] sampling"
  python sample_tu.py --cfg cfg/GraphVAEReq-diffusion.yaml --run-id "$seed" \
    out_dir "$result_root" name_tag "$tag" \
    dataset.name "$dataset" dataset.dir "$dataset_dir" \
    diffusion.first_stage_config "$encoder_ckpt" \
    seed "$seed" accelerator cuda:0 wandb.use False train.batch_size "$diffusion_batch" || true
  latest="$(ls -t "$diffusion_dir"/sampled/epoch_*_graphs.pkl 2>/dev/null | head -1)"
  if [[ -n "$latest" ]]; then
    python "$repo_dir/check_generated.py" "$latest"
  else
    echo "[$(date -Is)] sampling produced no graphs pickle" >&2
  fi
fi
echo "[$(date -Is)] dataset=$dataset seed=$seed complete"
