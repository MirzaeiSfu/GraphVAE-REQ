#!/usr/bin/env bash
set -euo pipefail
run_id=${1:?run id required}
seed=${2:?seed required}

repo=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
root=/home/mirzaei/aids_full_matrix_priority_20260910
env_dir=/home/mirzaei/miniconda3/envs/micro_env
env_archive=/project/cs-schulte-lab/ali/micro_env.tar.gz
motif_cache=/home/mirzaei/graphvae_motif_cache_20260910
aids_data=/project/cs-schulte-lab/ali/GraphVAE-REQ-aids-enzymes-top100-20260714/data_raw
config=configs/multihop_smoothed/aids_full_matrix_20260903.yaml
run_dir=$root/runs/seed_$seed
dataset_cache=$root/dataset_cache/seed_$seed
log=$root/logs/aids_full_seed_$seed.log
status=$root/status/aids_full_seed_$seed.status
mkdir -p "$run_dir" "$dataset_cache" "$root/logs" "$root/status"

if [[ ! -x "$env_dir/bin/python" ]]; then
  mkdir -p /home/mirzaei/miniconda3/envs
  (
    flock -x 9
    if [[ ! -x "$env_dir/bin/python" ]]; then
      mkdir -p "$env_dir"
      tar -xzf "$env_archive" -C "$env_dir"
    fi
  ) 9>/home/mirzaei/miniconda3/envs/.micro_env_extract.lock
fi

cd "$repo"
export PATH="$env_dir/bin:$PATH"
export PYTHONPATH="$repo/vendor/python:${PYTHONPATH:-}"
export DGL_DOWNLOAD_DIR=/home/mirzaei/.dgl
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

printf 'RUNNING host=%s job=%s seed=%s motif_weight=0.1 started=%s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$seed" "$(date --iso-8601=seconds)" | tee "$status"

set +e
srun "$env_dir/bin/python" -u main.py \
  --config "$config" \
  --seed "$seed" \
  --third_party_eval_seed "$seed" \
  --data_dir "$aids_data" \
  --train_batch_size 200 \
  --alpha_motif_loss 0.1 \
  --alpha_syntactic_literal_motif_loss 0.1 \
  --motif_cache_dir "$motif_cache" \
  --dataset_cache_dir "$dataset_cache" \
  --disable_dataset_cache false \
  --require_existing_dataset_cache false \
  --motif_prune_max_values_per_rule 64 \
  --motif_prune_score_threshold 5.0 \
  --motif_batch_size 50 \
  --checkpoint_interval_epochs 1000 \
  --resume_from_latest_checkpoint true \
  --graph_save_path "$run_dir" \
  --run_label "$run_id" 2>&1 | tee "$log"
code=${PIPESTATUS[0]}
set -e

if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s host=%s job=%s seed=%s exit_code=%s finished=%s\n' \
  "$state" "$(hostname)" "$SLURM_JOB_ID" "$seed" "$code" "$(date --iso-8601=seconds)" | tee "$status"
exit "$code"
