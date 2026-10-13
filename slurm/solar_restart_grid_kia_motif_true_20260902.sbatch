#!/bin/bash
#SBATCH --job-name=gvae_grid_restart
#SBATCH --account=cs-schulte
# The schedulable seven-day partition is cs-gpu-research.  Venus 09 and 13
# were selected on the first submission for Schulte priority; this generic
# copy is also used to place overflow work on any normally available Venus.
#SBATCH --partition=cs-gpu-research
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/grid_restart_%A_%a.out
#SBATCH --error=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/grid_restart_%A_%a.err

set -euo pipefail

repo=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
original_root="$repo/runs/solar_kia_bce_motif01_2seed"
restart_root="$repo/runs/solar_kia_bce_motif01_2seed_restarts_20260902"
env_dir=/home/mirzaei/miniconda3/envs/micro_env
env_archive=/project/cs-schulte-lab/ali/micro_env.tar.gz

case "$SLURM_ARRAY_TASK_ID" in
  0) mode=total_count; seed=0 ;;
  1) mode=total_count; seed=1 ;;
  2) mode=full_matrix; seed=0 ;;
  *) echo "Unexpected task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

if [[ ! -x "$env_dir/bin/python" ]]; then
  mkdir -p "$env_dir"
  tar -xzf "$env_archive" -C "$env_dir"
fi

run_dir="$restart_root/grid/$mode/seed_$seed"
if [[ -e "$run_dir/console.log" ]]; then
  echo "Refusing to overwrite $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir" "$repo/slurm_logs"

export PATH="$env_dir/bin:$PATH"
export PYTHONPATH="$repo/vendor/python:${PYTHONPATH:-}"
export DGL_DOWNLOAD_DIR="$repo/.dgl_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd "$repo"
{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
  echo "dataset=grid mode=$mode seed=$seed"
  echo "restart_of=preempted_20260831"
  echo "partition=$SLURM_JOB_PARTITION"
  echo "adjacency_bce_weight=50"
  echo "kl_weight=2000"
  echo "motif_weight=5"
  echo "best_model=$run_dir/best_validation_mmd_model"
} | tee "$run_dir/job_metadata.log"

srun "$env_dir/bin/python" -u main.py \
  --config "configs/linkcorr_motif_true_3seed/grid_${mode}.yaml" \
  --seed "$seed" \
  --third_party_eval_seed "$seed" \
  --use_graphvae_mm_bce_kl_weights true \
  --alpha_motif_loss 5 \
  --alpha_syntactic_literal_motif_loss 5 \
  --graph_save_path "$run_dir" \
  --dataset_cache_dir "$original_root/grid/dataset_cache" \
  --run_label "solar-kia-grid-${mode}-seed${seed}-restart-20260902" \
  --model_path "$run_dir/best_model" \
  2>&1 | tee "$run_dir/console.log"
