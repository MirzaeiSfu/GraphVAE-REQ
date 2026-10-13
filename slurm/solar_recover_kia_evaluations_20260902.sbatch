#!/bin/bash
#SBATCH --job-name=gvae_eval_recover
#SBATCH --account=cs-schulte
#SBATCH --partition=cs-gpu-research
#SBATCH --nodelist=cs-venus-09,cs-venus-13,cs-venus-14
#SBATCH --array=0-6%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/eval_recover_%A_%a.out
#SBATCH --error=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/eval_recover_%A_%a.err

set -euo pipefail

repo=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
root="$repo/runs/solar_kia_bce_motif01_2seed"
env_dir=/home/mirzaei/miniconda3/envs/micro_env

case "$SLURM_ARRAY_TASK_ID" in
  0) relative=lobster/full_matrix/seed_0; eval_seed=0 ;;
  1) relative=triangular_grid/total_count/seed_0; eval_seed=0 ;;
  2) relative=triangular_grid/full_matrix/seed_0; eval_seed=0 ;;
  3) relative=proteins/total_count/seed_0; eval_seed=0 ;;
  4) relative=proteins/total_count/seed_1; eval_seed=1 ;;
  5) relative=proteins/full_matrix/seed_0; eval_seed=0 ;;
  6) relative=proteins/full_matrix/seed_1; eval_seed=1 ;;
  *) echo "Unexpected task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

run_dir="$root/$relative"
test -f "$run_dir/best_validation_mmd_model"
test -f "$run_dir/Single_comp_generatedGraphs_adj_final_eval.npy"
test -f "$run_dir/testGraphs_adj_.npy"

export PATH="$env_dir/bin:$PATH"
export PYTHONPATH="$repo/vendor/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$repo"
srun "$env_dir/bin/python" -u scripts/evaluate_graph_realism_batch_fixed.py \
  --run-dir "$run_dir" \
  --json-filename graph_realism_random_gin_recovered.json \
  --summary-csv "$run_dir/graph_realism_recovered_summary.csv" \
  --repeats 10 \
  --max-graphs 1000 \
  --seed "$eval_seed" \
  --device cuda \
  2>&1 | tee "$run_dir/evaluation_recovery_20260902.log"
