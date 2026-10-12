#!/bin/bash
#SBATCH --job-name=gvae_kia_motif
#SBATCH --account=cs-schulte
#SBATCH --partition=cs-gpu-research
#SBATCH --nodelist=cs-venus-04,cs-venus-09,cs-venus-13,cs-venus-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-19%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/kia_motif_%A_%a.out
#SBATCH --error=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/slurm_logs/kia_motif_%A_%a.err

set -euo pipefail

REPO=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
RUN_ROOT="$REPO/runs/solar_kia_bce_motif01_2seed"
ENV_DIR=/home/mirzaei/miniconda3/envs/micro_env
ENV_ARCHIVE=/project/cs-schulte-lab/ali/micro_env.tar.gz

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    mkdir -p /home/mirzaei/miniconda3/envs
    (
        flock -x 9
        if [[ ! -x "$ENV_DIR/bin/python" ]]; then
            mkdir -p "$ENV_DIR"
            tar -xzf "$ENV_ARCHIVE" -C "$ENV_DIR"
        fi
    ) 9>/home/mirzaei/miniconda3/envs/.micro_env_extract.lock
fi

task="$SLURM_ARRAY_TASK_ID"
dataset_index=$((task / 4))
setting_index=$((task % 4))
seed=$((setting_index % 2))

if (( setting_index < 2 )); then
    mode=total_count
else
    mode=full_matrix
fi

case "$dataset_index" in
    0)
        dataset=grid
        bce_weight=50
        kl_weight=2000
        motif_weight=5
        config="configs/linkcorr_motif_true_3seed/grid_${mode}.yaml"
        ;;
    1)
        dataset=lobster
        bce_weight=40
        kl_weight=2000
        motif_weight=4
        config="configs/linkcorr_motif_true_3seed/lobster_${mode}.yaml"
        ;;
    2)
        dataset=triangular_grid
        bce_weight=50
        kl_weight=2000
        motif_weight=5
        config="configs/linkcorr_motif_true_3seed/triangular_grid_${mode}.yaml"
        ;;
    3)
        dataset=mutag
        bce_weight=4
        kl_weight=60
        motif_weight=0.4
        config="configs/multihop_smoothed/mutag_${mode}.yaml"
        ;;
    4)
        dataset=proteins
        bce_weight=50
        kl_weight=2000
        motif_weight=5
        config="configs/multihop_smoothed_3seed/proteins_${mode}.yaml"
        ;;
    *)
        echo "Unexpected array task: $task" >&2
        exit 2
        ;;
esac

run_dir="$RUN_ROOT/$dataset/$mode/seed_$seed"
if [[ -e "$run_dir/job_metadata.log" || -e "$run_dir/console.log" ]]; then
    echo "Refusing to overwrite existing run: $run_dir" >&2
    exit 1
fi

mkdir -p "$run_dir" "$REPO/slurm_logs"
cd "$REPO"

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO/vendor/python:${PYTHONPATH:-}"
export DGL_DOWNLOAD_DIR="$REPO/.dgl_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

"$ENV_DIR/bin/python" - "$config" "$dataset" "$mode" "$bce_weight" "$kl_weight" "$motif_weight" <<'PY'
import sys
import yaml
from loss_weight_utils import KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET

config_path, dataset, mode, bce, kl, motif = sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

dataset_name = config["data"]["dataset"].upper()
expected_bce, expected_kl = KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET[dataset_name]
assert config["motif"]["motif_loss"] is True
assert config["motif"]["motif_output_mode"] == mode
assert config["motif"]["motif_cp_table_source"] == "cp_smoothed"
assert config["motif"]["rule_prune"] is True
assert config["runtime"]["keep_best_validation_mmd"] is True
assert config["runtime"]["save_validation_checkpoints"] is False
assert float(bce) == expected_bce
assert float(kl) == expected_kl
assert float(motif) == 0.1 * expected_bce
PY

{
    echo "host=$(hostname)"
    echo "job_id=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID"
    echo "dataset=$dataset mode=$mode seed=$seed"
    echo "config=$config"
    echo "adjacency_bce_weight=$bce_weight"
    echo "kl_weight=$kl_weight"
    echo "motif_weight=$motif_weight"
    echo "best_model=$run_dir/best_validation_mmd_model"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    "$ENV_DIR/bin/python" --version
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$run_dir/job_metadata.log"

srun "$ENV_DIR/bin/python" -u main.py \
    --config "$config" \
    --seed "$seed" \
    --third_party_eval_seed "$seed" \
    --use_graphvae_mm_bce_kl_weights true \
    --alpha_motif_loss "$motif_weight" \
    --alpha_syntactic_literal_motif_loss "$motif_weight" \
    --graph_save_path "$run_dir" \
    --dataset_cache_dir "$RUN_ROOT/$dataset/dataset_cache" \
    --run_label "solar-kia-${dataset}-${mode}-seed${seed}" \
    --model_path "$run_dir/best_model" \
    2>&1 | tee "$run_dir/console.log"
