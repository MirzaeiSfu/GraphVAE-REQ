#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
    echo "Usage: $0 GPU_ID RUN_ROOT PYTHON DATASET:SEED [...]" >&2
    exit 2
fi

gpu_id="$1"
run_root="$2"
python_bin="$3"
shift 3

repo="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

for item in "$@"; do
    dataset="${item%%:*}"
    seed="${item##*:}"

    case "$dataset" in
        grid)
            config=configs/linkcorr_motif_true_3seed/grid_total_count.yaml
            bce=50
            kl=2000
            ;;
        lobster)
            config=configs/linkcorr_motif_true_3seed/lobster_total_count.yaml
            bce=40
            kl=2000
            ;;
        triangular_grid)
            config=configs/linkcorr_motif_true_3seed/triangular_grid_total_count.yaml
            bce=50
            kl=2000
            ;;
        mutag)
            config=configs/multihop_smoothed/mutag_total_count.yaml
            bce=4
            kl=60
            ;;
        proteins)
            config=configs/multihop_smoothed_3seed/proteins_total_count.yaml
            bce=50
            kl=2000
            ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            exit 2
            ;;
    esac

    run_dir="$run_root/$dataset/seed_$seed"
    if [[ -f "$run_dir/final_metrics_summary.json" ]]; then
        echo "[queue] already complete, skipping $run_dir"
        continue
    fi
    if [[ -e "$run_dir/console.log" ]]; then
        echo "[queue] refusing to overwrite incomplete run $run_dir" >&2
        exit 1
    fi

    mkdir -p "$run_dir" "$run_root/$dataset/dataset_cache"
    cd "$repo"

    {
        echo "host=$(hostname)"
        echo "gpu_physical=$gpu_id cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
        echo "dataset=$dataset seed=$seed motif_loss=false"
        echo "config=$config"
        echo "adjacency_bce_weight=$bce"
        echo "kl_weight=$kl"
        echo "motif_weight=0"
        echo "best_model=$run_dir/best_validation_mmd_model"
        "$python_bin" --version
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    } | tee "$run_dir/job_metadata.log"

    "$python_bin" -u main.py \
        --config "$config" \
        --seed "$seed" \
        --third_party_eval_seed "$seed" \
        --motif_loss false \
        --use_syntactic_literal_rules false \
        --alpha_motif_loss 0 \
        --alpha_syntactic_literal_motif_loss 0 \
        --use_graphvae_mm_bce_kl_weights true \
        --graph_save_path "$run_dir" \
        --dataset_cache_dir "$run_root/$dataset/dataset_cache" \
        --run_label "kia-false-${dataset}-seed${seed}" \
        --model_path "$run_dir/best_model" \
        --device cuda:0 \
        2>&1 | tee "$run_dir/console.log"
done
