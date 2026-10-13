#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_ID DATASET MODE SEED MOTIF_WEIGHT" >&2
  exit 2
fi
RUN_ID=$1
DATASET=$2
MODE=$3
SEED=$4
MOTIF_WEIGHT=$5
REPO=/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831
ENV_DIR=/home/mirzaei/miniconda3/envs/micro_env
ENV_ARCHIVE=/project/cs-schulte-lab/ali/micro_env.tar.gz
ROOT=/project/cs-schulte-lab/ali/ptc_aids_motif_true_20260903
MOTIF_CACHE=/project/cs-schulte-lab/ali/graphvae_motif_cache_20260903/multihop_smoothed
AIDS_DATA=/project/cs-schulte-lab/ali/GraphVAE-REQ-aids-enzymes-top100-20260714/data_raw
LOG_DIR=$ROOT/logs
STATUS_DIR=$ROOT/status
mkdir -p "$LOG_DIR" "$STATUS_DIR"
cd "$REPO"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO/vendor/python:${PYTHONPATH:-}"
export DGL_DOWNLOAD_DIR="$REPO/.dgl_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


srun --ntasks=1 --cpus-per-task=1 bash -lc '
set -euo pipefail
ENV_DIR="/home/mirzaei/miniconda3/envs/micro_env"
ENV_ARCHIVE="/project/cs-schulte-lab/ali/micro_env.tar.gz"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  mkdir -p /home/mirzaei/miniconda3/envs
  (
    flock -x 9
    if [[ ! -x "$ENV_DIR/bin/python" ]]; then
      rm -rf "$ENV_DIR"
      mkdir -p "$ENV_DIR"
      tar -xzf "$ENV_ARCHIVE" -C "$ENV_DIR"
    fi
  ) 9>/home/mirzaei/miniconda3/envs/.micro_env_extract.lock
fi
"$ENV_DIR/bin/python" --version
'

if [[ "$DATASET" == PTC ]]; then
  config="configs/multihop_smoothed/ptc_${MODE}.yaml"
  run_dir="$ROOT/runs/ptc/motif_${MOTIF_WEIGHT}/${MODE}/seed_${SEED}"
  dataset_cache="$ROOT/dataset_cache/ptc"
  data_args=()
  initial_motif_batch_size=512
  batch_args=(--train_batch_size 256)
elif [[ "$DATASET" == AIDS ]]; then
  config="configs/multihop_smoothed/aids_${MODE}_20260903.yaml"
  run_dir="$ROOT/runs/aids/motif_${MOTIF_WEIGHT}/${MODE}/seed_${SEED}"
  dataset_cache="$ROOT/dataset_cache/aids"
  data_args=(--data_dir "$AIDS_DATA")
  initial_motif_batch_size=50
  batch_args=(--train_batch_size 200)
else
  echo "unknown dataset $DATASET" >&2
  exit 2
fi
mkdir -p "$run_dir" "$dataset_cache"
status_file="$STATUS_DIR/${RUN_ID}.status"
log_file="$LOG_DIR/${RUN_ID}.log"
if [[ -f "$status_file" ]] && grep -q "^COMPLETE " "$status_file"; then
  echo "$RUN_ID already complete"
  exit 0
fi
{
  echo "RUNNING run_id=$RUN_ID dataset=$DATASET mode=$MODE seed=$SEED motif_weight=$MOTIF_WEIGHT host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date --iso-8601=seconds)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "config=$config"
  echo "run_dir=$run_dir"
  "$ENV_DIR/bin/python" --version
  srun --ntasks=1 --cpus-per-task=1 nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
} | tee "$status_file" "$run_dir/job_metadata.log"
run_attempt() {
  local cap=$1
  local mbatch=$2
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" "$ENV_DIR/bin/python" -u main.py \
    --config "$config" \
    --seed "$SEED" \
    --third_party_eval_seed "$SEED" \
    --use_graphvae_mm_bce_kl_weights false \
    --alpha_motif_loss "$MOTIF_WEIGHT" \
    --alpha_syntactic_literal_motif_loss "$MOTIF_WEIGHT" \
    --motif_cache_dir "$MOTIF_CACHE" \
    --dataset_cache_dir "$dataset_cache" \
    --require_existing_dataset_cache false \
    --disable_dataset_cache false \
    --motif_prune_max_values_per_rule "$cap" \
    --motif_batch_size "$mbatch" \
    --graph_save_path "$run_dir" \
    --run_label "ptc-aids-motif-${DATASET}-${MODE}-m${MOTIF_WEIGHT}-s${SEED}" \
    "${data_args[@]}" \
    "${batch_args[@]}"
}
set +e
run_attempt 256 "$initial_motif_batch_size" 2>&1 | tee "$log_file"
code=${PIPESTATUS[0]}
if [[ $code -ne 0 ]] && grep -Eqi "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$log_file"; then
  echo "OOM_RETRY cap=128 motif_batch_size=1 time=$(date --iso-8601=seconds)" | tee -a "$log_file" "$status_file"
  run_attempt 128 1 2>&1 | tee -a "$log_file"
  code=${PIPESTATUS[0]}
fi
if [[ $code -ne 0 ]] && grep -Eqi "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$log_file"; then
  echo "OOM_RETRY cap=64 motif_batch_size=1 time=$(date --iso-8601=seconds)" | tee -a "$log_file" "$status_file"
  run_attempt 64 1 2>&1 | tee -a "$log_file"
  code=${PIPESTATUS[0]}
fi
set -e
if [[ $code -eq 0 ]]; then
  echo "COMPLETE run_id=$RUN_ID dataset=$DATASET mode=$MODE seed=$SEED motif_weight=$MOTIF_WEIGHT host=$(hostname) finished=$(date --iso-8601=seconds)" | tee "$status_file"
else
  echo "FAILED run_id=$RUN_ID dataset=$DATASET mode=$MODE seed=$SEED motif_weight=$MOTIF_WEIGHT exit_code=$code host=$(hostname) finished=$(date --iso-8601=seconds)" | tee "$status_file"
fi
exit "$code"
