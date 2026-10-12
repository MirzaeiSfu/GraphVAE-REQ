#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?dataset}
gpu=${2:?gpu}
campaign=/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920
out=$campaign/archive_evaluation_20260921/$dataset
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/$dataset
mkdir -p "$out"
exec > >(tee -a "$out/pipeline.log") 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "FAILED exit=$code" | tee "$out/FAILED"; fi; rsync -a "$out/" "mirzaei@cs-cl-19.cmpt.sfu.ca:$archive/evaluations/lgd_seed_2/" || true' EXIT
export CUDA_VISIBLE_DEVICES=$gpu
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled
export PATH="$campaign/env/bin:$PATH"
export LD_LIBRARY_PATH="$campaign/env/lib:${LD_LIBRARY_PATH:-}"
cd "$campaign/repo"
python /local-scratch/localhome/mirzaei/lgd_archive_sample.py "$dataset" "$out"
ssh mirzaei@cs-cl-18.cmpt.sfu.ca "mkdir -p /local-scratch2/mirzaei/LGD_ARCHIVE_EVALUATION_20260921/$dataset"
rsync -a "$out/samples.pkl" "mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_ARCHIVE_EVALUATION_20260921/$dataset/"
ssh mirzaei@cs-cl-18.cmpt.sfu.ca "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /local-scratch2/mirzaei/miniconda3/envs/micro/bin/python /local-scratch/localhome/mirzaei/lgd_archive_evaluate.py /local-scratch2/mirzaei/LGD_ARCHIVE_EVALUATION_20260921/$dataset"
rsync -a "mirzaei@cs-cl-18.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_ARCHIVE_EVALUATION_20260921/$dataset/" "$out/"
date -Is > "$out/COMPLETE"
