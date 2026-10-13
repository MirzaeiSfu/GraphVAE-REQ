#!/usr/bin/env bash
set -euo pipefail
campaign=$1
dataset=$2
seed=$3
gpu=$4
helper=$5
out=$campaign/archive_evaluation_20260921/${dataset}/seed_${seed}
remote=/local-scratch2/mirzaei/LGD_ARCHIVE_EVALUATION_20260921/${dataset}/seed_${seed}
archive=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/${dataset}
mkdir -p "$out"
exec > >(tee -a "$out/pipeline.log") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "FAILED exit=$rc" > "$out/FAILED"; fi; ssh mirzaei@cs-cl-19.cmpt.sfu.ca "mkdir -p $archive/evaluations/lgd_seed_$seed"; rsync -a "$out/" "mirzaei@cs-cl-19.cmpt.sfu.ca:$archive/evaluations/lgd_seed_$seed/"; ssh mirzaei@cs-cl-19.cmpt.sfu.ca "flock /tmp/mirzaei_archive_report.lock python3 /local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/update_archive_reports.py"' EXIT
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PATH="$campaign/env/bin:$PATH"
export LD_LIBRARY_PATH="$campaign/env/lib:${LD_LIBRARY_PATH:-}"
cd "$campaign/repo"
python "$helper/lgd_multi_sample.py" "$campaign" "$dataset" "$seed" "$out"
ssh mirzaei@cs-cl-18.cmpt.sfu.ca "mkdir -p $remote"
rsync -a "$out/samples.pkl" "mirzaei@cs-cl-18.cmpt.sfu.ca:$remote/"
ssh mirzaei@cs-cl-18.cmpt.sfu.ca "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /local-scratch2/mirzaei/miniconda3/envs/micro/bin/python /local-scratch/localhome/mirzaei/lgd_multi_evaluate.py $remote"
rsync -a "mirzaei@cs-cl-18.cmpt.sfu.ca:$remote/" "$out/"
