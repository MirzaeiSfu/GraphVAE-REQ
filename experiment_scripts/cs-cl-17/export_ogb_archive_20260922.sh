#!/usr/bin/env bash
set -euo pipefail
seed=$1
gpu=$2
root=/localhome/mirzaei/defog_ogbg_3seed_20260920
if [ "$seed" = 2 ]; then py=$root/env/defog-benchmark/bin/python; else py=/localhome/mirzaei/qm9_defog_lab24_20260914/env/defog-benchmark/bin/python; fi
out=$root/archive_evaluation_20260922/seed_$seed
mkdir -p "$out"
exec > "$out/pipeline.log" 2>&1
trap 'echo FAILED > "$out/status"' ERR
echo RUNNING > "$out/status"
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled
repo=$root/source/DeFoG
export PYTHONPATH="$repo:$repo/src"
cd "$repo"
"$py" -u src/main.py dataset=ogbg_molbbbp +experiment=ogbg_molbbbp dataset.datadir=data/ogbg_molbbbp_frozen/ train.seed="$seed" train.batch_size=32 general.test_only="$root/runs/seed_${seed}_vizfix/checkpoints/ogbg_molbbbp_seed_${seed}_vizfix/last.ckpt" general.generation_seed="$seed" general.final_model_samples_to_generate=405 general.final_model_samples_to_save=405 general.final_model_chains_to_save=0 general.num_sample_fold=1 general.evaluate_all_checkpoints=false general.name="ogb_archive_seed_$seed" general.wandb=disabled hydra.run.dir="$out"
test -s "$out/generated_graphs.pt"
echo COMPLETE > "$out/status"
