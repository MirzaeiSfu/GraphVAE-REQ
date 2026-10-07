#!/usr/bin/env bash
set -euo pipefail

seed=$1
root=/localhome/mirzaei/defog_ogbg_3seed_20260920
out=$root/archive_evaluation_20260922/seed_$seed
repo=$root/source/DeFoG

if [[ "$seed" == 2 ]]; then
  py=$root/env/defog-benchmark/bin/python
  graph_eval_src=/localhome/mirzaei/defog_ptc_metrics_20260907/repo/graph_evaluation/src
else
  py=/localhome/mirzaei/qm9_defog_lab24_20260914/env/defog-benchmark/bin/python
  graph_eval_src=/localhome/mirzaei/fb/GraphVAE-REQ/graph_evaluation/src
fi

exec >"$out/export_repair.log" 2>&1
echo REPAIR_RUNNING >"$out/status"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled
export PYTHONPATH="$repo:$repo/src:$graph_eval_src"
cd "$repo"

"$py" -u src/main.py \
  dataset=ogbg_molbbbp \
  +experiment=ogbg_molbbbp \
  dataset.datadir=data/ogbg_molbbbp_frozen/ \
  train.seed="$seed" \
  train.batch_size=32 \
  general.test_only="$root/runs/seed_${seed}_vizfix/checkpoints/ogbg_molbbbp_seed_${seed}_vizfix/last.ckpt" \
  general.generated_path="$out/generated_samples_rank0.pkl" \
  general.final_model_samples_to_generate=405 \
  general.final_model_samples_to_save=405 \
  general.final_model_chains_to_save=0 \
  general.num_sample_fold=1 \
  general.evaluate_all_checkpoints=false \
  general.name="ogb_archive_seed_${seed}_export_repair" \
  general.wandb=disabled \
  hydra.run.dir="$out"

test -s "$out/generated_graphs.pt"
echo COMPLETE >"$out/status"
echo "EXPORT_REPAIR_COMPLETE seed=$seed"
