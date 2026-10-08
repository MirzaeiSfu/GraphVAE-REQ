#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/QM9_FULL_TEST_20260925
py=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
remote=mirzaei@cs-cl-17.cmpt.sfu.ca
old=/localhome/mirzaei/qm9_defog_lab24_20260914
campaign=/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916
mkdir -p "$root/defog_source" "$root/defog_site" "$root/defog_data" "$root/checkpoints"
echo STAGING > "$root/defog_status.txt"
rsync -a -e 'ssh -o BatchMode=yes' "$remote:$old/work/source/" "$root/defog_source/"
rsync -a -e 'ssh -o BatchMode=yes' "$remote:$old/env/defog-benchmark/lib/python3.8/site-packages/" "$root/defog_site/"
rsync -a -e 'ssh -o BatchMode=yes' "$remote:$old/work/data/seed_0/" "$root/defog_data/"
for seed in 0 1 2; do
  rsync -a -e 'ssh -o BatchMode=yes' "$remote:$campaign/runs/seed_$seed/checkpoints/qm9_seedfixed250_seed_$seed/epoch=239.ckpt" "$root/checkpoints/defog_seed_$seed.ckpt"
done
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$root/defog_site:$root/defog_source/src:$root/defog_source:/local-scratch2/mirzaei/fb/GraphVAE-REQ/graph_evaluation/src"
cd "$root/defog_source"
for seed in 0 1 2; do
  out="$root/defog/seed_$seed"
  mkdir -p "$out"
  echo "GENERATING seed=$seed" > "$root/defog_status.txt"
  "$py" -u /local-scratch/localhome/mirzaei/qm9_defog_fulltest_entry_20260925.py \
    +experiment=qm9_with_h dataset.datadir="$root/defog_data" train.seed="$seed" train.batch_size=128 \
    general.test_only="$root/checkpoints/defog_seed_$seed.ckpt" general.generation_seed="$seed" \
    general.final_model_samples_to_generate=26167 general.final_model_samples_to_save=26167 \
    general.final_model_chains_to_save=0 general.num_sample_fold=1 general.evaluate_all_checkpoints=false \
    general.name="qm9_fulltest_seed_$seed" general.wandb=disabled general.gpus=1 hydra.run.dir="$out" \
    > "$out/generation.log" 2>&1
  test -s "$out/generated_graphs.pt"
  env -u PYTHONPATH "$py" /local-scratch2/mirzaei/qm9_common_eval_20260917/convert_defog_raw_to_common_dgl.py \
    --input "$out/generated_graphs.pt" --output "$out/generated.bin" --metadata "$out/conversion.json" --seed "$seed"
  reference="$root/true_full/seed_0/reference_attributed_graphs.bin"
  until test -s "$root/true_full/seed_0/generation.json"; do sleep 30; done
  echo "EVALUATING seed=$seed" > "$root/defog_status.txt"
  env -u PYTHONPATH "$py" -u /local-scratch/localhome/mirzaei/qm9_full_test_20260925.py external \
    --generated "$out/generated.bin" --reference "$reference" --output "$out/structural.json" > "$out/structural.log" 2>&1
  echo COMPLETE > "$out/COMPLETE"
done
echo COMPLETE > "$root/defog_status.txt"
