#!/usr/bin/env bash
set -euo pipefail
gpu=${1:-1}
root=/local-scratch2/mirzaei/common_eval_fix_20260923/AIDS
repo=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
convert=/local-scratch2/mirzaei/edge_feature_random_gin_20260917/convert_pyg_collection_to_dgl.py
common=/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluate_dgl_common_metrics.py
allmodes=/local-scratch2/mirzaei/edge_feature_random_gin_20260917/evaluate_dgl_all_modes.py
canonical=/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae/false/seed_0/reference_attributed_graphs.bin
mkdir -p "$root/stage" "$root/results"
exec >"$root/pipeline.log" 2>&1
trap 'echo FAILED >"$root/status"' ERR
cp -p "$canonical" "$root/stage/reference.bin"
for seed in 3 4 5; do
  if [ "$seed" = 3 ]; then campaign=aids_defog_independent_20260919; else campaign=aids_defog_independent_20260922; fi
  remote=/local-scratch2/mirzaei/$campaign/seed_$seed/generation/generated_graphs.pt
  echo WAITING_seed_${seed} >"$root/status"
  until ssh -o BatchMode=yes mirzaei@cs-cl-09.cmpt.sfu.ca "test -s $remote"; do sleep 30; done
  mkdir -p "$root/stage/defog/seed_$seed" "$root/results/defog/seed_$seed"
  rsync -a "mirzaei@cs-cl-09.cmpt.sfu.ca:$remote" "$root/stage/defog/seed_$seed/generated_graphs.pt"
  "$python" "$convert" --input "$root/stage/defog/seed_$seed/generated_graphs.pt" --output "$root/stage/defog/seed_$seed/generated.bin"
  echo RUNNING_seed_${seed} >"$root/status"
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$common" --repo "$repo" --generated "$root/stage/defog/seed_$seed/generated.bin" --reference "$root/stage/reference.bin" --output "$root/results/defog/seed_$seed/structural.json" --label aids_defog_corrected --seed "$seed" --device cuda --repeats 10 --skip-random-gin
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$allmodes" --repo "$repo" --generated "$root/stage/defog/seed_$seed/generated.bin" --reference "$root/stage/reference.bin" --output "$root/results/defog/seed_$seed/random_gin_all_modes.json" --label aids_defog_corrected --seed "$seed" --device cuda --repeats 10
done
sha256sum "$root/stage/reference.bin" "$root/stage/defog"/seed_*/generated.bin >"$root/SHA256SUMS"
rsync -a -e 'ssh -o BatchMode=yes' "$root/" mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/evaluations/common_fixed_20260923/
echo COMPLETE >"$root/status"
