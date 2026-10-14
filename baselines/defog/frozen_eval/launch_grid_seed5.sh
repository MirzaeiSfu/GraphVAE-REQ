#!/usr/bin/env bash
set -euo pipefail

repo=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
runner="$repo/baselines/defog/frozen_eval/run_defog_job.py"
manifest="$repo/baselines/defog/frozen_eval/manifest_seed5.yaml"
campaign="$repo/baselines/defog/frozen_eval/campaign.yaml"
defog_root="$repo/runs/defog/source"
artifact_root="$repo/runs/defog/frozen_eval"
run_root="$artifact_root/jobs"
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
log="$run_root/grid_seed5_launcher.log"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

"$python_bin" -u "$runner" \
  --manifest "$manifest" \
  --campaign "$campaign" \
  --defog-root "$defog_root" \
  --artifact-root "$artifact_root" \
  --run-root "$run_root" \
  --dataset GRID \
  --seed 5 \
  --python "$python_bin" \
  --stage all 2>&1 | tee "$log"
