#!/usr/bin/env bash
set -euo pipefail

REPO=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
PYTHON=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
ROOT=/local-scratch2/mirzaei/defog_ptc_frozen_20260906
MANIFEST=$ROOT/manifest_ptc.yaml
ARTIFACT_ROOT=$ROOT/artifacts
RUN_ROOT=$ROOT/jobs
RANDOM_GIN_ROOT=$ROOT/random_gin
AGG_ROOT=$ROOT/aggregate
LOG=$ROOT/eval_after_training.log
GPU=${1:-1}

cd "$REPO"
mkdir -p "$RANDOM_GIN_ROOT" "$AGG_ROOT"

complete_seed() {
  local seed=$1
  local record="$RUN_ROOT/ptc/seed_$seed/job_record.json"
  [[ -f "$record" ]] && "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' "$record"
}

{
  echo "START watcher at $(date --iso-8601=seconds) on $(hostname), gpu=$GPU"
  while true; do
    if complete_seed 0 && complete_seed 1 && complete_seed 2; then
      echo "ALL_SEEDS_COMPLETE at $(date --iso-8601=seconds)"
      break
    fi
    echo "WAIT $(date --iso-8601=seconds)"
    sleep 1800
  done

  PYTHONPATH=graph_evaluation/src "$PYTHON" \
    baselines/defog/frozen_eval/verify_campaign.py \
    --manifest "$MANIFEST" \
    --artifact-root "$ARTIFACT_ROOT" \
    --dataset PTC

  for seed in 0 1 2; do
    echo "RANDOM_GIN seed=$seed at $(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=graph_evaluation/src "$PYTHON" \
      baselines/defog/frozen_eval/run_random_gin.py \
      --manifest "$MANIFEST" \
      --artifact-root "$ARTIFACT_ROOT" \
      --dataset PTC \
      --training-seed "$seed" \
      --output-root "$RANDOM_GIN_ROOT" \
      --device cuda \
      --python "$PYTHON"
  done

  PYTHONPATH=graph_evaluation/src "$PYTHON" \
    baselines/defog/frozen_eval/aggregate.py \
    --input-root "$RANDOM_GIN_ROOT" \
    --dataset PTC \
    --output-dir "$AGG_ROOT/ptc"

  echo "COMPLETE eval at $(date --iso-8601=seconds)"
} 2>&1 | tee -a "$LOG"
