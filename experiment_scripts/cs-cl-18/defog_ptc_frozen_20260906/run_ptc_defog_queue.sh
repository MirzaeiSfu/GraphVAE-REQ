#!/usr/bin/env bash
set -uo pipefail

REPO=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ
PYTHON=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
DEFOG_ROOT=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/source
ROOT=/local-scratch2/mirzaei/defog_ptc_frozen_20260906
MANIFEST=$ROOT/manifest_ptc.yaml
CAMPAIGN=$ROOT/campaign_ptc.yaml
ARTIFACT_ROOT=$ROOT/artifacts
RUN_ROOT=$ROOT/jobs
LOG_ROOT=$ROOT/queue_logs
DATASET=PTC

GPU=${1:-1}
shift || true
if (( $# > 0 )); then
  seeds=("$@")
else
  seeds=(0 1 2)
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$REPO" || exit 2

for seed in "${seeds[@]}"; do
  job_root="$RUN_ROOT/ptc/seed_$seed"
  record="$job_root/job_record.json"
  if [[ -f "$record" ]] && "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' "$record"; then
    echo "SKIP PTC seed $seed: already complete"
    continue
  fi

  available_kb=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < 15 * 1024 * 1024 )); then
    echo "STOP PTC seed $seed: less than 15 GiB free on $(hostname)" >&2
    exit 3
  fi

  log="$LOG_ROOT/ptc_seed_${seed}.log"
  echo "START PTC seed $seed on $(hostname) GPU $GPU at $(date --iso-8601=seconds)"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    baselines/defog/frozen_eval/run_defog_job.py \
    --manifest "$MANIFEST" \
    --campaign "$CAMPAIGN" \
    --defog-root "$DEFOG_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --run-root "$RUN_ROOT" \
    --dataset "$DATASET" \
    --seed "$seed" \
    --python "$PYTHON" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  if (( status != 0 )); then
    echo "FAILED PTC seed $seed with exit $status at $(date --iso-8601=seconds)" >&2
    continue
  fi
  if [[ ! -e "$job_root/best_validation.ckpt" || ! -e "$job_root/final_epoch.ckpt" ]]; then
    echo "FAILED PTC seed $seed: checkpoint aliases are missing" >&2
    continue
  fi
  sha256sum "$job_root/best_validation.ckpt" "$job_root/final_epoch.ckpt" > "$job_root/checkpoint_sha256.txt"
  echo "COMPLETE PTC seed $seed at $(date --iso-8601=seconds)"
done
