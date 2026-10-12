#!/usr/bin/env bash
set -uo pipefail

if (( $# < 6 )); then
  echo "usage: $0 REPO PYTHON DEFOG_ROOT DATASET GPU SEED [SEED ...]" >&2
  exit 2
fi

repo=$1
python=$2
defog_root=$3
dataset=${4^^}
gpu=$5
shift 5
seeds=("$@")

cd "$repo"
log_root="runs/defog/frozen_eval/queue_logs"
run_root="runs/defog/frozen_eval/jobs"
mkdir -p "$log_root" "$run_root"

for seed in "${seeds[@]}"; do
  job_root="$run_root/${dataset,,}/seed_$seed"
  record="$job_root/job_record.json"
  if [[ -f "$record" ]] && "$python" -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' \
      "$record"; then
    echo "SKIP $dataset seed $seed: already complete"
    continue
  fi

  available_kb=$(df -Pk "$repo" | awk 'NR == 2 {print $4}')
  if (( available_kb < 15 * 1024 * 1024 )); then
    echo "STOP $dataset seed $seed: less than 15 GiB free on $(hostname)" >&2
    exit 3
  fi

  log="$log_root/${dataset,,}_seed_${seed}.log"
  echo "START $dataset seed $seed on $(hostname) GPU $gpu at $(date --iso-8601=seconds)"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$python" \
    baselines/defog/frozen_eval/run_defog_job.py \
    --manifest baselines/defog/frozen_eval/manifest.yaml \
    --campaign baselines/defog/frozen_eval/campaign.yaml \
    --defog-root "$defog_root" \
    --artifact-root runs/defog/frozen_eval \
    --run-root "$run_root" \
    --dataset "$dataset" \
    --seed "$seed" \
    --python "$python" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  if (( status != 0 )); then
    echo "FAILED $dataset seed $seed with exit $status at $(date --iso-8601=seconds)" >&2
    continue
  fi
  if [[ ! -e "$job_root/best_validation.ckpt" || ! -e "$job_root/final_epoch.ckpt" ]]; then
    echo "FAILED $dataset seed $seed: checkpoint aliases are missing" >&2
    continue
  fi
  sha256sum "$job_root/best_validation.ckpt" "$job_root/final_epoch.ckpt" \
    > "$job_root/checkpoint_sha256.txt"
  echo "COMPLETE $dataset seed $seed at $(date --iso-8601=seconds)"
done
