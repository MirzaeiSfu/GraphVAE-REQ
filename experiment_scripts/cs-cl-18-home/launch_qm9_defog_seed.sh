#!/usr/bin/env bash
set -euo pipefail
seed=${1:?seed}
gpu=${2:?gpu}
root=/local-scratch2/mirzaei/qm9_defog_matched_split_20260914
source_dir=$root/source
env_dir=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark
run_dir=$root/runs/seed_$seed
mkdir -p "$run_dir" "$root/data/seed_$seed" "$root/logs" "$root/status"
cd "$source_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_dir:${PYTHONPATH:-}"
printf 'RUNNING seed=%s gpu=%s host=%s started=%s\n' "$seed" "$gpu" "$(hostname)" "$(date --iso-8601=seconds)" > "$root/status/seed_$seed.status"
set +e
"$env_dir/bin/python" -u src/main.py \
  +experiment=qm9_with_h \
  dataset.datadir="$root/data/seed_$seed" \
  train.seed="$seed" \
  train.n_epochs=250 \
  train.batch_size=512 \
  train.save_model=true \
  general.name="qm9_matched_split_seed_$seed" \
  general.wandb=disabled \
  general.gpus=1 \
  hydra.run.dir="$run_dir" 2>&1 | tee "$root/logs/seed_$seed.log"
code=${PIPESTATUS[0]}
set -e
if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s gpu=%s exit_code=%s finished=%s\n' "$state" "$seed" "$gpu" "$code" "$(date --iso-8601=seconds)" > "$root/status/seed_$seed.status"
exit "$code"
