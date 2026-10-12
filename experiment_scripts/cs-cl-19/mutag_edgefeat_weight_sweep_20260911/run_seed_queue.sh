#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:-1}
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
config=configs/mutag_edgefeat_campaign/full_matrix.yaml
sweep=/local-scratch2/mirzaei/mutag_edgefeat_weight_sweep_20260911
prior_status=/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911/status/seed_${seed}.status
queue_status=$sweep/status/seed_${seed}_queue.status

mkdir -p "$sweep/status" "$sweep/logs"
printf 'WAITING_FOR_0.2 seed=%s host=%s\n' "$seed" "$(hostname)" > "$queue_status"
until grep -q '^COMPLETE' "$prior_status" 2>/dev/null; do sleep 60; done

for spec in '005 0.05' '015 0.15' '030 0.30'; do
  set -- $spec
  tag=$1
  weight=$2
  campaign=$sweep/motif_${tag}
  run_dir=$campaign/runs/full_matrix/seed_$seed
  log=$campaign/logs/seed_${seed}.log
  status=$campaign/status/seed_${seed}.status
  mkdir -p "$run_dir" "$campaign/logs" "$campaign/status"
  printf 'RUNNING weight=%s seed=%s host=%s gpu=%s started=%s\n' \
    "$weight" "$seed" "$(hostname)" "$gpu" "$(date --iso-8601=seconds)" > "$status"
  printf 'RUNNING weight=%s seed=%s host=%s\n' "$weight" "$seed" "$(hostname)" > "$queue_status"
  cd "$repo" || exit 2
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONUNBUFFERED=1
  export WANDB_MODE=disabled
  "$python" -u main.py \
    --config "$config" \
    --seed "$seed" \
    --third_party_eval_seed "$seed" \
    --device cuda:0 \
    --graph_save_path "$run_dir" \
    --run_label "mutag-edgefeat-cpsmoothed-full-motif${tag}-seed-${seed}" \
    --motif_loss true \
    --motif_output_mode full_matrix \
    --motif_cp_table_source cp_smoothed \
    --rule_prune true \
    --alpha_motif_loss "$weight" \
    --alpha_syntactic_literal_motif_loss "$weight" \
    --checkpoint_interval_epochs 0 \
    2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  if (( code != 0 )); then
    printf 'FAILED weight=%s seed=%s host=%s exit_code=%s finished=%s\n' \
      "$weight" "$seed" "$(hostname)" "$code" "$(date --iso-8601=seconds)" > "$status"
    printf 'FAILED weight=%s seed=%s host=%s exit_code=%s\n' "$weight" "$seed" "$(hostname)" "$code" > "$queue_status"
    exit "$code"
  fi
  printf 'COMPLETE weight=%s seed=%s host=%s exit_code=0 finished=%s\n' \
    "$weight" "$seed" "$(hostname)" "$(date --iso-8601=seconds)" > "$status"
done

printf 'COMPLETE seed=%s host=%s finished=%s\n' "$seed" "$(hostname)" "$(date --iso-8601=seconds)" > "$queue_status"
