#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
campaign=/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
config=configs/mutag_edgefeat_campaign/full_matrix.yaml
run_dir=$campaign/runs/full_matrix/seed_$seed
log=$campaign/logs/seed_${seed}.log
status=$campaign/status/seed_${seed}.status

mkdir -p "$run_dir" "$campaign/logs" "$campaign/status"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

printf 'RUNNING host=%s gpu=%s seed=%s started=%s\n' \
  "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
cd "$repo" || exit 2
"$python" -u main.py \
  --config "$config" \
  --seed "$seed" \
  --third_party_eval_seed "$seed" \
  --device cuda:0 \
  --graph_save_path "$run_dir" \
  --run_label "mutag-edgefeat-cpsmoothed-full-motif02-seed-${seed}" \
  --motif_loss true \
  --motif_output_mode full_matrix \
  --motif_cp_table_source cp_smoothed \
  --rule_prune true \
  --alpha_motif_loss 0.2 \
  --alpha_syntactic_literal_motif_loss 0.2 \
  --checkpoint_interval_epochs 0 \
  2>&1 | tee "$log"
code=${PIPESTATUS[0]}
state=FAILED
(( code == 0 )) && state=COMPLETE
printf '%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
  "$state" "$(hostname)" "$gpu" "$seed" "$code" \
  "$(date --iso-8601=seconds)" > "$status"
exit "$code"
