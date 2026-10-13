#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:-0}
source_root=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
campaign_root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910
checkpoint="$campaign_root/runs/seed_$seed/checkpoints/proteins_3seed_20260910_seed_$seed/last.ckpt"
training_status="$campaign_root/status/proteins_seed_$seed.status"
output_dir="$campaign_root/final_eval/seed_$seed"
pipeline_status="$campaign_root/status/proteins_seed_${seed}_third_party.status"
log="$campaign_root/logs/proteins_seed_${seed}_final_and_third_party.log"
reference=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/proteins/real_test_graphs.pt

mkdir -p "$output_dir" "$(dirname "$pipeline_status")"
printf 'WAITING seed=%s host=%s\n' "$seed" "$(hostname)" > "$pipeline_status"
until grep -q '^COMPLETE' "$training_status" 2>/dev/null && test -s "$checkpoint"; do sleep 60; done

printf 'GENERATING seed=%s host=%s gpu=%s\n' "$seed" "$(hostname)" "$gpu" > "$pipeline_status"
cd "$source_root" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:$source_root/src"
export WANDB_MODE=disabled
export DGL_DOWNLOAD_DIR="$source_root/.dgl_cache"

"$python_bin" src/main.py \
  dataset=proteins +experiment=proteins \
  dataset.datadir=data/proteins_3seed_20260910/ \
  train.seed="$seed" \
  general.name="proteins_seed_${seed}_final_eval" \
  general.wandb=disabled general.test_only="$checkpoint" \
  general.final_model_samples_to_generate=210 \
  general.final_model_samples_to_save=30 \
  hydra.run.dir="$output_dir" >> "$log" 2>&1 || exit $?

printf 'EVALUATING seed=%s host=%s\n' "$seed" "$(hostname)" > "$pipeline_status"
"$python_bin" "$campaign_root/run_external_random_gin.py" \
  --generated "$output_dir/generated_graphs.pt" \
  --reference "$reference" \
  --output "$output_dir/third_party" \
  --device cpu >> "$log" 2>&1
code=$?
if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s host=%s exit_code=%s finished=%s\n' "$state" "$seed" "$(hostname)" "$code" "$(date --iso-8601=seconds)" > "$pipeline_status"
exit "$code"
