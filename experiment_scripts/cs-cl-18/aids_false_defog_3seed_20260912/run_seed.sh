#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}
mode=${3:-both}
root=/local-scratch2/mirzaei/aids_false_defog_3seed_20260912
gv_repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/GraphVAE-REQ
df_repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
gv_python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
df_python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
config=$root/configs/aids_graphvae.yaml
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 WANDB_MODE=disabled
mkdir -p "$root/status" "$root/logs" "$root/runs/graphvae_false/seed_$seed" "$root/runs/defog/seed_$seed"

run_gv() {
  local status=$root/status/graphvae_false_seed_${seed}.status
  printf 'RUNNING method=graphvae_false host=%s gpu=%s seed=%s started=%s\n' "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
  cd "$gv_repo" || return 2
  "$gv_python" -u main.py --config "$config" --seed "$seed" --third_party_eval_seed "$seed" \
    --data_dir "$gv_repo/data_raw" --device cuda:0 \
    --motif_loss false --alpha_motif_loss 0 --alpha_syntactic_literal_motif_loss 0 \
    --dataset_cache_dir "$root/dataset_cache/shared" --disable_dataset_cache false --require_existing_dataset_cache true \
    --checkpoint_interval_epochs 0 --graph_save_path "$root/runs/graphvae_false/seed_$seed" \
    --run_label "aids-graphvae-false-seed-$seed" 2>&1 | tee "$root/logs/graphvae_false_seed_${seed}.log"
  local code=${PIPESTATUS[0]}; local state=FAILED; ((code==0)) && state=COMPLETE
  printf '%s method=graphvae_false host=%s seed=%s exit_code=%s finished=%s\n' "$state" "$(hostname)" "$seed" "$code" "$(date --iso-8601=seconds)" > "$status"
  return "$code"
}

run_defog() {
  local status=$root/status/defog_seed_${seed}.status
  printf 'RUNNING method=defog host=%s gpu=%s seed=%s env=%s started=%s\n' "$(hostname)" "$gpu" "$seed" "$df_python" "$(date --iso-8601=seconds)" > "$status"
  cd "$df_repo" || return 2
  export PYTHONPATH="$df_repo:$df_repo/src"
  "$df_python" src/main.py dataset=aids +experiment=aids dataset.datadir=data/aids_3seed_20260912/ \
    train.seed="$seed" train.n_epochs=1592 train.batch_size=16 train.save_model=true \
    general.name="aids_defog_seed_$seed" general.wandb=disabled \
    general.final_model_samples_to_generate=400 \
    hydra.run.dir="$root/runs/defog/seed_$seed" 2>&1 | tee "$root/logs/defog_seed_${seed}.log"
  local code=${PIPESTATUS[0]}; local state=FAILED; ((code==0)) && state=COMPLETE
  printf '%s method=defog host=%s seed=%s env=%s exit_code=%s finished=%s\n' "$state" "$(hostname)" "$seed" "$df_python" "$code" "$(date --iso-8601=seconds)" > "$status"
  return "$code"
}

if [[ "$mode" == both || "$mode" == graphvae ]]; then run_gv || true; fi
if [[ "$mode" == both || "$mode" == defog ]]; then run_defog || true; fi
printf 'LANE_FINISHED host=%s seed=%s finished=%s\n' "$(hostname)" "$seed" "$(date --iso-8601=seconds)" > "$root/status/lane_seed_${seed}.status"
