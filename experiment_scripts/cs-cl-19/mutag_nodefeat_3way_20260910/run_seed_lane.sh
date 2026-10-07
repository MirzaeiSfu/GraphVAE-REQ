#!/usr/bin/env bash
set -uo pipefail

seed=${1:?seed required}
gpu=${2:?gpu required}

campaign=/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910
source_campaign=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910
graphvae_repo=$source_campaign/source/GraphVAE-REQ
defog_repo=$source_campaign/source/DeFoG
graphvae_python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
defog_python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
graphvae_config=configs/mutag_edgefeat_campaign/full_matrix.yaml

mkdir -p "$campaign/logs" "$campaign/status" \
  "$campaign/runs/graphvae_motif_true_full/seed_$seed" \
  "$campaign/runs/graphvae_motif_false/seed_$seed" \
  "$campaign/runs/defog/seed_$seed"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

run_graphvae() {
  local method=$1
  shift
  local run_dir=$campaign/runs/$method/seed_$seed
  local log=$campaign/logs/${method}_seed_${seed}.log
  local status=$campaign/status/${method}_seed_${seed}.status

  printf 'RUNNING method=%s host=%s gpu=%s seed=%s started=%s\n' \
    "$method" "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" > "$status"
  cd "$graphvae_repo" || return 2
  "$graphvae_python" -u main.py \
    --config "$graphvae_config" \
    --seed "$seed" \
    --third_party_eval_seed "$seed" \
    --device cuda:0 \
    --graph_save_path "$run_dir" \
    --run_label "mutag-nodefeat-3way-${method}-seed-${seed}" \
    --checkpoint_interval_epochs 0 \
    "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  local state=FAILED
  (( code == 0 )) && state=COMPLETE
  printf '%s method=%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
    "$state" "$method" "$(hostname)" "$gpu" "$seed" "$code" \
    "$(date --iso-8601=seconds)" > "$status"
  return "$code"
}

run_defog() {
  local method=defog
  local run_dir=$campaign/runs/$method/seed_$seed
  local log=$campaign/logs/${method}_seed_${seed}.log
  local status=$campaign/status/${method}_seed_${seed}.status
  local old_checkpoint=$source_campaign/runs/defog/seed_$seed/checkpoints/mutag_edgefeat_seed_$seed/last.ckpt
  local resume_args=()
  if [[ -s "$old_checkpoint" ]]; then
    resume_args+=("general.resume=$old_checkpoint")
  fi

  printf 'RUNNING method=%s host=%s gpu=%s seed=%s resume=%s started=%s\n' \
    "$method" "$(hostname)" "$gpu" "$seed" \
    "${old_checkpoint:-none}" "$(date --iso-8601=seconds)" > "$status"
  cd "$defog_repo" || return 2
  export PYTHONPATH="$defog_repo:$defog_repo/src"
  "$defog_python" src/main.py \
    dataset=mutag \
    +experiment=mutag \
    dataset.datadir=data/mutag_edgefeat/ \
    train.seed="$seed" \
    train.n_epochs=20000 \
    train.batch_size=131 \
    train.save_model=true \
    general.name="mutag_nodefeat_3way_seed_${seed}" \
    general.wandb=disabled \
    general.check_val_every_n_epochs=1000 \
    general.sample_every_val=1 \
    general.samples_to_generate=40 \
    general.final_model_samples_to_generate=39 \
    hydra.run.dir="$run_dir" \
    "${resume_args[@]}" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  local state=FAILED
  (( code == 0 )) && state=COMPLETE
  printf '%s method=%s host=%s gpu=%s seed=%s exit_code=%s finished=%s\n' \
    "$state" "$method" "$(hostname)" "$gpu" "$seed" "$code" \
    "$(date --iso-8601=seconds)" > "$status"
  return "$code"
}

# Run all methods even if an earlier method fails; each method has its own status.
run_graphvae graphvae_motif_true_full \
  --motif_loss true \
  --motif_output_mode full_matrix \
  --motif_cp_table_source cp_smoothed \
  --rule_prune true \
  --alpha_motif_loss 0.1 \
  --alpha_syntactic_literal_motif_loss 0.1 || true

run_graphvae graphvae_motif_false \
  --motif_loss false \
  --alpha_motif_loss 0 \
  --alpha_syntactic_literal_motif_loss 0 || true

run_defog || true

printf 'LANE_FINISHED host=%s gpu=%s seed=%s finished=%s\n' \
  "$(hostname)" "$gpu" "$seed" "$(date --iso-8601=seconds)" \
  > "$campaign/status/lane_seed_${seed}.status"
