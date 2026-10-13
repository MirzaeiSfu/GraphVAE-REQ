#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index}
root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
gv_repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
status=$root/status/defog.status

while ! grep -q '^COMPLETE' "$root/status/prepare.status" 2>/dev/null; do sleep 30; done
# Share GPU 0 only after the motif=True GraphVAE evaluation releases it.
while ! grep -Eq '^(COMPLETE|FAILED)' "$root/status/graphvae_true_full.status" 2>/dev/null; do sleep 30; done
printf 'RUNNING gpu=%s started=%s\n' "$gpu" "$(date --iso-8601=seconds)" > "$status"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$repo/src:$repo:$gv_repo/graph_evaluation/src:${PYTHONPATH:-}"
cd "$repo"

for seed in 1 2; do
  run=$root/defog/raw/seed_$seed
  mkdir -p "$run" "$root/evaluations/defog/seed_$seed"
  if [[ ! -s "$run/generated_graphs.pt" ]]; then
    "$python" -u src/main.py dataset=aids +experiment=aids \
      dataset.datadir="$repo/data/aids_3seed_20260912/" \
      train.seed="$seed" train.batch_size=16 \
      general.test_only="$root/defog/checkpoints/seed_$seed.ckpt" \
      +general.generation_seed="$seed" \
      general.final_model_samples_to_generate=400 \
      general.final_model_samples_to_save=400 \
      general.final_model_chains_to_save=0 general.num_sample_fold=1 \
      general.evaluate_all_checkpoints=false general.name="aids_defog_seed_${seed}_common" \
      general.wandb=disabled general.gpus=1 hydra.run.dir="$run"
  fi
  "$python" "$root/scripts/convert_pyg_to_dgl.py" \
    --input "$run/generated_graphs.pt" \
    --output "$root/evaluations/defog/seed_$seed/generated_attributed_graphs.bin" \
    --metadata "$root/evaluations/defog/seed_$seed/conversion.json" --seed "$seed"
done
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
