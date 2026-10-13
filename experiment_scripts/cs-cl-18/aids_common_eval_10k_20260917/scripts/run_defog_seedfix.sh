#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-0}
root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
repo=/local-scratch2/mirzaei/mutag_edgefeat_campaign_20260910/source/DeFoG
gv_repo=$root/source/GraphVAE-REQ
python=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
campaign=$root/defog/seedfix_20260919

mkdir -p "$campaign/logs" "$campaign/raw" "$campaign/evaluations"
export CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$repo/src:$repo:$gv_repo/graph_evaluation/src:${PYTHONPATH:-}"
cd "$repo"

# Use visibly distinct inference seeds. The training checkpoint number and the
# generation seed are intentionally recorded separately.
for checkpoint_seed in 1 2; do
  generation_seed=$((9100 + checkpoint_seed))
  run=$campaign/raw/checkpoint_${checkpoint_seed}_generation_${generation_seed}
  eval_dir=$campaign/evaluations/checkpoint_${checkpoint_seed}_generation_${generation_seed}
  mkdir -p "$run" "$eval_dir"

  "$python" -u src/main.py dataset=aids +experiment=aids \
    dataset.datadir="$repo/data/aids_3seed_20260912/" \
    train.seed="$checkpoint_seed" train.batch_size=16 \
    general.test_only="$root/defog/checkpoints/seed_${checkpoint_seed}.ckpt" \
    general.generation_seed="$generation_seed" \
    general.final_model_samples_to_generate=400 \
    general.final_model_samples_to_save=400 \
    general.final_model_chains_to_save=0 general.num_sample_fold=1 \
    general.evaluate_all_checkpoints=false \
    general.name="aids_defog_ckpt_${checkpoint_seed}_genseed_${generation_seed}" \
    general.wandb=disabled general.gpus=1 hydra.run.dir="$run" \
    2>&1 | tee "$campaign/logs/checkpoint_${checkpoint_seed}_generation_${generation_seed}.log"

  "$python" "$root/scripts/convert_pyg_to_dgl.py" \
    --input "$run/generated_graphs.pt" \
    --output "$eval_dir/generated_attributed_graphs.bin" \
    --metadata "$eval_dir/conversion.json" --seed "$generation_seed"

  sha256sum "$run/generated_graphs.pt" > "$eval_dir/generated_graphs.sha256"
done

sha256sum "$campaign"/raw/*/generated_graphs.pt > "$campaign/generated_graphs_all.sha256"
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$campaign/status.txt"
