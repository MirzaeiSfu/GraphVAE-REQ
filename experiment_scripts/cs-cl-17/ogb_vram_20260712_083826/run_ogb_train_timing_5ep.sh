#!/usr/bin/env bash
set -u
cd /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check || exit 2
mkdir -p runs/ogb_timing_5ep
export OGB_DATA_ROOT=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-MM/dataset
export MOTIF_CACHE_DIR=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/cache_motifs_ogb_cap8
export PYTHONPATH=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/python_deps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
LOG=runs/ogb_timing_5ep/train_5ep.log
START=$(date +%s)
CUDA_VISIBLE_DEVICES=1 /localhome/mirzaei/miniconda3/envs/micro/bin/python main.py \
  --config configs/cluster_tests/ogbg_molbbbp_graphvae.yaml \
  --dataset ogbg-molbbbp \
  --database_name ogbg_molbbbp_feat_snap \
  --motif_loss true \
  --use_syntactic_literal_rules true \
  --syntactic_literal_rule_mode both \
  --rule_prune true \
  --motif_prune_max_values_per_rule 8 \
  --split_mode legacy_80_20 \
  --train_fraction 0.8 \
  --val_fraction 0.0 \
  --bfs_strategy all_components \
  --disable_dataset_cache true \
  --motif_cache_dir /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/cache_motifs_ogb_cap8 \
  --dataset_cache_dir /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/runs/ogb_timing_5ep/cache \
  --graph_save_path /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/runs/ogb_timing_5ep/out \
  --epoch_number 5 \
  --vis_step 100000 \
  --third_party_eval false \
  --plot_test_graphs false \
  --save_validation_checkpoints false \
  --checkpoint_interval_epochs 100000 \
  --motif_batch_size 50 \
  > "$LOG" 2>&1
status=$?
END=$(date +%s)
echo "status=$status elapsed_sec=$((END-START)) log=$LOG"
grep -E "Epoch:|Batch:|TrainingData|rule_prune|Loaded [0-9]+ motif rules|Batch +[0-9]|Done|Traceback|RuntimeError" "$LOG" | tail -80
exit "$status"
