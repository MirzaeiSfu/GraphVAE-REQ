#!/usr/bin/env bash
set -u
cd /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check || exit 2
mkdir -p runs/ogb_vram_benchmark_99
export OGB_DATA_ROOT=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-MM/dataset
export MOTIF_CACHE_DIR=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/cache_motifs_ogb_cap8
export PYTHONPATH=/local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/python_deps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
LOG=runs/ogb_vram_benchmark_99/ogb_99cols_vram.log
MON=runs/ogb_vram_benchmark_99/ogb_99cols_vram_monitor.tsv
GPU_MON=runs/ogb_vram_benchmark_99/ogb_99cols_gpu_total.tsv
: > "$MON"
: > "$GPU_MON"
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
  --sanity_check_only \
  --disable_dataset_cache true \
  --motif_cache_dir /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/cache_motifs_ogb_cap8 \
  --dataset_cache_dir /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/runs/ogb_vram_benchmark_99/cache \
  --graph_save_path /local-scratch/localhome/mirzaei/ogb_vram_20260712_083826/GraphVAE-REQ-main-check/runs/ogb_vram_benchmark_99/out \
  --third_party_eval false \
  --plot_test_graphs false \
  --motif_batch_size 50 \
  > "$LOG" 2>&1 &
pid=$!
echo "pid=$pid"
max=0
gmax=0
while kill -0 "$pid" 2>/dev/null; do
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v p="$pid" '$1+0==p {gsub(/ /,"",$2); print $2+0}')
  gmem=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1+0}')
  ts=$(date +%s)
  if [[ -n "${mem:-}" ]]; then
    echo -e "${ts}\t${mem}" >> "$MON"
    if (( mem > max )); then max=$mem; fi
  fi
  if [[ -n "${gmem:-}" ]]; then
    echo -e "${ts}\t${gmem}" >> "$GPU_MON"
    if (( gmem > gmax )); then gmax=$gmem; fi
  fi
  sleep 1
done
wait "$pid"
status=$?
echo "status=$status max_process_mem_mib=$max max_gpu1_total_mib=$gmax log=$LOG mon=$MON gpu_mon=$GPU_MON"
grep -E "rule_prune=True|Loaded [0-9]+ motif rules|Batch|SanityCheck|Counts match" "$LOG" | tail -80
exit "$status"
