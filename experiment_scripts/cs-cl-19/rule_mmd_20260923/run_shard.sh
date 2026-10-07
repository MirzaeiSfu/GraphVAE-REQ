#!/usr/bin/env bash
# Rule-MMD sharded worker for large state spaces (run in tmux on any lab machine).
#   run_shard.sh <DATASET> <GPU> <work root> <python> select [motif_batch_size]
#   run_shard.sh <DATASET> <GPU> <work root> <python> shard <i>/<n>
# select -> datasets/<DS>/selection.json ; shard -> datasets/<DS>/counts/*.shard<i>of<n>.npz
# Outputs are pushed to the hub (cs-cl-18); status in hub status/<DS>_<tag>.
set -euo pipefail

DS=$1; GPU=$2; WORK=$3; PY=$4; MODE=$5; ARG=${6:-}
HUB=mirzaei@cs-cl-18.cmpt.sfu.ca
HUB_ROOT=/local-scratch2/mirzaei/rule_mmd_20260923
DIR=$WORK/datasets/$DS
TAG=$MODE${ARG:+_${ARG//\//of}}
STATUS_LOCAL=$WORK/status_${DS}_$TAG

hub_status() { echo "$1 host=$(hostname) gpu=$GPU time=$(date --iso-8601=seconds)" > "$STATUS_LOCAL";
               scp -q "$STATUS_LOCAL" "$HUB:$HUB_ROOT/status/${DS}_$TAG" || true; }
trap 'hub_status "FAILED line=$LINENO"' ERR

mkdir -p "$WORK/datasets"
if [ "$(hostname)" != "cs-cl-18" ]; then
  hub_status "SYNCING"
  rsync -a "$HUB:$HUB_ROOT/rule_mmd.py" "$HUB:$HUB_ROOT/run_shard.sh" "$WORK/"
  rsync -a --delete "$HUB:$HUB_ROOT/source/" "$WORK/source/"
  rsync -a --exclude counts --exclude logs --exclude 'generated_alt' --exclude 'source_copies' \
    "$HUB:$HUB_ROOT/datasets/$DS/" "$DIR/"
fi
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$DIR/counts"

if [ "$MODE" = select ]; then
  hub_status "RUNNING select"
  "$PY" "$WORK/rule_mmd.py" select --dataset-dir "$DIR" --device cuda ${ARG:+--motif-batch-size $ARG}
  [ "$(hostname)" = "cs-cl-18" ] || rsync -a "$DIR/selection.json" "$HUB:$HUB_ROOT/datasets/$DS/"
else
  hub_status "RUNNING shard $ARG"
  OUT=$WORK/shard_out/$DS/$TAG
  mkdir -p "$OUT"
  "$PY" "$WORK/rule_mmd.py" count-all --dataset-dir "$DIR" --device cuda --shard "$ARG" --output-dir "$OUT"
  rsync -a --exclude partial "$OUT/" "$HUB:$HUB_ROOT/datasets/$DS/counts/"
fi
hub_status "COMPLETE"
