#!/usr/bin/env bash
# Rule-MMD worker: run in tmux on any lab machine.
#   run_dataset.sh <DATASET> <GPU index> <work root on this host> <python>
# Pulls the dataset's staged inputs from the hub (cs-cl-18), runs
# select -> count (train, test, every generated item) -> score, and pushes
# selection/counts/results/log back to the hub. Existing counts are reused.
set -euo pipefail

DS=$1; GPU=$2; WORK=$3; PY=$4
HUB=mirzaei@cs-cl-18.cmpt.sfu.ca
HUB_ROOT=/local-scratch2/mirzaei/rule_mmd_20260923
DIR=$WORK/datasets/$DS
STATUS_LOCAL=$WORK/status_$DS

hub_status() { echo "$1 host=$(hostname) gpu=$GPU time=$(date --iso-8601=seconds)" > "$STATUS_LOCAL";
               scp -q "$STATUS_LOCAL" "$HUB:$HUB_ROOT/status/$DS" || true; }
trap 'hub_status "FAILED line=$LINENO"' ERR

mkdir -p "$WORK/datasets"
if [ "$(hostname)" != "cs-cl-18" ]; then
  rsync -a "$HUB:$HUB_ROOT/rule_mmd.py" "$HUB:$HUB_ROOT/run_dataset.sh" "$WORK/"
  rsync -a --delete "$HUB:$HUB_ROOT/source/" "$WORK/source/"
  rsync -a --exclude counts --exclude logs "$HUB:$HUB_ROOT/datasets/$DS/" "$DIR/"
  rsync -a "$HUB:$HUB_ROOT/datasets/$DS/counts" "$DIR/" 2>/dev/null || true
fi
mkdir -p "$DIR/counts" "$DIR/logs"
export CUDA_VISIBLE_DEVICES=$GPU
DEV=cuda

hub_status "RUNNING select"
[ -s "$DIR/selection.json" ] || "$PY" "$WORK/rule_mmd.py" select --dataset-dir "$DIR" --device $DEV

run_count() {  # name [generated path]
  local name=$1 gen=${2:-}
  [ -s "$DIR/counts/$name.npz" ] && { echo "skip $name"; return; }
  hub_status "RUNNING count $name"
  if [ -n "$gen" ]; then
    "$PY" "$WORK/rule_mmd.py" count --dataset-dir "$DIR" --device $DEV --name "$name" \
      --generated "$gen" --output "$DIR/counts/$name.npz"
  else
    "$PY" "$WORK/rule_mmd.py" count --dataset-dir "$DIR" --device $DEV --name "$name" \
      --output "$DIR/counts/$name.npz"
  fi
  rsync -a "$DIR/counts/$name.npz" "$HUB:$HUB_ROOT/datasets/$DS/counts/" 2>/dev/null || true
}

run_count train
run_count test
while IFS=$'\t' read -r name path; do
  run_count "$name" "$path"
done < <("$PY" - "$DIR" "$WORK" <<'PY'
import json, sys
d, work = sys.argv[1], sys.argv[2]
text = open(f"{d}/inputs.json").read().replace("/local-scratch2/mirzaei/rule_mmd_20260923", work)
for item in json.loads(text)["items"]:
    print(f"{item['method']}_seed_{item['seed']}\t{item['path']}")
PY
)

hub_status "RUNNING score"
"$PY" "$WORK/rule_mmd.py" score --dataset-dir "$DIR"
if [ "$(hostname)" != "cs-cl-18" ]; then
  rsync -a "$DIR/selection.json" "$DIR/rule_mmd_results.json" "$DIR/counts" "$HUB:$HUB_ROOT/datasets/$DS/"
fi
hub_status "COMPLETE"
