#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/qm9_common_eval_20260917
source_repo=/local-scratch2/mirzaei/fb/GraphVAE-REQ
eval_repo=$root/source/GraphVAE-REQ

while ! grep -q '^COMPLETE' "$root/status/sync.status" 2>/dev/null; do
  sleep 30
done

printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$root/status/source.status"
mkdir -p "$eval_repo"
rsync -a \
  --exclude='.git/' \
  --exclude='runs/' \
  --exclude='cache_motifs/' \
  --exclude='data_raw/' \
  --exclude='__pycache__/' \
  "$source_repo/" "$eval_repo/"

# These two modules differed from the training checkout.  Use the exact Solar
# versions that produced the QM9 checkpoints and motif cache.
cp "$root/provenance/data.py" "$eval_repo/data.py"
mkdir -p "$eval_repo/motif_counting"
cp "$root/provenance/motif_counter.py" "$eval_repo/motif_counting/motif_counter.py"

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$root/status/source.status"
