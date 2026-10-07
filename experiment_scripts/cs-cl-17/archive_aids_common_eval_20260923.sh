#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/common_eval_fix_20260923/AIDS
while [ ! -f "$root/status" ] || ! grep -q COMPLETE "$root/status"; do
  sleep 30
done
destination=/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/AIDS/evaluations/common_fixed_20260923
ssh -o BatchMode=yes mirzaei@cs-cl-19.cmpt.sfu.ca "mkdir -p $destination"
rsync -a -e 'ssh -o BatchMode=yes' "$root/" "mirzaei@cs-cl-19.cmpt.sfu.ca:$destination/"
