#!/usr/bin/env bash
set -euo pipefail
root=/local-scratch2/mirzaei/aids_common_eval_10k_20260917
status=$root/status/report.status
for required in downstream motif_tv; do
  while ! grep -q '^COMPLETE' "$root/status/$required.status" 2>/dev/null; do sleep 30; done
done
printf 'RUNNING started=%s\n' "$(date --iso-8601=seconds)" > "$status"
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  "$root/scripts/build_aids_comparison_report.py" > "$root/logs/report_builder.log" 2>&1
printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$status"
