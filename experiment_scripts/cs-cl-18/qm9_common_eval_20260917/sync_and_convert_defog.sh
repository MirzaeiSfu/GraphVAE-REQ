#!/usr/bin/env bash
set -euo pipefail

root=/local-scratch2/mirzaei/qm9_common_eval_20260917
remote_root=/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916/raw_export_fixed
python=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
mkdir -p "$root/defog/raw" "$root/evaluations/defog" "$root/status"
printf 'WAITING started=%s\n' "$(date --iso-8601=seconds)" > "$root/status/defog_export.status"

for seed in 1 2; do
  while ! ssh mirzaei@cs-cl-17.cmpt.sfu.ca "grep -q '^COMPLETE' '$remote_root/seed_$seed.status'"; do
    if ssh mirzaei@cs-cl-17.cmpt.sfu.ca "grep -q '^FAILED' '$remote_root/seed_$seed.status'"; then
      echo "DeFoG raw export seed $seed failed" >&2
      exit 1
    fi
    sleep 30
  done
  rsync -a --partial \
    "mirzaei@cs-cl-17.cmpt.sfu.ca:$remote_root/seed_$seed/generated_graphs.pt" \
    "$root/defog/raw/seed_$seed.pt"
  "$python" "$root/convert_defog_raw_to_common_dgl.py" \
    --input "$root/defog/raw/seed_$seed.pt" \
    --output "$root/evaluations/defog/seed_$seed/generated_attributed_graphs.bin" \
    --metadata "$root/evaluations/defog/seed_$seed/conversion.json" \
    --seed "$seed"
done

printf 'COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" > "$root/status/defog_export.status"
