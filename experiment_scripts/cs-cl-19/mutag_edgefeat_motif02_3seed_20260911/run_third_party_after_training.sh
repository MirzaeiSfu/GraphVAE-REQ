#!/usr/bin/env bash
set -uo pipefail
seed=${1:?seed required}
root=/local-scratch2/mirzaei/mutag_edgefeat_motif02_3seed_20260911
run="$root/runs/full_matrix/seed_$seed"
training_status="$root/status/seed_$seed.status"
status="$root/status/seed_${seed}_third_party.status"
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
evaluator=/local-scratch2/mirzaei/proteins_defog_3seed_20260910/run_external_random_gin.py
reference="/local-scratch2/mirzaei/mutag_nodefeat_3way_20260910/runs/graphvae_motif_true_full/seed_$seed/real_test_graphs.pt"
output="$root/third_party/seed_$seed"
log="$root/logs/seed_${seed}_third_party.log"

printf 'WAITING seed=%s host=%s\n' "$seed" "$(hostname)" > "$status"
until grep -q '^COMPLETE' "$training_status" 2>/dev/null; do sleep 60; done
until test -s "$run/generated_graphs.pt" && test -s "$run/graph_realism_random_gin.json"; do sleep 30; done
mkdir -p "$output"
printf 'EVALUATING seed=%s host=%s\n' "$seed" "$(hostname)" > "$status"
"$python_bin" "$evaluator" --generated "$run/generated_graphs.pt" --reference "$reference" --output "$output" --device cpu >> "$log" 2>&1
code=$?; if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=%s host=%s exit_code=%s finished=%s\n' "$state" "$seed" "$(hostname)" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
