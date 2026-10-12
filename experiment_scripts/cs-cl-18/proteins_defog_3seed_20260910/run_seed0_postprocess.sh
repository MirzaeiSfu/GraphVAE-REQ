#!/usr/bin/env bash
set -uo pipefail
root=/local-scratch2/mirzaei/proteins_defog_3seed_20260910
python_bin=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
status="$root/status/proteins_seed_0_third_party.status"
until ! tmux has-session -t proteins_defog_s0_final_eval 2>/dev/null; do sleep 30; done
generated="$root/final_eval/seed_0/generated_graphs.pt"
reference=/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ/runs/defog/proteins/real_test_graphs.pt
if ! test -s "$generated"; then printf 'FAILED missing=%s\n' "$generated" > "$status"; exit 3; fi
printf 'EVALUATING seed=0 host=%s\n' "$(hostname)" > "$status"
"$python_bin" "$root/run_external_random_gin.py" --generated "$generated" --reference "$reference" --output "$root/final_eval/seed_0/third_party" --device cpu >> "$root/logs/proteins_seed_0_third_party.log" 2>&1
code=$?; if (( code == 0 )); then state=COMPLETE; else state=FAILED; fi
printf '%s seed=0 host=%s exit_code=%s finished=%s\n' "$state" "$(hostname)" "$code" "$(date --iso-8601=seconds)" > "$status"
exit "$code"
