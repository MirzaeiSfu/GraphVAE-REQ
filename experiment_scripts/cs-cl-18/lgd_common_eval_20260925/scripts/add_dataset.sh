#!/usr/bin/env bash
# ONE COMMAND to add a dataset (e.g. GRID or QM9 once LGD training finishes):
#   tmux new -d -s lgd_eval_GRID "bash /local-scratch2/mirzaei/lgd_common_eval_20260925/scripts/add_dataset.sh GRID 0"
#   tmux new -d -s lgd_eval_QM9  "bash /local-scratch2/mirzaei/lgd_common_eval_20260925/scripts/add_dataset.sh QM9 1"
# Steps: fetch pickles -> sanity gate (reproduce DeFoG seed 0; abort on mismatch) ->
# verify reference/label mapping -> stage -> evaluate -> RESULTS.md.  Run on cs-cl-18.
set -euo pipefail
dataset=${1:?GRID|QM9|...}; gpu=${2:-0}; shift 2 || true; seeds=${*:-0 1 2}
S=/local-scratch2/mirzaei/lgd_common_eval_20260925/scripts
bash "$S/fetch_lgd.sh" "$dataset" $seeds
bash "$S/sanity.sh" "$dataset" "$gpu"
bash "$S/run_lgd_eval.sh" "$dataset" "$gpu" $seeds
echo "Done. Review /local-scratch2/mirzaei/lgd_common_eval_20260925/$dataset/reference_verification.json (must say REFERENCE_EQUAL / IDENTITY) and RESULTS.md"
