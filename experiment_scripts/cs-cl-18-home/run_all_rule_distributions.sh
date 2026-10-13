#!/usr/bin/env bash
set -euo pipefail
group=$1
shift
root=/local-scratch2/mirzaei/all_rule_distributions_20260924
mkdir -p "$root/logs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec >> "$root/logs/$group.log" 2>&1
echo "START $(date -Is) $*"
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python -u /local-scratch/localhome/mirzaei/all_rule_distribution_eval.py "$@"
echo "EXIT $(date -Is); check dataset status.txt"
