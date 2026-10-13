#!/usr/bin/env bash
set -euo pipefail
group=$1
shift
root=/local-scratch2/mirzaei/metapath_instance_mmd_20260923
mkdir -p "$root/logs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec > >(tee -a "$root/logs/$group.log") 2>&1
echo "START $(date -Is) datasets=$*"
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python -u \
  /local-scratch/localhome/mirzaei/metapath_instance_mmd.py "$@"
echo "WORKER EXIT $(date -Is); see each dataset status.txt for COMPLETE/PARTIAL/BLOCKED"
