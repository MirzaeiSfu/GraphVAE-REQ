#!/usr/bin/env bash
# Count DeFoG synthetic-fix runs as they appear; when all 9 are counted, score and compare.
R=/local-scratch2/mirzaei/rule_mmd_20260923; P=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
cd $R
while true; do
  bash $R/synthetic_fix/count_available.sh 1 2>&1 | grep -E "nnz=|Traceback|Error" >> $R/synthetic_fix/watch_defog.log
  n=$(ls $R/datasets_fix/*/counts/defog_seed_*.npz 2>/dev/null | wc -l)
  echo "$(date +%H:%M) defog counted: $n/9" >> $R/synthetic_fix/watch_defog.log
  if [ "$n" -ge 9 ]; then
    for DS in GRID TRIANGULAR_GRID LOBSTER; do $P rule_mmd.py score --dataset-dir datasets_fix/$DS --references train,test,reference_new_seed_0 > /dev/null; done
    PYTHONPATH=. $P synthetic_fix/synfix_compare.py > $R/synthetic_fix/compare_final.txt 2>&1
    echo "ALLDONE $(date)" >> $R/synthetic_fix/watch_defog.log; break
  fi
  sleep 120
done
