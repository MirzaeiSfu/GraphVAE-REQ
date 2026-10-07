set -x
bash /local-scratch2/mirzaei/lgd_common_eval_20260925/scripts/add_dataset.sh LOBSTER 0 0 1 2
export LGD_EVAL_BASE=/local-scratch2/mirzaei/lgd_common_eval_20260925_tri_seeds345
bash /local-scratch2/mirzaei/lgd_common_eval_20260925_tri_seeds345/scripts/add_dataset.sh TRIANGULAR_GRID 0 3 4 5
echo EXTRA_GPU0_DONE $(date)
