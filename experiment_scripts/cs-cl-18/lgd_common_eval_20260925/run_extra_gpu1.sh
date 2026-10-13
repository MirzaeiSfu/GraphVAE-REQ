set -x
bash /local-scratch2/mirzaei/lgd_common_eval_20260925/scripts/add_dataset.sh QM9 1 0 1
export LGD_EVAL_BASE=/local-scratch2/mirzaei/lgd_common_eval_20260925_tri_bigbatch
S=/local-scratch2/mirzaei/lgd_common_eval_20260925_tri_bigbatch/scripts
LGD_TAG_SUFFIX=_b32 bash $S/fetch_lgd.sh TRIANGULAR_GRID 0
LGD_TAG_SUFFIX=_b16 bash $S/fetch_lgd.sh TRIANGULAR_GRID 1
LGD_TAG_SUFFIX=_b32 bash $S/fetch_lgd.sh TRIANGULAR_GRID 2
bash $S/sanity.sh TRIANGULAR_GRID 1
bash $S/run_lgd_eval.sh TRIANGULAR_GRID 1 0 1 2
echo EXTRA_GPU1_DONE $(date)
