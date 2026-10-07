C=~/synfix_defog_ctl
until [ -f $C/done_TRIANGULAR_GRID_1 ]; do sleep 30; done
if [ ! -f $C/done_TRIANGULAR_GRID_3 ]; then
  ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh TRIANGULAR_GRID 3 20260924 1 500'
  echo "TRIANGULAR_GRID 3 18 TRIANGULAR_GRID_seed3_g20260924" >> $C/streams.txt; date
fi
if [ ! -f $C/done_GRID_5 ]; then
  ssh -n mirzaei@cs-cl-19.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh GRID 5 20260925 0 500'
  echo "GRID 5 19 GRID_seed5_g20260925" >> $C/streams.txt; date
fi
