C=~/synfix_defog_ctl
until [ -f $C/done_TRIANGULAR_GRID_3 ]; do sleep 30; done
if [ ! -f $C/done_GRID_5 ]; then
  ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh GRID 5 20260926 1 500'
  echo "GRID 5 18 GRID_seed5_g20260926" >> $C/streams.txt; date
  ssh -n mirzaei@cs-cl-19.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh GRID 5 20260927 1 500'
  echo "GRID 5 19 GRID_seed5_g20260927" >> $C/streams.txt; date
fi
