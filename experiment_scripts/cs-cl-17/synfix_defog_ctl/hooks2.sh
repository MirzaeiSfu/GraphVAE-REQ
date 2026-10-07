C=~/synfix_defog_ctl
h1=0; h2=0
while [ $h1 = 0 ] || [ $h2 = 0 ]; do
  if [ $h1 = 0 ] && [ -f $C/done_GRID_4 ] && [ ! -f $C/done_GRID_5 ]; then
    ssh -n mirzaei@cs-cl-13.cmpt.sfu.ca 'SYNFIX_W=$HOME/synfix_defog_work bash ~/synfix_defog_work/bundle/launch.sh GRID 5 20260924 0 500 $HOME/defog_ogbg_3seed_20260920/env/defog-benchmark/bin/python'
    echo "GRID 5 13 GRID_seed5_g20260924" >> $C/streams.txt; date; h1=1
  fi
  if [ $h2 = 0 ] && [ -f $C/done_TRIANGULAR_GRID_0 ] && [ ! -f $C/done_TRIANGULAR_GRID_1 ]; then
    ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh TRIANGULAR_GRID 1 20260924 1 500'
    echo "TRIANGULAR_GRID 1 18 TRIANGULAR_GRID_seed1_g20260924" >> $C/streams.txt; date; h2=1
  fi
  sleep 30
done
