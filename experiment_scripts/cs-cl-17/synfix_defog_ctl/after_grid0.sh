C=~/synfix_defog_ctl
until [ -f $C/done_GRID_0 ]; do sleep 30; done
ssh -n mirzaei@cs-cl-13.cmpt.sfu.ca 'SYNFIX_W=$HOME/synfix_defog_work bash ~/synfix_defog_work/bundle/launch.sh GRID 4 20260924 0 500 $HOME/defog_ogbg_3seed_20260920/env/defog-benchmark/bin/python'
echo "GRID 4 13 GRID_seed4_g20260924" >> $C/streams.txt
ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca 'bash /local-scratch2/mirzaei/synfix_defog_work/bundle/launch.sh TRIANGULAR_GRID 0 20260924 1 500'
echo "TRIANGULAR_GRID 0 18 TRIANGULAR_GRID_seed0_g20260924" >> $C/streams.txt
date
