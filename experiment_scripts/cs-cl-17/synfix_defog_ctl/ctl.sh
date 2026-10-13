#!/bin/bash
# Controller: stop each DeFoG run at N accepted graphs (summed over its streams, in streams.txt order),
# collect raw progress to cs-cl-18 and convert to rule_mmd synthetic_fix/generated/<DS>/defog/seed_k/.
N=${N:-100}
C=~/synfix_defog_ctl
S=/local-scratch2/mirzaei/rule_mmd_20260923/synthetic_fix/generated/scripts_defog
PY=/local-scratch2/mirzaei/Abdolreza/venvs/defog-benchmark/bin/python
COL=/local-scratch2/mirzaei/synfix_defog_work/collected
rdir() { [ "$1" = 13 ] && echo '~/synfix_defog_work/runs' || echo /local-scratch2/mirzaei/synfix_defog_work/runs; }
while true; do
  for run in $(awk '{print $1":"$2}' $C/streams.txt | awk '!s[$0]++'); do
    ds=${run%%:*}; sd=${run##*:}
    [ -f $C/done_${ds}_${sd} ] && continue
    tot=0
    while read d s h st; do
      [ "$d:$s" = "$run" ] || continue
      n=$(timeout 90 ssh -n -o ConnectTimeout=20 mirzaei@cs-cl-$h.cmpt.sfu.ca "grep -a '\[synfix\] batch' $(rdir $h)/$st.log | sed -E 's/.*size ([0-9]+).*/\1/' | paste -sd+ | bc")
      tot=$((tot + ${n:-0}))
    done < $C/streams.txt
    echo "$(date +%T) $run total=$tot"
    if [ $tot -ge $N ]; then
      streams=""
      while read d s h st; do
        [ "$d:$s" = "$run" ] || continue
        R=$(rdir $h); dst=$COL/${st}@cs-cl-$h
        ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca "mkdir -p $dst"
        for f in progress_batches.pkl synfix_invocation.json; do
          ssh -n mirzaei@cs-cl-$h.cmpt.sfu.ca "cat $R/$st/$f" | ssh mirzaei@cs-cl-18.cmpt.sfu.ca "cat > $dst/$f"
        done
        ssh -n mirzaei@cs-cl-$h.cmpt.sfu.ca "cat $R/$st/.hydra/config.yaml" | ssh mirzaei@cs-cl-18.cmpt.sfu.ca "cat > $dst/hydra_config.yaml"
        streams="$streams $dst"
      done < $C/streams.txt
      out=/local-scratch2/mirzaei/rule_mmd_20260923/synthetic_fix/generated/$ds/defog/seed_$sd
      if ssh -n mirzaei@cs-cl-18.cmpt.sfu.ca "cd /local-scratch2/mirzaei/synfix_defog_work && $PY $S/convert_defog.py --dataset $ds --train-seed $sd --streams $streams --n $N --ckpt bundle/ckpt/${ds}_seed_${sd}.ckpt --out-dir $out" >> $C/convert_${ds}_${sd}.log 2>&1; then
        while read d s h st; do
          [ "$d:$s" = "$run" ] || continue
          ssh -n mirzaei@cs-cl-$h.cmpt.sfu.ca "tmux kill-session -t synfix_defog_$st" && echo "$(date +%T) killed $st on $h"
        done < $C/streams.txt
        touch $C/done_${ds}_${sd}; echo "$(date +%T) DONE $run -> $out"
      else
        echo "$(date +%T) convert failed for $run (see convert log); will retry"
      fi
    fi
  done
  sleep 120
done
