#!/usr/bin/env bash
# Wait until all 3 seeds of DATASET have GATE.json on Solar, then run add_dataset.sh DATASET GPU.
DS=$1; GPU=$2; E=/local-scratch2/mirzaei/lgd_common_eval_20260925
while true; do
  n=$(ssh -p 24 -o BatchMode=yes mirzaei@solar.cs.sfu.ca "ls ~/LGD_FIX_20260924/results/GraphVAEReq-diffusion-${DS}_seed{0,1,2}/*/sampled/GATE.json 2>/dev/null | wc -l" 2>/dev/null | tail -1)
  echo "$(date +%F_%H:%M) $DS finished seeds: ${n:-?}/3"
  [ "${n:-0}" -ge 3 ] && break
  sleep 600
done
bash $E/scripts/add_dataset.sh $DS $GPU
echo "AUTO_ADD_DONE $DS $(date)"
