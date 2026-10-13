#!/usr/bin/env bash
# Copy finished LGD sample pickles into <BASE>/raw_pickles/<DATASET>_seed<k>/ (read-only on the source).
# Usage: fetch_lgd.sh DATASET [SEEDS...]   Run on cs-cl-18.  A seed is fetched only if its run has
# the COMPLETE marker and sampled/epoch_${E}_graphs.pkl + GATE.json exist.  Override the source
# with LGD_SRC="host:/path/to/results" (host may be 'solar' or 'local').
set -euo pipefail
dataset=${1:?dataset}; shift; seeds=${*:-0 1 2}
B=${LGD_EVAL_BASE:-/local-scratch2/mirzaei/lgd_common_eval_20260925}
SOLAR_SSH="ssh -p 24 -o BatchMode=yes"
default_src() {  # dataset seed -> host:path
  case "$1:$2" in
    PTC:0|PTC:1) echo "local:/local-scratch2/mirzaei/LGD_FIX_20260924/results" ;;
    PTC:2) echo "mirzaei@cs-cl-16.cmpt.sfu.ca:/local-scratch/mirzaei/LGD_FIX_20260924/results" ;;
    TRIANGULAR_GRID:0|TRIANGULAR_GRID:2) echo "mirzaei@cs-cl-09.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_FIX_20260924/results" ;;
    LOBSTER:0|LOBSTER:1|TRIANGULAR_GRID:5) echo "local:/local-scratch2/mirzaei/LGD_FIX_20260924/results" ;;
    LOBSTER:2|TRIANGULAR_GRID:3|TRIANGULAR_GRID:4) echo "mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_FIX_20260924/results" ;;
    TRIANGULAR_GRID:1) echo "mirzaei@cs-cl-19.cmpt.sfu.ca:/local-scratch2/mirzaei/LGD_FIX_20260924/results" ;;
    *) echo "solar:/home/mirzaei/LGD_FIX_20260924/results" ;;   # PROTEINS, GRID, QM9
  esac
}
# final sampled epoch: 2000-epoch runs -> 1999; QM9 (300 epochs) -> 299. Override with LGD_EPOCH.
if [ "$dataset" = QM9 ]; then E=${LGD_EPOCH:-299}; else E=${LGD_EPOCH:-1999}; fi
for seed in $seeds; do
  src=${LGD_SRC:-$(default_src "$dataset" "$seed")}
  host=${src%%:*}; path=${src#*:}
  run="$path/GraphVAEReq-diffusion-${dataset}${LGD_TAG_SUFFIX:-}_seed${seed}/${seed}"
  dst=$B/raw_pickles/${dataset}_seed${seed}; mkdir -p "$dst"
  case "$host" in
    local) test -f "$run/COMPLETE" && test -f "$run/sampled/epoch_${E}_graphs.pkl" || { echo "seed $seed not finished at $run"; exit 1; }
           cp -p "$run/sampled/epoch_${E}_graphs.pkl" "$run/sampled/GATE.json" "$dst/" ;;
    solar) mirzaei_solar=mirzaei@solar.cs.sfu.ca
           $SOLAR_SSH $mirzaei_solar "test -f $run/COMPLETE && test -f $run/sampled/epoch_${E}_graphs.pkl && test -f $run/sampled/GATE.json" || { echo "seed $seed not finished at solar:$run"; exit 1; }
           rsync -a -e "$SOLAR_SSH" "$mirzaei_solar:$run/sampled/epoch_${E}_graphs.pkl" "$mirzaei_solar:$run/sampled/GATE.json" "$dst/" ;;
    *) ssh -o BatchMode=yes "$host" "test -f $run/COMPLETE && test -f $run/sampled/epoch_${E}_graphs.pkl" || { echo "seed $seed not finished at $host:$run"; exit 1; }
       rsync -a "$host:$run/sampled/epoch_${E}_graphs.pkl" "$host:$run/sampled/GATE.json" "$dst/" ;;
  esac
  echo "fetched $dataset seed $seed -> $dst ($(grep -o '"status": "[A-Z]*"' "$dst/GATE.json"))"
done
