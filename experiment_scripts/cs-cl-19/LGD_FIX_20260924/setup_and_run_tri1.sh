#!/usr/bin/env bash
set -euo pipefail
N=/local-scratch2/mirzaei/LGD_FIX_20260924
H=mirzaei@cs-cl-18.cmpt.sfu.ca
rsync -a --exclude __pycache__ $H:/local-scratch2/mirzaei/LGD_FIX_20260924/repo/ $N/repo/
if [ ! -x $N/env/bin/python ]; then
  rsync -a $H:/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920/lgd_env.tar.gz $N/lgd_env.tar.gz
  tar -xzf $N/lgd_env.tar.gz -C $N/env
  [ -x $N/env/bin/conda-unpack ] && $N/env/bin/conda-unpack
  rm -f $N/lgd_env.tar.gz
fi
$N/env/bin/python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available())"
LGD_CAMPAIGN_ROOT=$N LGD_ENV_DIR=$N/env bash $N/repo/run_lgd_fixed.sh TRIANGULAR_GRID 1 0 > $N/logs/TRIANGULAR_GRID_seed1.log 2>&1
