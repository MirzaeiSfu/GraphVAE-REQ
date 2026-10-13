#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${REPO_ROOT}/runs/table3_reproduction/grid_graphvae_table3_nearest_run"
REPORT_DIR="${REPO_ROOT}/runs/table3_reproduction/grid_graphvae_table3_nearest_eval"
MPL_DIR="/tmp/matplotlib-${USER:-codex}"
XDG_DIR="/tmp/xdg-cache-${USER:-codex}"

mkdir -p "${MPL_DIR}" "${XDG_DIR}" "${REPORT_DIR}"
export MPLCONFIGDIR="${MPL_DIR}"
export XDG_CACHE_HOME="${XDG_DIR}"

cd "${REPO_ROOT}"

echo "[run] training config reference: configs/kiarash_graphvae/grid_graphvae_table3_nearest.yaml"
echo "[run] run dir: ${RUN_DIR}"
echo "[run] report dir: ${REPORT_DIR}"
echo "[run] env: micro"
echo "[run] device: cuda:0"

conda run -n micro python main.py \
  --config configs/kiarash_graphvae/grid_graphvae_table3_nearest.yaml

conda run -n micro python scripts/reproduce_table3.py \
  --dataset GRID \
  --mode all \
  --run-dir "${RUN_DIR}" \
  --paper-row GraphVAE-MM \
  --row-label grid_graphvae_table3_nearest \
  --device cuda \
  --output-dir "${REPORT_DIR}"

echo "[done] table3 report: ${REPORT_DIR}/table3_grid_reproduction.md"
