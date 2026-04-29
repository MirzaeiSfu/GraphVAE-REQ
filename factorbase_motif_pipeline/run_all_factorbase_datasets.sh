#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for dataset in GRID LOBSTER TRIANGULAR_GRID QM9 PROTEINS; do
  python run_factorbase_pipeline.py "$dataset"
done
