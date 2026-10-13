#!/usr/bin/env bash
set -euo pipefail

src=/tmp/motif_true_sources
dst=/local-scratch/localhome/mirzaei/motif_true_effective_configs_20260903
mkdir -p "$dst"

for input in "$src"/*.yaml; do
  name=$(basename "$input")
  cp "$input" "$dst/$name"
done

# Make the YAML files self-contained versions of the effective Solar runs.
# The dataset-specific GraphVAE-MM switch maps GRID/TRIANGULAR_GRID to
# BCE=50, KL=2000 and LOBSTER to BCE=40, KL=2000.
sed -i '/alpha_adj_recon: 0.0/a\  use_graphvae_mm_bce_kl_weights: true' "$dst"/*.yaml
sed -i 's/alpha_motif_loss: 0.1/alpha_motif_loss: 5.0/' "$dst"/grid_*.yaml "$dst"/triangular_grid_*.yaml
sed -i 's/alpha_motif_loss: 0.1/alpha_motif_loss: 4.0/' "$dst"/lobster_*.yaml
sed -i 's#runs/linkcorr_motif_true_3seed/#runs/linkcorr_motif_true_effective/#g' "$dst"/*.yaml
sed -i 's/linkcorr-motif-true-/linkcorr-motif-true-effective-/g' "$dst"/*.yaml

cat > "$dst/README.txt" <<'EOF'
Effective motif=True configuration templates, 2026-09-03.

Run seeds 0, 1, and 2 by overriding both data.seed and
runtime.third_party_eval_seed on the command line. These YAMLs already encode
motif_loss=true, cp_smoothed, pruning, calibrated_gaussian, no node/edge
feature loss, and the effective dataset-specific GraphVAE-MM BCE/KL switch.

Effective weights:
GRID: BCE 50, KL 2000, motif 5, node 0, edge 0
LOBSTER: BCE 40, KL 2000, motif 4, node 0, edge 0
TRIANGULAR_GRID: BCE 50, KL 2000, motif 5, node 0, edge 0
EOF
