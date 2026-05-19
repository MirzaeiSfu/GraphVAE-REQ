# Table 3 Reproduction

This path reproduces the GNN-based metrics from Table 3 in `Kia paper with appendix.pdf`.

It uses the same vendored Random-GIN evaluator as `scripts/evaluate_graph_realism_batch.py`,
but wraps it in a paper-aligned report that compares current outputs against the Table 3 paper values.

## 1. Compute the `50/50 split` ideal row

Example for Grid:

```bash
python scripts/reproduce_table3.py \
  --dataset GRID \
  --mode ideal-50-50
```

By default this writes to:

- `runs/table3_reproduction/grid/metrics.json`
- `runs/table3_reproduction/grid/table3_grid_reproduction.md`

## 2. Compare a saved run against the paper row

Example for the saved Grid run:

```bash
python scripts/reproduce_table3.py \
  --dataset GRID \
  --mode evaluate-generated \
  --run-dir runs/table2_reproduction/grid_graphvae \
  --paper-row GraphVAE-MM \
  --row-label grid_graphvae_current
```

This reads:

- `Single_comp_generatedGraphs_adj_final_eval.npy`
- `testGraphs_adj_.npy`

from the run directory unless `--generated` or `--test-graphs` are given explicitly.

## 3. Compute both rows in one report

```bash
python scripts/reproduce_table3.py \
  --dataset LOBSTER \
  --mode all \
  --run-dir runs/table2_reproduction/lobster_graphvae \
  --paper-row GraphVAE-MM \
  --row-label lobster_current
```

## Notes

- The script reports Table 3 metrics only: `MMD RBF` and `F1 PR`.
- The report also includes the paper benchmark rows (`GraphRNN-S`, `GraphRNN`, `GRAN`, `BiGG`) as reference values.
- `TRIANGULAR_GRID`, `LOBSTER`, `GRID`, and `PROTEINS` can compute the ideal `50/50 split` row directly through `list_graph_loader`.
- For `ogbg-molbbbp`, the raw loader path is not enabled in this repo right now, so use `--source-graphs <saved_graphs.npy>` if you want the ideal `50/50 split` row from a pre-saved graph collection.
