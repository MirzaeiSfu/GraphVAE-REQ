# Grid GraphVAE Baseline vs Paper

This note compares the saved baseline GraphVAE run in `runs/grid_simple` against the `Grid / GraphVAE` row in Table 2 of `Kia paper with appendix.pdf`.

## Split clarification

- In the paper, the `50/50 split` row is an ideal/reference score, not the GraphVAE training split.
- The paper states that model experiments use `train (70%), validation (10%) and test (20%)` splits.
- The `50/50 split` row is described as a score computed from a `50/50 split of the data set` and is used as a lower-bound reference.
- Therefore the `Grid / GraphVAE` row should not be read as a `50/50 split` experiment.

## Reproduction workflow

- Reproduction changes are isolated on branch `reproduce-table2-grid`.
- Use `configs/reproduce_table2/grid_graphvae_table2.yaml` for the paper-style `70/10/20` GraphVAE run.
- The reproduction config uses the paper-style `legacy_first_component` BFS strategy.
- Use `scripts/reproduce_table2_grid.py` to compute the Table 2 `50/50 split` row and to compare saved generated graphs.
- The computed Grid `50/50 split` result is saved in `runs/table2_reproduction/grid_50_50/table2_grid_reproduction.md`.

## Which saved result is comparable to the paper?

- `runs/grid_simple/mmd.log` contains validation-set checkpoint metrics written during training.
- The paper table should be compared against the final test-set evaluation that runs after training completes.
- In this codebase, the final test-set evaluation is the `EvalTwoSet(model, test_list_adj, ..., _f_name="final_eval")` call in `main.py`.
- Therefore the current-run numbers below come from the last `result for subgraph with maximum connected componnent` block in `runs/grid_simple/train.log`.

## Important reproducibility caveat

- The current code does not appear to match the paper's stated `70/10/20` split exactly.
- In `data.py`, `data_split(...)` uses an `80/20` train/test split.
- In `main.py`, `val_adj` is then taken from the training portion, so validation examples are not held out in the same way as the paper description suggests.
- Because of this, the comparison below is still useful as a baseline check, but it is not a strict reproduction of the paper setup.

## Table 2 comparison

Lower is better for all MMD values.

| Metric     | Paper `Grid / GraphVAE` | Current saved baseline | Difference `(current - paper)` | Match? |
| ---        | ---:     | ---:     | ---:               | --- |
| Degree     | 0.062000 | 0.108757 | +0.046757 (+75.4%) | No |
| Clustering | 0.055000 | 0.186573 | +0.131573 (+239.2%)| No |
| Orbit      | 0.515000 | 0.612729 | +0.097729 (+19.0%) | No |
| Spectral   | 0.018000 | 0.013805 | -0.004195 (-23.3%) | No, current is lower |
| Diameter   | 0.143000 | 0.101150 | -0.041850 (-29.3%) | No, current is lower |

## Summary

- The paper's `50/50 split` row is an ideal lower-bound reference, not the GraphVAE experiment split.
- The current baseline does not reproduce the paper's `Grid / GraphVAE` Table 2 results exactly.
- The current run is worse on degree, clustering, and orbit MMD.
- The current run is better on spectral and diameter MMD.
- The final saved evaluation also reports a higher average edge count in generated graphs than in the test set: `507.95` vs `409.7`, which is consistent with denser generated graphs.

## Sources used

- Paper source: `Kia paper with appendix.pdf`, Table 2, `Grid / GraphVAE` row.
- Current run source: `runs/grid_simple/train.log`, final `result for subgraph with maximum connected componnent` block after training.
- Validation checkpoint metrics in `runs/grid_simple/mmd.log` were intentionally not used for the paper comparison.

## Scope note

This note compares the statistics-based evaluation from Table 2 only. The saved baseline artifacts do not include the paper's Table 1 GNN-based metrics (`MMD RBF`, `F1 PR`).
