# Grid Baseline vs Paper Table 2

The previous saved Grid experiment was `runs/grid_simple`. That was the **old baseline**, before the paper-style changes. It used the older split behavior: effectively `80/20` train/test, with validation taken from training, not a clean paper `70/10/20` split.

Compared with the paper and the new run:

| Metric | Paper GraphVAE | Old `runs/grid_simple` | New Table2 Repro |
| --- | ---: | ---: | ---: |
| Degree | 0.062 | 0.108757 | 0.046014 |
| Clustering | 0.055 | 0.186573 | 0.061631 |
| Orbit | 0.515 | 0.612729 in old log | 0.451460 |
| Spectral | 0.018 | 0.013805 | 0.011637 |
| Diameter | 0.143 | 0.101150 | 0.093923 |

So yes, the changes were useful for matching Table 2 better.

The biggest improvements are:

- Degree moved from `0.1088` to `0.0460`, much closer to paper.
- Clustering moved from `0.1866` to `0.0616`, much closer to paper.
- Orbit moved from worse-than-paper to better/lower than paper.
- Generated graph edge count improved a lot: old generated average was `507.95` edges vs test `409.7`; new generated average is `432.6` vs test `409.7`.

Small caveat: when I re-evaluated the old saved graph files with the new locked script, the old orbit metric came out even worse, around `0.7826`, while the old raw log says `0.6127`. That likely comes from the old ORCA temp-file instability. Either way, the conclusion is the same: the new reproduction setup is clearly closer to the paper.
