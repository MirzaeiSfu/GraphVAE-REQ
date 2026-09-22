# Metrics where GraphVAE-REQ motif=True beats DeFoG

This report compares the aggregate means in each committed dataset report. Higher is better for precision, recall, and F1-PR; lower is better for MMD, error, and total-variation metrics. “Full” means the motif=True full-matrix model; “total” means the motif=True total-count model.

These are directional wins, not statistical-significance claims. A result is listed only when both methods appear under the same stated evaluation protocol. Missing or non-comparable cells are not wins.

## Summary

| Dataset | Motif=True variant | Metrics better than DeFoG | Qualification |
|---|---|---|---|
| GRID | Full | Degree, clustering, orbit, spectral, diameter and triangle MMD; topology F1-PR, recall and MMD-RBF | DeFoG uses healthy seeds 0 and 4 (`n=2`); three GraphVAE seeds |
| GRID | Total | Degree, clustering, orbit, spectral and diameter MMD; topology recall and MMD-RBF | Same `n=2` DeFoG qualification |
| LOBSTER | Full | Diameter MMD; topology F1-PR and recall; held-out pruned-rule TV | Three seeds; historical training-reference TV instead favors DeFoG |
| LOBSTER | Total | Topology F1-PR and recall | Three seeds |
| TRIANGULAR_GRID | Full | Degree, clustering, orbit, spectral, diameter, triangle and sparsity MMD; topology F1-PR, precision, recall, MMD-RBF and linear-MMD mean; pruned-rule TV | Healthy DeFoG seeds 0, 1 and replacement 3; seed 2 excluded as documented outlier |
| TRIANGULAR_GRID | Total | Degree, clustering, orbit, diameter and triangle MMD; topology F1-PR, precision, recall and MMD-RBF | Same outlier policy |
| MUTAG | Full | Topology recall | Primary matched topology table; historical DeFoG seed 0/1 outputs were duplicated |
| PTC | Full | Degree, clustering, orbit, spectral, diameter and triangle metrics; primary topology-derived GIN F1-PR, precision, recall, MMD-RBF and all reported linear-MMD summaries; constant-input F1-PR, precision and recall | Three seeds; DeFoG wins constant-input MMD-RBF and hard pruned-state TV |
| PTC | Total | Mean-edge absolute error | Three seeds |
| QM9 | Full | Topology recall; node-feature recall; mean-edge absolute error | Three seeds; DeFoG wins the other common aggregate metrics |
| AIDS | Full | Topology embedding MMD-RBF; degree MMD | DeFoG has only corrected seeds 1 and 2 and identical exported metrics; provisional |
| PROTEINS | Full, alpha 0.03 | Topology-control recall | Historical DeFoG seeds 0 and 1 are duplicate collections |
| OGB | — | None claimed | No verified common three-way comparison |

## Dataset-level interpretation

### GRID

Motif=True full is stronger than DeFoG on six of eight structural metrics and on three primary topology RandomGIN metrics. DeFoG remains better on sparsity, edge-count error, precision, the linear-MMD summaries, and pruned-rule TV. Because only two healthy DeFoG seeds are retained, this comparison is less stable than a full three-seed comparison.

### LOBSTER

DeFoG dominates most structural and embedding-distance metrics. Motif=True full wins diameter MMD, F1-PR, and recall. On held-out pruned-rule TV, full is `0.004021` versus DeFoG `0.006782`; on the separate historical training-reference protocol, DeFoG is better (`0.011229` versus `0.018693`).

### TRIANGULAR_GRID

This is the broadest Motif=True advantage. Full beats DeFoG on seven structural MMD metrics, five primary RandomGIN metrics, and pruned-rule TV. DeFoG remains better on edge-count error and two robust linear-MMD summaries.

### MUTAG

In the matched topology table, Motif=True wins only recall (`0.998291` versus `0.958120`). Corrected attributed DeFoG seeds were later evaluated, but there are only two, so the committed dataset report retains the protocol and independence warnings rather than presenting them as a complete three-seed replacement.

### PTC

Motif=True full wins most structural and primary topology-derived RandomGIN metrics. DeFoG is better on sparsity, constant-input MMD-RBF, hard pruned-state TV, complete-state TV, and the retained-rule count-error diagnostics. Thus PTC is a strong topology/structure result for Motif=True, but not a universal motif-distribution win.

### QM9

Motif=True beats DeFoG only on topology recall, node-feature recall, and mean-edge absolute error. DeFoG is substantially better on most structural, embedding, and rule-state-TV metrics.

### AIDS

Motif=True beats DeFoG on topology embedding MMD-RBF (`0.003375` versus `0.037542`) and degree MMD (`0.001448` versus `0.005885`). The DeFoG comparison is provisional because only two corrected seeds are available and their exported metrics are identical.

### PROTEINS

In the latest native-feature supplement, Motif=True beats DeFoG only on topology-control recall (`0.975599` versus `0.951675`). Other listed topology and node-feature RandomGIN metrics favor DeFoG. Duplicate historical DeFoG collections prevent treating the nominal three-seed SD as independent uncertainty.

### OGB

No win/loss claim is made. The archived supplement contains DeFoG-only attempted evaluation output, and all three newer OGB evaluation jobs are marked failed. A valid three-way common evaluation is required.

## Counting policy

The table inventories metric names, not independent scientific hypotheses. Closely related metrics such as F1, precision, recall, and multiple MMD summaries are reported separately because that is how the dataset reports present them. No global “win percentage” is calculated, since datasets have different available metrics and unequal seed quality.
