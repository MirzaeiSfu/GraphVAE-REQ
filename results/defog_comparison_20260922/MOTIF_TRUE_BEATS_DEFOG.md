# Metrics where GraphVAE-REQ motif=True beats DeFoG

This report compares the aggregate means in each committed dataset report. Higher is better for precision, recall, and F1-PR; lower is better for MMD, error, and total-variation metrics. “Full” means the motif=True full-matrix model; “total” means the motif=True total-count model.

These are directional wins, not statistical-significance claims. A result is listed only when both methods appear under the same stated evaluation protocol. Missing or non-comparable cells are not wins.

## Summary

| Dataset | Motif=True variant | Metrics better than DeFoG | Qualification |
|---|---|---|---|
| GRID | Full | Degree, clustering, orbit, spectral, diameter and triangle MMD; topology MMD-RBF | Corrected common 20-graph reference; DeFoG seeds 0/4/5 |
| GRID | Total | Degree, clustering, orbit, spectral and diameter MMD; topology recall and MMD-RBF | Same `n=2` DeFoG qualification |
| LOBSTER | Full | Diameter MMD; topology F1-PR and recall; held-out pruned-rule TV | Three seeds; historical training-reference TV instead favors DeFoG |
| LOBSTER | Total | Topology F1-PR and recall | Three seeds |
| TRIANGULAR_GRID | Full | Degree, clustering, orbit, spectral, diameter, triangle and sparsity MMD; topology F1-PR, precision and MMD-RBF; pruned-rule TV | Corrected common 20-graph reference; recall tied; DeFoG seeds 0/1/3 |
| TRIANGULAR_GRID | Total | Degree, clustering, orbit, diameter and triangle MMD; topology F1-PR, precision, recall and MMD-RBF | Same outlier policy |
| MUTAG | Full | Topology recall | Primary matched topology table; historical DeFoG seed 0/1 outputs were duplicated |
| PTC | Full | Degree, clustering, orbit, spectral, diameter and triangle metrics; primary topology-derived GIN F1-PR, precision, recall, MMD-RBF and all reported linear-MMD summaries; constant-input F1-PR, precision and recall | Three seeds; DeFoG wins constant-input MMD-RBF and hard pruned-state TV |
| PTC | Total | Mean-edge absolute error | Three seeds |
| QM9 | Full | Topology recall; node-feature recall; mean-edge absolute error | Three seeds; DeFoG wins the other common aggregate metrics |
| AIDS | Full | Topology, node, edge and node+edge MMD-RBF; degree MMD | Corrected independent DeFoG seeds 3/4/5; common 400-graph reference |
| PROTEINS | Full, alpha 0.03 | Topology-control recall; orbit, diameter and triangle MMD | Corrected independent DeFoG outputs; common 209-graph reference |
| OGB | Setting 03 | None claimed against DeFoG | Common terminal-chain evaluation completed and favors DeFoG throughout; historical final-output GraphVAE results are a different protocol |

## Dataset-level interpretation

### GRID

Under the corrected common-reference evaluation, Motif=True full is stronger than DeFoG on six structural MMD metrics and topology MMD-RBF. DeFoG now wins F1-PR (`0.956353` versus `0.913467`), precision, recall, and mean-edge error. DeFoG uses three independent seeds 0, 4, and 5.

### LOBSTER

DeFoG dominates most structural and embedding-distance metrics. Motif=True full wins diameter MMD, F1-PR, and recall. On held-out pruned-rule TV, full is `0.004021` versus DeFoG `0.006782`; on the separate historical training-reference protocol, DeFoG is better (`0.011229` versus `0.018693`).

### TRIANGULAR_GRID

This remains the broadest Motif=True advantage. Under the corrected common reference, full beats DeFoG on all seven listed structural MMD metrics, F1-PR (`0.927428` versus `0.794559`), precision, and MMD-RBF. Recall is tied at `1.0`; DeFoG remains better on mean-edge error. The separate pruned-rule-TV result is unchanged.

### MUTAG

In the matched topology table, Motif=True wins only recall (`0.998291` versus `0.958120`). Corrected attributed DeFoG seeds were later evaluated, but there are only two, so the committed dataset report retains the protocol and independence warnings rather than presenting them as a complete three-seed replacement.

### PTC

Motif=True full wins most structural and primary topology-derived RandomGIN metrics. DeFoG is better on sparsity, constant-input MMD-RBF, hard pruned-state TV, complete-state TV, and the retained-rule count-error diagnostics. Thus PTC is a strong topology/structure result for Motif=True, but not a universal motif-distribution win.

### QM9

Motif=True beats DeFoG only on topology recall, node-feature recall, and mean-edge absolute error. DeFoG is substantially better on most structural, embedding, and rule-state-TV metrics.

### AIDS

With independent corrected DeFoG seeds 3, 4, and 5, Motif=True beats DeFoG on degree MMD (`0.001448` versus `0.003440`) and RandomGIN MMD-RBF in every mode: topology (`0.003375` versus `0.029275`), node (`0.007786` versus `0.016448`), edge (`0.001140` versus `0.029580`), and node+edge (`0.004214` versus `0.017807`). DeFoG wins every corresponding F1-PR result and most structural metrics.

### PROTEINS

The corrected independent-seed evaluation removes the duplicate DeFoG uncertainty. Motif=True beats DeFoG on topology-control recall (`0.968262` versus `0.948485`), orbit MMD, triangle MMD, and slightly diameter MMD. DeFoG wins topology and decoded-node F1-PR, precision, and MMD-RBF, as well as degree, clustering, spectral, sparsity, and mean-edge error.

### OGB

The topology-only common terminal-chain evaluation completed with three generator seeds, 289 aligned graphs per collection, one serialized reference, and 10 RandomGIN evaluator seeds. DeFoG F1-PR is `0.968970`, versus `0.577909` for Motif=True setting 03 and `0.661598` for Motif=False setting 01; DeFoG also wins every reported structural metric in that protocol.

The historical GraphVAE final-output experiment remains separately valid under its own protocol: Motif=True setting 03 achieved F1-PR `0.7757` (`n=3`) versus `0.7259` for Motif=False setting 01 (`n=2`). It also improved degree, clustering, and orbit MMD. Those historical GraphVAE values must not be compared directly with DeFoG's `0.968970`, because they evaluate different generated artifacts. A final-output three-way reevaluation is still required before claiming whether Motif=True beats DeFoG on OGB; no retraining is required if the saved final outputs are used.

## Counting policy

The table inventories metric names, not independent scientific hypotheses. Closely related metrics such as F1, precision, recall, and multiple MMD summaries are reported separately because that is how the dataset reports present them. No global “win percentage” is calculated, since datasets have different available metrics and unequal seed quality.
