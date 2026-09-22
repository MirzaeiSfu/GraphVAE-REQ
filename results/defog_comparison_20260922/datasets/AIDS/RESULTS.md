# AIDS topology-only report: motif=True full at epoch 10,000 vs motif=False vs DeFoG

Generated automatically after evaluation. This is a common-protocol comparison, not a merge of historical evaluator outputs.

## Common evaluation protocol

- Generator seeds: motif=True full = 0, 1, 2; motif=False = 0, 1, 2; DeFoG = corrected independent seeds 1 and 2.
- Exactly 400 generated graphs per seed and all 400 graphs from the frozen held-out AIDS test split as reference.
- Split: paper 70/10/20, split seed 123; the reference bundle is shared by all methods.
- RandomGIN: 10 evaluator seeds (0–9) per generator seed. Each table entry first averages evaluator repeats within a generator seed, then reports mean ± sample SD across generator seeds.
- Structural metrics use the same generated/reference graph bundles. MMD metrics are lower-is-better.
- DeFoG has only two corrected seeds, so its uncertainty is less stable and remains provisional relative to the three-seed GraphVAE results. Their currently exported metrics are identical, producing zero observed SD; this should be disclosed rather than interpreted as perfect stability.

### RandomGIN without node features (topology only)

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (2 seeds: 1, 2) |
|---|:---:|---:|---:|---:|
| Precision | ↑ | 0.880417 ± 0.017590 | 0.674333 ± 0.021825 | 0.888500 ± 0 |
| Recall | ↑ | 0.908917 ± 0.015121 | 0.939583 ± 0.004785 | 0.974250 ± 0 |
| F1-PR | ↑ | 0.893297 ± 0.016140 | 0.782748 ± 0.015110 | 0.929203 ± 0 |
| Embedding MMD-RBF | ↓ | 0.003375 ± 0.000180 | 0.001768 ± 0.000136 | 0.037542 ± 0 |
| Embedding MMD-linear | ↓ | 10806472658.979687 ± 5309290385.637910 | 5839111992.211458 ± 3362106065.365885 | 2.281236 ± 0 |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Precision | Motif=True (full) | s0=0.867750, s1=0.900500, s2=0.873000 |
| Precision | Motif=False | s0=0.698000, s1=0.655000, s2=0.670000 |
| Precision | DeFoG | s1=0.888500, s2=0.888500 |
| Recall | Motif=True (full) | s0=0.897250, s1=0.926000, s2=0.903500 |
| Recall | Motif=False | s0=0.941000, s1=0.943500, s2=0.934250 |
| Recall | DeFoG | s1=0.974250, s2=0.974250 |
| F1-PR | Motif=True (full) | s0=0.882170, s1=0.911809, s2=0.885912 |
| F1-PR | Motif=False | s0=0.799661, s1=0.770582, s2=0.777999 |
| F1-PR | DeFoG | s1=0.929203, s2=0.929203 |
| Embedding MMD-RBF | Motif=True (full) | s0=0.003184, s1=0.003398, s2=0.003543 |
| Embedding MMD-RBF | Motif=False | s0=0.001921, s1=0.001721, s2=0.001661 |
| Embedding MMD-RBF | DeFoG | s1=0.037542, s2=0.037542 |
| Embedding MMD-linear | Motif=True (full) | s0=13499164345.196875, s1=4690365499.907812, s2=14229888131.834375 |
| Embedding MMD-linear | Motif=False | s0=2351665288.980469, s1=6105661077.241406, s2=9060009610.412500 |
| Embedding MMD-linear | DeFoG | s1=2.281236, s2=2.281236 |

### Structural metrics

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (2 seeds: 1, 2) |
|---|:---:|---:|---:|---:|
| Degree MMD | ↓ | 0.001448 ± 0.000974 | 0.012433 ± 0.002313 | 0.005885 ± 0 |
| Clustering MMD | ↓ | 0.064662 ± 0.013072 | 0.376930 ± 0.038837 | 0.015050 ± 0 |
| Orbit MMD | ↓ | 0.002175 ± 0.000740 | 0.011010 ± 0.003254 | 0.001462 ± 0 |
| Spectral MMD | ↓ | 0.007434 ± 0.001110 | 0.010378 ± 0.000839 | 0.006290 ± 0 |
| Diameter MMD | ↓ | 0.040486 ± 0.006629 | 0.037079 ± 0.002603 | 0.011564 ± 0 |
| Triangle MMD | ↓ | 0.001108 ± 0.000556 | 0.000689 ± 0.000374 | 2.890e-07 ± 0 |
| Sparsity MMD | ↓ | 3.070e-07 ± 6.586e-08 | 2.267e-07 ± 4.507e-08 | 2.932e-09 ± 0 |
| Absolute error in mean edge count | ↓ | 109.708333 ± 32.317548 | 84.944167 ± 29.271318 | 1.680000 ± 0 |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Degree MMD | Motif=True (full) | s0=0.001585, s1=0.000413, s2=0.002347 |
| Degree MMD | Motif=False | s0=0.009822, s1=0.013255, s2=0.014222 |
| Degree MMD | DeFoG | s1=0.005885, s2=0.005885 |
| Clustering MMD | Motif=True (full) | s0=0.079274, s1=0.054078, s2=0.060634 |
| Clustering MMD | Motif=False | s0=0.333712, s1=0.388171, s2=0.408907 |
| Clustering MMD | DeFoG | s1=0.015050, s2=0.015050 |
| Orbit MMD | Motif=True (full) | s0=0.001955, s1=0.001571, s2=0.003000 |
| Orbit MMD | Motif=False | s0=0.007281, s1=0.013278, s2=0.012470 |
| Orbit MMD | DeFoG | s1=0.001462, s2=0.001462 |
| Spectral MMD | Motif=True (full) | s0=0.007232, s1=0.006439, s2=0.008630 |
| Spectral MMD | Motif=False | s0=0.009709, s1=0.010104, s2=0.011319 |
| Spectral MMD | DeFoG | s1=0.006290, s2=0.006290 |
| Diameter MMD | Motif=True (full) | s0=0.045092, s1=0.032888, s2=0.043477 |
| Diameter MMD | Motif=False | s0=0.036319, s1=0.034939, s2=0.039977 |
| Diameter MMD | DeFoG | s1=0.011564, s2=0.011564 |
| Triangle MMD | Motif=True (full) | s0=0.001306, s1=0.000481, s2=0.001539 |
| Triangle MMD | Motif=False | s0=0.000257, s1=0.000888, s2=0.000921 |
| Triangle MMD | DeFoG | s1=2.890e-07, s2=2.890e-07 |
| Sparsity MMD | Motif=True (full) | s0=3.445e-07, s1=2.309e-07, s2=3.455e-07 |
| Sparsity MMD | Motif=False | s0=1.778e-07, s1=2.357e-07, s2=2.666e-07 |
| Sparsity MMD | DeFoG | s1=2.932e-09, s2=2.932e-09 |
| Absolute error in mean edge count | Motif=True (full) | s0=125.212500, s1=72.560000, s2=131.352500 |
| Absolute error in mean edge count | Motif=False | s0=51.670000, s1=96.440000, s2=106.722500 |
| Absolute error in mean edge count | DeFoG | s1=1.680000, s2=1.680000 |

## Result locations

- Evaluation root: `/local-scratch2/mirzaei/aids_common_eval_10k_20260917`
- Motif=True full per-seed results: `/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae/true_full/seed_<0|1|2>`
- Motif=False per-seed results: `/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/graphvae/false/seed_<0|1|2>`
- DeFoG per-seed results: `/local-scratch2/mirzaei/aids_common_eval_10k_20260917/evaluations/defog/seed_<1|2>`
- GraphVAE checkpoints/configs: `/local-scratch2/mirzaei/aids_common_eval_10k_20260917/graphvae`

## Interpretation caution

This reduced report intentionally includes only topology RandomGIN and structural metrics. Native node- and edge-feature RandomGIN modes are excluded by request. The longer-running motif-TV evaluation is also not treated as final here.
