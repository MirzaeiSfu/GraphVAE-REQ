# PROTEINS fresh-evaluation randomness comparison

The same three saved best models were evaluated twice. Repeat 1 used generation seed `20260716`; repeat 2 used `20260717`. Each repeat regenerated validation/test graphs and reinitialized both local and third-party Random-GIN evaluators 10 times.

## Test local GIN metrics

| Seed | Test F1-PR R1 | Test F1-PR R2 | Test MMD RBF R1 | Test MMD RBF R2 | Precision R1 → R2 | Recall R1 → R2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.935756 | 0.938140 | 0.090163 | 0.001074 | 0.967619 → 0.921429 | 0.907143 → 0.955714 |
| 1 | 0.941810 | 0.967768 | 0.073418 | 0.071857 | 0.953810 → 0.953810 | 0.931429 → 0.982381 |
| 2 | 0.961373 | 0.956159 | 0.000715 | 0.000541 | 0.946890 → 0.932381 | 0.977619 → 0.981429 |

## Test structural MMD

| Seed | Degree R1 → R2 | Clustering R1 → R2 | Orbit R1 → R2 | Spectral R1 → R2 | Diameter R1 → R2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.042429 → 0.041877 | 0.029670 → 0.029447 | 0.016391 → 0.024897 | 0.021645 → 0.021591 | 0.035770 → 0.046181 |
| 1 | 0.044003 → 0.042909 | 0.030039 → 0.026246 | 0.011760 → 0.011407 | 0.024184 → 0.021372 | 0.043626 → 0.045632 |
| 2 | 0.043565 → 0.047418 | 0.029113 → 0.029405 | 0.022161 → 0.007695 | 0.021042 → 0.023645 | 0.032887 → 0.028282 |

## Third-party Random-GIN

| Seed | F1-PR R1 → R2 | MMD RBF R1 → R2 | Linear MMD trimmed R1 → R2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.866901 → 0.909155 | 0.084491 → 0.001497 | 18.612 → 5,042,080.938 |
| 1 | 0.911043 → 0.918766 | 0.067197 → 0.063000 | 29.101 → 23.088 |
| 2 | 0.911554 → 0.890122 | 0.001315 → 0.002909 | 3,445,981.484 → 1,097,755.211 |

## Generated test density

The real held-out mean is 54.61 edges, and the dense threshold is 179.68 edges.

| Seed | Mean edges R1 → R2 | Maximum edges R1 → R2 | Dense graphs R1 → R2 |
| ---: | ---: | ---: | ---: |
| 0 | 44.57 → 157.59 | 109 → 4,821 | 0/210 → 5/210 |
| 1 | 49.12 → 49.82 | 212 → 150 | 2/210 → 0/210 |
| 2 | 138.49 → 105.07 | 4,814 → 4,556 | 5/209 → 4/210 |

## Interpretation

- Seed 0 is extremely generation-dependent. Repeat 1 contains no dense graph and looks ordinary under linear MMD; repeat 2 contains five near-complete outliers, increasing trimmed linear MMD from `18.6` to `5.04 million`. Its test RBF MMD simultaneously improves from `0.0902` to `0.0011`, demonstrating that RBF MMD can look better while the dense tail becomes much worse.
- Seed 1 is the most repeatable. Mean edge count stays near the real distribution, RBF MMD changes only modestly, and third-party trimmed linear MMD remains in the same small range (`29.1` versus `23.1`).
- Seed 2 is consistently heavy-tailed. Both repeats contain near-complete graphs and million-scale linear MMD, although the exact value changes by more than two million.
- Structural MMD values are much less volatile than density-sensitive linear MMD. Largest-component structural statistics can therefore conceal rare, catastrophic dense graphs.

The two-repeat evidence strengthens the earlier recommendation: seed 1 is the safest of these models, while seeds 0 and 2 require rejection/edge-budget post-processing. Model assessment should use repeated generated sets rather than one final sample.
