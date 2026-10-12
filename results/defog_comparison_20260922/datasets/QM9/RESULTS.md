# QM9: motif=True full vs motif=False vs DeFoG

Updated: 2026-09-22. This is a common-protocol comparison of frozen saved models, not a merge of historical evaluator outputs. Positive improvement means better; negative means worse. SD is across generator seeds, not evaluator repeats.

## Common evaluation protocol

- Generator seeds: motif=True full, motif=False, and corrected DeFoG each use seeds 0, 1, 2.
- Exactly 512 generated graphs per seed and the same first 512 graphs from the frozen held-out QM9 test split as reference.
- Split: paper 70/10/20, split seed 123; the reference bundle is shared by all methods.
- RandomGIN: 10 evaluator seeds (0–9) per generator seed. Each table entry first averages evaluator repeats within a generator seed, then reports mean ± sample SD across generator seeds.
- `without node features` uses topology only. `with node features` uses the aligned 9-dimensional QM9 representation: atom type (5) plus hydrogen-count category (4).
- Structural metrics use the same generated/reference graph bundles. MMD metrics are lower-is-better.
- Motif correlation is hard aggregate normalized full-state total variation (TV), lower-is-better, on the exact loaded `_CP_smoothed` rule-state universe used by the GraphVAE runs: no rule pruning, 10 loaded rules / 53 loaded states; 3 multi-atom rules are eligible for this correlation summary.
- Saved GraphVAE selections are epoch 250; DeFoG exports use epoch=239.ckpt (240 epochs, zero-based filename). This checkpoint-budget difference must be disclosed; equal epochs are not equal compute. Separately running LGD jobs are not included as finished benchmark results.

### RandomGIN without node features (topology only)

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (3 seeds) | True improvement vs False | True improvement vs DeFoG |
|---|:---:|---:|---:|---:|---:|---:|
| Precision | ↑ | 0.779167 ± 0.022071 | 0.749805 ± 0.010357 | 0.979427 ± 0.005176 | +3.92% | -20.45% |
| Recall | ↑ | 0.988086 ± 0.002890 | 0.985286 ± 0.010308 | 0.974740 ± 0.010213 | +0.28% | +1.37% |
| F1-PR | ↑ | 0.869974 ± 0.012855 | 0.850344 ± 0.004288 | 0.977038 ± 0.007150 | +2.31% | -10.96% |
| Embedding MMD-RBF | ↓ | 0.054759 ± 0.000448 | 0.062479 ± 0.006605 | 0.006296 ± 0.001724 | +12.36% | -769.73% |
| Embedding MMD-linear | ↓ | 17.577647 ± 3.905999 | 32.894597 ± 3.409953 | 0.663968 ± 0.313167 | +46.56% | -2547.36% |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Precision | Motif=True (full) | s0=0.758984, s1=0.802734, s2=0.775781 |
| Precision | Motif=False | s0=0.750195, s1=0.739258, s2=0.759961 |
| Precision | DeFoG | s0=0.979492, s1=0.974219, s2=0.984570 |
| Recall | Motif=True (full) | s0=0.989453, s1=0.984766, s2=0.990039 |
| Recall | Motif=False | s0=0.992188, s1=0.990234, s2=0.973437 |
| Recall | DeFoG | s0=0.983398, s1=0.963477, s2=0.977344 |
| F1-PR | Motif=True (full) | s0=0.857801, s1=0.883416, s2=0.868705 |
| F1-PR | Motif=False | s0=0.853125, s1=0.845407, s2=0.852501 |
| F1-PR | DeFoG | s0=0.981402, s1=0.968787, s2=0.980927 |
| Embedding MMD-RBF | Motif=True (full) | s0=0.055191, s1=0.054788, s2=0.054297 |
| Embedding MMD-RBF | Motif=False | s0=0.055415, s1=0.063520, s2=0.068502 |
| Embedding MMD-RBF | DeFoG | s0=0.008198, s1=0.004836, s2=0.005854 |
| Embedding MMD-linear | Motif=True (full) | s0=18.869213, s1=13.189442, s2=20.674286 |
| Embedding MMD-linear | Motif=False | s0=29.119391, s1=35.751038, s2=33.813361 |
| Embedding MMD-linear | DeFoG | s0=0.778260, s1=0.309708, s2=0.903935 |

### RandomGIN with node features

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (3 seeds) | True improvement vs False | True improvement vs DeFoG |
|---|:---:|---:|---:|---:|---:|---:|
| Precision | ↑ | 0.450065 ± 0.017254 | 0.467578 ± 0.033765 | 0.969531 ± 0.000517 | -3.75% | -53.58% |
| Recall | ↑ | 0.975521 ± 0.003247 | 0.983008 ± 0.003405 | 0.952669 ± 0.010954 | -0.76% | +2.40% |
| F1-PR | ↑ | 0.614553 ± 0.016334 | 0.632864 ± 0.030919 | 0.960982 ± 0.005762 | -2.89% | -36.05% |
| Embedding MMD-RBF | ↓ | 0.059790 ± 0.004650 | 0.049211 ± 0.003964 | 0.005307 ± 0.000980 | -21.50% | -1026.63% |
| Embedding MMD-linear | ↓ | 31.285457 ± 4.177455 | 24.667934 ± 5.844856 | 0.620310 ± 0.168671 | -26.83% | -4943.52% |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Precision | Motif=True (full) | s0=0.455078, s1=0.464258, s2=0.430859 |
| Precision | Motif=False | s0=0.461133, s1=0.437500, s2=0.504102 |
| Precision | DeFoG | s0=0.969336, s1=0.969141, s2=0.970117 |
| Recall | Motif=True (full) | s0=0.978516, s1=0.972070, s2=0.975977 |
| Recall | Motif=False | s0=0.986914, s1=0.980664, s2=0.981445 |
| Recall | DeFoG | s0=0.958398, s1=0.940039, s2=0.959570 |
| F1-PR | Motif=True (full) | s0=0.620298, s1=0.627238, s2=0.596122 |
| F1-PR | Motif=False | s0=0.628252, s1=0.604510, s2=0.665830 |
| F1-PR | DeFoG | s0=0.963812, s1=0.954352, s2=0.964782 |
| Embedding MMD-RBF | Motif=True (full) | s0=0.057201, s1=0.057010, s2=0.065158 |
| Embedding MMD-RBF | Motif=False | s0=0.046997, s1=0.053787, s2=0.046848 |
| Embedding MMD-RBF | DeFoG | s0=0.005955, s1=0.005786, s2=0.004180 |
| Embedding MMD-linear | Motif=True (full) | s0=30.751246, s1=27.400804, s2=35.704321 |
| Embedding MMD-linear | Motif=False | s0=26.378345, s1=29.466773, s2=18.158685 |
| Embedding MMD-linear | DeFoG | s0=0.810318, s1=0.562355, s2=0.488257 |

### Structural metrics

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (3 seeds) | True improvement vs False | True improvement vs DeFoG |
|---|:---:|---:|---:|---:|---:|---:|
| Degree MMD | ↓ | 0.018796 ± 0.000885 | 0.019460 ± 0.000349 | 0.000225 ± 6.153e-05 | +3.41% | -8254.82% |
| Clustering MMD | ↓ | 0.100142 ± 0.005400 | 0.150461 ± 0.009369 | 0.002981 ± 0.001241 | +33.44% | -3258.88% |
| Orbit MMD | ↓ | 0.000875 ± 0.000195 | 0.003066 ± 2.299e-05 | 0.000102 ± 8.099e-05 | +71.45% | -754.23% |
| Spectral MMD | ↓ | 0.003839 ± 0.000248 | 0.003261 ± 0.000453 | 0.001061 ± 0.000197 | -17.73% | -262.01% |
| Diameter MMD | ↓ | 0.004010 ± 0.000575 | 0.002164 ± 0.001742 | 0.000775 ± 0.000758 | -85.33% | -417.28% |
| Triangle MMD | ↓ | 4.703e-06 ± 8.696e-07 | 9.147e-06 ± 1.178e-06 | 1.432e-08 ± 6.346e-09 | +48.59% | -32732.80% |
| Sparsity MMD | ↓ | 1.941e-08 ± 1.888e-09 | 1.703e-08 ± 4.936e-09 | 1.048e-10 ± 1.380e-10 | -13.99% | -18427.03% |
| Absolute error in mean edge count | ↓ | 0.418620 ± 0.089383 | 0.507161 ± 0.254434 | 0.466797 ± 0.103294 | +17.46% | +10.32% |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Degree MMD | Motif=True (full) | s0=0.018698, s1=0.017965, s2=0.019727 |
| Degree MMD | Motif=False | s0=0.019074, s1=0.019751, s2=0.019554 |
| Degree MMD | DeFoG | s0=0.000191, s1=0.000188, s2=0.000296 |
| Clustering MMD | Motif=True (full) | s0=0.097850, s1=0.096266, s2=0.106310 |
| Clustering MMD | Motif=False | s0=0.157135, s1=0.139750, s2=0.154499 |
| Clustering MMD | DeFoG | s0=0.003627, s1=0.003767, s2=0.001550 |
| Orbit MMD | Motif=True (full) | s0=0.001028, s1=0.000655, s2=0.000943 |
| Orbit MMD | Motif=False | s0=0.003044, s1=0.003090, s2=0.003064 |
| Orbit MMD | DeFoG | s0=0.000196, s1=6.013e-05, s2=5.141e-05 |
| Spectral MMD | Motif=True (full) | s0=0.003651, s1=0.003748, s2=0.004120 |
| Spectral MMD | Motif=False | s0=0.003336, s1=0.002775, s2=0.003672 |
| Spectral MMD | DeFoG | s0=0.001286, s1=0.000918, s2=0.000978 |
| Diameter MMD | Motif=True (full) | s0=0.003626, s1=0.003733, s2=0.004672 |
| Diameter MMD | Motif=False | s0=0.004174, s1=0.001093, s2=0.001225 |
| Diameter MMD | DeFoG | s0=0.000399, s1=0.000280, s2=0.001648 |
| Triangle MMD | Motif=True (full) | s0=5.075e-06, s1=3.709e-06, s2=5.325e-06 |
| Triangle MMD | Motif=False | s0=7.819e-06, s1=1.006e-05, s2=9.558e-06 |
| Triangle MMD | DeFoG | s0=1.676e-08, s1=1.909e-08, s2=7.120e-09 |
| Sparsity MMD | Motif=True (full) | s0=2.138e-08, s1=1.761e-08, s2=1.926e-08 |
| Sparsity MMD | Motif=False | s0=1.142e-08, s1=1.897e-08, s2=2.070e-08 |
| Sparsity MMD | DeFoG | s0=1.547e-12, s1=5.133e-11, s2=2.615e-10 |
| Absolute error in mean edge count | Motif=True (full) | s0=0.324219, s1=0.501953, s2=0.429688 |
| Absolute error in mean edge count | Motif=False | s0=0.800781, s1=0.369141, s2=0.351562 |
| Absolute error in mean edge count | DeFoG | s0=0.531250, s1=0.347656, s2=0.521484 |

### Motif correlation

Aggregate = mean ± sample SD across generator training seeds.

| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (3 seeds) | True improvement vs False | True improvement vs DeFoG |
|---|:---:|---:|---:|---:|---:|---:|
| Mean rule-state TV | ↓ | 0.105975 ± 0.009783 | 0.109578 ± 0.009929 | 0.043598 ± 0.006764 | +3.29% | -143.07% |

Per-generator-seed values:

| Metric | Method | Seed values |
|---|---|---|
| Mean rule-state TV | Motif=True (full) | s0=0.096182, s1=0.105994, s2=0.115749 |
| Mean rule-state TV | Motif=False | s0=0.118916, s1=0.099149, s2=0.110669 |
| Mean rule-state TV | DeFoG | s0=0.042923, s1=0.050675, s2=0.037197 |

## Result locations

- Evaluation root: `/local-scratch2/mirzaei/qm9_common_eval_20260917`
- Motif=True full per-seed results: `/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations/graphvae/true_full/seed_<0|1|2>`
- Motif=False per-seed results: `/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations/graphvae/false/seed_<0|1|2>`
- DeFoG per-seed results: `/local-scratch2/mirzaei/qm9_common_eval_20260917/evaluations/defog/seed_<0|1|2>`
- DeFoG checkpoints on system17: `/localhome/mirzaei/qm9_defog_seedfixed250_batch128_20260916/runs/seed_S/checkpoints/qm9_seedfixed250_seed_S/epoch=239.ckpt`.
- GraphVAE checkpoints/configs: `/local-scratch2/mirzaei/qm9_common_eval_20260917/graphvae`

## Interpretation caution

The motif TV above is a state-distribution comparison on the shared rule universe. It is not the Gaussian training loss itself, and only three loaded rules are multi-atom rules eligible for this particular correlation calculation. Accordingly, it should be reported alongside—rather than substituted for—the structural and RandomGIN results.
