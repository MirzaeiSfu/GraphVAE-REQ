# LOBSTER complete verified experiment report

## Archive status (2026-09-21)

LGD evaluation has been launched on system 09 in tmux `lgd_eval_lobster_s2`.
It uses the completed seed-2 diffusion checkpoint at epoch 1999 (2,000 epochs).
Outputs will be copied automatically to [LGD results](evaluations/lgd_seed_2/LGD_RESULTS.md), with raw samples, `metrics.json`, and `pipeline.log` alongside it.
This is one generator training seed, evaluated using ten Random-GIN initializations.
The evaluation reports structural metrics and four-state pruned-rule TV against test and training references. The training-reference calculation uses the same generated collection as the test evaluation and is labeled accordingly.

- Archive root: `/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921/LOBSTER` on `cs-cl-19`.
- The transfer runs in tmux session `archive_lobster_20260921`; its live log is `manifests/transfer.log`.
- GraphVAE motif=False, motif=True total-count, and motif=True full-matrix each archive seeds 0, 1, and 2, including their saved models and run artifacts.
- DeFoG archives healthy seeds 0, 1, and 2. No LOBSTER DeFoG seed is excluded.
- LGD encoder and diffusion training for seed 2 are complete and their checkpoints are being archived under `experiments/lgd/`. No LGD samples or downstream structural/Random-GIN/motif-correlation results were found, so LGD is not included in the numerical comparison below.
- A completed transfer creates `manifests/transfer_finished.txt`, `manifests/file_manifest.tsv`, and `manifests/archive_size.txt`.

### Archive layout

| Content | Relative directory |
| --- | --- |
| GraphVAE motif=False | `experiments/graphvae/motif_false/seed_{0,1,2}` |
| GraphVAE motif=True total count | `experiments/graphvae/motif_true_total_count/seed_{0,1,2}` |
| GraphVAE motif=True full matrix | `experiments/graphvae/motif_true_full_matrix/seed_{0,1,2}` |
| DeFoG | `experiments/defog/seed_{0,1,2}` |
| LGD trained seed 2 | `experiments/lgd/{encoder_seed_2,diffusion_seed_2}` |
| Evaluation evidence | `metrics/` |
| Reports | `reports/` |

This 2026-09-17 copy is the requested all-three-seed report. LOBSTER has no
native node or edge features, so no native-feature Random-GIN result is
included. The reported Random-GIN channels are derived from adjacency only.

Verified 2026-09-14. The documented multi-metric outlier audit found **no
DeFoG outlier for LOBSTER**: seeds 0, 1, and 2 fall within the observed range
across F1-PR, structural MMD, and edge-count error. All three are therefore
retained. Excluding a seed merely because its motif TV is weaker would be
post-hoc cherry-picking.

The primary motif-correlation support is the exact LinkCorrelation training
rule `edges(nodes0,nodes1) AND edges(nodes1,nodes2)`. All four binary states
`FF`, `TT`, `FT`, and `TF` are retained, so the four-state TV below is both the
complete state table and the identical pruned training support. Counts are
summed across each graph collection before one within-rule normalization.

Generated 2026-09-13. Lower is better for MMD, error, and total-variation
metrics; higher is better for Random-GIN precision, recall, and F1-PR. Every
aggregate is the arithmetic mean and sample SD across generator-training seeds
0, 1, and 2.

## Dataset and protocol

- Dataset: topology-only synthetic LOBSTER; no native node or edge attributes.
- Split: paper 70/10/20 with split seed 123.
- Motif=True database with link correlation: `lobster_undir_feat_snap_85093d_multi_linkcorr`.
- Joint rule: `edges(nodes0,nodes1) AND edges(nodes1,nodes2)`.
- GraphVAE-REQ: 20,000 epochs, batch 64, learning rate 0.0003, latent dimension 128.
- Motif weight: 0.1; kernel/BCE weight: 1; node- and edge-feature weights: 0.
- DeFoG uses the frozen topology-only LOBSTER campaign.

## Phase 1 — Structural metrics

Primary setting: LinkCorrelation ON.

| Metric | Motif=False | True total | True full | DeFoG | Best |
|---|---:|---:|---:|---:|---|
| Degree MMD | 0.058797 ± 0.027880 | 0.013939 ± 0.007342 | 0.011193 ± 0.005709 | **0.000605 ± 0.000165** | DeFoG |
| Clustering MMD | 0.418071 ± 0.134877 | 0.041915 ± 0.034351 | 0.044990 ± 0.028643 | **0.002989 ± 0.002803** | DeFoG |
| Orbit MMD | 0.184583 ± 0.066306 | 0.023375 ± 0.014481 | 0.025109 ± 0.007520 | **0.017164 ± 0.008945** | DeFoG |
| Spectral MMD | 0.027167 ± 0.008180 | 0.020993 ± 0.006800 | 0.019118 ± 0.004837 | **0.007768 ± 0.001874** | DeFoG |
| Diameter MMD | 0.096861 ± 0.009322 | 0.078950 ± 0.019176 | **0.047856 ± 0.034634** | 0.205590 ± 0.033288 | True full |
| Triangle MMD | 1.7e-5 ± 2.27e-6 | 5.583e-7 ± 4.477e-7 | 6.831e-7 ± 4.528e-7 | **1.653e-8 ± 2.052e-8** | DeFoG |
| Sparsity MMD | 6.332e-9 ± 8.588e-9 | 7.922e-8 ± 4.086e-8 | 1.312e-7 ± 5.551e-8 | **5.030e-9 ± 6.676e-9** | DeFoG |
| Mean-edge absolute error | 21.950 ± 5.975 | 12.833 ± 5.605 | 8.683 ± 3.184 | **5.050 ± 1.228** | DeFoG |

The fixed reference contains 45.85 mean edges. Generated means are 67.80
(false), 33.02 (true total), 37.17 (true full), and 47.73 (DeFoG).

## Phase 2 — Random-GIN without native node features

These are degree, clustering, and square-clustering channels computed from
adjacency. They are not native LOBSTER attributes.

| Metric | Motif=False | True total | True full | DeFoG | Best |
|---|---:|---:|---:|---:|---|
| F1-PR | 0.752521 ± 0.071307 | **0.984352 ± 0.005134** | 0.975381 ± 0.010674 | 0.970886 ± 0.024528 | True total |
| Precision | 0.620000 ± 0.087178 | 0.970000 ± 0.010000 | 0.953333 ± 0.020207 | **0.981667 ± 0.015275** | DeFoG |
| Recall | 0.993333 ± 0.011547 | **1.000000 ± 0** | **1.000000 ± 0** | 0.963333 ± 0.046188 | True total/full |
| MMD-RBF | 0.173007 ± 0.053576 | 0.131193 ± 0.047214 | 0.116534 ± 0.017920 | **0.100850 ± 0.000773** | DeFoG |
| Linear MMD median | 88.3278 ± 47.9113 | 10.3573 ± 10.0146 | 4.8681 ± 3.5311 | **3.3173 ± 2.8287** | DeFoG |
| Linear MMD trimmed | 102.2776 ± 46.8731 | 10.3817 ± 10.0914 | 4.8643 ± 3.3103 | **3.3052 ± 2.7339** | DeFoG |

### Constant-input topology-only ablation

| Metric | Motif=False | True total | True full | DeFoG |
|---|---:|---:|---:|---:|
| F1-PR | 0.837029 ± 0.043195 | **0.997401 ± 0.002632** | 0.991324 ± 0.008988 | N/A |
| Precision | 0.721667 ± 0.062517 | **0.995000 ± 0.005000** | 0.983333 ± 0.017559 | N/A |
| Recall | 1.000000 ± 0 | 1.000000 ± 0 | 1.000000 ± 0 | N/A |
| MMD-RBF | 0.161025 ± 0.054453 | 0.134000 ± 0.057649 | **0.126955 ± 0.020895** | N/A |

DeFoG lacks an archived result from this older local constant-input path. A
future frozen common-evaluator rerun is required rather than mixing protocols.

## Phase 3 — Motif-correlation metrics on the identical pruned rule set

Counts for the four joint states of
`edges(nodes0,nodes1) AND edges(nodes1,nodes2)` are summed over each complete
graph collection, normalized once within the rule, and compared as
`0.5 * sum_s |p(s)-q(s)|`.

### Historical training-reference protocol

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Improvement vs false |
|---|---:|---:|---:|---:|---:|
| Motif=False | — | — | — | 0.032247 ± 0.002789 | — |
| True total | — | — | — | 0.022316 ± 0.004500 | 30.80% |
| True full | — | — | — | 0.018693 ± 0.001786 | 42.03% |
| **DeFoG** | **0.004381** | **0.007815** | **0.021490** | **0.011229 ± 0.009051** | **65.18%** |

The DeFoG values were newly computed on 2026-09-13 with the exact historical
GraphVAE training collection and the same retained four-state rule. DeFoG has
the lowest aggregate TV, although seed 2 is much weaker than seeds 0 and 1.

### Held-out 20-graph protocol

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| True total | 0.005731 | 0.016652 | 0.001948 | 0.008110 ± 0.007636 |
| True full | 0.001635 | 0.003277 | 0.007151 | **0.004021 ± 0.002832** |
| DeFoG | 0.000341 | 0.003165 | 0.016840 | 0.006782 ± 0.008825 |

LinkCorrelation-OFF contains only unary rules, so joint full-state motif TV is
not defined for that setting.

## LinkCorrelation OFF structural summary

| Metric | True total OFF | True full OFF | True total ON | True full ON |
|---|---:|---:|---:|---:|
| Degree MMD | 0.013805 ± 0.004634 | 0.013903 ± 0.007266 | 0.013939 ± 0.007342 | **0.011193 ± 0.005709** |
| Clustering MMD | 0.046822 ± 0.025679 | 0.046722 ± 0.023552 | **0.041915 ± 0.034351** | 0.044990 ± 0.028643 |
| Orbit MMD | 0.026832 ± 0.012564 | 0.030378 ± 0.014301 | **0.023375 ± 0.014481** | 0.025109 ± 0.007520 |
| Spectral MMD | 0.021769 ± 0.005797 | **0.019064 ± 0.004112** | 0.020993 ± 0.006800 | 0.019118 ± 0.004837 |
| Diameter MMD | 0.077767 ± 0.019188 | **0.046835 ± 0.034890** | 0.078950 ± 0.019176 | 0.047856 ± 0.034634 |
| Edge-count error | 13.117 ± 5.544 | 8.850 ± 3.144 | 12.833 ± 5.605 | **8.683 ± 3.184** |

## Archive layout

Archive host: `cs-cl-09.cmpt.sfu.ca`

Archive root: `/local-scratch2/mirzaei/LOBSTER`

- `graphvae_motif_false`: three motif=False runs and selected models.
- `graphvae_motif_true_linkcorr`: total-count and full-matrix runs, models,
  configurations, generated graphs, and evaluation outputs.
- `defog/frozen_campaign`: DeFoG jobs, checkpoints, generated graphs, and frozen
  split artifacts.
- `evaluations/historical_training_reference_tv`: new per-seed DeFoG TV JSON.
- `reproducibility`: the exact evaluation scripts and copied input manifests.
- `reports`: consolidated and prior reports retained for provenance.

## Original result addresses

- Motif=False: system 16,
  `/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster/seed_<s>`.
- Motif=True: system 16,
  `/local-scratch/localhome/mirzaei/fb/GraphVAE-REQ/runs/linkcorr_motif_true_3seed/lobster/{total_count,full_matrix}/seed_<s>`.
- DeFoG: system 16,
  `/local-scratch/mirzaei/defog_frozen_benchmark_20260903/GraphVAE-REQ-full/runs/defog/frozen_eval/jobs/lobster/seed_<s>`.
- New historical-TV outputs: system 18,
  `/local-scratch2/mirzaei/lobster_defog_historical_tv_20260913/seed_<s>.json`.
