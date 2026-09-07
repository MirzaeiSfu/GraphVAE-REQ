# AIDS Kiarash-weight comparison

## Question and experiment

This experiment asked whether the reconstruction scaling used by Kiarash helps
plain GraphVAE under our practical AIDS budget. Graph-statistics and motif
losses were disabled. We compared the uniform vector `[1, 1, 1, 1]` with three
vectors in `[adjacency BCE, KL, edge feature, node feature]` order:

- `[50, 2000, 1, 1]`;
- `[50, 2000, 0.1, 1]`;
- `[50, 2000, 1, 0.1]`.

Every new candidate used 250 epochs, training seed 0, generation seeds 123,
124, and 125, and the same ten fixed Random-GIN evaluators. The objective was
exactly `evaluation.modes.decoded_node_edge.summary.f1_pr.mean` on all 184
validation graphs. No test or held-out graph was accessed.

The byte-identical uniform training-seed-0 checkpoint from the frozen AIDS
evaluator bakeoff was reused. This avoided unnecessary retraining and made the
comparison use the same dataset, split, feature schema, generation seeds, and
evaluator ensemble.

## Results

| BCE, KL, edge, node | Seed 123 | Seed 124 | Seed 125 | Three-seed mean | Difference from uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1, 1, 1, 1]` | 0.713957 | 0.675759 | 0.678339 | **0.689352** | — |
| `[50, 2000, 1, 1]` | 0.000011 | 0.000013 | 0.000011 | 0.000012 | -0.689340 |
| `[50, 2000, 0.1, 1]` | 0.000010 | 0.000011 | 0.000013 | 0.000011 | -0.689340 |
| `[50, 2000, 1, 0.1]` | 0.000012 | 0.000010 | 0.000011 | 0.000011 | -0.689341 |

The conclusion is `no_improvement`. All three Kiarash-style candidates are
near the numerical floor of the Attr-F1PR evaluator and are worse than uniform
for every generation seed. The predeclared improvement threshold was +0.02;
the best candidate instead differed from uniform by about -0.68934.

## Interpretation

These weights were successful in a different GraphVAE-MM loss system, but they
do not transfer to the statistics-free AIDS GraphVAE objective. Here,
multiplying adjacency BCE by 50 and KL by 2000 changes their scale relative to
the node and edge decoder losses by orders of magnitude. The generated graphs
retained some recall in several runs, but evaluator precision was zero in every
new cell. That pattern indicates failure to generate attributed graphs that
overlap the validation distribution closely enough for the fixed Random-GIN
ensemble; it is not ordinary generation-seed noise.

This experiment isolates the requested weights, not the complete GraphVAE-MM
method. It therefore shows that the four-number scaling alone is harmful under
this exact 250-epoch AIDS contract. It does not contradict results obtained
when those weights interact with GraphVAE-MM's graph-statistics terms,
architecture, preprocessing, or other training settings.

## Reproducibility and safety

The PostgreSQL study consumed exactly three reservations: all completed and
none were replaced. The six additional validation evaluations completed on
cs-cl-17 after training. The study was collected, frozen, and restored from its
portable SQLite snapshot with byte-identical aggregate outputs. The AIDS cache
remained read-only with SHA-256
`6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb`.
Credential scans found no protected material, and every recorded access flag
remained false for test and held-out data.

Exact machine-readable evidence is recorded in
`configs/bayesian_optimization/aids_kia_bce_kl_comparison_completion.json`.
