# LOBSTER GraphVAE reconstruction-weight optimization summary

## Short answer

Yes, the non-QM9 work is complete. The distributed Bayesian-optimization
system was implemented and exercised from failure handling through a real
multi-GPU LOBSTER search. The search produced a meaningful nonuniform weight
pair, but the matched multi-seed confirmation did **not** show that it improves
GraphVAE over uniform reconstruction weights.

The scientifically supported LOBSTER default is therefore still:

```yaml
alpha_node_feat: 1.0
alpha_edge_feat: 1.0
```

The result is not that Bayesian optimization can never help. It is that this
particular search protocol and selected pair did not provide reproducible
evidence of improvement.

## What is complete

The following work was completed, audited, frozen, restored from portable
snapshots, tested, committed, and pushed:

- Gate 5 deployment qualification on the dedicated `cs-cl-13` and `cs-cl-17`
  roots, including protected PostgreSQL authentication, exact runtime and
  source fingerprints, immutable cache checks, and GPU isolation.
- R04: two simultaneous mock workers on different hosts with exact reservation
  accounting and unique trial/artifact identities.
- R05: two simultaneous tiny real LOBSTER trials, one on each host.
- R06: definite-prelaunch and ambiguous-postlaunch failure handling without
  blind duplicate dispatch.
- R07: safe orphan-process recovery and stale-trial reconciliation to one
  consumed `FAIL`, with no replacement.
- Gate 6 R08 hardware repeatability, the bounded five-reservation LOBSTER
  pilot, R09 clean-snapshot restoration, and R10 post-freeze test-only
  evaluation.
- A nine-worker concurrency and hardware-signal qualification. Concurrency
  worked, but heterogeneous GPUs did not reproduce the same numerical result.
- A homogeneous three-worker TITAN RTX qualification with exact numerical
  agreement.
- A fresh 30-trial, 2,000-epoch LOBSTER weight search on three TITAN RTX GPUs.
- A matched 10,000-epoch confirmation of the selected and uniform weights at
  training seeds 0, 1, and 2.
- One explicitly authorized production selected-versus-uniform held-out
  LOBSTER comparison after the validation result and decision rule were
  frozen.
- Final credential, storage-URL, test-access, cache, artifact, lifecycle,
  PostgreSQL-isolation, and portable-restore audits.
- The complete non-PostgreSQL distributed suite (83 tests) and isolated
  PostgreSQL suite (19 tests), both passing at the final checkpoint.

No full-QM9 BO or Gate 7 production study was run. No new PROTEINS weight
study has yet been run.

## What experiment was run

The frozen LOBSTER cache contains 100 graphs split into 70 training, 10
validation, and 20 held-out test graphs. It has 14 node-feature channels and 11
edge-feature channels. Every optimization and confirmation trial used both the
node-feature and edge-feature decoders.

The search sampled these two loss weights independently on a log scale from
`0.01` through `10`:

```text
alpha_node_feat
alpha_edge_feat
```

The first reservation was the uniform pair `(1, 1)`. The other reservations
followed the frozen Optuna TPE plan. All 30 search trials used training seed 0,
2,000 epochs, all 10 validation graphs, generation seed 123, and evaluator
seeds 0 through 9. The search used three simultaneous workers on homogeneous
NVIDIA TITAN RTX GPUs.

The selected pair was:

```yaml
alpha_node_feat: 5.229045672015893
alpha_edge_feat: 0.05386414830134693
```

Its node-to-edge ratio is about 97.1:1, compared with 1:1 for the uniform
pair.

## Objective values and where they came from

The exact optimization objective was preserved as:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

It was calculated on the **validation split**, not on the training data and
not on the test data. During optimization, `skip_final_evaluation=true` and
`test_access=false` prevented automatic test evaluation.

For each evaluator repeat, generated and reference validation graphs are
embedded by a seeded random attributed GIN. Precision is the fraction of
generated embeddings covered by the reference manifold; recall is the
fraction of reference embeddings covered by the generated manifold. Their
harmonic F1-PR value is computed, and the reported objective is the arithmetic
mean over the ten evaluator seeds. `decoded_node_edge` means that adjacency,
decoded node attributes, and decoded edge attributes all participate.

### Search result: 2,000 epochs, seed 0

| Candidate | Node weight | Edge weight | Validation Attr-F1PR |
| --- | ---: | ---: | ---: |
| BO-selected trial 7 | 5.2290456720 | 0.0538641483 | 0.8012845288 |
| Uniform trial 0 | 1.0 | 1.0 | 0.6827769426 |

The apparent search-time advantage was `+0.1185075863`. This was a valid
result for that exact seed and short training budget, but it was only the
selection result, not proof of general improvement.

Across all 30 completed search trials, the reported validation objectives
ranged from `0.5216515747` to `0.8012845288`.

### Matched confirmation: 10,000 epochs

| Training seed | BO-selected | Uniform | Selected minus uniform |
| ---: | ---: | ---: | ---: |
| 0 | 0.7514671231 | 0.7294436358 | +0.0220234873 |
| 1 | 0.5589719479 | 0.5755313142 | -0.0165593664 |
| 2 | 0.6664272412 | 0.7064799634 | -0.0400527222 |
| **Mean** | **0.6589554374** | **0.6704849711** | **-0.0115295337** |

The paired 95% t interval was
`[-0.0893880672, 0.0663289998]`. It crosses zero, and two of the three paired
differences are negative. The predeclared superiority rule therefore returned
`no_improvement`.

This does not establish that the selected weights are intrinsically harmful:
the interval is too wide for that claim. It establishes that an improvement
over uniform weights was not demonstrated.

### Held-out comparison after freezing

The one-time production-confirmation test evaluation was performed only after
the validation ranking and conclusion were frozen. It did not train a model,
create an Optuna trial, or rerank a candidate.

| Training seed | BO-selected test F1-PR | Uniform test F1-PR | Difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.6069668367 | 0.5074371585 | +0.0995296782 |
| 1 | 0.4363844157 | 0.5240760169 | -0.0876916012 |
| 2 | 0.5255151063 | 0.4872250320 | +0.0382900744 |
| **Mean** | **0.5229554529** | **0.5062460691** | **+0.0167093838** |

This small positive held-out mean is mixed across seeds and is secondary
descriptive evidence. It cannot overturn the frozen validation conclusion or
justify selecting the weights after seeing the test data.

## Is the objective deterministic?

It is deterministic under the complete frozen execution contract, but it is
not a seed-independent constant.

The code seeds Python, NumPy, PyTorch, CUDA, and DGL; disables cuDNN; requires
PyTorch deterministic algorithms; fixes the graph-generation seed; and fixes
the ten evaluator seeds. This worked on the homogeneous pool: three independent
TITAN RTX workers reproduced the uniform 2,000-epoch objective
`0.682776942562006` exactly.

There are two important qualifications:

1. Changing the training seed intentionally changes model initialization and
   optimization, so it can change the objective substantially. That is
   scientific seed sensitivity, not a failure of deterministic replay.
2. Different GPU models produced different results despite the same software
   fingerprint. The nine-worker hardware trial ranged from about `0.56045` to
   `0.68278`, so the final search correctly used only homogeneous TITAN RTX
   devices.

The evaluator itself is repeatable when its graphs and seeds are identical,
but its ten random-GIN values vary considerably. For the search-time uniform
trial, the per-repeat F1-PR standard deviation was `0.2943`; for the selected
trial it was `0.1271`. Averaging ten fixed repeats makes replay deterministic,
but it does not create ten independent training runs or ten independent
validation datasets.

## Why the selected weights were worse in confirmation

Several effects acted together. The evidence does not isolate one single
cause, but it strongly supports the following explanation.

### 1. The search optimized a short, single-seed proxy

Every search candidate was judged at 2,000 epochs and training seed 0. The
scientific comparison was made at 10,000 epochs and three training seeds. The
selected pair's seed-0 advantage shrank from `+0.1185` at 2,000 epochs to only
`+0.0220` at 10,000 epochs. The ranking therefore depended strongly on
training horizon. Seeds 1 and 2 then reversed the sign.

In other words, BO successfully optimized the exact proxy it was given, but
that proxy was not sufficiently predictive of robust, fully trained behavior.

### 2. Thirty choices were ranked on only ten validation graphs

With ten reference and ten generated validation graphs, precision and recall
move in coarse increments of 0.1 before the harmonic mean is taken. One
uniform evaluator repeat had precision zero and F1-PR near `0.00002`, while
another reached `0.94738`. This makes candidate ordering sensitive to a small
number of graphs and random-GIN representations.

Selecting the maximum of 30 noisy estimates also creates a winner's-curse
effect: the top observed value tends to include favorable seed/evaluator
variation. Reusing the same ten validation graphs for all proposals can
additionally overfit the BO process to that small split.

### 3. The two raw weights mixed two different questions

The training loss contains:

```text
kernel_cost
+ alpha_node_feat * node_feature_loss
+ alpha_edge_feat * edge_feature_loss
```

Sampling the two weights independently changes both their ratio and the total
feature-reconstruction strength relative to `kernel_cost`. BO was therefore
asked to learn a node-versus-edge trade-off and a global loss scale at the same
time.

At epoch 2,000, the uniform trial's logged weighted node and edge terms were
about `0.009308` and `0.004045`. The selected trial's were about `0.026409`
and `0.000117`. The winner made the edge contribution roughly 35 times smaller
than the uniform trial at that checkpoint while increasing the node
contribution. That can look favorable to a small validation embedding sample
without being a stable solution for a metric that explicitly requires both
decoded node and edge attributes.

### 4. The search surface was weakly identified

Many high-scoring trials suppressed the edge weight, but their node weights
were spread across very different scales. Among the top search results were
node weights near `0.01`, `0.04`, `1`, `5`, `7`, and `8`. This is not the
shape of a sharply identified optimum. It suggests that several parameter
combinations can receive a favorable single-seed score and that the TPE model
was learning noise and broad tendencies as well as signal.

### 5. Only the winning search trial received full confirmation

The protocol correctly avoided post-hoc reranking, but it also means that a
potentially more stable second- or third-ranked region was not compared at
10,000 epochs. A better design should promote a predeclared shortlist and use
multi-seed information before choosing one winner.

### 6. Parallelism was not the main scientific failure

The exact reservation budget, PostgreSQL locking, failure consumption, and
artifact identities all behaved correctly. The homogeneous three-GPU baseline
also reproduced exactly. Parallel TPE can be less sample-efficient because
some proposals are made before earlier results return, but that is a smaller
issue than the single-seed, short-horizon, ten-graph objective.

## Was the small dataset the cause?

The small LOBSTER dataset was probably a major contributor, especially its
ten-graph validation split, but this experiment cannot prove that it was the
only cause.

Dataset size explains the coarse precision/recall values, validation
overfitting risk, and weak estimate of generalization. It does not by itself
explain the training-horizon mismatch, extreme loss ratio, or sensitivity to
model initialization. Those problems would still need attention on a larger
dataset.

A larger dataset is a useful next experiment only if it has meaningful node
and edge attributes. The repository's current PROTEINS loader sets edge
features to `None`, and the official TUDataset inventory reports 1,113
PROTEINS graphs with node labels/attributes but no edge labels or edge
attributes. Therefore ordinary PROTEINS cannot satisfy the required
`decoded_node_edge` objective without changing the scientific question. The
official inventory is available at
<https://chrsmrrs.github.io/datasets/docs/datasets/>.

The repository already has a more suitable larger-dataset path: AIDS. The
official inventory reports 2,000 graphs with node labels, node attributes, and
edge labels, and the local loader already maps those edge labels into the edge
decoder schema. It still needs a new immutable cache and full qualification
before use.

## Why the BO work took a long time

The model training portion was bounded and considerably shorter than the whole
engineering exercise:

| Study | Trials | Epochs/trial | GPU time | Trial-span wall time | Median trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heterogeneous hardware signal | 9 | 2,000 | 0.664 h | 5.84 min | 4.25 min |
| TITAN RTX qualification | 3 | 2,000 | 0.207 h | 4.27 min | 4.20 min |
| Weight search | 30 | 2,000 | 2.077 h | 68.89 min | 4.20 min |
| Matched confirmation | 6 | 10,000 | 1.556 h | 32.98 min | 15.86 min |
| **Total** | **48** | — | **4.504 h** | **about 1.87 h** | — |

The 30-trial search required eleven bounded waves because the frozen maximum
parallelism was three. Its ideal ten full waves were also separated by
collection, audit, and relaunch work. The confirmation required two more
waves.

The full project took longer because it also included protected deployment,
runtime and hardware qualification, mock and real failure drills, PostgreSQL
state reconciliation, source/cache verification, collection, portable
snapshots, restore tests, credential scans, and a commit/push checkpoint after
each completed step. The four main study roots contain about 66 GiB of model
and evaluation artifacts, so copying and hashing evidence was not negligible.

### Time estimate for another dataset

Graph count alone is not enough to predict training time. Runtime also depends
on graph size, maximum padded node count, edge density, feature dimensions,
the number of training batches per epoch, generation cost, and checkpoint I/O.
The correct estimate begins with one qualified timing trial on the new cache.

For three workers, a useful planning formula is:

```text
search wall time       ~= ceil(number_of_fits / 3) * measured_search_fit_time
confirmation wall time ~= ceil(number_of_fits / 3) * measured_full_fit_time
total                  ~= search + confirmation + 25% operational margin
```

For reference, if a new dataset needs 10 minutes for a search fit and 40
minutes for a full fit, 30 search fits plus six confirmation fits need roughly
3 hours before qualification overhead. At 30 and 120 minutes, the same design
needs roughly 9 hours. AIDS or a modified PROTEINS pipeline must be calibrated;
the LOBSTER 4.2-minute result should not be multiplied by graph count and
treated as a promise.

## How to use the code for weight optimization

### 1. Prepare a BO-safe configuration

Start from a dataset-specific GraphVAE YAML. It must retain both feature
decoders and use a real paper-style validation split. Keep these invariants:

```yaml
data:
  use_feature: true
  split_mode: paper_70_10_20
  deterministic: true
  deterministic_warn_only: false

runtime:
  require_existing_dataset_cache: true
  skip_final_evaluation: true
  ideal_Evalaution: false
  tiny_overfit: false
  sanity_check: false
```

The cache must contain nonempty node and edge schemas, be hashed in a manifest,
and remain read-only during the study.

### 2. Run a local mock trial

This checks Optuna state and artifacts without training:

```bash
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  scripts/tune_graphvae_attribute_weights.py \
  --base-config configs/bayesian_optimization/lobster_graphvae_attr_f1pr_signal.yaml \
  --study-name lobster_weight_example_mock \
  --output-dir runs/bayesian_optimization/lobster_weight_example_mock \
  --trials 2 \
  --device cpu \
  --mock
```

Always use a fresh study name and output root for a new scientific question.

### 3. Run a bounded local real search

For an exact reproduction of the LOBSTER search limits:

```bash
/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python \
  scripts/tune_graphvae_attribute_weights.py \
  --base-config configs/bayesian_optimization/lobster_graphvae_attr_f1pr_signal.yaml \
  --study-name lobster_weight_example_real \
  --output-dir runs/bayesian_optimization/lobster_weight_example_real \
  --trials 30 \
  --alpha-node-feat-min 0.01 \
  --alpha-node-feat-max 10 \
  --alpha-edge-feat-min 0.01 \
  --alpha-edge-feat-max 10 \
  --split-seed 123 \
  --training-seed 0 \
  --generation-seed 123 \
  --evaluator-seed 0 \
  --evaluator-repeats 10 \
  --max-graphs 0 \
  --device cuda:0
```

The result is written to `best_trial.json`, `best_config.yaml`, `trials.csv`,
and `SUMMARY.md` inside the output directory. The optimization command never
evaluates the test split.

### 4. Use the distributed controller for multiple GPUs

The production workflow is:

1. Create and hash one immutable cache with
   `scripts/prepare_graphvae_attr_bo_cache.py`.
2. Create a fresh PostgreSQL study with
   `scripts/run_distributed_graphvae_attr_bo.py init` and an exact reservation
   budget.
3. Run `preflight` with the qualified repository, Python, credential-path, and
   GPU-slot mappings.
4. Launch bounded waves with `run`; after any ambiguous launch, use `probe`
   before doing anything else.
5. Use `status`, collect every trial root, and call `finalize` only after all
   reserved trials are terminal.
6. Restore the portable snapshot without PostgreSQL and verify its aggregate
   hashes.

The complete command patterns and failure rules are in
[`attr_f1pr_bayesian_optimization.md`](attr_f1pr_bayesian_optimization.md).
Never place a storage URL or credential value on a command line or in a study
root.

### 5. Confirm before claiming improvement

Do not use the single best search score as the final answer. Freeze the winner
and comparison policy, refit the winner and uniform pair at the same full
budget on several paired training seeds, and compare the paired validation
objectives. Only after that decision is frozen may one explicit test evaluation
be performed. Test results must never trigger reranking or additional training.

The next proposed protocol is specified in
[`improving_BO_roadmap.md`](improving_BO_roadmap.md).

## Final conclusion

For the completed LOBSTER experiment, **Bayesian optimization did not
demonstrate an improvement over uniform GraphVAE reconstruction weights**. The
selected pair won the short seed-0 search, but its mean validation objective
was `0.01153` lower than uniform in the matched multi-seed confirmation, with a
wide interval spanning both benefit and harm. Uniform `(1,1)` remains the
supported default while the improved protocol is developed and tested.
