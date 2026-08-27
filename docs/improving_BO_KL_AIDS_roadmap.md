# Lightweight AIDS GraphVAE BO roadmap with KL-weight optimization

## Status and purpose

The evaluator-selection prerequisite is complete. A matched AIDS bake-off
selected the ten-seed Random-GIN ensemble over the ten-encoder GraphCL
ensemble. The KL-weight study itself has not been launched. This document does
not authorize or record held-out/test evaluation or a production claim.

The goal is to answer one narrow question with a bounded amount of work:

> On AIDS, does validation-only Bayesian optimization of the node-feature,
> edge-feature, and KL loss weights produce a reconstruction-weight setting
> that reproducibly improves GraphVAE over the uniform/default setting?

The recommended experiment is intentionally smaller and simpler than a broad
new campaign. It reuses the already qualified AIDS dataset, cache, evaluator,
runtime class, and distributed controller. It adds one parameter, `beta`,
which is already the model's supported KL coefficient. It does not introduce a
new model, a new evaluator, multi-objective optimization, pruning, learned
surrogates outside Optuna, or a complicated statistical package.

## Final recommendation

Use AIDS and the now directly qualified attributed Random-GIN validation
objective. Add
`beta` as a third log-scaled BO parameter, keep adjacency BCE fixed at `1.0`,
and keep the search close to the current default:

```text
alpha_node_feat in [0.5, 2.0]
alpha_edge_feat in [0.5, 2.0]
beta            in [0.25, 4.0]
```

Run exactly 15 search trials at 250 epochs and training seed 0:

- six fixed anchors that cover the important directions;
- nine adaptive TPE proposals;
- three homogeneous TITAN RTX workers, when available;
- all 184 validation graphs and ten fixed evaluator seeds `1000..1009`;
- generation seed 123 and no held-out/test access.

Only if the frozen search winner exceeds the same-study uniform anchor by at
least `0.03` validation Attr-F1PR should it enter confirmation. Confirm exactly
that one winner against uniform at training seeds 1, 2, and 3. Re-evaluate the
six resulting checkpoints at generation seeds 124 and 125; this adds only
evaluation work, not training. Claim improvement only if the selected-minus-
uniform difference remains positive across every training-seed average and
every generation-seed average, and the overall mean improvement is at least
`0.02`.

This design is expected to take about 7--8 wall-clock hours on three qualified
TITAN RTX GPUs, including operational margin. It is comparable to the previous
AIDS study, not to the much heavier GraphCL/LOBSTER campaign.

## What the completed experiments teach us

### LOBSTER Random-GIN BO

The 30-trial, 2,000-epoch search selected approximately
`(alpha_node_feat=5.229, alpha_edge_feat=0.0539)` and showed a search-time gain
of `+0.1185` over uniform. At 10,000 epochs and matched training seeds 0, 1,
and 2, the selected mean was `0.65896` versus `0.67048` for uniform, a mean
difference of `-0.01153`.

The infrastructure worked. The scientific proxy did not transfer from the
short, single-seed search to the longer, multi-seed comparison. LOBSTER's ten
validation graphs also made the objective coarse and easy to overfit.

### AIDS Random-GIN BO

AIDS removed the most obvious dataset-size problem: it has 1,294 training,
184 validation, and 371 held-out graphs, with both node and edge attributes.
The 14-trial, 100-epoch search selected approximately
`(alpha_node_feat=1.424, alpha_edge_feat=2.469)` and showed `+0.05964` over the
100-epoch uniform anchor.

At the predeclared 250-epoch confirmation, the selected mean was `0.58102`
versus `0.64552` for uniform. The paired mean difference was `-0.06449`; two
of three training seeds favored uniform. Therefore dataset size was not the
main remaining problem. The 100-to-250-epoch fidelity mismatch and single-seed
winner selection remained decisive.

### LOBSTER GraphCL-F1PR qualification

At 10,000 epochs and generation seed 123, edge emphasis
`(alpha_node_feat=0.25, alpha_edge_feat=4.0)` exceeded uniform by `0.10872`.
That apparent signal did not qualify:

- the promoted ranking reversed between 2,000 and 10,000 epochs, with
  Spearman correlation `-1.0`;
- the maximum within-candidate generation-seed range was `0.29390`, much
  larger than the `0.10872` candidate difference;
- adaptive GraphCL BO was correctly not launched.

GraphCL is therefore not the economical evaluator for the next attempt. Its
current generated-sample sensitivity is too large, and its training/evaluation
stack adds substantial runtime and operational complexity.

### Matched AIDS Random-GIN versus GraphCL bake-off

The evaluator choice was subsequently tested directly on AIDS rather than
inferred only from LOBSTER. The bake-off reused all six frozen 250-epoch AIDS
checkpoints and generated 18 exact validation collections: selected and
uniform candidates at training seeds 0, 1, and 2 and generation seeds 123,
124, and 125. Every collection was scored by the same ten fixed Random-GIN
seeds and the same ten train-only GraphCL encoders. Both methods replayed the
predeclared uniform job exactly.

| Evaluator | Selected mean | Uniform mean | Difference | Mean paired SD | Mean generation range | Stable signs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random-GIN | 0.58160 | 0.64980 | -0.06820 | 0.04085 | 0.03062 | 9/9 |
| GraphCL | 0.60736 | 0.68366 | -0.07629 | 0.03267 | 0.03957 | 8/9 |

GraphCL reduced evaluator-to-evaluator paired dispersion by `20.027%`, just
enough to pass that condition. It nevertheless had a `29.218%` larger mean
generation-seed range and worse sign stability. It therefore failed two of
the predeclared selection conditions. Random-GIN is the frozen primary
evaluator for the proposed search. A disjoint fixed Random-GIN ensemble is
still required for confirmation; the bake-off does not claim that the old BO
weights improved GraphVAE.

### Combined diagnosis

Across the experiments, four conclusions are well supported:

1. Distributed scheduling, reservation accounting, artifact handling, and
   deterministic replay are not the main scientific limitations.
2. A short-fidelity, single-training-seed objective is not a reliable proxy for
   a longer, seed-robust result.
3. A larger validation set helps but does not remove training-seed sensitivity
   or winner's curse.
4. Tuning only node and edge feature weights changes their total scale relative
   to adjacency reconstruction and KL regularization. The BO search has never
   been allowed to correct that latent-regularization balance.

The fourth point motivates adding KL weight, but it does not justify a huge
three-dimensional range. A narrow, full-fidelity AIDS search is the most useful
next experiment.

## Why the KL term is a sensible addition

For the plain GraphVAE path, the base VAE loss contains adjacency BCE and KL
as the last two coefficients. The current default resolves them to `1.0` and
`1.0`. `main.py` already supports the configuration value `beta`; when it is
not `null`, it replaces the final KL coefficient. Therefore KL optimization
does not require changing the model's mathematical implementation.

The next study should use:

```yaml
model:
  beta: <sampled positive value>

loss:
  alpha_node_feat: <sampled positive value>
  alpha_edge_feat: <sampled positive value>
  use_graphvae_mm_bce_kl_weights: false
```

Adjacency BCE remains fixed at `1.0`. Motif and other auxiliary losses remain
off. This isolates one interpretable change: the strength of latent KL
regularization relative to adjacency, node-feature, and edge-feature
reconstruction.

Do not enable the legacy Kiarash `40/2000` or `50/2000` BCE/KL bundle. That
would simultaneously change adjacency BCE and KL by orders of magnitude and
would confound the proposed experiment. It is also unnecessarily far from the
qualified plain-GraphVAE AIDS baseline.

## Five plan/critique/revision passes

### Pass 1: broad three-dimensional BO

Initial plan: add KL weight and run 30 or more trials over several orders of
magnitude for all three parameters.

Critique: this repeats the weakly identified LOBSTER search in a larger space.
It is expensive, gives TPE many pathological scales to explore, and still does
not fix the fidelity or training-seed problem.

Revision: narrow the ranges around the supported default and solve fidelity
before increasing trial count.

### Pass 2: repeat a cheap 100-epoch AIDS search

Revised plan: use the narrow three-dimensional space but retain 100 epochs to
keep the run short.

Critique: the existing AIDS experiment already proves that 100-epoch rankings
can fail at 250 epochs. Repeating that proxy with an extra parameter is likely
to produce another attractive but non-transferable winner.

Revision: search and confirmation must use the same 250-epoch fidelity.

### Pass 3: average two training seeds inside every BO trial

Revised plan: make every adaptive objective the mean of training seeds 0 and 1
at 250 epochs.

Critique: this is scientifically stronger, but it doubles every search fit and
requires a new grouped Random-GIN trial implementation, aggregation contract,
failure semantics, and restoration tests. With a third parameter, a useful
search would become heavier than necessary for a first KL experiment.

Revision: use one seed for bounded search, but make confirmation independent,
mandatory, and strict. Do not include the search seed in the final paired
decision.

### Pass 4: replace BO with a fixed KL grid

Revised plan: run a small full-fidelity grid over feature and KL scales, then
confirm the best grid point.

Critique: a grid is simple, but it does not answer whether BO helps and spends
most of its budget on corners. A dense three-dimensional grid is not light; a
tiny grid is unlikely to locate an interaction.

Revision: use six scientifically chosen fixed anchors as TPE startup evidence,
then spend the remaining budget on nine real adaptive proposals.

### Pass 5: final bounded design

Final plan: AIDS, Random-GIN Attr-F1PR, 250 epochs everywhere, narrow direct
weights, six fixed anchors plus nine TPE trials, followed by an independent
three-training-seed confirmation and cheap generation-seed re-evaluation.

Final critique: the adaptive search still observes only training seed 0, and
nine adaptive proposals cannot map a complicated three-dimensional surface.
The design manages rather than eliminates winner's curse. It is acceptable
because the domain is deliberately narrow, the anchors cover the main effects,
and no claim is allowed without independent seed confirmation. If this design
fails, simply adding more TPE trials is not the recommended response; the
objective or model training stability should be improved first.

## Frozen scientific contract for the proposed study

The new study must preserve exactly:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

Required invariants:

- dataset: AIDS;
- cache: the already frozen 1,849-graph AIDS cache, reverified by SHA-256;
- split: 1,294 train / 184 validation / 371 held-out;
- optimization and confirmation split: validation only;
- `test_access=false`;
- `skip_final_evaluation=true`;
- node and edge feature decoders both required;
- all 184 validation graphs;
- ten fixed Random-GIN evaluator seeds `1000..1009` during search;
- a disjoint ten-seed Random-GIN ensemble `2000..2009` during confirmation;
- identical evaluator, graph-generation, runtime, and hardware contracts for
  search and confirmation;
- GraphVAE fidelity: 250 epochs for both search and confirmation;
- adjacency BCE fixed at `1.0`;
- motif losses off;
- legacy reparameterization unchanged, so the new experiment changes only the
  three declared weights;
- no held-out/test evaluation during this roadmap.

The search ranges are log scaled:

| Parameter | Low | High | Default anchor |
| --- | ---: | ---: | ---: |
| `alpha_node_feat` | 0.5 | 2.0 | 1.0 |
| `alpha_edge_feat` | 0.5 | 2.0 | 1.0 |
| `beta` (KL weight) | 0.25 | 4.0 | 1.0 |

These ranges are intentionally conservative. The previous AIDS candidate with
edge weight `2.469` failed confirmation, and the extreme LOBSTER ratio failed
to generalize. Broadening the ranges before a positive narrow-range result is
not justified.

## Minimal implementation roadmap

### I1. Add `beta` to the BO parameter contract

Implementation status (2026-08-27): complete in source. The bounded beta mock
and all 138 non-PostgreSQL distributed/attribute-BO tests pass. The isolated
PostgreSQL suite is collected but cannot currently qualify because the
protected Gate 5 login is rejected from both the controller and `cs-cl-17`.
No study or schema was created by those failed connection attempts. Protected
qualification credentials must be rotated or repaired before I3 or any real
reservation.

Extend the existing BO plumbing; do not modify the VAE loss formula.

Required code changes:

1. Add optional `beta` bounds to `SearchRanges`.
2. Sample `beta` with `trial.suggest_float(..., log=True)` when enabled.
3. Inject sampled `beta` into the configuration's `model` section, while node
   and edge weights remain in `loss`.
4. Add `beta` to distributed search-space, fixed-parameter, and immutable
   reservation-plan allowlists.
5. Add controller CLI support for beta range and fixed beta values.
6. Carry beta through worker contracts, trial JSON, CSV/SUMMARY output,
   portable snapshots, restoration, and audit code.
7. Keep beta absent or `null` for every historical contract; old studies must
   restore unchanged.
8. Update the deterministic mock objective so tests can prove beta is sampled,
   injected, persisted, and replayed.

Fail closed when:

- beta is non-finite or non-positive;
- beta is outside the immutable contracted range;
- a reservation supplies beta when the study does not contract it;
- `use_graphvae_mm_bce_kl_weights=true` in an AIDS KL-BO configuration;
- a resolved trial config does not contain the sampled beta exactly;
- an old study changes meaning after the implementation.

Focused tests must cover local and distributed sampling, fixed reservations,
mock execution, failure consumption, CSV/finalization, portable restoration,
and backward compatibility. Run the full non-PostgreSQL BO suite and isolated
PostgreSQL suite before any real reservation.

### I2. Create AIDS KL-BO configurations and manifests

Create separate files rather than editing the completed AIDS study contracts:

- `aids_graphvae_attr_f1pr_kl_smoke.yaml`;
- `aids_graphvae_attr_f1pr_kl_search.yaml`;
- `aids_graphvae_attr_f1pr_kl_confirmation.yaml`;
- a search policy and exact 15-reservation plan;
- a confirmation policy template and exact six-reservation plan generated only
  after the search winner is frozen;
- fresh study names and output roots.

Reuse the existing AIDS cache manifest only after its path, size, mode, SHA-256,
split fingerprint, and 56/3 node/edge schemas are reverified. Do not regenerate
the dataset or create a new split.

### I3. Qualify only what changed

Do not repeat the full historical lifecycle campaign. The controller has
already passed multi-host locking, ambiguity, stale-worker recovery, portable
restore, and PostgreSQL isolation tests.

Requalify only:

- clean source manifest on controller and workers;
- exact runtime fingerprint;
- protected `verify-full` PostgreSQL authentication;
- mode-`0444` AIDS cache identity on every worker;
- 56-dimensional node and 3-dimensional edge schemas;
- homogeneous TITAN RTX slot isolation;
- one mock beta trial;
- one real five-epoch beta smoke trial on validation, with no scientific claim.

If beta is not present exactly in the resolved config and training log, stop
before creating the scientific search.

## Execution roadmap

### E1. Freeze the 15-trial search before creation

Use exactly these six startup anchors, in order. They are frozen in
`aids_attr_f1pr_kl_search_reservations_15.json`:

| Budget index | Label | Node | Edge | KL beta |
| ---: | --- | ---: | ---: | ---: |
| 0 | uniform/default | 1.0 | 1.0 | 1.0 |
| 1 | low KL | 1.0 | 1.0 | 0.5 |
| 2 | high KL | 1.0 | 1.0 | 2.0 |
| 3 | weak feature reconstruction | 0.5 | 0.5 | 1.0 |
| 4 | strong feature reconstruction | 2.0 | 2.0 | 1.0 |
| 5 | mild edge emphasis with low KL | 1.0 | 2.0 | 0.5 |

Budget indexes 6--14 are nine adaptive TPE proposals. Freeze the TPE study
seed and use `n_startup_trials=6`. With three workers, the intended synchronous
waves are `3+3+3+3+3`. The first two waves are fully fixed; the last three are
adaptive. `max_parallel=3` is a wall-clock optimization only and may be reduced
if a slot is not qualified.

All 15 trials use training seed 0 and 250 epochs. A failed consumed reservation
is never replaced. Ambiguous work is probed and reconciled before any later
dispatch.

### E2. Freeze and audit the search

After exactly 15 terminal reservations:

1. collect and audit every launch, trial, checkpoint, and objective;
2. verify all 15 objectives use validation and `test_access=false`;
3. verify exact beta/node/edge values against each reservation;
4. finalize, freeze, snapshot, and restore without PostgreSQL;
5. select the maximum finite validation objective exactly once;
6. do not rerank after inspecting confirmation.

Proceed to confirmation only if:

- the winner is not the uniform anchor;
- winner minus same-study uniform is at least `0.03`;
- both validation precision and recall are finite and positive;
- every integrity, cache, decoder, reservation, and security audit passes.

Otherwise stop and report `no_improvement_at_search_gate`. This saves the six
confirmation fits when the search signal is too small to be credible.

### E3. Run independent matched confirmation

If E2 passes, create a different fresh study after the winner and its exact
weights are committed. Reserve exactly six fixed trials:

```text
training seed 1: selected, uniform
training seed 2: selected, uniform
training seed 3: selected, uniform
```

Use two waves of three workers, 250 epochs, generation seed 123, the same 184
validation graphs, and the disjoint fixed evaluator seeds `2000..2009`.
Training seed 0 is not
part of the final decision because it selected the winner.

No alternate candidate is substituted if confirmation fails. No failed
reservation is replaced.

### E4. Add cheap generation-seed stability

After the six confirmation checkpoints are frozen, evaluate each checkpoint at
generation seeds 124 and 125. Reuse the original seed-123 evaluation by exact
hash. This creates exactly 12 new evaluations and no new training runs.

For each candidate, compute the 3-by-3 matrix:

```text
training seeds   = 1, 2, 3
generation seeds = 123, 124, 125
```

Do not use SciPy significance tests. Report direct paired differences and use
the following simple predeclared rule.

### E5. Final decision rule

Report `validation_improvement_demonstrated` only when all conditions hold:

1. the overall selected-minus-uniform mean across the nine matched cells is at
   least `+0.02` Attr-F1PR;
2. for each training seed, the mean difference across generation seeds is
   strictly positive;
3. for each generation seed, the mean difference across training seeds is
   strictly positive;
4. all 18 candidate evaluations are finite, validation-only, decoder-complete,
   and integrity-clean;
5. the exact reservation, snapshot, cache, credential, and restore audits pass.

If any condition fails, report `no_improvement`. Do not fall back to the second
search candidate and do not add trials after seeing the outcome.

This rule is deliberately more interpretable than a three-point t interval. It
requires a practically meaningful average gain and consistent direction across
both sources of randomness that caused the earlier failures.

## Runtime and compute budget

The previous AIDS measurements give a 250-epoch fit time of approximately
51--57 minutes on a TITAN RTX. With three workers:

| Stage | Training fits | Waves | Estimated wall time |
| --- | ---: | ---: | ---: |
| mock and five-epoch smoke | 1 tiny real fit | 1 | under 15 minutes |
| 15-trial search | 15 | 5 | 4.3--4.8 hours |
| six-trial confirmation | 6 | 2 | 1.7--1.9 hours |
| 12 extra generation evaluations | 0 | bounded | under 10 minutes |
| audits, collection, freeze, restore | — | — | about 1 hour margin |
| **Maximum expected total** | **21 full fits** | **7 full waves** | **about 7--8 hours** |

If the search gate fails, stop after roughly 5--6 hours and avoid confirmation.
If only two GPUs are available, preserve the exact trial budget and lower
parallelism; expect longer wall time but the same scientific contract.

Do not increase the 15-trial budget during the run. A larger follow-up is
justified only if this bounded experiment produces a consistent positive
confirmation near a search-space boundary.

## What is intentionally excluded

- QM9;
- PROTEINS, because the current attributed objective requires real edge
  features and the existing PROTEINS path does not satisfy that contract;
- GraphCL for this next attempt;
- motif-weight optimization;
- adjacency-BCE optimization;
- learning-rate, architecture, decoder, or reparameterization changes;
- early-stopping/pruning/successive-halving logic;
- multi-objective BO;
- Gaussian-process or custom surrogate implementation;
- broad KL ranges such as `1e-3` to `2000`;
- held-out/test evaluation;
- post-hoc reranking, replacement trials, or extra trials after seeing results.

These exclusions keep the interpretation clear and implementation bounded.

## Audit and commit checkpoints

Commit and push separately after each completed step:

1. beta BO implementation and focused/full tests;
2. immutable AIDS KL configs, policies, and exact search reservations;
3. deployment/source/cache/runtime/smoke qualification;
4. search initialization and prelaunch proof;
5. completed/frozen/restored 15-trial search and search-gate decision;
6. confirmation contract, if authorized by the search gate;
7. completed/frozen/restored confirmation and generation-seed audit;
8. final plain-language conclusion.

Before every commit, inspect status, recent commit subjects, unstaged and staged
statistics, all tracked/untracked changes, and the complete staged diff. Never
commit credentials, URLs, caches, checkpoints, or generated trial artifacts.

## Expected interpretation

There are only three acceptable final outcomes:

1. `no_improvement_at_search_gate`: full-fidelity BO did not produce a large
   enough seed-0 signal to justify more compute.
2. `no_improvement`: a promising search winner failed independent training- or
   generation-seed confirmation.
3. `validation_improvement_demonstrated`: the selected node, edge, and KL
   weights exceed uniform by at least `0.02` and retain a positive direction
   across every training- and generation-seed average.

Even outcome 3 is validation evidence for AIDS under this exact model and
runtime. It is not a universal GraphVAE default and is not a held-out/test
claim. Any one-time held-out evaluation requires a separately frozen plan and
explicit authorization after the validation conclusion is committed.

## Why this plan is promising without being excessive

The plan directly fixes the two clearest design failures: it removes the
search/confirmation epoch mismatch and gives BO control over KL balance. It
uses AIDS because its validation set and attribute schemas are already
qualified. It uses the cheaper evaluator whose hardware determinism was
demonstrated, retains actual adaptive TPE proposals, and adds generation-seed
checking for almost no training cost.

At the same time, it limits implementation to one already supported model
parameter, limits the search to 15 trials near the current default, uses a
simple confirmation rule, and stops early when the search signal is weak. This
is the smallest next experiment that has a reasonable chance of producing a
more credible answer than the completed LOBSTER and AIDS searches.
