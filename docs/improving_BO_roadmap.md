# Roadmap for improving GraphVAE reconstruction-weight BO

## Purpose and final question

This roadmap defines the next bounded, non-QM9 study. Its final question is:

> Does Bayesian optimization of GraphVAE node- and edge-reconstruction loss
> weights improve validation Attr-F1PR over uniform `(1,1)` weights, and does
> that improvement remain stable across training seeds at the full training
> budget?

The current LOBSTER answer is **no improvement demonstrated**. This roadmap is
designed to distinguish a genuinely useful weight region from the short-run,
single-seed selection effect observed in the completed study.

The exact objective remains:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

Selection remains validation-only. `test_access` remains false throughout
search and confirmation. Both `node_feature_decoder` and
`edge_feature_decoder` are required. No final-test evaluation may run during
optimization, and test results may never cause reranking or more training.

## Evidence carried forward from the completed run

The design below is based on these observed facts rather than a generic BO
recipe:

1. The 30-trial search winner `(5.2290456720, 0.0538641483)` scored
   `0.8012845288` versus `0.6827769426` for uniform at seed 0 and 2,000 epochs.
2. At seed 0 and 10,000 epochs, that advantage shrank from `+0.11851` to
   `+0.02202`.
3. At 10,000 epochs, seeds 1 and 2 favored uniform by `0.01656` and `0.04005`.
4. The confirmed selected-minus-uniform mean was `-0.01153`, with 95% interval
   `[-0.08939, 0.06633]`.
5. The validation split had only ten graphs. The uniform trial's ten
   random-GIN repeats had F1-PR standard deviation `0.2943`.
6. The selected ratio was about 97:1. Its final logged weighted edge term was
   about `0.000117`, versus `0.004045` for uniform, even though the objective
   explicitly uses edge attributes.
7. Three homogeneous TITAN RTX workers reproduced the same uniform result
   exactly. Heterogeneous GPU models did not. Hardware homogeneity must remain
   part of the scientific contract.
8. A 2,000-epoch LOBSTER fit took about 4.2 minutes; a 10,000-epoch fit took
   about 15.9 minutes. The 30-trial search took about 69 wall-clock minutes on
   three workers, but its study tree was about 33 GiB.

These observations make the main risks explicit: fidelity mismatch,
training-seed overfitting, a small and repeatedly reused validation set, an
unhelpful loss parameterization, noisy winner selection, and excessive
artifact volume.

## Dataset decision: attributed data first

### Recommended primary new dataset: AIDS

Use AIDS as the first new-dataset candidate, subject to cache qualification.
The official TUDataset inventory reports 2,000 graphs with node labels, four
node attributes, and edge labels. The repository's existing TU loader already
maps its node information into the node schema and its edge labels into the
edge schema. This makes it conceptually compatible with `decoded_node_edge`.

Source: <https://chrsmrrs.github.io/datasets/docs/datasets/>

This is only a candidate until the locally produced cache proves the exact
retained graph count, split sizes, maximum graph size, node/edge dimensions,
and stable schema fingerprints.

### Why ordinary PROTEINS is not the primary choice

The official PROTEINS dataset has 1,113 graphs and meaningful node
labels/attributes, but it has no edge labels or edge attributes. The current
repository loader agrees: it sets `edge_feature_info = None` and appends no
edge features.

Therefore PROTEINS cannot be used for this exact objective as it stands. A
constant “edge exists” channel would make edge reconstruction nearly trivial
and would not answer whether meaningful edge-attribute weighting helps. Do not
silently add such a channel merely to pass schema checks.

PROTEINS may proceed only if a separately reviewed data source provides
scientifically meaningful, provenance-tracked edge types for the same graphs.
That would be a new dataset contract and must be tested independently. If no
such source is adopted, record PROTEINS as incompatible and continue with AIDS
or another dataset that has genuine node and edge semantics.

## Non-negotiable operational contract

Every phase must preserve the distributed guarantees already qualified:

- fresh study and output roots for every distinct contract;
- exact reservation counts and bounded `max_parallel`;
- immutable source, runtime, configuration, cache, split, schema, and seed
  fingerprints;
- homogeneous GPU model and exact runtime fingerprint within a study;
- deterministic dispatch seeds and unique worker/trial/artifact identities;
- PostgreSQL heartbeat, grace, and test isolation;
- one consumed budget slot for every claimed failure, with no replacement;
- probe-before-retry for ambiguous launches;
- atomic completion, collection, finalization, snapshot, and restore handling;
- protected credentials outside repository/cache/artifact roots and redacted
  storage identities;
- read-only cache verification before and after every study;
- no credentials, storage URLs, test split, or true `test_access` in search
  artifacts;
- a commit and push after every completed roadmap gate.

## Phase 0 — add the scientific controls that the first run lacked

### 0.1 Record loss-component scale and gradient information

Extend training diagnostics to record, at predetermined epochs:

- raw `kernel_cost`, node-feature loss, and edge-feature loss;
- weighted values of all three terms;
- decoder-specific gradient norms before optimizer update;
- node- and edge-decoder accuracy or cross-entropy by feature group;
- whether any decoder receives zero, non-finite, or negligible gradients.

Write a small atomic JSON summary, not a full tensor trace. Tests must verify
that diagnostics are finite, correctly associated with the trial contract,
and absent when unsupported. Use the uniform baseline to set search bounds
from observed scale rather than from an arbitrary five-order-of-magnitude
range.

Exit condition: a uniform LOBSTER timing/diagnostic trial explains how the
node and edge terms compare with the kernel term through training.

### 0.2 Reparameterize the search

Replace independent raw weights with an overall feature scale `g` and a
node-to-edge ratio `r`:

```text
alpha_node_feat = g * sqrt(r)
alpha_edge_feat = g / sqrt(r)
```

Then:

```text
g = sqrt(alpha_node_feat * alpha_edge_feat)
r = alpha_node_feat / alpha_edge_feat
```

This separates “how strong should feature reconstruction be relative to the
kernel loss?” from “how should that feature strength be divided between node
and edge reconstruction?” Uniform weights are exactly `g=1, r=1`.

Determine bounded log ranges after Phase 0.1. A provisional range for testing
is `g in [0.25, 4]` and `r in [0.25, 4]`; it must not become the production
range until the loss/gradient calibration is reviewed. Keep the old 97:1
winner as a diagnostic anchor outside the main region, not as an assumed
optimum.

Tests must prove that derived weights are finite, positive, within the frozen
contract, reproducible, and written into the resolved YAML and trial result.

### 0.3 Support a multi-fit candidate objective

One BO candidate must be able to own a predeclared group of model fits sharing
the same weights but using different training seeds. The candidate objective
must be computed only after every reserved member is terminal.

Implement a versioned candidate-group plan with:

- exact candidate index and member reservation indexes;
- shared derived weights;
- exact training, generation, evaluator, and split seeds;
- a fixed aggregation rule;
- explicit behavior when one member fails;
- no automatic replacement of failed members;
- an atomic candidate summary tied to all member artifact hashes.

The initial aggregation should be the arithmetic mean validation Attr-F1PR
across paired training seeds. Also report the minimum, sample standard
deviation, and standard error. Do not optimize a test value or silently omit a
failed member. A failed member makes that candidate fail closed unless the
predeclared policy assigns a conservative penalty.

### 0.4 Reduce artifact volume safely

The first search retained roughly 1.1 GiB per trial. Add a reviewed retention
mode that keeps the final checkpoint, resolved config, reproducibility record,
process identities, logs, generated/reference evaluator inputs needed for
audit, evaluator output, and hashes, while omitting redundant plots and
intermediate arrays. Never delete an artifact after it has entered a frozen
manifest. Apply retention during trial creation, not as post-hoc cleanup.

Exit condition for Phase 0: focused unit tests, launcher tests, integrity
tests, one mock candidate group, one tiny real group, portable restore, and
credential/test-access scans all pass. Commit and push before scientific use.

## Phase 1 — diagnose and repair the LOBSTER objective

LOBSTER is cheap and already qualified, so use it to validate the improved
method. Because its test split has already been opened, this phase is method
development and validation-only evidence; it cannot create a new pristine
LOBSTER test claim.

### 1.1 Freeze an outer split and internal validation partitions

Keep the existing 20 held-out graph identities fixed and inaccessible. Add a
split-plan format that reshuffles only the remaining 80 graphs into 70 training
and 10 validation graphs. Freeze three internal split seeds, for example 123,
456, and 789, only after verifying that no outer-test identity appears in any
training or validation partition.

This avoids the serious error of changing the split seed over all 100 graphs,
which could move previously held-out graphs into optimization.

### 1.2 Measure fidelity predictiveness

For uniform weights and a small set of calibrated anchors, run paired training
seeds at 500, 2,000, 5,000, and 10,000 epochs. Whenever implementation permits,
evaluate predetermined checkpoints from the same training trajectory so that
the comparison isolates training horizon without retraining identical early
epochs.

Compute rank correlations between early and 10,000-epoch objectives. A search
fidelity may be used only if it predicts the full-budget ranking well enough
under a predeclared threshold. If 2,000 epochs remains poorly predictive, use
5,000 epochs or a learning-curve promotion rule.

Exit condition: select and freeze one search fidelity using correlation and
runtime evidence, not convenience.

### 1.3 Run a structured calibration design

Before TPE, evaluate twelve predeclared `(g, r)` anchors, including:

- uniform `(1,1)`;
- balanced lower and higher global feature scales;
- node-favoring and edge-favoring ratios of equal magnitude;
- the current BO winner as an extreme diagnostic point;
- several space-filling log-domain points.

Use two training seeds per anchor and common generation/evaluator seeds. This
24-fit design tests whether the surface is smooth, whether the current extreme
ratio is repeatable, and whether the proposed bounds contain useful signal.

Exit condition: reject or refine the parameterization if nearby points do not
produce a coherent response surface or if decoder gradients vanish.

### 1.4 Run robust LOBSTER BO

Use a fresh immutable study with 24 candidate groups in total: the 12
structured anchors followed by 12 adaptive proposals. Each candidate uses two
paired training seeds at the frozen search fidelity. Aggregate the two
validation objectives per candidate. Seed the study with the structured
anchors, including uniform, before adaptive TPE proposals.

Keep `max_parallel=3` on the qualified TITAN RTX pool. A three-worker pool is
scientifically preferable to the available nine heterogeneous workers; the
nine-worker trial proved concurrency but failed numerical hardware
equivalence.

Predeclare the full candidate and member reservation budget. Parallel TPE must
retain `constant_liar=True`, but proposal batches should be as small as the
three-worker pool so completed information is incorporated frequently.

### 1.5 Promote a shortlist without reranking after the fact

Before the robust search starts, freeze this promotion rule:

1. retain the best three candidate groups by aggregate validation objective;
2. always retain uniform as a control;
3. train those four candidates at 5,000 or 10,000 epochs on three training
   seeds and the predeclared internal validation partitions;
4. choose at most one winner using the frozen aggregate rule.

The shortlist size and tie handling must be defined before results exist.
Failed reservations remain failed and consume their exact budget.

### 1.6 LOBSTER decision

Compare the frozen winner and uniform pair at 10,000 epochs on five paired
training seeds. Aggregate over the internal validation partitions without
opening the fixed outer test. Report per-seed/per-partition differences and a
paired confidence interval.

LOBSTER supports a method-level improvement only if:

- the mean paired difference is positive;
- every training-seed aggregate is positive;
- the 95% paired interval lower bound is positive;
- the mean gain exceeds a practical threshold frozen from the uniform pilot;
- neither decoder shows collapsed gradients or invalid outputs.

Otherwise record `no_improvement_demonstrated`. Do not search again using the
same validation results.

## Phase 2 — qualify a larger attributed dataset

### 2.1 Build the AIDS cache

Create a BO-specific YAML and cache using the existing AIDS loader. Before any
model training, record and verify:

- source dataset provenance and local raw-file hashes;
- loader options, maximum-node filter, and exact retained graph count;
- exact 70/10/20 train/validation/test identities;
- node and edge feature definitions, dimensions, categories, and schema
  fingerprints;
- graph-size and edge-count distributions;
- immutable cache path, byte size, SHA-256, and read-only mode;
- successful `decoded_node_edge` conversion on every split;
- absence of test loading in the BO preflight path.

Reject the dataset if edge labels are missing after filtering, mappings differ
between splits, or the edge decoder would learn a constant target.

### 2.2 Build a BO-safe AIDS configuration

Start from the same GraphVAE architecture only where its tensor dimensions are
valid for AIDS. Keep selection and evaluation settings fixed, including
`skip_final_evaluation=true`, full validation evaluation, generation seed 123,
ten evaluator repeats, and both feature decoders.

Do not copy the existing PROTEINS Table 2 config as a BO config: it enables
automatic evaluation behavior intended for table reproduction and has no edge
schema.

### 2.3 Calibrate runtime and memory before setting the budget

Run exactly one uniform training seed at 500 epochs, followed by one at the
proposed search fidelity. Measure:

- peak GPU memory;
- training, generation, and evaluation seconds;
- artifact bytes;
- accepted validation graph count;
- objective and per-repeat variance;
- component losses and decoder gradient norms.

Repeat the proposed-fidelity uniform trial once on every candidate GPU. Freeze
only a hardware-homogeneous slot set that meets the numerical tolerance.

Use these measurements to calculate, document, and approve the exact number of
fits. Do not infer the budget only from the 20x larger graph count.

### 2.4 Decide whether AIDS is statistically preferable

The AIDS validation set should be much larger than LOBSTER's ten graphs, but
the exact retained split controls the benefit. Compare:

- per-repeat evaluator standard deviation;
- paired training-seed standard deviation;
- precision/recall resolution;
- early-to-full fidelity rank correlation;
- wall time and artifact cost.

Proceed only if the objective is materially more stable or if the study budget
can compensate for the remaining variance.

## Phase 3 — run the definitive new-dataset search

The exact budget is frozen after Phase 2 timing. The provisional bounded design
is:

1. **Anchors:** 12 structured `(g,r)` candidates, two training seeds each.
2. **Adaptive search:** 24 additional candidate groups, two training seeds
   each, with TPE initialized by the anchor evidence.
3. **Promotion:** the top three candidates plus uniform, three seeds each at a
   higher fidelity.
4. **Confirmation:** one frozen winner versus uniform, five paired training
   seeds at the full budget.

This corresponds to 72 anchor/search fits, 12 promotion fits, and 10
confirmation fits before any optional split-robustness extension. The number
may be reduced only before study initialization and only from measured timing
evidence; it may never be silently reduced after failures.

If the evaluator remains noisy, increase independent training seeds or fixed
internal validation partitions before increasing raw TPE proposals. More noisy
candidates are less valuable than fewer candidates measured reliably.

### Estimated wall-time formula

Let `Tsearch`, `Tpromote`, and `Tfull` be the measured minutes for one fit at
each fidelity, and let `W` be the number of scientifically qualified
homogeneous workers. Then:

```text
optimistic_training_lower_bound_minutes =
    ceil(72 / W) * Tsearch
  + ceil(12 / W) * Tpromote
  + ceil(10 / W) * Tfull
```

Candidate-group and promotion barriers can make the real schedule longer than
this lower bound. Add at least 25% for those barriers, dispatch gaps,
collection, hashing, restore, and audits. For `W=3`, the lower-bound training
component is `24*Tsearch + 4*Tpromote + 4*Tfull`. Publish the estimate and
storage budget before initialization.

## Phase 4 — predeclared confirmation and final answer

### 4.1 Freeze the analysis policy

Before confirmation study creation, commit a versioned policy containing:

- selected candidate identity and weight derivation;
- uniform control `(1,1)`;
- full epoch budget and all seeds;
- exact validation split identities;
- objective JSON path;
- paired aggregation and confidence-interval method;
- a practical minimum effect chosen from the uniform pilot;
- superiority and no-improvement rules;
- prohibition on alternate-candidate fallback;
- `held_out_access=false`, `test_access=false`, and no reranking.

### 4.2 Make the primary decision on validation

Return **BO improves GraphVAE** only when all of the following are true:

- all confirmation reservations are complete and valid;
- the selected-minus-uniform mean is positive;
- every paired seed aggregate is positive;
- the 95% confidence-interval lower bound is above zero;
- the mean difference meets the frozen practical-effect threshold;
- source, cache, schema, hardware, and evaluator contracts all pass.

If any condition fails, return **BO improvement is not demonstrated; keep
uniform weights**. This is the required yes/no operational answer. A wide
interval should be described as uncertainty, not misreported as proof that BO
causes harm.

### 4.3 Open held-out test once, if authorized

Only after the validation decision, winner, weights, and report hashes are
frozen may a separate test-only plan evaluate the selected and uniform
checkpoints. It must run no training, create no Optuna trial, and cause no
selection, reranking, or follow-up training.

The held-out result is a generalization check. It may support or challenge the
validation conclusion descriptively, but it cannot retroactively change how
the candidate was selected.

## Phase 5 — final audit and deliverables

For every completed study:

1. collect all intended host artifacts with collision and checksum checks;
2. reconcile PostgreSQL trial/reservation/heartbeat state;
3. prove the exact budget, including every consumed failure;
4. finalize and seal the study read-only;
5. restore its portable snapshot without PostgreSQL;
6. compare aggregate and semantic hashes;
7. recheck source/runtime/cache/GPU identities;
8. scan for credentials, unredacted storage URLs, test split, and true
   `test_access` in pre-test artifacts;
9. run the full non-PostgreSQL distributed suite;
10. run the isolated protected PostgreSQL suite and prove zero residual test
    studies;
11. document objective values, timing, GPU hours, artifact bytes, failures,
    uncertainty, and the supported yes/no conclusion;
12. inspect changes with the commit-summary workflow, commit only intended
    source/config/documentation evidence, and push.

Generated caches, checkpoints, study databases, trial trees, credentials, and
storage URLs remain outside commits.

## Stop conditions

Stop before expensive search and record the reason if any of these occurs:

- no genuine edge attributes are available;
- node or edge decoder schema is missing or unstable;
- the objective cannot be reproduced on homogeneous hardware;
- early fidelity has no useful relationship with full-budget ranking;
- uniform runs show variance too large for the bounded confirmation budget;
- artifact/storage projections exceed the approved bound;
- protected PostgreSQL isolation or credential redaction fails;
- a launch is ambiguous and matching worker/DB state cannot be reconciled;
- any optimization path accesses the test split.

Stopping under these conditions is a valid scientific result. It is preferable
to producing a confident-looking BO winner from an invalid or underpowered
comparison.

## Expected final interpretation

The completed LOBSTER study already answers its own question: **the tested BO
weights do not improve GraphVAE over uniform reconstruction weights under the
matched validation contract**.

The next study should not try to force a positive result. It should use a
better-identified loss parameterization, multi-seed candidate objectives,
predictive training fidelity, a larger genuinely attributed validation set,
and a predeclared paired confirmation. If that design passes the superiority
rule, the answer becomes “yes” for the qualified dataset and contract. If it
does not, the correct final answer remains “no improvement demonstrated,” and
uniform weights remain the default.

## AIDS execution checkpoint (2026-08-24)

The user authorized the bounded non-QM9 weight study, and AIDS is selected for
qualification. The max-40-node contract retains 1,849 of the 2,000 official
graphs while bounding the fully connected decoder shape. It leaves 1,294
training, 184 validation, and 371 held-out test graphs. The loader audit found
118 disconnected retained source graphs, so `all_components` BFS is frozen to
preserve rather than silently discard their non-root components.

The source archive and all six relevant raw files have committed SHA-256
provenance. Every retained graph has node attributes and a real edge label;
the three observed edge states are `0`, `1`, and `2`. The canonical cache is
73,822,456 bytes with SHA-256
`6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb`.
Its validated node/edge dimensions are 56/3 and its split, graph, and schema
fingerprints are frozen in
`configs/bayesian_optimization/aids_attr_f1pr_cache_manifest.json`. The cache
is mode `0444`.

The cache bootstrap saved the cache atomically before its intentionally tiny
debug continuation enabled the legacy motif preset and failed against an
unavailable local MySQL service. It created no Optuna study or reservation.
The cache was subsequently loaded and fully re-fingerprinted by the dedicated
manifest verifier, so the unrelated post-save bootstrap failure is not hidden
or treated as model evidence.

The BO-safe 250-epoch timing configuration and four focused AIDS/config/cache
tests pass as part of a 22-test focused run. No training qualification, BO
reservation, validation objective, or held-out access has occurred. The next
step is a dedicated two-host deployment followed by one bounded uniform timing
and hardware-equivalence study; its measurement will freeze the search and
confirmation budgets.

The dedicated deployment is now qualified on three physical TITAN RTX slots:
cs-cl-13 GPU 0 and cs-cl-17 GPUs 0 and 1. Each 24,576 MiB physical device maps
to exactly one logical `cuda:0`, and no compute process was active during the
qualification check. The controller and both hosts reproduce runtime
fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`.
The clean `7fd1fec` source tree and canonical mode-`0444` cache were deployed
to new AIDS-only roots and independently verified on both hosts.

Separate protected AIDS controller/worker bundles exist outside every source,
cache, artifact, and repository root. Their directories are mode `0700` and
files mode `0600`; the controller and both workers authenticated to PostgreSQL
through the protected CA/password files under the mandatory `verify-full`
policy. No credential value or storage URL enters the committed qualification
record. Real training and held-out access still have not occurred. The next
irreversible action is the fresh, exact three-reservation uniform timing and
hardware-equivalence study.
