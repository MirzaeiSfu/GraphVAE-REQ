# LOBSTER GraphCL-F1PR Bayesian-optimization roadmap

## 1. Purpose and final question

This roadmap defines a new, validation-only LOBSTER experiment. It does not
reinterpret or modify the completed Random-GIN studies. Its final question is:

> When reconstruction weights are selected using a split-safe ensemble of
> contrastively trained GraphCL-GIN evaluators, does the selected GraphVAE
> configuration improve mean validation GraphCL-F1PR over uniform `(1,1)`
> reconstruction weights at matched training seeds and final fidelity?

The answer must be exactly one of:

- `improvement_confirmed`;
- `no_improvement`;
- `qualification_failed`, when the evaluator or fidelity proxy is too unstable
  to support a meaningful BO claim.

The experiment must never turn a promising search observation into an
improvement claim without a separately frozen, paired confirmation.

## 2. Scope and exclusions

Authorized work for this campaign is:

- export the exact LOBSTER training and validation collections to the strict
  PyG interchange format;
- train and freeze LOBSTER-specific GraphCL-GIN encoders using training graphs
  only;
- implement GraphCL-F1PR evaluation and distributed BO integration;
- run bounded mock, timing, anchor, search, and validation-confirmation studies;
- use only a newly qualified homogeneous GPU slot set;
- collect, audit, freeze, restore, test, document, commit, and push each gate.

Excluded unless the user later gives a separate explicit authorization:

- held-out/test evaluation;
- encoder training on validation or test graphs;
- QM9 BO;
- post-confirmation reranking or alternate-candidate fallback;
- changing or appending trials to any completed Random-GIN study;
- claiming that an old adaptively sampled Random-GIN study optimized GraphCL;
- unrestricted production studies.

## 3. Immutable dataset contract

The source cache is:

```text
cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_loaderseed-0_bfs-legacy_first_component_features-lobster-optimal_v2.pkl
```

Its immutable properties are:

- byte length: `59,295,793`;
- SHA-256: `928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`;
- file mode after deployment: `0444`;
- split: 70 training, 10 validation, 20 held-out/test graphs;
- split seed: `123`;
- loader seed: `0`;
- BFS: `legacy_first_component`;
- LOBSTER feature schema: `optimal_v2`;
- node feature dimension: `14`;
- edge feature dimension: `11`.

GraphCL encoder training may read only the 70 training graphs. BO and
confirmation evaluation may read only the 10 validation graphs. The 20 test
graphs must not be exported, loaded, hashed as evaluator input, or mentioned in
an executable evaluation plan during this campaign.

The PyG feature identity is frozen as:

```text
lobster-optimal_v2|export=decoded_node_edge
```

Every generated and reference collection must contain adjacency, grouped
one-hot node attributes, and grouped one-hot edge attributes with identical
schema hashes.

## 4. GraphCL evaluator contract

### 4.1 Upstream source

The external evaluator checkout remains outside the repository and must be a
clean detached checkout at:

```text
fb6bc26237eb21d7617fd41b22b4bb26ab29bf95
```

The campaign must fail closed if the revision differs, the checkout is dirty,
or required runtime dependencies are absent. The external source, dependency
bundle, and trained binary checkpoints are never committed.

### 4.2 Encoder ensemble

Train five independent GraphCL-GIN encoders with:

- encoder: `graphcl`;
- architecture: upstream GIN;
- feature mode: `decoded_node_edge`;
- seeds: `101`, `202`, `303`, `404`, `505`;
- epochs: `100`;
- layers: `3`;
- hidden dimension: `32`;
- initialization: `orthogonal`;
- released Lipschitz limiter: enabled, factor `1.0`;
- training collection: exact 70-graph PyG training artifact;
- training device class: the campaign-qualified homogeneous GTX TITAN X slot
  class on `cs-cl-09`.

The five checkpoints, training JSON files, stdout/stderr logs, environment
fingerprint, source revision, input collection digest, sizes, and SHA-256
digests form one immutable evaluator bundle. No encoder is retrained or replaced
after any candidate metric is observed.

### 4.3 Candidate evaluation

For each GraphVAE checkpoint:

1. generate exactly 10 accepted graphs from a frozen generation seed;
2. decode adjacency, node features, and edge features from the same latent draw;
3. export strict PyG generated graphs with the frozen LOBSTER schema;
4. compare them with the exact 10-graph validation reference collection;
5. compute PRDC precision, recall, and F1-PR independently in each frozen
   GraphCL-GIN embedding space;
6. aggregate across all five encoder checkpoints.

The primary objective artifact is the finite value at:

```text
summary.f1_pr.mean
```

The artifact must additionally assert:

```text
engine = contrastive-pyg-upstream
encoder = graphcl
feature_mode = decoded_node_edge
checkpoint_count = 5
reference split = validation
test_access = false
```

Higher is better. Precision, recall, FID, density, coverage, MMD-RBF, MMD-linear,
and per-checkpoint values are retained as diagnostics but cannot silently
replace the primary objective.

## 5. Determinism and uncertainty

There are three distinct sources of variability and none may be mislabeled:

- GraphCL encoder seed: represented by the five frozen checkpoints;
- GraphVAE training seed: represented by matched candidate/uniform training
  pairs;
- graph generation seed: fixed within a comparison and changed only according
  to a predeclared robustness plan.

The mean over encoder checkpoints reduces evaluator dependence on one learned
representation. It does not remove GraphVAE training variance or uncertainty
caused by only ten validation graphs.

Qualification must record per-encoder F1-PR, ensemble mean, population standard
deviation, range, and coefficient of variation. A metric is not accepted for
BO merely because it is finite.

## 6. Study identities and artifact boundaries

Every stage uses a fresh PostgreSQL study name and a fresh controller/output
root. Suggested prefixes are:

```text
lobster_graphcl_f1pr_mock_
lobster_graphcl_f1pr_timing_
lobster_graphcl_f1pr_anchor_
lobster_graphcl_f1pr_search_
lobster_graphcl_f1pr_confirmation_
```

Use the committed dedicated GraphCL-F1PR repository mapping. Do not reuse Gate
4, Gate 5, Gate 6, AIDS, or previous LOBSTER production roots. Credentials remain
outside repository/source/cache/artifact roots with directories mode `0700` and
files mode `0600`. PostgreSQL requires `PGPASSFILE`, the protected CA, and
`sslmode=verify-full`.

Immutable reservation counts, failure consumption, heartbeat/grace behavior,
bounded concurrency, deterministic dispatch seeds, atomic artifact publication,
portable restoration, and no-replacement rules remain inherited from the
distributed Attr-F1PR controller.

## 7. Gate 0 — roadmap and prerequisite audit

Tasks:

1. freeze this roadmap and the restart prompt;
2. verify repository cleanliness and source revision;
3. verify the cache mode, byte length, and digest;
4. verify the clean pinned contrastive upstream;
5. inventory local and remote Python, PyTorch, PyG, PyGCL, scatter/sparse, CUDA,
   GPU, disk, and dependency-bundle fingerprints;
6. identify dedicated host roots and exact physical/logical slots;
7. prove no LOBSTER GraphCL checkpoint is being reused from an incompatible
   split or schema.

Exit condition: all inputs and missing components are documented. Commit and
push before implementation.

## 8. Gate 1 — exact split export

Implement a cache-backed exporter rather than regenerating a new split from raw
data. It must:

- accept the exact frozen cache and its expected SHA-256;
- export only `real_train_graphs.pt` and `real_validation_graphs.pt` by default;
- require a separate explicit flag for any test export, which is forbidden in
  this campaign;
- normalize graphs through the existing strict PyG contract;
- preserve grouped one-hot node and edge channels;
- write atomic tensor-only artifacts and JSON manifests;
- record split fingerprints, collection digests, graph counts, dimensions,
  source cache digest, and `test_access=false`;
- verify that the training and validation graph-fingerprint sets are disjoint;
- verify source cache mode/hash before and after export.

Tests must cover missing attributes, wrong dimensions, schema mismatch,
train/validation overlap, cache mutation, attempted test export, unsafe pickle
fallback, and deterministic repeated export.

Exit condition: 70 training and 10 validation PyG graphs reproduce identical
digests on repeated export. Commit and push.

## 9. Gate 2 — encoder training and freeze

Run one bounded timing seed first. If healthy, train all five predeclared seeds.
The encoder training stage must:

- consume only `real_train_graphs.pt`;
- run in isolated worker processes through `ggm_eval`;
- preserve the pinned upstream without local edits;
- record input collection and schema identities in every checkpoint;
- fail closed on mixed dimensions, wrong feature mode, or nonfinite loss;
- write a bundle manifest with checkpoint sizes and hashes;
- make the frozen bundle read-only;
- restore and verify the bundle manifest without loading validation or test.

Exit condition: five independently seeded checkpoints exist, pass integrity
checks, and are immutable. Commit only manifests and evidence, never binary
checkpoints or external dependencies.

## 10. Gate 3 — GraphCL-F1PR GraphVAE evaluator

Implement a validation evaluator that reuses the existing same-latent GraphVAE
decoder path but exports PyG collections and invokes the frozen GraphCL bundle.
It must:

- require both feature decoders;
- reject topology-control or node-only fallback;
- accept only validation during optimization;
- verify cache, schema, split, source, checkpoint bundle, and runtime hashes;
- generate exactly 10 accepted graphs or fail;
- use deterministic generation seeds;
- retain generated/reference collection digests and per-encoder metrics;
- expose exact `summary.f1_pr.mean`;
- publish artifacts atomically;
- set `test_access=false` and `skip_final_evaluation=true`.

Add adversarial parser tests for wrong encoder, wrong feature mode, wrong
checkpoint count, repeated checkpoint, changed upstream, mismatched schemas,
test reference, nonfinite values, incomplete graph counts, and tampered files.

Exit condition: local unit/integration tests and a bounded real checkpoint
evaluation pass. Commit and push.

## 11. Gate 4 — distributed controller integration

Extend the study definition with a versioned evaluator backend block containing:

- backend `graphcl_f1pr`;
- objective path `summary.f1_pr.mean`;
- encoder-bundle manifest digest;
- ordered encoder checkpoint digests and seeds;
- PyG schema and collection digests;
- validation split fingerprint;
- checkpoint count five;
- nearest-k five;
- generation seed policy;
- test access false.

Worker commands must receive protected or staged paths, never embed secrets or
unredacted storage URLs. Collection and portable snapshots must audit GraphCL
artifacts as strictly as the existing Random-GIN artifacts.

Run bounded mock tests for two and three simultaneous workers, definite
prelaunch failure, ambiguous postlaunch state, stale-worker failure, safe
process recovery, no replacement, and portable restoration.

Exit condition: the complete non-PostgreSQL distributed suite and isolated
PostgreSQL suite pass. Commit and push.

## 12. Gate 5 — evaluator and fidelity qualification

Before adaptive BO, run fixed anchors only. At minimum evaluate:

- uniform `(1,1)`;
- previous Random-GIN winner `(5.2290456720, 0.0538641483)`;
- common weak `(0.25,0.25)` and strong `(4,4)` scales;
- node emphasis `(4,0.25)`;
- edge emphasis `(0.25,4)`.

Run these at 2,000 epochs and GraphVAE seeds 0 and 1. Promote uniform, the best
anchor, and one deliberately contrasting anchor to 10,000 epochs at the same
seeds. Measure:

- encoder-ensemble dispersion;
- paired GraphVAE-seed dispersion;
- 2,000-versus-10,000 rank correlation;
- repeated-generation-seed stability on at least uniform and the best anchor;
- exact runtime per phase.

Fail the campaign with `qualification_failed` instead of running BO if any of
the following holds:

- validation/test overlap or schema mismatch;
- missing feature decoder or encoder checkpoint;
- nonfinite objective;
- coefficient of variation above `0.20` for most anchors;
- 2,000-versus-10,000 rank correlation below `0.50` or a sign reversal for the
  best-anchor versus uniform comparison;
- generation-seed variation dominates the candidate difference;
- no meaningful finite signal above the PRDC floor.

Thresholds are frozen before anchor results are viewed. Exit condition: a
written go/no-go decision and timing-based bounded search budget. Commit and
push.

## 13. Gate 6 — fresh GraphCL-F1PR BO search

Only run if Gate 5 passes. The preferred robust search unit is one candidate
weight pair evaluated at GraphVAE training seeds 0 and 1, with the arithmetic
mean of their two GraphCL-F1PR ensemble means as the Optuna value. Both
replicates belong to one reservation; failure of either consumes that
reservation and the candidate is not partially scored or replaced.

Predeclared search design:

- exactly 18 candidate reservations;
- `max_parallel=2` on the selected homogeneous GTX TITAN X slots;
- six fixed anchors followed by 12 TPE candidates;
- TPE startup target six;
- bounded synchronous waves;
- log search ranges `[0.25,4]` for node and edge weights;
- 2,000 epochs only if Gate 5 qualifies that fidelity;
- exactly 10 validation references;
- five frozen GraphCL encoder checkpoints;
- no pruning from test data and no final-test evaluation;
- one immutable winner: maximum finite candidate-mean objective after freeze;
- no alternate-candidate fallback.

If multi-seed-per-reservation support is not safely implemented, do not silently
fall back to a single seed. Stop at Gate 5 and implement the missing contract.

Exit condition: exact budget is terminal, audited, frozen, restored, and one
candidate is selected using validation only. Commit and push.

## 14. Gate 7 — matched final-fidelity confirmation

Create a different fixed-plan study after the search winner and analysis rule
are committed. Compare only:

- the single selected GraphCL-F1PR weight pair;
- uniform `(1,1)`.

Use GraphVAE training seeds 0, 1, and 2, 10,000 epochs, the same generation
seed policy, all 10 validation graphs, and the same five frozen GraphCL
encoders. Reserve exactly six trials in three fixed waves of two.

For each seed compute selected minus uniform. The primary estimate is the mean
paired difference. `improvement_confirmed` requires:

1. all three differences are positive;
2. the mean difference is positive;
3. the lower endpoint of the two-sided 95% paired t interval is above zero.

Otherwise report `no_improvement`. Do not rerank another candidate.

Exit condition: paired report, frozen/restored study, cache and encoder bundle
integrity, credential/storage-URL/test-access scans, and a truthful conclusion.
Commit and push.

## 15. Gate 8 — final audit and handoff

Run:

- all GraphCL package tests;
- all GraphCL-F1PR integration tests;
- the complete non-PostgreSQL distributed BO suite;
- the isolated PostgreSQL suite in a disposable schema;
- portable restore audits for every new study;
- lifecycle/reservation/launch reconciliation;
- source, cache, collection, checkpoint, and contract hash verification;
- exhaustive credential and unredacted-storage-URL scans;
- explicit `test_access=true` and held-out path scans;
- zero-residual-test-study/schema checks.

Update this roadmap with exact study names, commits, hashes, counts, timings,
objectives, intervals, deviations, and the final answer. Preserve every failed
attempt and consumed reservation. Never rewrite prior evidence.

## 16. Commit discipline

Each completed gate receives its own commit and push. Before every commit:

```text
git status --short --branch
git log -5 --pretty=format:%s
git diff --stat
git diff --cached --stat
```

Inspect every tracked and untracked change, use the commit-summary skill, stage
only intended files, run `git diff --cached --check`, inspect the staged diff,
then commit and push. Never commit caches, GraphVAE checkpoints, GraphCL binary
checkpoints, generated collections, run trees, credentials, dependency bundles,
passwords, or storage URLs.

## 17. Current checkpoint

At the completed prerequisite audit:

- the roadmap/restart contract is committed and pushed at `cc65b58`;
- the exact LOBSTER cache is present, mode `0444`, and hash-verified;
- the pinned contrastive upstream exists outside the repository, is clean, and
  matches revision `fb6bc26237eb21d7617fd41b22b4bb26ab29bf95`;
- the local controller Python has PyTorch `2.1.2` and PyG but does not contain
  PyGCL/scatter/sparse in its base environment;
- `cs-cl-13` is rejected because its intended scratch filesystem has only about
  6 GiB free; `cs-cl-17` is not selected because it lacks the prior isolated
  GraphCL deployment and has only about 39 GiB free;
- `cs-cl-09` is selected with a new dedicated root on `/local-scratch2`, about
  952 GiB free, an existing clean pinned upstream/runtime, an already-deployed
  protected LOBSTER worker bundle, and two homogeneous 12,288 MiB GTX TITAN X
  slots; physical GPU 0 was partially occupied at audit and must be idle before
  any two-worker wave;
- candidate concurrency is therefore two, not the provisional three, and a
  fresh fixed-parameter hardware/timing qualification is mandatory before BO;
- no bundled LOBSTER GraphCL checkpoint exists;
- the generic real-split exporter and GraphCL runner exist, but an exact
  cache-backed, test-disabled export and BO objective integration remain to be
  implemented;
- no GraphCL-F1PR LOBSTER trial, encoder training, or held-out evaluation has
  started.

Gate 1 is complete locally. The hash-bound exporter emitted only the 70 training
and 10 validation graphs with feature identity
`lobster-optimal_v2|export=decoded_node_edge`, exact dimensions 14/11, zero
rejected graph, and zero training/validation fingerprint overlap. Repeated
exports reproduce collection digests
`8de6ccf86bb2ae994f0a7401217d57a814d5e71c6e49732e345ae2b242f569e4`
and `0a5ad40ab717440f1739f0b203df3df253a6318089202aa467dd4fc6ee5c1832`.
The published input files and manifests are mode `0444`; the source cache is
unchanged. Seven focused tests pass, including a guard that rejects test export
before cache access and a hash guard that precedes pickle loading. Generated
PyG inputs remain ignored runtime artifacts and are not committed.

Gate 2 is complete on the dedicated `cs-cl-09` root. Source commit `21bb0fb`
and its deployment manifest verified; protected PostgreSQL authentication used
`sslmode=verify-full`; physical GPU 1 was isolated as logical `cuda:0`. The
first launch is preserved as a pre-training failure because the inventoried
dependency directory contained only orphaned bytecode and could not export
`GCL.models.DualBranchContrast`. A complete existing PyGCL `0.1.2`, scatter
`2.1.2+pt21cu121`, and sparse `0.6.18+pt21cu121` bundle was checksum-staged into
the dedicated root; its 125-file dependency identity and exact model imports
then qualified.

The five 100-epoch seeds `101/202/303/404/505` completed in 9.63--10.74 seconds
each with finite losses. Their exact checkpoint hashes are recorded in
`lobster_graphcl_f1pr_encoder_qualification.json`. The train-only bundle is
mode-frozen, independently reopened, and has semantic SHA-256
`4cf22a9b204b3638ce6f63fc691ff9556986af1334b64b0994411f5bfd7ac8be`
under GraphCL runtime fingerprint
`980433997b5ae2df27e8be37a639d27c9b28670e0dcef04b81475d95cb44d4b7`.
The cache remained unchanged and credential/storage-URL/test-access scans pass.

Gate 3 is complete. Commit `8aad9e9` adds a validation-only evaluator that
reuses the same-latent GraphVAE adjacency/node/edge decode, exports exactly ten
strict PyG candidates, and aggregates the five frozen GraphCL encoders. Its
parser rejects wrong engine/mode/count, test access, missing decoders, repeated
or tampered checkpoints, changed source/runtime/schema, nonfinite metrics, and
inconsistent aggregation. Twenty-nine focused GraphCL campaign tests pass.

The bounded real integration used the already-frozen 2,000-epoch uniform
`(1,1)` seed-0 checkpoint from the prior Random-GIN study; it did not train,
reserve, search, or select a new model. At generation seed `123`, the ten-graph
validation-only GraphCL objective at `summary.f1_pr.mean` was
`0.6968191868090722` (encoder population standard deviation
`0.08772715594442737`, coefficient of variation `0.12589658494645403`). This is
technical evaluator qualification, not evidence that BO improves uniform
weights. The exact per-encoder metrics and hashes are frozen in
`lobster_graphcl_f1pr_evaluator_qualification.json`; the immutable 20-file
runtime tree has digest
`49445a987c05184efcfe557de0edb87095ee4ab05cff96925faa3c4daa8bd4c9`.

Gate 4 implementation is now complete locally, pending its bounded PostgreSQL
qualification. The immutable backend contract binds the exact validation
reference, five ordered encoder seeds/checkpoint digests, bundle/runtime/source
identities, two GraphVAE training seeds, nearest-k five, and both objective
views. One Optuna reservation now owns sequential GraphVAE seeds 0 and 1; it
publishes only their arithmetic mean after both GraphCL artifacts pass strict
portable audit. Either replicate failure consumes the reservation, preserves
both completed and failed attempt evidence, and cannot publish a partial score.
The legacy Random-GIN backend remains the default and its tests remain green.

Eighty-four focused backend and distributed tests pass, including successful
two-seed grouping, tampered-objective rejection, and a deliberate seed-1
failure with no partial score. The next action is to commit and deploy this
implementation, verify the real frozen inputs on `cs-cl-09`, then run the fresh
two-worker and three-worker bounded mock PostgreSQL studies plus the required
failure/recovery/restore qualifications before any fixed real anchors.
The dedicated mock configuration is separate from the legacy eight-graph smoke
contract and freezes exactly ten validation graphs; its focused configuration
and backend suite passes 11 tests.
Prelaunch review also separates the lifecycle mock cache fingerprint from the
real frozen GraphCL validation-split fingerprint. The worker and auditor now
bind each to its correct contract field; 43 focused tests pass with deliberately
different synthetic-cache and real-evaluator split identities.

Two definite prelaunch attempts are preserved. Study name
`lobster_graphcl_f1pr_mock2_20260825a` was created before definition assembly
failed because the rsync root has no `.git`; a direct read-only Optuna probe
verified zero trials and zero study attributes. Replacement name
`lobster_graphcl_f1pr_mock2_20260825b` initialized two WAITING reservations, but
dispatch stopped before any launch because the embedded deployment/cache
manifests had not been published as launcher staging files. The supported
preclaim interface retired it with both reservations WAITING and
`reservation_consumed=false`. Controller initialization now atomically
publishes the exact source and cache mappings already bound into its immutable
definition and rejects differing pre-existing files. Forty-two focused
launcher/backend/integrity tests pass. A fresh name must be used after deploy.

Fresh study `lobster_graphcl_f1pr_mock2_20260825c` reserved exactly two slots.
Its first wave remained PLANNED because `cs-cl-09` had not yet accepted its own
SSH host key; PostgreSQL showed two WAITING slots, and the recorded launch probe
proved both attempts `DEFINITE_PRELAUNCH` and `retry_safe=true` with no DB trial
claim. After host-key staging, wave 2 launched both GPU-isolated mock workers
with distinct dispatch sequences and sampler seeds. Both reservations reached
COMPLETE, each with GraphVAE seeds 0 and 1, and collection succeeded.

The first finalize attempt audited the grouped trial artifacts but stopped
before freezing because legacy aggregate generation expected one root-level
resolved config. Grouped final output now emits a candidate descriptor with
both seed configs, both replicate artifact records, the arithmetic-mean
objective, and both objective paths; it never treats one seed as canonical.
Sixty-nine focused backend/controller/launcher tests pass. The existing study
remains READY and terminal for a post-deploy finalize retry; no reservation may
be added or rerun.

That same terminal study was finalized after deploying `ba6d2cd`; no worker was
rerun. It is FROZEN with two COMPLETE reservations, zero FAIL/WAITING/RUNNING or
unreserved rows, and both worker completion markers. Offline restore initially
failed under an unsourced shell because its runtime identity differed, then
passed under the same protected environment used at initialization; it did not
access PostgreSQL. The restored aggregate files match byte-for-byte, snapshot
SHA-256 is
`a4c94e4f81bb7fe9bf8b99e9f5f103dc12f987f9be17e22964ff2cf4d85cb4ef`,
and semantic fingerprint is
`074ced681b66ee91c212312c9ca0ef6866c2222eede1dca88390491dd0a18724`.
Cache, validation reference, and encoder manifest hashes remain exact; storage
URL, credential-assignment, and `test_access=true` scans are clean. Full details
are frozen in `lobster_graphcl_f1pr_mock2_qualification.json`. Mock objectives
are synthetic lifecycle evidence and have no scientific interpretation.

The next action is a separate fresh three-worker lifecycle mock. It may use
three CPU/mock processes on `cs-cl-09` to qualify controller/database
parallelism, but it does not expand real GPU concurrency beyond the two frozen
physical slots. Then complete the remaining failure/recovery/restore cases
before any real fixed anchor.

The three-worker launcher support is now implemented and locally qualified.
Slot files may use the literal `mock-cpu` only when the immutable study
definition has `training.mock=true`; both preflight and dispatch fail closed if
such a slot is presented to any real study. CPU/mock commands omit both
`CUDA_VISIBLE_DEVICES` and `--physical-gpu` and explicitly select `--device
cpu`. Multiple lifecycle-only worker identities may therefore share
`cs-cl-09` without claiming nonexistent GPU capacity. The legacy physical-GPU
slot representation is unchanged, and 72 focused unit, launcher, and grouped
GraphCL backend tests pass. The next action is to commit and deploy this
support, then initialize and run a fresh exact-three-reservation mock study.

Study `lobster_graphcl_f1pr_mock3_20260825a` is preserved as a complete but
non-qualifying concurrency attempt. Contract
`ed8c6de2cc483c8b38fb7b8b55cebe2ab1fe80fd8decff2ae0c9cf047db68198`
reserved exactly three trials at `max_parallel=3`; one launch wave recorded
three distinct worker-run identities, dispatch sequences, deterministic seeds,
CPU devices, and SSH acknowledgements. All three reservations completed with
no failed, waiting, running, or unreserved row, and strict finalize plus offline
restore passed. The snapshot SHA-256 is
`2b740fa5464730bc8b6bbc28bce9ff6abe902a7647c6cae6a44cb1158a977d66`
and restored semantic fingerprint is
`e9c41d7cf4f41188f6f5f9e0289cdc79801a000c885265fb3e6102143ba056c8`.
The cache remained mode `0444` with its frozen hash, the reference and encoder
manifest hashes remained exact, and URL, credential-assignment, password, and
`test_access=true` scans each found zero files.

This attempt does not satisfy the concurrency exit condition: its grouped
trial intervals were `[1787678609.313, 1787678609.611]`,
`[1787678609.619, 1787678609.992]`, and
`[1787678610.032, 1787678610.397]`, so the ultra-short mock bodies ran
sequentially despite simultaneous launch acknowledgement. No consumed
reservation will be rerun or replaced. A bounded mock-only hold is now
implemented as an immutable training-contract field. It is limited to 0--30
seconds, forbidden for real studies, and occurs only after a mock reservation
has been claimed. Eighty-one focused tests pass, including hold execution and
the real-study fail-closed guard. The next action is to commit and deploy this
change, then use a new study name with a two-second hold per grouped replicate
to produce an auditable three-way overlap.
This section must be updated after every commit so a resumed agent never guesses
the campaign state.
