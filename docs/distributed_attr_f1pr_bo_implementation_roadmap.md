# Distributed GraphVAE Attr-F1PR Bayesian Optimization Roadmap

- Status: implementation specification
- Audience: the Codex agent implementing distributed Bayesian optimization in this repository
- Repository: `mirzaeisfu/graphvae_req` / GraphVAE-REQ
- Objective: maximize validation `Attr-F1PR`

## 1. Executive decision

Implement native multi-node Optuna backed by one network-reachable PostgreSQL
database. Add a thin controller for cluster preflight, bounded GPU dispatch,
artifact collection, exact budget enforcement, status, and finalization. Do not
implement a replacement Bayesian optimizer or a custom ask/tell trial-state
machine.

Installing Optuna on every worker changes the earlier recommendation because it
makes this design feasible, but installing Optuna alone is not sufficient. Every
worker must also have:

- the same pinned Optuna version and PostgreSQL client driver;
- network access to the same persistent PostgreSQL service;
- the same deployed source and dependency fingerprint;
- an identical, prebuilt dataset cache and feature-schema manifest;
- a host-local artifact directory that can be collected by the controller.

PostgreSQL is a hard prerequisite for distributed mode. Distributed mode must
reject SQLite, JournalStorage on NFS, copied databases, and independent per-host
studies. The existing local SQLite mode remains supported for serial smoke tests
and single-machine studies.

Native Optuna is the safer component for transactional trial allocation, unique
trial numbers, state transitions, concurrent sampling, and heartbeat updates.
It does not manage this repository's SSH/GPU placement, dataset cache, artifact
transfer, total-cost limit, aggregate reports, or held-out-test policy. The thin
controller owns only those responsibilities.

## 2. Answers to the design questions

### Does installing Optuna on every machine change the plan?

Yes, conditionally. It makes native multi-node Optuna preferable when a real
client/server database is available. It does not make a shared SQLite file safe
and does not eliminate cluster orchestration or artifact collection.

The current GraphVAE `micro` environment is Python 3.8. The latest Optuna 4.9
requires Python 3.9 or newer, while Optuna 4.2.1 supports Python 3.8. The initial
compatibility target is therefore `optuna==4.2.1`, not an unconstrained
`optuna>=3.6,<5`. Upgrading the complete GraphVAE environment to a newer Python
is a separate project and is not part of this roadmap.

### Must one worker be tested before several workers?

Yes. The one-worker qualification must use the final PostgreSQL storage, worker
command, remote repository, cache, GPU isolation, and artifact collection path.
It is a short qualification, not a full study. It isolates environment, decoder,
checkpoint, evaluator, path, and cache failures before concurrency makes them
harder to diagnose. It must be followed by a two-worker test because one worker
does not test locking, concurrent sampling, duplicate prevention, or collection
collisions.

### Is native multi-node Optuna safer than new orchestration code?

It is safer for the study state machine and trial allocation, so those parts
must remain native. It is not a complete job scheduler. A small amount of new
code is still required, but it must use public Optuna APIs and avoid modifying
Optuna's schema or recreating its concurrency mechanisms.

## 3. Non-negotiable scientific contract

The optimization target remains exactly:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

Its public name is `Attr-F1PR`. It is the mean validation `f1_pr` from the
attributed Random-GIN evaluator in `decoded_node_edge` mode. Generated adjacency,
node attributes, and edge attributes must come from GraphVAE's actual adjacency,
node-feature, and edge-feature decoders.

Distributed execution must preserve all existing objective safeguards:

- optimization reads validation data only;
- it never reads a test metric or selects a checkpoint using test data;
- topology-only `stat_rnn.py` F1-PR is never accepted as the target;
- generated node and edge attributes are used exactly as decoded;
- attributes are not replaced by degree or topology-derived features;
- generated labels are not repaired with reference attributes;
- no endpoint-validity post-processing is introduced;
- node and edge feature dimensions and channel meanings must match the cache;
- checkpoints without both feature decoder heads fail clearly;
- the number of accepted validation graphs is constant across trials;
- only `alpha_node_feat` and `alpha_edge_feat` are tuned by default;
- `alpha_motif_loss` is tuned only with the existing explicit opt-in;
- topology reconstruction, KL, training budget, split, and other settings stay
  fixed.

## 4. Reproducibility boundary

Two different guarantees must be named separately:

1. **Trial reproducibility:** fixed parameters, dataset/split, source,
   environment, training seed, generation seed, evaluator seeds, and hardware
   class should reproduce the same trial within an established numeric
   tolerance. This is required.
2. **Study-path reproducibility:** an identical sequence of TPE proposals is not
   guaranteed with parallel workers because trial claim and completion timing
   changes the history observed by each sampler. Optuna documents this as an
   inherent property of distributed optimization. Exact proposal replay requires
   `--max-parallel 1`.

Distributed mode must use `TPESampler(constant_liar=True)` to reduce suggestions
near running trials. Every one-trial worker process receives a distinct,
controller-generated sampler seed derived from the study seed and immutable
dispatch sequence. The sampler version, settings, seed, dispatch sequence, and
completion order are recorded.

Derive that seed exactly as follows, interpreting the first four digest bytes as
an unsigned big-endian integer. A pretrial relaunch reuses the same dispatch
sequence and therefore the same seed:

```python
material = f"graphvae-attr-f1pr-sampler-v1\0{study_seed}\0{dispatch_sequence}"
worker_sampler_seed = int.from_bytes(
    hashlib.sha256(material.encode("utf-8")).digest()[:4], "big"
)
```

Before mixing GPU models, run one fixed-parameter reproducibility pair on each
candidate GPU class. If Attr-F1PR or training outputs differ beyond the accepted
tolerance, use a homogeneous GPU subset for a study. Based on the current but
dated inventory, the three TITAN RTX slots on `cs-cl-13` and `cs-cl-17` are the
first production candidates; all GPUs must be re-probed before use.

## 5. Budget and failure semantics

Define `--trials N` as exactly `N` reserved scientific trial slots and at most
`N` expensive training starts. Each reserved slot may become `COMPLETE` or
`FAIL`; both consume one slot once Optuna changes it to `RUNNING`. This keeps the
GPU-cost ceiling bounded. `--trials` does not mean "continue until N successes,"
because repeated failures could make that cost unbounded.

The controller initializes the study and reserves exactly `N` empty `WAITING`
trials through the public `Study.enqueue_trial` API:

```python
study.enqueue_trial(
    {},
    user_attrs={
        "graphvae_bo_reserved": True,
        "budget_index": index,
        "study_contract_sha256": contract_sha256,
    },
)
```

An empty fixed-parameter mapping lets the worker's TPE sampler choose the loss
weights when the reservation is claimed. This behavior must pass the PostgreSQL
concurrency test before any real training is allowed.

Reservation initialization must itself be crash-safe. The initializer first
sets the study lifecycle to `INITIALIZING`, creates uniquely numbered
`budget_index` reservations, and changes the lifecycle to `READY` only after it
can prove that every index in `[0, N)` occurs exactly once. If initialization is
interrupted, a rerun inspects the existing indexes and creates only missing
ones. Duplicate, out-of-range, or unmarked trials fail initialization. Workers
must refuse a study whose lifecycle is not `READY`.

Workers run exactly one `study.optimize(..., n_trials=1)` call and then exit.
At objective entry they must reject any trial without the reservation marker or
with the wrong contract hash before starting training. This makes an accidentally
extra worker fail cheaply instead of launching an unbudgeted training run.

Optuna may create one extra unreserved database row before that objective guard
can reject an oversubscribed worker; public `study.optimize` has no
"claim WAITING or create nothing" operation. Such a row must contain no sampled
parameters, must be marked `FAIL` with `unreserved_guard=true`, and must never
affect TPE, the reserved budget, best selection, or aggregate scientific results.
It is included in an audit section of `trials.csv` and `SUMMARY.md`. The safety
guarantee is exactly `N` reserved slots and at most `N` expensive trainings, not
exactly `N` total PostgreSQL rows.

Version 1 does not automatically retry a trial after it reaches `RUNNING`:

- a scientific/model/evaluator failure is `FAIL` and consumes its reserved slot;
- a killed worker becomes `FAIL` after heartbeat expiry and consumes its slot;
- an SSH or preflight failure before a trial is claimed may be retried safely;
- an ambiguous launch is not retried until the original tmux session and worker
  marker are reconciled;
- replacing failed trial slots requires a separate, explicit future study
  extension and must never occur silently.

`MaxTrialsCallback` may be retained only as a defensive stop. The pre-created
reservations and one-trial worker process are the primary exact-budget controls.

## 6. Target architecture

```text
                           private cluster network

  Controller                                          PostgreSQL
  ---------------------------                          ----------
  initialize/freeze study  --------------------------> Optuna RDB
  reserve N WAITING trials                             trial states
  preflight and dispatch                               heartbeats
  poll/collect/finalize                                study contract
       |                                                     ^
       | SSH/tmux one-trial jobs                             |
       v                                                     |
  +------------------+  +------------------+  +------------------+
  | worker/GPU slot  |  | worker/GPU slot  |  | worker/GPU slot  |
  | study.optimize(1)|  | study.optimize(1)|  | study.optimize(1)|
  | local artifacts  |  | local artifacts  |  | local artifacts  |
  +------------------+  +------------------+  +------------------+
       |                       |                       |
       +-----------------------+-----------------------+
                               |
                       staged rsync + hashes
                               v
                     controller canonical output
```

Only the initializer may create the study contract and reservations. Workers may
only validate the contract and consume one reservation. Only the finalizer may
write `trials.csv`, `best_config.yaml`, `best_trial.json`, `SUMMARY.md`, the
portable study snapshot, or the frozen-study marker.

## 7. PostgreSQL and dependency contract

### PostgreSQL prerequisites

The operator must provide one persistent PostgreSQL service that:

- is reachable from every approved worker and the controller;
- uses a dedicated database and non-superuser role with only the privileges
  required inside that database;
- is restricted by firewall to approved cluster hosts;
- uses certificate and hostname verification (`sslmode=verify-full` with a
  trusted CA) by default; `sslmode=require` is allowed only as a documented
  infrastructure exception because it encrypts without authenticating the
  server;
- is backed up and has enough retention to recover the study;
- has synchronized host clocks.

The implementation must not install or administer PostgreSQL automatically.
That is an infrastructure action requiring separate authorization.

Prefer libpq's protected password file so the storage URL itself contains no
password:

```text
GRAPHVAE_BO_STORAGE_URL=postgresql+psycopg2://USER@HOST:5432/DATABASE?sslmode=verify-full
PGPASSFILE=/absolute/path/to/graphvae_bo.pgpass
```

`PGPASSFILE` and its parent directory must be accessible only to the account
running the workers, and the password file must be mode `0600`. If an operator
instead supplies a credential-bearing URL through the environment, it receives
the same protection and redaction requirements. Neither a password nor an
unredacted credential-bearing URL may appear in command-line arguments, YAML,
logs, manifests, Optuna attributes, JSON/CSV/Markdown output, shell history, or
Git. Remote environment files must be outside the repository and mode `0600`.
Diagnostics may report only a redacted dialect/host/database identity.

### Python dependency target

Add a BO-specific constraints file. Initial pins for the existing Python 3.8
environment are:

```text
optuna==4.2.1
psycopg2-binary==2.9.10
```

Resolve and pin the compatible SQLAlchemy and Alembic versions after the first
clean-environment import/connect test. Every worker must report an identical
dependency fingerprint. Do not let the current broad `optuna>=3.6,<5` select
different versions on different machines.

Construct `RDBStorage` with configurable defaults:

- `heartbeat_interval=60` seconds;
- `grace_period=600` seconds;
- SQLAlchemy `pool_pre_ping=True`;
- a finite connection timeout;
- no automatic stale-trial retry callback in version 1.

Optuna 4.2.1 names its stale callback argument `failed_trial_callback`. Do not
copy the renamed 4.9 API without a version guard. Use `study.optimize`, not
controller ask/tell, so native heartbeat recording remains active.

Official references:

- <https://optuna.readthedocs.io/en/v4.2.0/tutorial/10_key_features/004_distributed.html>
- <https://optuna.readthedocs.io/en/v4.2.0/reference/generated/optuna.storages.RDBStorage.html>
- <https://optuna.readthedocs.io/en/v4.2.0/reference/generated/optuna.samplers.TPESampler.html>
- <https://optuna.readthedocs.io/en/v4.2.0/faq.html>
- <https://pypi.org/project/optuna/4.2.1/>

## 8. Study definition and invariants

Initialization creates one canonical JSON object, serializes it with sorted keys
and stable separators, and hashes it with SHA-256. Store both the object and hash
in PostgreSQL study user attributes and write the same object atomically to the
controller output directory.

The definition must include at least:

- schema version;
- study name and generated study UUID;
- objective name and exact JSON path;
- direction `maximize`;
- primary evaluator mode `decoded_node_edge`;
- optimization split `validation` and test access `false`;
- base YAML content hash and resolved fixed configuration;
- search-space names, bounds, log flags, and motif opt-in;
- exact reserved-trial count;
- split, training, generation, evaluator, and sampler seeds;
- evaluator repeat seeds and repeat count;
- accepted validation graph count or fixed `max_graphs`;
- adjacency threshold, batch size, and nearest-k;
- training epoch/budget and subprocess timeouts;
- source commit plus deployment-tree SHA-256;
- Python and dependency fingerprint;
- Optuna and DB-driver versions;
- dataset-cache file SHA-256;
- split fingerprint;
- node- and edge-feature schema fingerprints, including group/channel meanings;
- approved hardware policy and numeric tolerance;
- heartbeat/grace settings;
- scheduler mode and maximum concurrency.

The study lifecycle (`INITIALIZING`, `READY`, or `FROZEN`) and controller
identity are stored separately as PostgreSQL study attributes so they can be
updated without changing the immutable scientific contract.

Workers compare their local values with this definition before connecting to a
trial. A mismatch must create no trial and start no training. Resumption with a
different definition must fail and require a new study name.

## 9. Dataset and source staging

One controller-side process creates the canonical dataset cache before workers
are launched. Workers must never concurrently create or replace it.

Add a cache manifest containing:

- relative cache filename and byte length;
- SHA-256 of the complete cache file;
- cache metadata already validated by `main.py`;
- split mode, seed, fractions, and graph counts;
- deterministic train/validation/test graph-identity fingerprints;
- node feature dimension, groups, and channel meanings;
- edge feature dimension, groups, and channel meanings;
- expected number of validation graphs passed to the evaluator.

Rsync the study-specific cache and required raw inputs to every worker using
checksums. Compute the manifest again remotely, compare it exactly, and make the
staged cache read-only. Add a distributed `require_existing_dataset_cache` guard
so a missing cache fails before an Optuna reservation is claimed instead of
regenerating it.

The existing code distributor excludes `.git`. Therefore create a deployment
manifest before rsync containing the controller Git commit, clean-worktree
status, and a deterministic hash of the deployed source/config files. Workers
recompute the tree hash after rsync. The controller must refuse to distribute a
dirty source tree, consistent with the existing cluster tooling.

Fingerprint algorithms must be implemented once in a shared module and used by
the cache preparer, worker, evaluator, collector, and tests:

- frame every hashed field with a domain tag and byte length to prevent
  concatenation ambiguity;
- deployment fingerprint: enumerate `git ls-files -z`, sort normalized POSIX
  relative paths by UTF-8 bytes, reject symlinks, and hash path, executable bit,
  byte length, and content; data, caches, runs, and `.git` are represented by
  their separate manifests rather than included;
- array fingerprint: encode a fixed dtype name, shape, little-endian C-contiguous
  bytes, and byte length; normalize sparse data by lexicographically sorting
  indices before encoding;
- graph fingerprint: preserve canonical cached dataset index order and hash the
  normalized adjacency, node attributes, edge attributes, and relation/channel
  axes for each graph; hash each split as a length-prefixed sequence of those
  per-graph hashes;
- feature-schema fingerprint: hash canonical JSON (`sort_keys=True`, compact
  separators, UTF-8) containing ordered group names, channel ranges, label or
  relation meanings, encoding, dtype, and total dimension.

The shared helper must include a schema/version tag so any future serialization
change forces a new study contract instead of silently changing hashes.

## 10. Artifact and metadata contract

Each worker writes only beneath its host-local study root. PostgreSQL trial
numbers are globally unique within the study, so the canonical trial path is:

```text
trials/trial_00000/
```

Paths stored in Optuna attributes and portable outputs must be relative to the
study root. Host-local absolute paths may be recorded as provenance but may not
be the only artifact locator.

Required worker metadata includes:

- trial number and reservation budget index;
- sampled parameters;
- study contract hash;
- worker run ID, hostname, physical GPU, logical CUDA device, GPU model/VRAM;
- source/deployment, environment, cache, split, and feature-schema hashes;
- all training/generation/evaluator seeds;
- resolved YAML;
- phase start/end timestamps and elapsed durations;
- checkpoint path and SHA-256;
- evaluator output path and SHA-256;
- validation Attr-F1PR, precision, recall, and accepted graph count;
- failure phase, exception type/message, traceback, and subprocess exit status;
- sampler seed and observed Optuna/driver versions.

Use atomic temp-file-plus-rename writes for JSON, YAML, CSV, summaries, and
completion markers. Never parse console text for the objective or worker state.

The attributed evaluator's structured JSON must be extended to report the cache
hash, split fingerprint, node/edge schema fingerprints, actual decoder output
dimensions, generation/evaluator seeds, repeat count, and separate generated and
reference accepted graph counts. The required count is:

```text
expected_count = validation_cache_count                         if max_graphs == 0
expected_count = min(max_graphs, validation_cache_count)        otherwise
```

Both generated and reference accepted counts must equal `expected_count`; a
single aggregate `accepted_per_collection` field is not sufficient for the
distributed integrity check.

Canonical controller output:

```text
runs/bayesian_optimization/<study>/
├── study_definition.json
├── deployment_manifest.json
├── dataset_cache_manifest.json
├── launch_manifests/
│   └── wave_0001.json
├── workers/
│   └── <worker-run-id>/
│       ├── RUN_INFO.json
│       ├── stdout.log
│       ├── exit_status.txt
│       └── COMPLETED | FAILED_PRETRIAL
├── trials/
│   └── trial_00000/
│       ├── resolved_config.yaml
│       ├── trial_result.json
│       ├── training_subprocess.log
│       ├── evaluation_subprocess.log
│       ├── training/seed_0/...
│       └── validation_evaluation/attributed_random_gin.json
├── trials.csv
├── best_config.yaml
├── best_trial.json
├── SUMMARY.md
├── FROZEN.json
├── study_snapshot.sqlite3
└── final_test/                 # created only by explicit post-selection command
```

Rsync into a controller staging directory. Verify every expected hash and the
trial/result/parameter/contract relationship before atomically promoting the
trial directory. A destination collision with different content is fatal. An
identical collision is an idempotent re-collection.

The portable database deliverable is specifically `study_snapshot.sqlite3`,
created with the public `optuna.copy_study` API after freeze; it is not a live
distributed store or a `pg_dump`. Copy to a new temporary SQLite file, reopen it,
compare study/trial semantic fingerprints with PostgreSQL, fsync it, then rename
it atomically. Re-finalization first validates an existing snapshot semantically
and leaves it unchanged when it matches. PostgreSQL administrative backup remains
an operator responsibility outside this repository workflow.

## 11. Process and failure safety

Replace the current blocking subprocess helper with a process-group-aware
runner for distributed workers:

- start training/evaluation in a new process session;
- write PID, process-group ID, command identity, cwd, and start time;
- forward SIGINT/SIGTERM to the exact process group;
- on timeout, send TERM, wait a configurable interval, then send KILL;
- never use a broad `pkill` or kill by partial command name;
- on recovery, verify PID start time, command, cwd, study, and worker-run ID
  before killing a possible orphan, preventing PID-reuse mistakes.

The outer worker supervisor—not a `tee` pipeline—holds the per-GPU `flock` and
updates an atomic repository worker-heartbeat marker every 30 seconds. A released
OS lock is not by itself proof that the GPU is safe: the controller's slot ledger
stays occupied until the exact recorded child process group is absent and the DB
and marker states are reconciled.

Worker and controller states are structural, not inferred from logs:

```text
preflight -> launched -> trial_claimed -> training -> evaluation
          -> terminal_db_state -> artifacts_ready -> collected -> audited
```

The worker wrapper writes `COMPLETED` only after `study.optimize(n_trials=1)`
returns and the claimed trial is terminal in PostgreSQL. A handled scientific
failure is a terminal worker completion with an Optuna `FAIL` trial; a
`FAILED_PRETRIAL` marker means no reservation was consumed.

Call `study.optimize(..., catch=(Exception,))` for objective-level failures,
capture the claimed trial number, reload it after `optimize` returns, and require
`COMPLETE` or `FAIL` before writing the structural worker marker. An unexpected
`BaseException` or orchestration error exits nonzero and is reconciled through
heartbeat/state inspection; it is never reported as a scientific success.

If heartbeat recovery marks a trial `FAIL` before complete artifacts exist, the
controller writes an atomic `trial_failure_tombstone.json` containing trial,
reservation, worker/run, DB state, reconciliation time, failure category, and
the list/reason for missing artifacts. Finalization may accept a verified
tombstone for a `FAIL` trial only. Every `COMPLETE` trial still requires the full
result, evaluator JSON, resolved config, and checkpoint/hash contract.

The controller periodically calls the public experimental Optuna 4.2.1 API
`optuna.storages.fail_stale_trials(study)`, reconciles DB states with tmux
sessions and worker markers, and reports:

- reserved/WAITING;
- RUNNING plus the age of the repository's separate atomic worker heartbeat
  marker when reachable;
- COMPLETE finite;
- FAIL;
- collected and audited;
- missing or conflicting artifacts.

Do not read Optuna private heartbeat tables: the public trial API does not expose
the last database heartbeat timestamp. Status must label the separately measured
worker-marker age accurately and treat an unreachable marker as unknown until
the native stale sweep changes the DB state.

If a database/network partition causes native heartbeat expiry while a training
child is still alive, the DB `FAIL` state is authoritative and must never be
changed back to `COMPLETE`. Locally completed output from that attempt is retained
but is not selectable. The slot remains quarantined—regardless of whether its
`flock` was released—until the recorded process group is proven absent or is
safely terminated. No replacement work is launched on that GPU during the
ambiguous period.

Database-outage outcomes are deliberately conservative:

- before claim: the reservation stays `WAITING` and the worker records
  `FAILED_PRETRIAL`;
- transient outage during work: the native optimize call may complete only if
  its normal database operations recover before the configured timeout;
- terminal-commit uncertainty: poll the existing trial identity without
  relaunching it; accept it only if PostgreSQL already says `COMPLETE`, otherwise
  let native stale handling produce `FAIL` and write a tombstone;
- version 1 never repairs an uncertain native-optimize trial with a later manual
  `study.tell`, and never trains the same reserved slot twice.

A controller crash must not invalidate active trials. On restart it acquires the
single-controller lock, reloads PostgreSQL and launch manifests, probes remote
sessions, collects terminal artifacts, and only then dispatches more WAITING
reservations. A second controller must refuse to run against the same study
root.

Every mutating controller command holds an OS `flock` on
`<output-dir>/.controller.lock` and a PostgreSQL advisory lock derived from a
domain-separated 64-bit hash of the redacted database identity plus study name;
this key exists even before the study is created. The advisory lock is held on
one dedicated psycopg connection for the full operation and releases
automatically if that connection dies. The stored controller UUID and
output-root fingerprint must also match. A different root/controller may take
over only through an explicit recovery command after proving that the prior
controller and workers are quiescent. Active-active controllers remain out of
scope. Loss of the advisory-lock connection makes the controller stop all new
dispatch/publication immediately; it must reacquire both locks and perform a full
reconciliation before mutating again.

## 12. Scheduling policy

Use bounded synchronous waves by default:

1. select up to `--max-parallel` verified free slots;
2. launch one one-trial worker on each slot;
3. wait for every launched worker to reach a reconciled terminal/pretrial state;
4. collect and audit all wave artifacts;
5. only then launch the next wave.

Recommended production default after qualification: `--max-parallel 3` on a
homogeneous GPU class. With `n_startup_trials=5`, launch startup waves of at most
`3` and then `2`; thereafter size each startup wave as
`min(max_parallel, 5 - usable_observations, remaining_reservations)`. In Optuna
4.2.1, `FAIL` does not advance TPE's startup observation count, so failed startup
reservations may cause additional random proposals. Continue until five
`COMPLETE` observations exist or the reserved budget is exhausted, and record
the actual count. For maximum proposal reproducibility, set `--max-parallel 1`.

An asynchronous keep-slots-full mode is explicitly out of scope for version 1.
It can be added later after the wave implementation is stable and must be labeled
non-replayable at the study-path level.

Each slot is identified by `HOST GPU WORKER_ID`. Use
`CUDA_VISIBLE_DEVICES=<physical-index>` and pass logical `--device cuda:0`, as
the existing launcher does. Protect each remote GPU with an advisory `flock`
whose lifetime is the worker process. Validate duplicate host/GPU or worker IDs
before launch.

## 13. CLI contracts

Keep the existing serial command backward compatible. Add distributed behavior
through a new orchestration script and a narrow worker entry point rather than
embedding SSH scheduling in `main.py`.

### Controller

Add `scripts/run_distributed_graphvae_attr_bo.py` with subcommands:

```text
init       validate contract, create/load PostgreSQL study, reserve N trials
preflight  verify DB, workers, GPUs, deployment, environment, and cache
run        dispatch/reconcile bounded one-trial waves and collect artifacts
status     print status and optionally write stable JSON
collect    idempotently collect and audit terminal artifacts
finalize   freeze a quiescent study and write final study outputs/snapshot
```

Common required options:

```text
--base-config PATH
--study-name NAME
--output-dir PATH
--storage-env GRAPHVAE_BO_STORAGE_URL
--repo-paths FILE
--python-paths FILE
--slots FILE
--trials N
--max-parallel N
--sampler-seed N
--heartbeat-interval SEC
--grace-period SEC
```

`status --json PATH` must produce a stable schema suitable for tests and scripts.
`init` is idempotent only when every contract field and the exact reservation
count match. It must never append reservations during ordinary resume.

`finalize` holds both controller locks, confirms all reserved trials are
terminal, all worker slots are reconciled, and all required artifacts are
audited. It then changes the PostgreSQL lifecycle attribute to `FROZEN`. Workers
check this attribute before `optimize` and again at objective entry. Finalization
rechecks reserved/unreserved cardinality and terminal states immediately before
and after snapshot creation; any change aborts publication. `FROZEN.json` is an
atomic local mirror, not the sole enforcement mechanism.

### Worker

Add `scripts/run_graphvae_attr_bo_worker.py`. It accepts non-secret trial
settings plus the name of the environment variable containing the storage URL:

```text
--study-name NAME
--base-config PATH
--artifact-root PATH
--study-contract-sha256 HASH
--worker-id ID
--worker-run-id ID
--sampler-seed N
--device cuda:0
--storage-env GRAPHVAE_BO_STORAGE_URL
--heartbeat-interval SEC
--grace-period SEC
--mock
```

The worker performs preflight before `study.optimize`. It runs exactly one
reserved trial and never writes study-level summary/best/freeze files.

### Slots file

Add `CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt`:

```text
# HOST GPU WORKER_ID
cs-cl-13 0 cs-cl-13-gpu0
cs-cl-17 0 cs-cl-17-gpu0
cs-cl-17 1 cs-cl-17-gpu1
```

The sample is not evidence that a slot is currently available; preflight must
re-probe it.

### Final test

Keep held-out test evaluation separate and explicit. It must require a valid
`FROZEN.json`, a fully collected best checkpoint, matching hashes, no WAITING or
RUNNING reservations, and the same study contract. It must not create an Optuna
trial or alter the selected best trial.

## 14. File-level implementation roadmap

### Modify

`scripts/tune_graphvae_attribute_weights.py`

- separate storage construction from study construction;
- preserve SQLite only for serial mode;
- add PostgreSQL RDB construction, heartbeat, and distributed TPE settings;
- expose the existing search space and `execute_trial` for the worker;
- make trial paths portable and writes atomic;
- add reservation and study-contract validation at objective entry;
- record worker/cache/source/environment metadata;
- make subprocess execution process-group safe;
- make summary generation consume only collected, audited artifacts;
- make distributed finalization and test freeze checks explicit.

`scripts/cluster_distribute_code.sh`

- create/sync a deployment manifest;
- add a BO-specific cache/input staging mode without changing ordinary training
  defaults;
- verify hashes remotely rather than trusting rsync success alone.

`scripts/cluster_collect_results.sh`

- support an exact local destination;
- collect through a staging directory;
- verify manifests/hashes before promotion;
- treat differing collisions as fatal and identical content as idempotent.

`scripts/evaluate_attributed_graph_realism_checkpoints.py`

- emit the cache, split, and node/edge schema fingerprints derived from the
  actual loaded evaluator inputs;
- emit actual decoder output dimensions, seed/repeat metadata, and separate
  generated/reference accepted counts;
- keep the existing structural `decoded_node_edge` result and validation/test
  split declarations backward compatible.

`docs/attr_f1pr_bayesian_optimization.md`

- retain serial examples;
- add distributed initialize/preflight/run/status/finalize commands;
- document PostgreSQL, credentials, reproducibility, budget, and rollout rules.

`requirements.txt`

- prevent the broad Optuna range from defeating the tested BO pin, or clearly
  defer to the BO constraints file.

### Add

- `scripts/run_distributed_graphvae_attr_bo.py`;
- `scripts/run_graphvae_attr_bo_worker.py`;
- `scripts/prepare_graphvae_attr_bo_cache.py`;
- `scripts/graphvae_attr_bo_fingerprints.py`;
- `requirements-bo-py38.txt`;
- `CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt`;
- `configs/bayesian_optimization/lobster_graphvae_attr_f1pr_smoke.yaml` with an
  explicit tiny epoch/batch/graph budget and subprocess timeouts for Gates 4-5;
- `tests/test_distributed_graphvae_attr_bo_unit.py`;
- `tests/test_distributed_graphvae_attr_bo_postgres.py`;
- `tests/test_distributed_graphvae_attr_bo_launcher.py`;
- `tests/fakes/fake_graphvae_attr_trial.py`.

Do not modify `main.py` except for the smallest necessary
`require_existing_dataset_cache`/atomic-cache safety hook. Do not put cluster or
Optuna orchestration into `main.py`.

## 15. Test infrastructure

Add pytest markers:

```text
unit       no network, DB, GPU, or real training
postgres   requires GRAPHVAE_BO_TEST_STORAGE_URL
remote     requires an explicitly approved test slot file
gpu        invokes CUDA
slow       invokes real tiny training/evaluation
```

The fake objective must support deterministic score, configurable sleep,
training/evaluation failure, timeout, malformed JSON, non-finite metrics, wrong
split, topology-only output, missing decoder heads, and post-write corruption.
SSH, rsync, and tmux commands must be injectable so launcher tests can use fakes.

PostgreSQL tests must use a unique study name per test and clean only their own
study. They must never delete a shared database or another study. Tests requiring
PostgreSQL may skip only when the explicit test URL is absent; they are mandatory
at the PostgreSQL acceptance gate.

## 16. Defined test cases

### Unit and contract tests

| ID | Definition and steps | Required result |
| --- | --- | --- |
| U01 | Sample node/edge weights, inject them into a deep copy, then repeat with motif opt-in. | Only requested weights change; topology, KL, budget, split, and fixed motif settings do not. |
| U02 | Build the study definition twice; then mutate every field class individually. | Exact definitions resume; changes to objective, config, range, seeds, budget, source, cache, schema, evaluator, hardware policy, or version fail. |
| U03 | Parse valid and adversarial evaluator payloads. | Only finite validation `evaluation.modes.decoded_node_edge.summary.f1_pr.mean` with node and edge features is accepted. |
| U04 | Supply topology-only, node-only, edge-only, wrong-split, test-split, missing-feature, malformed, and non-finite payloads. | Every payload is rejected before trial completion. |
| U05 | Validate a result, then tamper with trial number, budget index, parameters, seeds, config/cache/schema/checkpoint hash, repeats, or graph count. | Every mismatch fails audit. |
| U06 | Move a complete trial tree to a different root and delete the original. | Finalization succeeds using relative collected paths only. |
| U07 | Interrupt each JSON/YAML/CSV/marker write before rename. | Readers observe an old complete file or a new complete file, never a partial file. |
| U08 | Generate study/trial/worker paths concurrently and provide traversal or shell metacharacters. | Paths remain unique and under their root; unsafe identifiers are rejected/sanitized without command injection. |
| U09 | Render launcher commands, then force PostgreSQL DNS, authentication, and TLS failures while a sentinel username/password/URL is configured. Scan exceptions, stdout/stderr, logs, status JSON, manifests, CSV, and summaries. | No sentinel credential or unredacted URL appears anywhere. |
| U10 | Validate slot files with duplicate host/GPU, duplicate worker ID, unknown host, invalid GPU, and malformed rows. | Preflight fails before SSH or Optuna access. |
| U11 | Try finalization and final-test evaluation with WAITING/RUNNING trials or without `FROZEN.json`. | Both refuse without reading test metrics. |
| U12 | Run a poison test object that raises if test data is accessed during optimization. | Validation optimization completes without touching the test object. |
| U13 | Compute worker sampler seeds for fixed study/dispatch inputs, restart, and compute them again; retry one pretrial dispatch. | Mapping matches the specified SHA-256 formula, is stable across restart, distinct across tested dispatches, and reused for the retry. |

### Local worker tests

| ID | Definition and steps | Required result |
| --- | --- | --- |
| L01 | Run one mock trial through the actual worker command. | Resolved YAML, result, hashes, logs, timings, provenance, and terminal DB state are present. |
| L02 | Inject training failure, evaluation failure, timeout, missing checkpoint, wrong evaluator schema, and non-finite score. | Each reservation becomes `FAIL`; its logs/reason survive; later reservations still run. |
| L03 | Launch a worker with a mismatched contract, source, dependency, cache, or feature-schema hash. | It writes `FAILED_PRETRIAL`, creates/claims no trial, and starts no training. |
| L04 | Launch an unauthorized worker when no reserved WAITING trial remains. | It starts no training; any defensive extra DB row is parameter-free, audited, and excluded without blocking safe finalization. |
| L05 | Run identical mock parameters/seeds twice; canonicalize by removing only timestamps, elapsed durations, host-local absolute paths, hostname/GPU, and worker-run ID. | The remaining result object and resolved scientific configuration are exactly equal. |
| L06 | Send TERM during training and evaluation; then force KILL in a separate test. | Exact child process groups are terminated or reconciled; unrelated processes survive. |

### PostgreSQL concurrency tests

| ID | Definition and steps | Required result |
| --- | --- | --- |
| P01 | Initialize a study and enqueue empty reservations on pinned Optuna; run one worker. | The worker claims a WAITING reservation and TPE samples both log-scale weights normally. |
| P02 | Start two worker processes simultaneously with a barrier and reversed sleep order. | They claim different reserved trial numbers and complete without locking errors. |
| P03 | Interrupt initialization after part of eight reservations, resume it, launch a worker, kill only the controller while that worker is RUNNING, leave the worker alive, then restart. | Indexes `0..7` occur once; the active worker finishes; restart dispatches nothing until DB/session/artifact reconciliation and then consumes only WAITING reservations. |
| P04 | Deliberately bypass the controller and start more workers than remaining reservations. | No unreserved training or sampling starts; any guard-created unreserved FAIL row has no params, is audited/excluded, and does not prevent safe finalization. |
| P05 | Replace `TPESampler` with a recording test double and construct several worker studies. | Every worker passes `constant_liar=True`, the configured startup count, and a distinct deterministic dispatch seed; settings are recorded in trial metadata. |
| P06 | Run equal studies with different worker completion order. | Each is internally valid; documentation/status labels proposal order as non-replayable when parallel. |
| P07 | With short test-only heartbeat/grace values, SIGKILL a worker after RUNNING; bounded-poll and call `optuna.storages.fail_stale_trials`. | State changes `RUNNING -> FAIL` within the test timeout; no permanent zombie remains. |
| P08 | Drop DB connectivity before claim, during the objective, and during terminal commit. | Before claim stays WAITING/FAILED_PRETRIAL; recovered native commit may be COMPLETE; unresolved commit becomes FAIL+tombstone; no reservation trains twice. |
| P09 | Start two mutating controllers locally, then from distinct roots/identities, against one study. | Filesystem plus PostgreSQL advisory locks allow one authority; the other exits before mutation; reservations and outputs remain coherent. |
| P10 | Freeze, use `optuna.copy_study` into a temporary SQLite file, reopen/compare it, then re-finalize. | Trial numbers, states, params, values, attrs, and best identity match; atomic publication is idempotent. |
| P11 | Set `n_startup_trials=5`, fail one of the early reservations, and inspect subsequent sampling through a recording sampler/fixed study history. | FAIL is not counted as a usable TPE observation; startup/random sampling continues until five COMPLETE observations or budget exhaustion. |

### Data and artifact integrity tests

| ID | Definition and steps | Required result |
| --- | --- | --- |
| D01 | Start with a missing distributed cache. | Preflight fails before a reservation is claimed; no cache is generated. |
| D02 | Change one byte while keeping plausible cache metadata. | SHA-256 verification rejects it. |
| D03 | Stage the canonical cache on two workers and recompute all fingerprints. | Cache, split, graph-count, node schema, and edge schema fingerprints match exactly. |
| D04 | Keep dimensions equal but alter a feature group/channel meaning. | Schema fingerprint mismatch is rejected. |
| D05 | Make the cache read-only and run a mock/tiny trial. | The trial succeeds without changing content or mtime. |
| D06 | Interrupt rsync and leave partial result/checkpoint files. | Staging is not promoted; hashes fail; recollection remains safe. |
| D07 | Collect the same artifact twice, then collect different bytes at the same trial path. | Identical recollection is idempotent; differing collision is fatal and quarantined. |
| D08 | Let all reserved trials fail, including one heartbeat failure with no result artifact. | A verified tombstone satisfies the failed-trial audit; finalizer writes a failure summary, no false best trial, and returns nonzero. |

### Remote and GPU tests

| ID | Definition and steps | Required result |
| --- | --- | --- |
| R01 | Dry-run a multi-host slot file with physical GPU indices. | Commands set intended `CUDA_VISIBLE_DEVICES` and logical `cuda:0`; no secret or test-split option appears. |
| R02 | Run and collect one remote mock worker using PostgreSQL. | Remote marker, DB state, and collected artifact contract agree. |
| R03 | Run one remote GPU with the committed smoke YAML and explicit tiny timeouts/graph limit. | The bounded run uses all three decoders and produces structurally valid validation Attr-F1PR without loading the production budget. |
| R04 | Run two simultaneous mock workers on different machines. | Trial/artifact IDs are unique, DB locking is clean, and the reserved budget is exact. |
| R05 | Run one tiny real trial on each of two machines concurrently. | Both complete/fail independently; collection and finalization remain coherent. |
| R06 | Make one host unreachable before launch and ambiguous immediately after launch. | Definite prelaunch failure is retryable; ambiguous work is probed and never blindly duplicated. |
| R07 | Kill a worker parent and verify orphan cleanup using its recorded process identity. | Only the trial's process group is removed and the reservation reaches `FAIL`. |
| R08 | Run fixed parameters on two intended GPU classes and compare against Gate 0's named `attr_f1pr_abs_tolerance` and optional training-loss tolerances. | Automated comparison passes every recorded tolerance, or the production slot list is restricted to the passing GPU class; checkpoint byte equality is not assumed. |
| R09 | Reopen the final portable SQLite snapshot in a clean environment and regenerate aggregate outputs from it. | Best selection and aggregate outputs match the original. |
| R10 | After freeze, explicitly run held-out test evaluation. | It evaluates only the selected checkpoint, creates no trial, and cannot change BO ranking. |

## 17. Acceptance gates and rollout

### Gate 0: infrastructure and contract

1. Choose/provision the PostgreSQL endpoint and dedicated credentials.
2. Verify TCP/TLS connectivity from every candidate host.
3. Install the pinned BO dependencies in a disposable copy of the Python 3.8
   environment and confirm GraphVAE imports still work.
4. Record executable cross-GPU criteria in the contract, including at least
   `attr_f1pr_abs_tolerance`, any training-loss absolute/relative tolerances, and
   an explicit statement that checkpoint byte equality is or is not expected.
5. Freeze budget semantics, heartbeat/grace values, artifact retention, and the
   initial three-slot production pool.

The following version-1 values are frozen before implementation:

- `attr_f1pr_abs_tolerance = 0.02` for fixed-parameter repeatability pairs. This
  is an absolute two-percentage-point tolerance on validation Attr-F1PR, not a
  license to merge evaluator seeds or alter the objective. A pair outside this
  tolerance excludes that hardware/environment class from the study; the
  tolerance must not be relaxed after observing BO results.
- When final training loss is compared, require
  `abs(a - b) <= max(1e-3, 0.05 * max(abs(a), abs(b)))`. Checkpoint byte equality
  is not expected across GPU executions; checkpoint semantic/schema validation
  and downstream metric tolerance are required instead.
- `heartbeat_interval = 60` seconds and `grace_period = 600` seconds, as defined
  in Section 7. Failed/stale RUNNING trials consume their reserved budget slot
  and are never silently replaced.
- Canonical controller artifacts, manifests, configs, evaluator JSON, logs, and
  checkpoints are retained for every reserved trial for the life of the study.
  Version 1 performs no automatic canonical deletion. Host-local copies become
  eligible for explicit cleanup only after terminal state, verified atomic
  collection, hash agreement, study freeze, and a restorable database snapshot.
- The initial homogeneous production pool is `cs-cl-13:cuda:0`,
  `cs-cl-17:cuda:0`, and `cs-cl-17:cuda:1` (TITAN RTX). This pool remains
  conditional on exact runtime/source/cache preflight and the R08 repeatability
  test; the current NumPy/runtime-metadata skew must be normalized or the
  failing slot excluded before production.
- Production starts with exactly 30 reserved scientific slots and
  `--max-parallel 3`. Gate 6 uses a separate four-to-eight-slot pilot study and
  cannot be counted as part of the production study.

Exit condition: PostgreSQL and dependency decisions are recorded without secrets,
and no unresolved infrastructure choice changes the code architecture.

### Gate 1: fast local suite

1. Run all existing BO and attributed evaluator tests.
2. Run U01-U13 and L01-L06.
3. Run lint/syntax checks for new Python and shell files.

Exit condition: all mandatory local tests pass; no required test is skipped.

### Gate 2: native storage qualification

1. Use a disposable PostgreSQL study/database namespace.
2. First run P01 on pinned Optuna 4.2.1 and prove that an empty enqueued
   reservation retains its user attributes, is claimed atomically, and samples
   missing parameters normally. If it fails, stop for an architecture decision;
   do not silently switch to random search, custom Optuna schema calls, or an
   unbounded worker loop.
3. Run P02-P11, including true concurrent processes and heartbeat failure.
4. Reopen the produced portable SQLite snapshot.

Exit condition: exact reservation count, concurrency, crash recovery, and restore
are demonstrated against real PostgreSQL.

### Gate 3: deployment, cache, and transport

1. Build one canonical tiny cache and manifest.
2. Run D01-D08.
3. Run launcher dry-run R01.

Exit condition: code/cache mismatches fail before trial claim, credentials are
absent from artifacts, and collection is atomic/idempotent.

### Gate 4: one-worker qualification

1. Use a new LOBSTER-only smoke study name and output root. This qualification
   dataset has 100 deterministic graphs with both node and edge attributes; it
   is not a QM9 proxy or production study.
2. Run R02 with a remote mock worker.
3. Run R03 with one tiny real GraphVAE trial using only the committed LOBSTER
   smoke YAML. Preserve the exact
   `evaluation.modes.decoded_node_edge.summary.f1_pr.mean` objective.
4. Collect, audit, finalize, restore, and inspect the result.

Exit condition: one worker completes the final end-to-end path using PostgreSQL,
the real LOBSTER qualification cache, checkpoint, and attributed evaluator.
This gate does not authorize or validate full-QM9 BO.

### Gate 5: two-worker concurrency qualification

1. Use another new smoke study.
2. Run R04 with simultaneous mock workers on two hosts.
3. Run R05 with one tiny real trial per host.
4. Run R06-R07 failure scenarios.

Exit condition: native trial allocation, heartbeat, exact budget, transport, and
restart behavior pass under real concurrency.

### Gate 6: hardware and production pilot

1. Run R08 on intended hardware.
2. Re-probe all slots and select a homogeneous initial pool.
3. Run four to eight low-budget reservations with `--max-parallel 3`.
4. Re-run finalization from a clean controller process.
5. Manually inspect every trial's config, cache hash, checkpoint hash, evaluator
   mode, graph count, and split.

Exit condition: no DB errors, unaudited guard rows, artifact mismatch, test
access, or unexplained hardware divergence.

### Gate 7: production study

1. Create a fresh production study; never reuse a smoke study.
2. Reserve the intended 30 trials exactly once.
3. Run bounded startup waves until five COMPLETE observations exist or the
   reserved budget is exhausted, following Section 12.
4. Continue with three-worker waves.
5. Resume safely after interruption using the same study contract.
6. Collect and audit every terminal trial.
7. Finalize only when all reservations are terminal and no worker is active.
8. Run held-out test evaluation only through the later explicit command.

## 18. Example target commands

Exact option spelling may be adjusted during implementation, but the completed
interfaces must support an equivalent workflow.

```bash
export GRAPHVAE_BO_STORAGE_URL='postgresql+psycopg2://...'

python scripts/run_distributed_graphvae_attr_bo.py init \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --trials 30 \
  --sampler-seed 0 \
  --max-parallel 3

python scripts/run_distributed_graphvae_attr_bo.py preflight \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --repo-paths CLUSTER_REPO_PATHS.txt \
  --python-paths CLUSTER_MICRO_PYTHON_PATHS.txt \
  --slots CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt

python scripts/run_distributed_graphvae_attr_bo.py run \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --repo-paths CLUSTER_REPO_PATHS.txt \
  --python-paths CLUSTER_MICRO_PYTHON_PATHS.txt \
  --slots CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt \
  --max-parallel 3

python scripts/run_distributed_graphvae_attr_bo.py status \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr

python scripts/run_distributed_graphvae_attr_bo.py finalize \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr

python scripts/tune_graphvae_attribute_weights.py \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --evaluate-best-on-test \
  --device cuda
```

The storage URL is shown as an exported placeholder only; real credentials must
come from the protected environment mechanism and must not be pasted into shell
history.

## 19. Definition of done

Implementation is complete only when all of the following are true:

- serial SQLite behavior and existing tests remain backward compatible;
- distributed mode uses native Optuna RDBStorage with PostgreSQL and rejects
  SQLite/non-client-server storage;
- every approved worker uses the same pinned Optuna, DB driver, GraphVAE
  environment, source, and cache fingerprints;
- exactly `N` reserved scientific trial records exist and no unreserved training
  can start;
- workers write only trial-local artifacts and never race on aggregate outputs;
- heartbeat expiry and targeted orphan cleanup are demonstrated;
- controller restart, idempotent collection, and finalization are demonstrated;
- every selectable trial has audited config/cache/schema/checkpoint/evaluator
  hashes and finite validation Attr-F1PR;
- best selection uses only validation
  `evaluation.modes.decoded_node_edge.summary.f1_pr.mean`;
- finalization refuses while any reservation is WAITING/RUNNING; every COMPLETE
  trial requires full audited artifacts, while a FAIL with missing artifacts
  requires a verified controller-generated tombstone;
- the frozen study produces `best_config.yaml`, `best_trial.json`, `trials.csv`,
  `SUMMARY.md`, `FROZEN.json`, and a restorable database snapshot;
- held-out test evaluation remains a separate explicit action and cannot affect
  the optimizer;
- one-worker, two-worker, failure-injection, and small real pilot gates pass
  before a full study is allowed;
- documentation contains exact environment setup, PostgreSQL preflight, smoke,
  production, resume, status, collect, finalize, and final-test commands.

## 20. Out of scope for version 1

- provisioning or administering PostgreSQL;
- active-active or multiple simultaneous controllers;
- asynchronous keep-all-slots-busy scheduling;
- automatic replacement of failed scientific trials;
- pruning or early stopping based on validation Attr-F1PR;
- changing GraphVAE architecture, dataset split, evaluator, or loss search space;
- automatic held-out test evaluation;
- using every heterogeneous GPU merely because it is available;
- object-store artifact backends or Optuna gRPC proxy at this cluster scale.

## 21. Environment qualification record (2026-08-23)

This section records the dependency and worker feasibility check performed
before implementation. It is evidence about the current machines, not a
substitute for the mandatory preflight on the day a study starts.

### Installed BO client stack

The following Python 3.8-compatible packages were installed into the `micro`
environment on `cs-cl-09`, `cs-cl-13`, `cs-cl-16`, `cs-cl-17`, `cs-cl-18`,
`cs-cl-19`, `cs-cl-26`, and `cs-cl-36` from one offline wheelhouse:

```text
optuna==4.2.1
psycopg2-binary==2.9.10
SQLAlchemy==2.0.52
alembic==1.14.1
Mako==1.3.12
colorlog==6.12.0
greenlet==3.1.1
importlib-metadata==8.5.0
importlib-resources==6.4.5
zipp==3.20.2
```

All sixteen resolved wheel files had identical SHA-256 hashes after transfer to
every machine. Existing NumPy, PyYAML, packaging, tqdm, typing-extensions, and
MarkupSafe installations satisfied the resolver and were not intentionally
upgraded. `pip check` passed on every machine.

`cs-cl-26` and `cs-cl-36` initially had zero available filesystem blocks. The
installation was retried only after `python -m pip cache purge` removed their
disposable pip download caches (32,793.8 MB and 9,740.8 MB respectively). Their
Conda environments, source, datasets, and experiment artifacts were not
deleted or replaced.

### Functional results

- The repository's focused serial BO suite passed: `10 passed` in
  `tests/test_tune_graphvae_attribute_weights.py`.
- Every machine completed an Optuna RDBStorage API smoke test using local
  SQLite, a seeded `TPESampler`, one pre-enqueued empty reservation, log-scale
  sampling of both default loss weights, and one terminal COMPLETE trial.
- Every machine imported Optuna 4.2.1, psycopg2 2.9.10, SQLAlchemy 2.0.52,
  Torch 2.1.2, and DGL 2.1.0+cu121 together.
- Every remote GPU completed a real Torch CUDA matrix operation and a DGL CUDA
  graph operation. The currently visible inventory is eleven GPU slots: two
  GTX TITAN X on `cs-cl-09`; one TITAN RTX on `cs-cl-13`; one Quadro RTX 4000
  on `cs-cl-16`; two TITAN RTX on `cs-cl-17`; two GTX 1080 Ti on `cs-cl-19`;
  one GTX 1080 Ti and one TITAN X (Pascal) on `cs-cl-26`; and one RTX 2080 on
  `cs-cl-36`.
- `cs-cl-18` has no CUDA device and is suitable as a controller, not as a GPU
  worker.
- Git, rsync, SSH, tmux, `nvidia-smi`, and `sha256sum` are available on every
  remote worker.

### Issues that the implementation must gate

The workers' `micro` environments are not exact clones. Direct imports showed
the intended Torch and DGL versions everywhere, but NumPy imported as 1.24.4 on
`cs-cl-13` and 1.24.3 elsewhere. Several environments also contain duplicate or
stale distribution metadata: for example, package metadata can report Torch
2.4.1 or NetworkX 2.5 while Python actually imports Torch 2.1.2 and NetworkX
3.1. Therefore a raw `pip freeze` hash is neither identical across workers nor
a trustworthy runtime fingerprint.

The implementation must fingerprint direct runtime imports, resolved module
paths, and selected source files in addition to recording package metadata. A
production study must either normalize the selected environments or restrict
itself to a preflight-approved set whose actual runtime stack passes the
fixed-parameter reproducibility gate. It must not silently mix the current
environments based only on their common `micro` name.

The GraphVAE repository is not currently present at
`/local-scratch2/mirzaei/Abdolreza/GraphVAE-REQ` on any remote worker. Source,
configuration, and cache deployment/verification are consequently required;
workers cannot run directly from the controller's local working tree.

### PostgreSQL status and implementation go/no-go

The cluster module named `TOOLS/POSTGRESQL` is a service-request placeholder; it
does not provide `postgres`, `initdb`, `pg_ctl`, or `psql`. It instructs the user
to request a managed PostgreSQL database from `iti-dbhelp@sfu.ca`. No existing
GraphVAE BO storage endpoint was found in the controller environment.

A temporary TCP listener was also bound to all interfaces on
`cs-cl-18.cmpt.sfu.ca` at an unprivileged test port. Direct connection attempts
from all seven candidate workers timed out, while SSH to those machines
continued to work. Thus installing PostgreSQL on `cs-cl-18` is not sufficient by
itself: a self-hosted design additionally requires an authorized host/network
firewall rule for the PostgreSQL port, or long-lived SSH tunnels with their own
failure monitoring. The managed SFU database remains the lower-risk production
choice. The temporary listener was terminated after the test.

The Python client side is qualified, and implementation can proceed. A real
distributed storage/concurrency test and any multi-worker study remain blocked
until the operator supplies all of the following:

- a PostgreSQL hostname and port reachable from the controller and approved
  workers;
- a dedicated database and non-superuser role;
- the trusted CA and verified TLS connection settings, or a documented cluster
  exception;
- a protected `PGPASSFILE` or equivalent credential injection mechanism.

Do not substitute shared SQLite for this missing service. After PostgreSQL is
provisioned, Gate 2 must prove schema initialization, concurrent reservation
claims, heartbeat expiry, and restart/resumption against that actual service
before one-worker GraphVAE qualification begins.

### Self-hosted PostgreSQL follow-up (2026-08-23)

PostgreSQL 16.15 was subsequently installed on `cs-cl-18`. The `16/main`
cluster is enabled, online, and accepts local Unix-socket and
`127.0.0.1:5432` connections. The installation has `ssl = on`; its initial
self-signed certificate is valid for the short DNS name `cs-cl-18`, not the
fully qualified hostname.

The dedicated `graphvae_bo` role and `graphvae_attr_bo` database were then
created, ownership and local password authentication were verified, and the
server was configured to listen on both `127.0.0.1:5432` and
`142.58.213.253:5432`. This remains a successful server installation, not yet a
distributed Optuna endpoint: certificate-verified TLS probes from all seven
workers timed out before reaching PostgreSQL.

The cause was localized to the controller's active `firewall.service`, which
loads `/etc/firewall.bash`. SFU source ranges are allowed only to TCP ports 22
and 48555 before a final INPUT drop, and no rule permits TCP 5432. Gate 2
therefore requires exact `/32` firewall rules for the seven approved worker
addresses and a repeated certificate-verified TLS probe. Because the firewall
script identifies itself as Ansible-managed, the operator must arrange
persistence with the system administrator or configuration-management source
rather than assuming a manual edit will remain. If worker access cannot be
authorized, use the managed service rather than silently falling back to shared
SQLite.

Temporary exact `/32` IPv4 firewall rules were subsequently installed for the
seven approved workers. Every worker then completed a hostname-verified
PostgreSQL TLS 1.3 handshake to `cs-cl-18:5432` using the server's public
certificate and negotiated `TLS_AES_256_GCM_SHA384`. A passwordless libpq probe
from every worker reached the expected SCRAM challenge for database
`graphvae_attr_bo` and role `graphvae_bo`; no worker was rejected by
`pg_hba.conf`, and no passwordless login was possible. Direct network, TLS, and
HBA routing are therefore qualified. Credential-protected authenticated access,
firewall persistence, and Optuna concurrent state tests remain before Gate 2 is
complete.

### Local Gate 1/2 qualification follow-up (2026-08-23)

The completed Section 14 implementation passed the mandatory fast local suite
in the qualified `micro` environment: `118 passed, 14 deselected`. The
deselected cases were the separately gated PostgreSQL, remote, GPU, and slow
tests; no mandatory Gate 1 case was skipped. Targeted Python compilation, shell
syntax, and Git whitespace checks also passed.

Gate 2 was then run against a disposable PostgreSQL 16.15 cluster bound only to
`127.0.0.1`. The cluster used a unique temporary data directory, and each test
created and deleted only its own UUID-named Optuna study. All fourteen selected
PostgreSQL tests passed with no skips. This includes P01-P11 and the local
actual-worker cases L01, L02, and L04, covering empty reservation claims,
multi-process allocation, interrupted initialization, oversubscription guards,
deterministic constant-liar TPE settings, heartbeat expiry, outage-before-claim,
controller advisory locking, portable snapshot reopening, and failed startup
observations. The temporary server was stopped and its data directory removed
after the run.

This result qualifies the implemented native PostgreSQL state machine locally.
It does not authorize remote credentials, replace the required remote
`sslmode=verify-full` preflight, prove firewall-rule persistence, launch remote
GraphVAE training, or approve a pilot or production study.

### Gate 3 data-integrity qualification follow-up (2026-08-23)

One deterministic synthetic qualification cache and canonical manifest were
built with `dataset-cache-v4` metadata, a fixed dataset-loader seed, 21 training
graphs, 3 validation graphs, 6 test graphs, and 30 unique graph fingerprints.
The fixture is restricted to local cache/transport qualification and does not
replace Gate 4's real LOBSTER qualification cache.

D01-D08 passed. Missing and byte-modified caches failed before PostgreSQL trial
claim; two independently staged copies recomputed identical file, split,
graph-count, and feature-schema identities; an equal-dimension channel-meaning
change was rejected; a read-only cache survived an actual mock worker without
content or mtime changes; partial collection was not promoted and could be
retried safely; identical collection was idempotent while differing bytes were
quarantined; and an all-failed study reconciled an artifactless heartbeat
failure through a verified tombstone, wrote no false best trial, froze a
portable snapshot, and returned nonzero.

R01 then passed through the actual controller `init`, `preflight --dry-run`, and
`run --dry-run` commands for two synthetic hosts using physical GPU indexes 3
and 7. Both commands mapped the selected physical device to logical `cuda:0`,
preserved one-trial worker execution, inherited only the storage environment
variable name, and contained no storage URL, password material, held-out test
option, or remote-execution acknowledgement. No SSH command ran. Mock-versus-
real execution, heartbeat/grace values, and maximum parallelism are now checked
against the immutable study contract before rendering a worker command.

The final Gate 3 regression results were `123 passed, 17 deselected` for the
non-PostgreSQL/non-remote/non-GPU/non-slow suite and `17 passed` for the complete
disposable-PostgreSQL suite. Gate 3 is complete locally. This does not authorize
Gate 4's remote mock worker or tiny real GraphVAE trial.

### Gate 4 qualification-cache preparation follow-up (2026-08-23)

The committed Gate 4 smoke configuration now uses the deterministic LOBSTER
qualification dataset instead of attempting an unbounded full-QM9 training
split. Its normal data pipeline produced a 59,295,793-byte, read-only canonical
cache with 70 training, 10 validation, and 20 held-out test graphs. The manifest
accepts at most eight validation graphs and records 14 node-feature channels,
11 edge-feature channels, every graph fingerprint, both schema fingerprints,
and cache SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.
Independent manifest verification passed without changing the cache. The cache
publisher also enforces the smoke contract's exact 100-graph total so a loader
or filtering change fails before study initialization.

No model was constructed or trained while preparing this cache. No PostgreSQL
study, remote worker, credential copy, held-out evaluation, pilot, or production
run was started. The cache is a Gate 4 infrastructure and attributed-decoder
qualification input only; it is not evidence about QM9 model quality.

### Gate 4 single-host staging follow-up (2026-08-23)

The clean committed source at `6db554484fd5f322aeccda8e3a6259fb05333142`
was staged to the dedicated
`cs-cl-13:/local-scratch/graphvae-req-work/GraphVAE-REQ-gate4-lobster`
qualification directory. The remote deployment contains 939 source files and
recomputed tree fingerprint
`a9d860eb9d9d864c19352c7eee34233a707518a717772278113be7285f5f8c54`.
Exact source-manifest verification passed after transport.

The canonical LOBSTER cache and manifest were staged separately. Remote
verification recomputed the cache, split, graph, node-schema, and edge-schema
identities, matched cache SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`,
and left the 59,295,793-byte cache read-only at mode `0444`. The approved GPU
was re-probed as one NVIDIA TITAN RTX with 24,576 MiB. Staging used the committed
one-host mapping and reported zero failures.

This step did not deploy credentials, access PostgreSQL, create a study, launch
tmux or a worker, construct/train a model, or access held-out test data.

### Gate 4 protected-credential preflight follow-up (2026-08-23)

The controller and the single approved `cs-cl-13` worker now each have a
host-local credential directory outside every repository. Directories are mode
`0700`; the libpq passfile, trusted self-signed root certificate, and environment
file are regular non-symlink files at mode `0600`. The environment files expose
the storage URL and libpq paths only inside the worker process environment; the
password is present only in the protected passfiles.

Both hosts passed an authenticated Optuna storage connection using
`sslmode=verify-full`, the `cs-cl-18` hostname certificate, role `graphvae_bo`,
and database `graphvae_attr_bo`. Each server-side connection reported TLS
enabled. No credential value, URL, passfile content, or passfile hash was
printed or written to a command manifest, repository, or collected artifact.
These credentials are authorized for Gate 4 qualification only and must be
rotated before any later pilot or production gate.

### Gate 4 runtime normalization follow-up (2026-08-23)

Pre-initialization runtime comparison failed closed because the controller
imported NumPy 1.24.3 while `cs-cl-13` imported 1.24.4, even though the worker's
installed-package metadata incorrectly reported 1.24.3. No study or reservation
was created with that mismatch. All other pinned dependency module hashes were
already identical.

The worker's disposable pip download cache was 31.6 GB and prevented a clean
reinstall. Only `python -m pip cache purge` was used; it removed 2,480
recoverable cache entries and did not delete environments, source, datasets,
caches used by GraphVAE, runs, or artifacts. NumPy was then force-reinstalled
without dependencies or a download cache into the existing qualified `micro`
environment, using a task-specific temporary directory on local scratch. The
controller was normalized to the same pinned wheel. NumPy, Torch, DGL, Optuna,
psycopg2, SQLAlchemy, Alembic, and PyYAML imports passed on both hosts.

The controller and worker now produce the identical semantic runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`.
This exact fingerprint will be frozen into the R02 and R03 study contracts.

### Gate 4 first R02 attempt and fail-closed remediation (2026-08-23)

The first one-slot R02 mock study failed closed and was retained under its
original immutable study name. Its initial worker attempt stopped before trial
claim because the per-study definition had not yet been staged on the worker;
the marker recorded `reservation_consumed=false`, PostgreSQL remained at one
WAITING reservation, and the attempt was preserved before a deterministic retry
with the same dispatch sequence and sampler seed. That retry consumed exactly
one reservation and reached a reconciled `FAIL` because the synthetic evaluator
still emitted its historical 4-node/3-edge fixture dimensions while the real
LOBSTER cache contract requires 14 node and 11 edge channels.

The failed study was collected, audited, and frozen without a false best trial.
Its summary preserves the exact
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean` objective and records no
test access; its portable SQLite snapshot reopens with one `FAIL` trial and a
`FROZEN` lifecycle. A credential-material scan of its artifacts passed.

The remediation makes mock evaluator dimensions follow the immutable cache
contract, stages and verifies the three public per-study inputs before remote
launch, records launch intent before SSH, and quotes the complete worker shell as
tmux's single command argument. A new study name is required for the passing R02
rerun; the consumed failed reservation is never replaced or rewritten.

### Current execution checkpoint

The roadmap is design-ready and may be used as the implementation authority.
Gate 0 is complete enough to begin local implementation: the storage
architecture, database/role, pinned client stack, TLS/HBA path, budget semantics,
heartbeat/grace values, tolerances, retention, and initial slot pool are fixed.
No implementation may weaken those decisions silently.

The following operational items remain explicit acceptance gates rather than
design questions:

- obtain explicit authorization before copying the protected `PGPASSFILE` to
  the seven worker homes, then prove authenticated `verify-full` access without
  logging credentials;
- make the current exact `/32` port-5432 firewall rules persistent in the
  Ansible-managed source or through SFU research support;
- normalize the runtime environment and stage verified source/cache inputs
  before Gates 4-6.

Section 14 and Gates 1-3 are complete locally. The first remote worker
acceptance test in Gate 4 still requires initialization of new, separate R02
mock and R03 real smoke studies. Protected
credential deployment and authenticated `verify-full` preflight now pass;
source, cache, and the one approved GPU slot are staged and verified. Full QM9
BO remains blocked pending a separately reviewed staged or multi-fidelity
budget.
