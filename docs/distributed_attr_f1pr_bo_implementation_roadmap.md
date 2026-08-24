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
probe      record tmux/worker/DB state for prior launches without dispatch
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
- `scripts/recover_graphvae_attr_bo_process.py`;
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

The first source restage after this change also failed closed when the
distributor's implicit local `python` lacked NumPy. Remote deployment-manifest
verification rejected the resulting empty file before worker launch. The
distributor now accepts an explicit `--local-python`, treats manifest-generation
failure or an empty manifest as fatal before any transport, and checks remote
cache-directory creation before copying cache bytes.

### Gate 4 R02 remote mock qualification follow-up (2026-08-23)

The fresh one-slot study
`lobster_attr_f1pr_gate4_r02_mock_20260823b` passed R02 using committed source
`f2b56bb8f291538be374bbb570e49ea0d866133f`. The controller staged and
hash-verified the immutable study definition, deployment manifest, and LOBSTER
cache manifest before launching one detached `cs-cl-13` tmux worker. The launch
manifest records one acknowledged dispatch, fixed sampler seed, bounded
synchronous execution, and `dry_run=false`.

The remote `COMPLETED` marker and PostgreSQL agree on exactly one reserved
`COMPLETE` trial, zero failures, zero WAITING/RUNNING trials, and zero unreserved
guard rows. The structured evaluator used validation only, both GraphVAE
attribute decoders, 14 node and 11 edge channels, five evaluator repeats, and
eight generated plus eight reference accepted graphs. The value at the exact
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean` path was
`0.7847403306199212`; this deterministic mock value qualifies orchestration and
artifact integrity only and is not a model-quality result.

Checksum collection, controller audit, finalization, and an independent reopen
of `study_snapshot.sqlite3` all passed. The reopened snapshot preserves one
`COMPLETE` trial, best trial 0, the same value and objective path, and a `FROZEN`
lifecycle. No held-out artifact exists, optimization records `test_access=false`,
and a credential-material scan of the collected study root passed.

### Gate 4 first R03 attempt and GPU-metadata remediation (2026-08-23)

The first real two-epoch LOBSTER study
`lobster_attr_f1pr_gate4_r03_real_20260823a` completed one reserved GPU trial,
was checksum-collected, audited, frozen, and independently restored. Its exact
validation objective was finite at `0.000019999800003999925`; the evaluator used
both attribute decoders, 14 node and 11 edge channels, five repeats, and eight
generated plus eight reference validation graphs. This tiny smoke value is not
model-quality evidence.

The run did not execute final evaluation or create a `final_test` artifact. A
training file named `testGraphs_adj_.npy` was inspected at source level and is a
legacy filename written by `EvalTwoSet(model, val_adj, ...)` during validation;
the held-out branch is guarded by `skip_final_evaluation=true`. Acceptance still
uses the structured evaluator's validation split and fingerprint, never that
legacy filename.

R03a is retained as qualification-incomplete because its otherwise valid trial
result recorded `gpu_model=null` and `gpu_vram_bytes=null`, violating the
required worker metadata contract. The remediation probes the selected physical
GPU with `nvidia-smi` before trial claim, fails pretrial if one exact model/VRAM
row cannot be verified, propagates both values through worker, trial, and Optuna
metadata, and makes collection audit reject a GPU result with missing identity.
A new one-slot R03 study is required; R03a is never rewritten or extended.

### Gate 4 R03 remote real qualification follow-up (2026-08-23)

The fresh one-slot study
`lobster_attr_f1pr_gate4_r03_real_20260823b` passed R03 using committed source
`24c103e17ce4210dddbada34dd8ea2660ebe184f`. Its one acknowledged bounded
dispatch ran two real GraphVAE epochs on physical GPU 0/logical `cuda:0`. The
preclaim worker record, trial result, and restored Optuna attributes all agree on
`NVIDIA TITAN RTX` with 25,769,803,776 VRAM bytes and runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`.

PostgreSQL and the remote `COMPLETED` marker agree on one reserved `COMPLETE`
trial, no failures, no WAITING/RUNNING trials, and no unreserved guard row. Real
training finished in 209.71 seconds and validation evaluation in 6.05 seconds,
inside both fixed 600-second phase caps. The sampled weights were
`alpha_node_feat=66.3603784360122` and
`alpha_edge_feat=26.984670498189587`.

The finite value at the exact
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean` path was
`0.000019999800003999925`, with precision `0.0`, recall `1.0`, and eight
accepted validation graphs. The structured evaluator attests to
`decoded_node_edge`, both GraphVAE attribute decoders, 14 node and 11 edge
channels, five repeats, and eight generated plus eight reference graphs. Cache,
validation-split, source, runtime, and both feature-schema fingerprints passed
collection audit. The selected checkpoint SHA-256 is
`fceb1311eac789dc4213d67388596e3a2639e4f8ae20342e16daf4fb24b9cdc3`.
R03b exactly reproduced R03a's fixed-seed objective and checkpoint bytes; this
same-slot observation does not substitute for the later R08 hardware test, and
checkpoint byte equality remains non-required.

Checksum collection, finalization, and independent SQLite reopen passed with a
`FROZEN` lifecycle and the same best trial, value, objective, and GPU metadata.
No final-test artifact exists, optimization records `test_access=false`, and a
credential-material scan passed. After the run, the canonical 59,295,793-byte
cache remained mode `0444` with unchanged SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

Gate 4 is complete. This qualification does not start or authorize Gate 5
multi-worker tests, R08, a pilot, production, held-out evaluation, or full-QM9
Bayesian optimization.

### Gate 5 authorization and two-host staging contract (2026-08-23)

Gate 5 is explicitly authorized for bounded two-worker qualification on
`cs-cl-13:cuda:0` and `cs-cl-17:cuda:0`. The dedicated repository roots,
pinned Python mappings, and physical-GPU slot identities are frozen in the
three `CLUSTER_GRAPHVAE_ATTR_BO_GATE5_*` files. Gate 4 source roots and worker
identities are not reused.

Read-only predeployment probes found the dedicated Gate 5 source roots absent
on both machines, so deployment cannot accidentally merge with an earlier
qualification. Both hosts expose the same pinned NumPy, Torch, DGL, Optuna,
psycopg2, SQLAlchemy, Alembic, and PyYAML versions used by Gate 4. Physical GPU
0 on each host is an NVIDIA TITAN RTX with 24,576 MiB of reported VRAM.
Available local scratch exceeded the source plus the immutable 59,295,793-byte
LOBSTER cache on both workers at the time of the probe.

The authorization includes protected PostgreSQL credential deployment to the
new `cs-cl-17` qualification root and R04-R07 only. It does not authorize R08,
the Gate 6 pilot, a production study, held-out evaluation, or full-QM9 BO.
Qualification credentials remain outside version control and must be rotated
before any pilot or production use.

### Gate 5 two-host deployment qualification follow-up (2026-08-23)

Separate Gate 5 controller and worker credential files were derived from the
authorized protected qualification material and placed outside every source,
cache, run, and artifact root. The controller, `cs-cl-13`, and `cs-cl-17` each
use host-local directories at mode `0700` with regular non-symlink environment,
passfile, and CA-certificate files at mode `0600`. All three hosts authenticated
as the qualification role and database through `sslmode=verify-full`; a
server-side `pg_stat_ssl` check reported TLS enabled for every connection. No
credential value, unredacted storage URL, passfile content, or credential hash
was printed, logged, committed, or written into qualification artifacts.

Clean committed source `605eca17408250626c08b4a1238537fb9eeeecb1` was staged
to both dedicated Gate 5 repository roots. Each independently verified all 942
deployment files and tree fingerprint
`4bf23dce7549b88b525efcb4c58a2bc7d94c655f21cdd4a0857f83916deab8a7`.
The LOBSTER cache also independently reverified on each host with 70/10/20
splits, 14 node channels, 11 edge channels, mode `0444`, size 59,295,793 bytes,
and SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

The first exact runtime comparison failed closed on `cs-cl-17`: its installed
NumPy reported version 1.24.3 but its top-level module bytes differed from the
frozen controller and `cs-cl-13` runtime. Every other pinned module hash and
version already matched, and no study or reservation existed. After confirming
the shared environment had no active Python executable, only NumPy was
force-reinstalled without dependencies or a download cache from the exact
1.24.3 binary wheel whose initializer matched the frozen hash. Temporary wheel
staging was then removed. The controller and both workers now reproduce runtime
fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`.

Physical GPU 0 on both workers re-probed as one NVIDIA TITAN RTX with 24,576
MiB reported VRAM. With `CUDA_VISIBLE_DEVICES=0`, each pinned worker Python saw
exactly one NVIDIA TITAN RTX at logical `cuda:0`. This step created no study,
claimed no reservation, launched no worker, trained no model, and performed no
held-out evaluation. The qualified deployment is restricted to R04-R07.

### Gate 5 R04 two-host mock concurrency follow-up (2026-08-23)

The fresh immutable mock study
`lobster_attr_f1pr_gate5_r04_mock_20260823a` used committed source
`26ed2863fb356c46113fca3a0a81666f625e732a` and contract SHA-256
`75e845870b041c8f39d442ee8629cd103517a57ca6f91be946ab46ef3c06edd9`.
It reserved exactly two scientific slots with `max_parallel=2`, heartbeat 60,
grace 600, startup target five, and no replacement of failed reservations.
Before launch, PostgreSQL reported exactly two reserved `WAITING` trials and no
other row.

One bounded wave launched `cs-cl-13-gate5-gpu0-dispatch-1000000` and
`cs-cl-17-gate5-gpu0-dispatch-1000001` on the two dedicated physical GPU 0
slots. Their recorded worker lifetimes overlapped by approximately 2.216
seconds. PostgreSQL assigned distinct trial numbers and budget indexes 0 and 1;
the SHA-256 seed derivation reproduced dispatch seeds 2,935,504,862 and
1,180,592,659 exactly. Both workers recorded the frozen runtime fingerprint,
NVIDIA TITAN RTX identity, logical `cuda:0`, and separate artifact identities.

Both reservations reached `COMPLETE` with no locking error, `FAIL`,
`WAITING`, `RUNNING`, unreserved guard row, duplicate, or replacement. Their
deterministic mock validation Attr-F1PR values were 0.6760063945173143 and
0.8995487816889531. Both artifacts use the exact
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean` objective, validation
split, five repeats, eight accepted graphs, both attribute decoders, and the
frozen 14-node/11-edge schemas. These mock values qualify orchestration only and
are not model-quality evidence; parallel proposal order remains explicitly
non-replayable.

Checksum collection from both hosts merged without a differing collision.
Controller audit accepted both trial and worker marker trees, finalized the
study as `FROZEN`, and atomically published a portable SQLite snapshot. An
independent reopen reproduced both `COMPLETE` trials, best trial 1, and semantic
study fingerprint
`626021351641f9c5960072823cf8e76f632257750fc564aaf9a52d060f3d1599`;
the live PostgreSQL study matched that fingerprint exactly. Credential-material
and generic storage-URL scans passed, optimization retained
`test_access=false`, and no final-test artifact exists. Both remote caches
remain mode `0444`, size 59,295,793 bytes, with unchanged SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

### Gate 5 R05 two-host real LOBSTER follow-up (2026-08-23)

The different fresh real study
`lobster_attr_f1pr_gate5_r05_real_20260823a` used committed source
`912854e24f6cea8140807ada618a976ce629d117` and immutable contract SHA-256
`a57a8e8d60e1f1da73234a7af7b77ead9e0f9f27ccbc7afff1fd6505d50592d1`.
It reserved exactly two scientific slots with `max_parallel=2`, heartbeat 60,
grace 600, and no failed-slot replacement. The committed LOBSTER smoke config
retained two epochs, at most eight evaluation graphs, five evaluator repeats,
600-second training and evaluation limits, and `skip_final_evaluation=true`.
PostgreSQL reported exactly two reserved `WAITING` trials before the only
bounded launch wave.

The dedicated `cs-cl-13` and `cs-cl-17` GPU-0 workers used dispatch sequences
1,000,000 and 1,000,001 and deterministic sampler seeds 3,729,395,558 and
2,192,993,003. Their worker lifetimes overlapped for more than 222 seconds.
PostgreSQL assigned distinct trial numbers and budget indexes 0 and 1 with no
locking error, duplicate, guard row, or replacement. Both workers recorded
logical `cuda:0`, NVIDIA TITAN RTX with 25,769,803,776 VRAM bytes, and the exact
frozen runtime fingerprint.

Trial 0 sampled `alpha_node_feat=45.91347191422007` and
`alpha_edge_feat=20.220667075373278`; training took 210.80 seconds and
evaluation took 8.21 seconds. Trial 1 sampled
`alpha_node_feat=0.36294870524446543` and
`alpha_edge_feat=0.01199297988354986`; training took 233.91 seconds and
evaluation took 9.10 seconds. Both independently reached `COMPLETE` inside the
fixed phase limits. Their checkpoints are distinct artifacts, as expected for
different sampled parameters and hosts.

Both structured evaluator payloads produce the same finite tiny-smoke value
`0.000019999800003999925` at exactly
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`, with precision 0.0,
recall 1.0, and eight accepted validation graphs. Each attests to validation
split, `decoded_node_edge`, both GraphVAE attribute decoders, 14 node and 11
edge channels, five repeats, and eight generated plus eight reference graphs.
This tiny two-epoch result is qualification evidence only, not a model-quality
claim.

Checksum collection merged both host trees without a differing collision.
Controller audit accepted both results and worker markers, froze the study, and
atomically published the portable SQLite snapshot. Independent reopen preserved
two `COMPLETE` reservations, best trial 0 under the tied objective, and semantic
study fingerprint
`9a7b70db1f284a51ae536d8e416c0abec1026acb166aa8c0971d9e885283f932`;
the live PostgreSQL study matched it exactly. Credential-material and generic
storage-URL scans passed, `test_access=false` remained frozen, and no final-test
artifact exists. Both deployed caches remain mode `0444`, size 59,295,793
bytes, with unchanged canonical SHA-256.

### Gate 5 R06 launch-ambiguity follow-up (2026-08-23)

R06 first added and committed a fail-closed launch-reconciliation interface in
`6cbd64c03f79ee569f69d276a6e2fa3d5a0c0743`. The controller now records the
phase of every remote attempt and refuses a later wave after `ATTEMPTING`,
`SSH_ERROR`, or `SSH_ACKNOWLEDGED` until `probe` matches the exact tmux name,
sanitized terminal marker, worker-run identity, reserved PostgreSQL trial
number, budget index, and database state. Probe results remain non-retryable
while the host is unreachable, the tmux session or DB trial is active, evidence
is missing, or marker identities conflict. The two test-only launch faults are
hidden and require `GRAPHVAE_BO_ENABLE_TEST_FAULTS=1`; normal invocations cannot
enable them accidentally. Focused launcher, unit, integrity, and smoke-config
coverage passed with 56 tests before qualification.

The fresh mock study
`lobster_attr_f1pr_gate5_r06a_prelaunch_20260823a` used immutable contract
SHA-256 `201e62cb13873a1c9708139fec3132c2e0251817d4bc6ac3f77a779f1f2acea9`,
one reserved trial, and `max_parallel=1`. Its injected cs-cl-13 failure occurred
before remote input staging and before any SSH/tmux launch call. Wave 1 retained
launch state `PLANNED`; its probe found no tmux, worker marker, or matching DB
trial and classified the identity `DEFINITE_PRELAUNCH` and retry-safe. The same
single reservation remained `WAITING` and unclaimed.

All attempt evidence was retained. The first retry exposed that the public
deployment and cache manifests had been supplied to `init` but not copied into
the new controller study root. That operator staging miss also failed before
SSH, remained `PLANNED`, had no DB claim, and was independently probed as a
second `DEFINITE_PRELAUNCH` identity. After the exact immutable manifests were
installed mode `0444`, wave 3 launched the unchanged reservation once with
dispatch sequence 3,000,000 and derived sampler seed 1,666,357,180. Its unique
worker run claimed trial 0/budget index 0 and completed the bounded mock
objective. No reservation was appended or replaced.

The separate fresh mock study
`lobster_attr_f1pr_gate5_r06b_ambiguous_20260823a` used contract SHA-256
`b6df141108d480d1287cd4a5cd8dbb444b39f9dc675e6ae534a530d472b93984`,
one reserved trial, and `max_parallel=1`. The controller received the remote
tmux acknowledgement and then injected the ambiguous failure, truthfully
recording wave 1 as `SSH_ERROR`, `AMBIGUOUS_SSH_ERROR`, and
`injected_after_remote_ack=true`. An immediate unprobed `run` failed closed and
created no wave 2. The original worker alone claimed trial 0/budget index 0
with dispatch sequence 1,000,000 and derived seed 807,335,553. The first probe
matched its `COMPLETED` marker to the reserved PostgreSQL `COMPLETE` row and
classified it `RECONCILED_TERMINAL`; no duplicate or replacement was launched.

Checksum collection retained the attempt, probe, worker, and trial trees for
both studies. Controller audit froze each study and atomically published its
portable SQLite snapshot. Independent reopen matched live PostgreSQL exactly:
R06a semantic fingerprint
`e314a9b7636aca984e49a4f202899268520c0e7a645d5a62231cfd92670ccf3f`
and R06b fingerprint
`af1c47ccbb1d5cdc4ee956f54eefc165cb530d8f1a9cd8be19b07a72d73a125e`.
Both definitions preserve the exact validation objective path, validation
split, and `test_access=false`. A combined 57-file scan found no protected
credential material, unredacted storage URL, test access, or final-test
artifact. The cache on both hosts remains mode `0444`, size 59,295,793 bytes,
and SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

### Gate 5 R07 orphan-process recovery follow-up (2026-08-23)

R07 added the probe-first recovery interface in
`a532410876e72cbb67e92f8a6eba531dc391473c`. It derives the only eligible
identity file from the immutable study root, worker-run ID, trial number, and
training/evaluation phase. A probe sends no signal. Termination additionally
requires `--terminate --execute` and exact agreement between the recorded and
live PID, process-group leader, PID start ticks, command, cwd, contract hash,
worker-run identity, trial number, and phase. PID reuse, a missing leader with
live group members, or any identity mismatch fails closed. Focused tests proved
that a mismatch sends no signal, the matching group is removed, and an
unrelated same-command group survives.

The fresh real study `lobster_attr_f1pr_gate5_r07_orphan_20260823a` used worker
source `a532410876e72cbb67e92f8a6eba531dc391473c`, immutable contract SHA-256
`3d6338dd97aad3890609f6137e3dd0a87d7210cceb04934b6b94c74a8fb44117`,
exactly one reserved trial, and `max_parallel=1`. It retained the committed
LOBSTER smoke configuration: two epochs, eight evaluation graphs, five
repeats, 600-second training/evaluation limits, and validation-only selection
with `test_access=false`. The only launch used cs-cl-13 physical GPU 0,
dispatch sequence 1,000,000, derived sampler seed 805,986,020, and worker run
`cs-cl-13-gate5-gpu0-dispatch-1000000`.

After PostgreSQL reported the sole reservation `RUNNING`, the training identity
was durably present with PID and PGID 280432, start ticks 335885710, cwd equal
to the dedicated Gate 5 repository, the exact study/worker/trial/phase fields,
and command SHA-256
`cf304f7baec51f18b16622168c16f5c1867f0338899582c621be6c44dd54728f`.
The worker's exact tmux parent session was killed; probes immediately before
and after that action both found the recorded child `MATCHING_LIVE`.

The first unrelated-sentinel harness command was malformed: tmux rejected its
format argument, no sentinel PID was obtained, and that attempt was not counted
as survival evidence. Before child cleanup, a correctly quoted unrelated
`sleep` session was then created as PID/PGID 292282 with start ticks 335896445.
The recovery interface moved only the recorded child from `MATCHING_LIVE` to
`ABSENT` and verified the complete group had no live member. The unrelated
sentinel retained its original PID, PGID, start ticks, and command through that
cleanup, then was removed separately. Final probes found both PIDs absent.

Process cleanup did not edit Optuna state. The reservation remained `RUNNING`
until the 18th bounded 30-second controller poll called Optuna's native stale
sweep under the frozen heartbeat 60/grace 600 contract; it then transitioned
to `FAIL`. There was still exactly one launch manifest, one reserved trial, no
guard row, no replacement, and no `WAITING` or `RUNNING` work. The pre-tombstone
launch probe stayed `MISSING_AMBIGUOUS` and non-retryable rather than trusting
DB `FAIL` alone.

Collection retained the partial training log, process identity, recovery
records, resolved config, and the identity-bound `status=RUNNING` trial result.
This exposed a finalizer gap, fixed in `6973720`: the controller now atomically
retains that exact record as `trial_result.interrupted.json`, verifies its
trial/budget/parameters/contract/worker identity, and binds its SHA-256
`b48fdf4f7292ed7523fdbb5c0b02ad765837631f631194104edf521ddc525c4b`
into the failure tombstone. The canonical `trial_result.json` path is verified
absent. Tamper and idempotence coverage brought the focused suite to 58 tests.

The all-failed finalizer returned its documented nonzero status while still
freezing the quiescent study, writing `RECONCILED_FAIL`, publishing a portable
snapshot, and truthfully selecting no best trial. The final probe matched the
tombstone marker to trial 0/budget index 0/DB `FAIL` and became
`RECONCILED_TERMINAL`. Independent reopen matched live PostgreSQL at semantic
fingerprint
`7e9054ba0f7771215fdd7b98a962b643ad74d1f41c9e935f851750aa59b0fc5e`.
A 73-file scan found no protected credential material, unredacted storage URL,
test access, or final-test artifact. Both host caches remain mode `0444`, size
59,295,793 bytes, and canonical SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

### Gate 5 final acceptance audit (2026-08-23)

The complete non-PostgreSQL distributed BO suite passed 58 tests across the
unit, integrity, launcher, and smoke-configuration files. The separately
marked PostgreSQL suite passed all 17 tests through the dedicated protected
Gate 5 qualification connection. It used UUID-prefixed isolated studies only;
a post-suite database query found zero remaining `graphvae_bo_pytest_*`
studies. No test changed or deleted an R04-R07 qualification study.

A single read-only cross-study audit then reopened all five frozen live
PostgreSQL studies and their five independently portable SQLite snapshots. It
revalidated every immutable contract, exact reservation index, terminal state,
launch identity and derived dispatch seed, result artifact, evaluator payload,
and R07 failure tombstone. The aggregate budget remains exactly seven
reservations: six `COMPLETE`, the one intended R07 `FAIL`, and zero `WAITING`,
`RUNNING`, other-state, guard, duplicate, missing, or replacement rows. The
live/snapshot semantic fingerprints remain exactly those recorded in the R04,
R05, R06, and R07 sections above. The corresponding portable snapshot
SHA-256 values are:

- R04: `7ec77d127b9ec96fb17ee6042d1198f57ff64f53bee072ae4ca5d83476c1ef96`;
- R05: `2074b8af7c20f834f08fa28dc1ace60f8ff23a333761181532d25b96e81d7594`;
- R06 definite-prelaunch: `b65814d2d549cbd81bff63e156c2f482b9ac348cadc3cf72b4d0bd364e9a12a2`;
- R06 post-launch ambiguity: `beaba6421a2fa6cd94b15e789288a66e41a79ebac60104d54633f7dc4943378a`;
- R07: `e0c44d96ecb42717f8c0dcfe35f3d8b803ffef3a17c75720db3642264789df7b`.

Every completed artifact was re-derived from exactly
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean` and attests to
validation split, `decoded_node_edge`, both `node_feature_decoder` and
`edge_feature_decoder`, five repeats, eight generated plus eight reference
graphs, and the immutable 14-node/11-edge LOBSTER schemas. Every definition
retains `test_access=false`, `skip_final_evaluation=true`, heartbeat 60, grace
600, non-replacement of failed slots, and `sslmode=verify-full`. The audit read
374 files totaling 2,438,001,453 bytes and found zero protected credential
material, unredacted PostgreSQL URL, `test_access=true`, or final-test
artifact.

Final host-side checks found the cache on both `cs-cl-13` and `cs-cl-17` still
mode `0444`, exactly 59,295,793 bytes, and SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.
Their current deployment manifests agree at SHA-256
`a2d9220ffb2d3fff8be5f0e3b16e6dbeecf590b8ffcad851b6968153129e24f9`,
the R07 source deployment, and their read-only cache manifests agree at
`ffe65e9ef38f10c4bd2390804c4db248834881263b2c3e06ca03cb7789fd3a46`.
The controller and both worker credential directories remain mode `0700`, and
all protected environment, `PGPASSFILE`, and CA files remain mode `0600`.

Gate 5 is complete. This qualification does not authorize Gate 6, R08, a
held-out/test evaluation, production studies, or full-QM9 optimization. The
qualification credentials must be rotated before Gate 6 or production use.

### Gate 6 R08 implementation prerequisite (2026-08-23)

The controller now accepts a fixed `alpha_node_feat` and `alpha_edge_feat` pair
only when both finite values are supplied inside their contracted search
ranges. The pair is part of the immutable study definition and is pre-enqueued
through Optuna for every reserved R08 trial. Ordinary studies continue to
enqueue empty fixed-parameter maps and use the unchanged TPE sampling path.

The new `hardware-audit` command requires a matching `FROZEN.json`, one audited
`COMPLETE` result per reservation, identical contracted weights, distinct
host/GPU identities, intact checkpoint and source/cache/environment/schema
hashes, validation-only `decoded_node_edge` evaluator evidence, and both
attribute decoders. It compares every recorded slot pair against the frozen
absolute Attr-F1PR tolerance `0.02`; the optional training-loss formula is used
only when every result records that value. Checkpoint byte equality remains
explicitly unnecessary. A failing comparison publishes no eligible slot list.

The expanded non-PostgreSQL distributed suite passes 60 tests. A separate
isolated PostgreSQL test proved that two fixed reservations retain exactly the
same pre-enqueued parameters through native Optuna claims and then deleted its
UUID-named study. This is interface qualification only: no Gate 6 study or GPU
training has started, and the protected qualification credentials still must
be rotated before R08.

### Gate 6 authorization, credentials, and staging contract (2026-08-23)

The user subsequently authorized every remaining roadmap action except
full-QM9 Bayesian optimization. This authorizes the non-QM9 Gate 6 R08 and
pilot, clean-snapshot R09, and the separate explicit post-freeze LOBSTER R10
held-out evaluation. It does not authorize the Gate 7 30-reservation QM9 study
or any substitute presented as a QM9 production result.

Before any Gate 6 study was created, the dedicated PostgreSQL role password was
rotated with a generated value that was never written to the repository or
printed. A new protected `gate6` generation was created outside all repository,
source, cache, and artifact roots. The controller verified the new generation
through the `verify-full` storage constructor and separately proved the Gate 5
password is rejected. Only `cs-cl-13` and `cs-cl-17` received the Gate 6 worker
generation. All three protected directories are mode `0700`; environment,
`PGPASSFILE`, CA, and rotation-metadata files are mode `0600`.

The initial protected worker template inherited controller-side Gate 4
`TMPDIR` and Matplotlib paths. Remote authentication still passed, but creation
of those paths correctly failed. Before any source deployment or study, the
controller and worker path variables were atomically changed to dedicated
Gate 6 trees. Both workers now verify those host-local paths and their mode
`0700`; no Python fallback is accepted as the Gate 6 runtime contract.

The committed Gate 6 mappings select dedicated repository roots at
`/local-scratch/graphvae-req-work/GraphVAE-REQ-gate6-lobster`, the pinned
`micro` Python, and exactly these three intended TITAN RTX slots:

- `cs-cl-13` physical GPU 0 as `cs-cl-13-gate6-gpu0`;
- `cs-cl-17` physical GPU 0 as `cs-cl-17-gate6-gpu0`;
- `cs-cl-17` physical GPU 1 as `cs-cl-17-gate6-gpu1`.

R08 will use the fresh study
`lobster_attr_f1pr_gate6_r08_fixed_20260823a`, exactly three reservations,
`max_parallel=3`, sampler seed 43, and the predeclared fixed parameters
`alpha_node_feat=2.0` and `alpha_edge_feat=3.0`. One bounded real LOBSTER smoke
trial will run on each slot in one wave. The separate pilot will use
`lobster_attr_f1pr_gate6_pilot_20260823a`, exactly five reservations,
`max_parallel=3`, and sampler seed 47. Both retain the committed two-epoch
LOBSTER smoke configuration, eight-graph validation limit, five repeats,
600-second phase limits, exact validation objective, both attribute decoders,
and `skip_final_evaluation=true`. R09 and R10 may begin only after that pilot is
audited and frozen.

The first Gate 6 source deployment attempt stopped before transfer because its
dedicated controller `TMPDIR` did not yet exist. This exposed that
`cluster_distribute_code.sh` used `set -uo pipefail`: a failed `mktemp` left an
empty manifest path and the script continued into manifest generation. The
deployment entry point now uses `set -euo pipefail`, so failure to allocate its
temporary manifest exits immediately. The exact missing-parent case is covered
by a bounded shell regression check. No remote Gate 6 source root or study was
created by the failed attempt.

### Gate 6 three-slot deployment qualification (2026-08-23)

After the fail-closed fix was committed, both dedicated Gate 6 roots were
staged from clean source commit
`96da63dcbeb5e96af8aa57bd5079f4137a2ee1c5`. Each independently verifies
source tree SHA-256
`738d52ce5f05dc506b2dc7f434ee4c469eb511d2372ea33b65420012e32c3e75`;
the two deployment-manifest files agree at SHA-256
`3236e4cb86ceacd3775c18a2acc36ea0e4bd0eb03de79c3200f2a579fd81f877`.
No Gate 4, Gate 5, or general-purpose source root was reused.

Both pinned worker Pythons reproduce runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`
and authenticate with the rotated protected generation through the exact
heartbeat 60/grace 600 verify-full storage constructor. Their cache manifests
are mode `0444` and agree at SHA-256
`ffe65e9ef38f10c4bd2390804c4db248834881263b2c3e06ca03cb7789fd3a46`.
Both cache files remain mode `0444`, exactly 59,295,793 bytes, and canonical
SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

Physical probing found `cs-cl-13` GPU 0 and `cs-cl-17` GPUs 0 and 1 each as an
NVIDIA TITAN RTX with 24,576 MiB reported VRAM. Setting
`CUDA_VISIBLE_DEVICES` separately to each selected physical index made the
pinned Python see exactly one NVIDIA TITAN RTX at logical `cuda:0`. This step
created no study or reservation, launched no worker, trained no model, and read
no held-out data. R08 initialization is now permitted.

### Gate 6 R08 fixed-parameter qualification (2026-08-23)

The fresh real study `lobster_attr_f1pr_gate6_r08_fixed_20260823a` used source
commit `96da63dcbeb5e96af8aa57bd5079f4137a2ee1c5`, immutable contract SHA-256
`c6a0e7d53bda12670edb35ce87332f3ef7a02bcd56ca3f9b344e9b4935bfa345`,
exactly three reserved trials, and `max_parallel=3`. Before launch, all three
PostgreSQL rows were uniquely indexed `WAITING` reservations whose native
Optuna fixed maps were exactly `alpha_node_feat=2.0` and
`alpha_edge_feat=3.0`; no other row existed.

One bounded wave launched each qualified slot once. Dispatch sequences
1,000,000, 1,000,001, and 1,000,002 reproduced derived sampler seeds
2,329,217,710, 1,290,239,270, and 1,865,563,458. The three worker lifetimes
overlapped for 215.643 seconds. Every worker independently reached `COMPLETE`
with the identical contracted parameters and validation Attr-F1PR
`0.000019999800003999925`. There was no `FAIL`, `WAITING`, `RUNNING`, guard,
duplicate, missing, or replacement row.

All results attest to exactly
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`, validation split,
`decoded_node_edge`, both GraphVAE attribute decoders, five repeats, eight
generated plus eight reference graphs, and the immutable 14-node/11-edge
LOBSTER schemas. The automated report compared all three slot pairs; its
maximum absolute Attr-F1PR difference is `0.0` against the frozen `0.02`
tolerance, so all three TITAN RTX slots are eligible for the pilot. Final
training loss was not a structured trial-result field and was therefore
truthfully marked `not_recorded` rather than inferred from logs. All three
checkpoint hashes happened to match, but checkpoint byte equality was not an
acceptance condition.

A clean controller process audited the merged host trees, froze the study, and
published its portable snapshot. Independent reopen matched live PostgreSQL at
semantic fingerprint
`66c09bc84ec4b84fced6a9a2850a5d52905389e7efe4b97293593793b8239260`;
the snapshot SHA-256 is
`badba66977f81c9f5fa2b2a8b6635c4a780ec44ed2f0306a72e4498f3d62bbda`.
A 271-file, 3,241,538,596-byte scan found no Gate 5 or Gate 6 credential
material, unredacted storage URL, `test_access=true`, or final-test artifact.
All exact worker sessions are absent. Both deployed caches remain mode `0444`,
size 59,295,793 bytes, and canonical SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.

### Gate 6 bounded five-reservation LOBSTER pilot (2026-08-23)

The first pilot study, `lobster_attr_f1pr_gate6_pilot_20260823a`, used immutable
contract SHA-256
`9b9837146ec92a07c9720e97b063cfbbeb6edba820af7e23e0ee400200a0f00b`.
Its first bounded wave received three SSH/tmux acknowledgements, but each worker
failed local preflight before claiming a reservation because real R08 training
had overwritten the tracked source file `kernelVGAE_Log.png`. All three
structured markers recorded `reservation_consumed=false`; the launch probe
matched no PostgreSQL trial or active tmux session and classified every attempt
`RECONCILED_PRETRIAL`. PostgreSQL retained exactly five `WAITING` reservations,
zero `RUNNING`, `COMPLETE`, `FAIL`, other, or guard rows. No blind redispatch
occurred.

The root cause was fixed in commit
`e94443d8b0161639d03eb5d529fac254ab748f98`: the live Plotter now writes beneath
the trial's configured `graph_save_path` instead of the immutable source root.
An AST regression test pins that destination, and 38 focused tuning and
distributed-integrity tests passed. Both dedicated roots were refreshed without
deleting their excluded run or cache trees. They then independently verified
commit `e94443d8b0161639d03eb5d529fac254ab748f98`, source tree SHA-256
`c96b0060529bbefe28af24127f69bdebb6d054e5e398e30c7460e8fb16b9d098`, and
deployment-manifest SHA-256
`09a3c92c82c4565e3ab8885f748f7be6d56972268c91bbce88b710344996d6cb`.

Because the corrected source differs from the first immutable contract, the
original study was not reused. Commit
`1eced03` added a fail-closed `RETIRED_PRECLAIM` lifecycle. Retirement requires
every exact reservation to remain `WAITING` and every attempted launch to have
a retry-safe `RECONCILED_PRETRIAL` probe with no tmux session or database trial;
it consumes no reservation, is idempotent, and blocks initialization and worker
dispatch. After checksum collection preserved all three worker failure trees,
the original study was retired with reason `source-contract-superseded`.
A deliberate dry-run dispatch was rejected before creating a wave: the launch
manifest count remained one. Its audit covers 30 files and 991,247 bytes with
zero protected-credential, storage-URL, test-access, test-split, or held-out
evaluation finding.

The replacement study `lobster_attr_f1pr_gate6_pilot_20260823b` binds the
corrected source under immutable contract SHA-256
`88e4032f8a708cd2520e76c0ae271cff373ece406836fd8e647f0224a6af952c`.
It reserved exactly five scientific trials with `max_parallel=3`, sampler seed
47, TPE startup target five, heartbeat 60, grace 600, and no replacement.
Preflight reported exactly five `WAITING` rows and no other trial. The committed
LOBSTER smoke configuration retained two epochs, at most eight validation
graphs, five evaluator repeats, 600-second training and evaluation limits, and
`skip_final_evaluation=true`.

Wave 1 launched the three qualified slots with dispatch sequences 1,000,000,
1,000,001, and 1,000,002 and reproduced sampler seeds 2,095,308,481,
499,541,639, and 2,412,751,057. It reached exactly three `COMPLETE` plus two
`WAITING` reservations and was terminal-probed and checksum-collected before
wave 2 began. Wave 2 launched only cs-cl-13 GPU 0 and cs-cl-17 GPU 0 with
dispatch sequences 2,000,000 and 2,000,001 and reproduced seeds 608,841,255 and
2,224,220,736. The within-wave worker intervals overlap by 212.036 and 213.198
seconds respectively, proving bounded real concurrency. Final PostgreSQL state
is exactly five `COMPLETE`, zero `FAIL`, `WAITING`, `RUNNING`, other, or guard
rows, with five unique worker, trial, budget, checkpoint, and artifact
identities.

All five real two-epoch trials produce the finite tiny-smoke value
`0.000019999800003999925` at exactly
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`. Each structured
evaluator uses validation split only, `decoded_node_edge`, both
`node_feature_decoder` and `edge_feature_decoder`, 14 node and 11 edge channels,
five repeats, and eight generated plus eight reference accepted graphs. Best
trial 0 is the deterministic first choice under the five-way tied validation
objective. These results qualify the bounded machinery only and are not a
model-quality claim.

Final terminal probes match all five worker markers one-to-one with reserved
PostgreSQL `COMPLETE` rows. Checksum collection and atomic controller merge
passed; a separate finalizer published the best trial/config, CSV/SUMMARY,
portable SQLite snapshot, and `FROZEN.json`. Independent reopen matches live
PostgreSQL at semantic fingerprint
`310f27b80fd2029ed1d2e568b8d04a36b0d407ef42aebfca3d1bf853d9f19518`;
the snapshot SHA-256 is
`add5bdddc0222e5299319c4296001a720c5b55f0848b0816bc6f71e2b2a94221`.
All five checkpoints pass explicit feature-head validation.

An exhaustive 458-file, 5,402,977,806-byte scan found no Gate 4, Gate 5, or
Gate 6 credential material, unredacted storage URL, `test_access=true`, test
split, or held-out/final-test evaluation indicator. Both remote source
manifests reverify after training, all exact study sessions are absent, and both
deployed caches remain mode `0444`, size 59,295,793 bytes, and canonical
SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`.
The pilot is complete and frozen. R09 clean-snapshot restoration is next; R10
remains prohibited until R09 passes.

### Gate 6 R09 clean-snapshot restoration (2026-08-23)

Commit `a3c53f3` added the fail-closed `restore` controller command. It accepts
only a frozen source tree and a fresh, separate destination: it copies the
portable SQLite snapshot into an atomic staging directory, verifies the exact
study contract and runtime fingerprint, and performs no PostgreSQL access. The
copied study must be quiescent and semantically identical before the command
regenerates `trials.csv`, `best_trial.json`, `best_config.yaml`, and
`SUMMARY.md` from the copied snapshot. Every regenerated aggregate is then
byte-compared with the frozen source, both snapshot copies are rehashed, and
the destination is atomically published with `RESTORED.json`. Equal, nested,
or already-existing destinations are rejected. Fifty-nine focused controller
and integrity tests passed.

R09 ran from a fresh controller process with all `GRAPHVAE_BO_*`,
`PGPASSFILE`, `PGPASSWORD`, and PostgreSQL environment variables explicitly
absent. It restored `lobster_attr_f1pr_gate6_pilot_20260823b` into the new root
`lobster_attr_f1pr_gate6_pilot_20260823b_r09_restore_20260823a` without opening
PostgreSQL. The copied snapshot retained SHA-256
`add5bdddc0222e5299319c4296001a720c5b55f0848b0816bc6f71e2b2a94221` and
reopened with the exact semantic fingerprint
`310f27b80fd2029ed1d2e568b8d04a36b0d407ef42aebfca3d1bf853d9f19518`.
It contains exactly five `COMPLETE` reservations with budget indexes `0..4`
and no other trial state.

The regenerated aggregate SHA-256 values are
`1c771503e16eac3c7e613fe6dca8ca820a1f2a15395d35eaf361b83b8b107d1e`
for `trials.csv`,
`c26bb1fac4de03f58653484b27c7c35ddcaf37ed6c5315c0431b2edb92f3fb68`
for `best_trial.json`,
`2f38001e94a19abfe2b675be992ab6751f1b9b3edcc9b12bbfbbd4ac55469fb8`
for `best_config.yaml`, and
`d3c7fa4774a85aec37f7bc3c9612898f0c73afa3fd19267534fb8234f41f5bbc`
for `SUMMARY.md`; each matches the original frozen byte stream. Best trial 0
and its value `0.000019999800003999925` are unchanged. Selection remains
validation-only at exactly
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`, with
`test_access=false`.

An independent eight-file, 574,791-byte restore-tree audit tested eleven pieces
of protected Gate 4-6 credential material and found zero credential or
unredacted-storage-URL match and zero test-access, test-split, or held-out
evaluation indicator. A deliberate second restore to the now-existing
destination was rejected before mutation. R09 passes; the separate explicit
post-freeze LOBSTER R10 evaluation may now begin.

### Gate 6 R10 explicit post-freeze LOBSTER evaluation (2026-08-23)

R10 used the existing explicit `--evaluate-best-on-test` interface only after
the pilot freeze and successful R09 restore. To leave the canonical pilot tree
immutable, the controller created the fresh sibling root
`lobster_attr_f1pr_gate6_pilot_20260823b_r10_heldout_20260823a` on cs-cl-13
and staged read-only copies of only `FROZEN.json`, `best_trial.json`,
`best_config.yaml`, the portable snapshot, and selected trial 0 checkpoint.
Every staged input matched the frozen controller copy before launch. In
particular, the snapshot SHA-256 remained
`add5bdddc0222e5299319c4296001a720c5b55f0848b0816bc6f71e2b2a94221`
and the selected checkpoint SHA-256 was
`e50bfa1bc3f8caf6348b706cbb03800ac7e458e8b95e69a1db797469a0f64fa0`.

The first and only explicit evaluation ran on cs-cl-13 physical GPU 0 with
`CUDA_VISIBLE_DEVICES=0` and logical `cuda:0`. All PostgreSQL and protected BO
credential environment variables were absent. The exact deployed evaluator
source from the frozen pilot contract was retained, its deployment manifest
verified before and after, and its `tune_graphvae_attribute_weights.py` hash was
`6c0fcb91fc3e152908646c11c360e168a649e61ad313280455c746fe2ffb4f64`.
The run was bounded to eight held-out reference/generated graphs, generation
batch size four, five evaluator repeats with seeds `0..4`, and a 600-second
phase limit. No optimization or training command ran.

The result explicitly records split `test` and primary mode
`decoded_node_edge`. It used both `node_feature_decoder` and
`edge_feature_decoder`, exact node/edge dimensions 14/11, cache SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`,
and held-out split fingerprint
`c327e6e90ebd3dd76608387907ac3b5536e1a9dd4049fcba221ec13bcc188229`.
The tiny qualification result is Attr-F1PR
`0.000019999800003999925`, precision `0.0`, and recall `1.0`; this is bounded
mechanism evidence, not a model-quality claim. Its evaluator JSON SHA-256 is
`d1f71c066ee0d9b9a5d9634ee8aa0c385ee0c6fa2d05c2f1c9cf63377929934f`
and its selection-record SHA-256 is
`eeeee0d0392e5a707f43364ac1aa142ec823f265839b7fcaa34d01f5d3d9cec5`.

Post-run reopen still reports exactly five `COMPLETE` reservations, no other
state, best trial 0, and semantic fingerprint
`310f27b80fd2029ed1d2e568b8d04a36b0d407ef42aebfca3d1bf853d9f19518`.
Thus R10 created zero trial and could not change the validation ranking or its
exact selection objective
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`;
`test_access_during_optimization` remains false. A six-file, 16,301-byte
evaluation-output scan tested eleven protected credential materials and found
zero credential, password-assignment, storage-URL, or `test_access=true`
finding. The evaluator child is reaped, the LOBSTER cache remains mode `0444`
with its exact size and hash, and both the remote and collected R10 evidence
trees are sealed read-only. Audit SHA-256 is
`f7282def0fa2ef770d6615328ccc5ae8bf8eea865bca3e250405338639ee6652`;
the R10 freeze-manifest SHA-256 is
`ebd559cdaba8e84e1119d98de3fdd5d22131a11daed4d4c515daa071ab46cc28`.
R10 passes.

### Gate 6 final non-QM9 acceptance audit (2026-08-23)

The complete non-PostgreSQL distributed BO suite passed all 62 current tests
across the unit, integrity, launcher, and smoke-configuration files. The
separately marked PostgreSQL suite passed all 18 current tests through the
rotated protected Gate 6 controller connection. It created and removed only
UUID-named `graphvae_bo_pytest_*` studies; a post-suite query found zero such
study remaining and no qualification study changed.

The final read-only live/snapshot audit found exactly the three intended Gate 6
PostgreSQL study names. R08 remains `FROZEN` with exactly three `COMPLETE`
reservations, the superseded pilot remains `RETIRED_PRECLAIM` with its exact
five unconsumed `WAITING` reservations, and replacement pilot B remains
`FROZEN` with exactly five `COMPLETE` reservations. There are zero guard or
replacement rows. All eight completed trial artifacts have unique
contract-scoped worker-run and checkpoint-path identities, reproduce their
derived dispatch seeds, and match their recorded checkpoint and evaluator
hashes. Both live/snapshot semantic fingerprints and portable snapshot hashes
remain exactly those recorded in the R08 and pilot sections.

Every completed optimization artifact revalidates validation split,
`decoded_node_edge`, both attribute decoders, five repeats, eight generated and
reference graphs, 14 node and 11 edge channels, the canonical cache, and exact
objective
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`. R09 still matches all
four aggregates without PostgreSQL. R10 still records zero created trial and an
unchanged validation ranking before its separate test-only result. The final
778-file, 8,646,105,843-byte scan tested eleven protected credential materials
and found zero credential, password-assignment, storage-URL,
`test_access=true`, or premature-held-out indicator.

Both cs-cl-13 and cs-cl-17 finally reverified protected `verify-full`
authentication, runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`,
current deployment source tree
`c96b0060529bbefe28af24127f69bdebb6d054e5e398e30c7460e8fb16b9d098`,
and deployment-manifest SHA-256
`09a3c92c82c4565e3ab8885f748f7be6d56972268c91bbce88b710344996d6cb`.
Their caches remain mode `0444`, exactly 59,295,793 bytes, and canonical hash;
their Gate 6 credential directories/files remain `0700`/`0600`. No Gate 6 tmux
session remains, and the qualified TITAN RTX pool is still cs-cl-13 GPU 0 plus
cs-cl-17 GPUs 0 and 1.

The ignored canonical closure tree is sealed read-only at
`gate6_non_qm9_final_audit_20260823a`. Its final audit SHA-256 is
`2f387696fac88b759ba02fc1e42644e34a6888d1220323291ca8b7eed883101f` and
its `GATE6_COMPLETE.json` SHA-256 is
`579b3c88f5c1e9a7808c101d112f4afee1a451a76ad390d8f62f29e8750dc7fb`.
Gate 6 is complete within the authorized non-QM9 scope. Gate 7 and full-QM9
Bayesian optimization were not executed.

### User-authorized LOBSTER production-equivalent extension (2026-08-23)

The user authorized broader distributed and scientific qualification on
LOBSTER while continuing to exclude full-QM9 Bayesian optimization. This
extension exercises the production machinery and seeks meaningful LOBSTER
attribute-loss weights; it is neither a substitute for QM9 data nor evidence
about QM9 model quality.

The two-epoch Gate 6 pilot is mechanism evidence only. All five objective
values were tied at the numerical floor, so those trials do not establish a
useful weight choice. The next campaign therefore uses the production GraphVAE
architecture (`graphEmDim=1024`, full-batch training) with the immutable
`optimal_v2` LOBSTER cache and a bounded 2,000-epoch signal/search budget.
Validation uses all ten graphs, ten generation/evaluator repeats, and the exact
selection objective
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`. Optimization retains
`skip_final_evaluation=true`, validation selection, `test_access=false`, and
both attribute decoders.

Capacity qualification begins with nine candidate physical GPUs on six hosts:
cs-cl-09 GPU 1; cs-cl-13 GPU 0; cs-cl-16 GPU 0; cs-cl-17 GPUs 0 and 1;
cs-cl-19 GPUs 0 and 1; and cs-cl-26 GPUs 0 and 1. Inclusion in the candidate
mapping is not eligibility. A slot enters the final pool only after protected
verify-full authentication, exact source/runtime/cache/schema checks, logical
GPU isolation, sufficient local retention space, and a fresh real
fixed-parameter hardware study. Heterogeneous hardware must meet the existing
absolute Attr-F1PR tolerance of 0.02; failures consume their reservations and
are never replaced. cs-cl-36 is excluded because its available scratch space
is insufficient for retained campaign artifacts, and the occupied cs-cl-09
GPU 0 is excluded.

The pinned Python is `/localhome/mirzaei/miniconda3/envs/micro/bin/python` on
all candidate hosts except cs-cl-19, whose qualified environment is
`/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python`.

Before staging or study creation, the dedicated PostgreSQL role was rotated to
a newly generated protected `lobster-production` credential generation. The
new controller authenticated through `sslmode=verify-full`, and the Gate 6
generation was explicitly rejected. The six host-specific worker environments,
passfiles, CA files, and rotation metadata remain outside repository/source/
cache/artifact roots; credential directories are mode `0700` and files mode
`0600`. cs-cl-09, cs-cl-13, cs-cl-16, cs-cl-17, and cs-cl-26 use protected
directories beneath `/localhome`. Because cs-cl-19 had zero free bytes there,
its protected directory uses `/local-scratch2/mirzaei`, separate from its
deployment and artifact roots.

The first remote authentication probe revealed that the password-free storage
URL still embedded the controller CA path, overriding each worker's
`PGSSLROOTCERT`; it also exposed the incorrect initially committed cs-cl-19
Python path. The probe reached no authenticated database session. Its
connection exception unfortunately printed the password-free database endpoint
to transient controller output, violating the URL-redaction rule, although it
contained no password, passfile content, or credential hash and was not saved
to a repository or study artifact. Host-specific protected URLs were corrected
to their actual CA paths, the cs-cl-19 Python mapping was fixed, and subsequent
exception-suppressed probes passed verify-full authentication and permission
checks on all six hosts. No source/cache deployment, study, reservation, or
training occurred in this credential step.

Before the 30-reservation search, one fresh fixed uniform-weight calibration
must prove that 2,000 epochs produces a finite, non-floor validation signal. If
it does not, the search must not launch: a higher bounded LOBSTER budget is
frozen and calibrated instead. The eventual search uses exactly 30
reservations, TPE startup count five, log ranges `[0.01, 10]` for
`alpha_node_feat` and `alpha_edge_feat`, and no more concurrent workers than
the eligible pool. Reservation zero is the predetermined uniform pair `(1,1)`;
the other 29 reservations use the contracted TPE path. This mixed reservation
schedule requires a fail-closed, immutable interface and isolated PostgreSQL
tests before study creation.

The controller now accepts that interface as a versioned JSON reservation
plan. The plan must contain exactly one entry for every budget index, cover
`0..N-1` without gaps or duplicates, contain only contracted finite parameters,
and assign an unsigned 32-bit training seed. It is embedded in the study hash.
Initialization pre-enqueues each exact fixed map (including an empty map for a
TPE reservation) and records its seed on the reservation. The worker verifies
both fields before suggesting any parameter and resolves its training seed from
the matching entry. Study-wide fixed parameters and a reservation plan are
mutually exclusive.

The exact 30-entry search plan is committed at
`configs/bayesian_optimization/lobster_attr_f1pr_search_reservations_30.json`.
Seventy-two non-PostgreSQL distributed tests pass. Nineteen isolated
PostgreSQL tests pass, including preservation of mixed fixed/empty maps and
per-reservation seeds through native claims; the post-suite check found zero
residual `graphvae_bo_pytest_*` study.

Staging found two byte-distinct NumPy 1.24.3 builds: cs-cl-13, cs-cl-17, and
the controller reproduced the previously qualified runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`,
while cs-cl-09, cs-cl-16, cs-cl-19, and cs-cl-26 initially produced
`1cde83f9ca07f35c1492bd26d15ce3a29eb924e57e3bab9d31820111d9179572`.
The fingerprint requirement was not weakened. Instead, the exact qualified
NumPy package and native-library directories were staged read-only beneath
each dedicated source root's `.runtime/python_overlay`, with user-site loading
and bytecode writes disabled. Source synchronization now excludes `.runtime/`
so a later clean deployment cannot silently remove the qualified overlay. All
six hosts subsequently reproduced the exact `e142...` fingerprint without
modifying their shared Conda environments.

Clean source commit `174a844` is staged in the six dedicated roots at tree
SHA-256 `1fdaa75ccbac2d6cb78c75631f47aba0515cb0cd3c86991e1c7e111509be4db9`.
Every host verifies that source, the exact runtime, protected storage
construction, and the canonical read-only cache: 59,295,793 bytes, SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`,
70/10/20 splits, all ten validation graphs, and 14/11 node/edge dimensions.
Deployment and cache manifests are read-only. All nine candidate GPUs were
idle and each physical index mapped to exactly one logical `cuda:0`; their
models and reported MiB are GTX TITAN X 12,288 (cs-cl-09), TITAN RTX 24,576
(cs-cl-13 and both cs-cl-17 slots), Quadro RTX 4000 8,192 (cs-cl-16), GTX
1080 Ti 11,264 (both cs-cl-19 slots and cs-cl-26 GPU 0), and TITAN X Pascal
12,288 (cs-cl-26 GPU 1).

The fresh study `lobster_attr_f1pr_production_hw_signal_20260823a` will combine
the real fixed-parameter hardware check and the pre-search signal calibration:
exactly nine reservations, `max_parallel=9`, sampler seed 53, fixed uniform
weights `(1,1)`, and one 2,000-epoch trial on every candidate slot. It uses the
committed candidate hardware policy, full ten-graph validation, ten repeats,
and no held-out access. Every reservation is consumed exactly once. The
30-reservation search remains blocked unless every eventual production slot
completes and meets the 0.02 Attr-F1PR tolerance and the resulting uniform
objective is finite and strictly above the previous numerical floor.

Initialization of that study correctly created nine fixed `(1,1)` `WAITING`
reservations, but prelaunch audit showed that startup count five would cap its
first synchronous wave at five workers. With no launch attempt or claim, study
A was preserved and moved through the supported `RETIRED_PRECLAIM` lifecycle;
all nine reservations remain unconsumed `WAITING`. Replacement study
`lobster_attr_f1pr_production_hw_signal_20260823b` has the same scientific
contract and exact budget but freezes startup count nine, so all nine fixed
reservations can launch in one wave. It is `READY` with indexes `0..8`, contract
SHA-256 `f1dd8d355c93d8e65ce1b4bc3cadd54d1ded2260f61ef12db028ee9dd9def424`,
and its six-host/nine-slot preflight passes.

That preflight also exposed that one global remote credential path cannot
represent cs-cl-19's protected scratch fallback and the other five hosts'
`/localhome` paths. The controller now accepts a mutually exclusive per-host
credential-environment mapping, requires its host set to equal the repository
mapping, and rejects relative/control-character paths before dispatch. The
committed mapping contains paths only, never credential contents. Sixty-two
focused unit and launcher tests pass. At that commit checkpoint study B was
entirely unlaunched and all nine reservations remained `WAITING`.

The nine-slot launch qualification for study B is now complete and frozen.
Wave 1 failed definitely before any SSH/tmux launch or PostgreSQL claim because
the controller output root was missing its public deployment and cache
manifests; its evidence is retained and all nine reservations remained
`WAITING`. After those immutable public inputs were installed, wave 2 launched
all nine slots simultaneously with dispatch sequences `2000000..2000008` and
nine unique deterministic sampler seeds. The immediate audit found exactly nine
`RUNNING` reservations, no waiting reservation, no unreserved guard row, and no
retry-safe launch. The terminal audit found exactly nine `COMPLETE`
reservations, nine unique worker-run identities and atomic completion markers,
zero failures, and no replacement or duplicate dispatch.

Study `lobster_attr_f1pr_production_hw_signal_20260823b` is `FROZEN` at contract
SHA-256 `f1dd8d355c93d8e65ce1b4bc3cadd54d1ded2260f61ef12db028ee9dd9def424`.
All nine fixed uniform `(1,1)` trials produced a finite full-validation
`evaluation.modes.decoded_node_edge.summary.f1_pr.mean`, so the 2,000-epoch
signal budget is scientifically usable and is far above the previous numerical
floor. The exact observed values were `0.5604518010256058` on the GTX TITAN X,
`0.6452243612592389` on the Quadro RTX 4000, `0.682776942562006` on all three
TITAN RTX slots, and `0.6017607207239262` on the GTX 1080 Ti and TITAN X Pascal
slots. The nine-slot objective range `0.12232514153640028` exceeds the frozen
absolute tolerance `0.02`; consequently the heterogeneous nine-slot hardware
audit correctly fails and publishes no eligible slot set. This is a scientific
pool-qualification failure, not a concurrency failure. The three TITAN RTX
slots agree exactly, but a fresh hardware-homogeneous qualification must freeze
that pool before the 30-reservation search; the failed all-hardware study is
never rewritten or used to rank weights.

Collection from all six dedicated roots completed with zero manifest or
collision failures. Finalization verified every checkpoint and evaluator
artifact, the exact validation objective path, both feature decoders, and
`test_access=false`. A PostgreSQL-independent restore reproduced all aggregate
outputs under runtime fingerprint
`e142a6b3516ef87ac4f0aa29092a41cf26ecfa91aa08a8c2702edbbcff12a1e1`.
Post-trial checks on all six hosts reverified source tree
`1fdaa75ccbac2d6cb78c75631f47aba0515cb0cd3c86991e1c7e111509be4db9`,
the read-only 59,295,793-byte cache at SHA-256
`928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660`,
and read-only deployment/cache manifests. Text artifact scans found no storage
URL, password/private-key material, held-out split, or true test-access flag.
The frozen study, restore tree, checkpoints, and generated trial artifacts
remain ignored operational evidence and are not committed.

The repository inventory contains no additional TITAN RTX slot: the only other
mapped hosts expose two GTX 1080 Ti GPUs (`cs-cl-18`) and one RTX 2080
(`cs-cl-36`). The committed production slot file therefore freezes the
hardware-homogeneous pool by model identity as `cs-cl-13:gpu0`,
`cs-cl-17:gpu0`, and `cs-cl-17:gpu1`. Its separate immutable hardware policy
retains the same `0.02` objective tolerance, training-loss tolerance, and exact
runtime fingerprint. A fresh three-reservation fixed-uniform study must pass
that policy before these slots can run the 30-reservation LOBSTER search.

That homogeneous qualification now passes. Fresh study
`lobster_attr_f1pr_production_hw_titanrtx_20260823a` froze exactly three
uniform `(1,1)` reservations at contract SHA-256
`bf61e72c7283c9ac4df72213a50e543e94428c78288a49de4d60f3a0cb49a4cc`.
One simultaneous worker ran on each frozen slot with unique dispatch sequence,
sampler seed, worker-run identity, trial identity, and atomic terminal marker.
All three reservations completed; no waiting, failed, unreserved, replacement,
or duplicate trial exists. Each independently reproduced validation Attr-F1PR
`0.682776942562006`, so the objective range and every pairwise difference are
exactly zero against the immutable `0.02` tolerance. The hardware audit marks
all three slots eligible.

The two-host collection completed without collision or manifest failure, the
study is `FROZEN`, and its PostgreSQL-independent restore reproduced all
aggregate hashes under the exact runtime fingerprint. Every evaluator used the
validation split, `decoded_node_edge`, and both GraphVAE feature decoders; no
test split was accessed. Credential/storage/test-access scans are clean. Final
source verification and the read-only cache SHA-256/size check pass on both
hosts. Dedicated qualified repository, Python, and protected-environment path
mappings now constrain subsequent production search dispatch and collection to
these two hosts; the mappings contain paths only.

The BO ranking is frozen before confirmation. The selected pair and the
predetermined uniform pair are then refit under identical 10,000-epoch budgets
and training seeds 0, 1, and 2. Their validation evaluations use identical
generation and evaluator seed schedules. Only after those candidates and the
comparison rule are frozen may an explicit held-out LOBSTER evaluation run;
held-out results cannot select, rerank, or trigger more training. The final
report includes per-seed outcomes, paired differences, means, uncertainty,
all failed attempts, portable restoration, cache/source integrity, lifecycle
state, and credential/test-access scans.

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
  before Gate 6.

Section 14 and Gates 1-3 are complete locally, and Gate 4 now passes with its
separate R02 mock and R03 real LOBSTER studies. Protected credential deployment,
authenticated `verify-full` preflight, source/cache staging, GPU isolation,
collection, audit, freeze, and restore all pass on the one approved Gate 4 slot.
Gate 5 is authorized, its dedicated two-host mapping is frozen, and protected
authentication plus source/cache/runtime/GPU qualification now pass on both
workers. R04 now passes with its separate frozen two-reservation mock study;
R05 passes with a different frozen two-reservation real LOBSTER study, and R06
passes with separate frozen definite-prelaunch and post-launch-ambiguity mock
studies. R07 passes with one consumed failed reservation, exact orphan cleanup,
native stale reconciliation, and a verified tombstone. The final full-suite,
isolated PostgreSQL, cross-study snapshot/artifact, cache, and redaction audits
all pass, so Gate 5 is complete. Gate 6 preparation has begun: credentials are
rotated, the three-slot mapping and bounded non-QM9 study contracts are frozen,
and the dedicated source/cache/runtime/auth/GPU deployment passes. R08 passes on
all three intended slots. The first pilot attempt was safely retired preclaim
after exposing source-tree plot mutation; the corrected replacement pilot is
frozen with exactly five `COMPLETE` reservations and all audits passing. R09
reopens that portable snapshot without PostgreSQL and byte-reproduces every
aggregate output. R10 evaluates only its frozen selected checkpoint on bounded
LOBSTER held-out data, creates no trial, and leaves the validation ranking
unchanged. The final suites, cross-study audit, redaction pass, and two-host
integrity checks all pass, so Gate 6 is complete in the authorized non-QM9
scope. Gate 7/full-QM9 BO remains explicitly excluded and was not executed.
