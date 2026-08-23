# GraphVAE attribute-loss Bayesian optimization

`scripts/tune_graphvae_attribute_weights.py` maximizes **Attr-F1PR** on a
fixed validation split. The exact objective is:

```text
evaluation.modes.decoded_node_edge.summary.f1_pr.mean
```

This is feature-aware Random-GIN precision/recall over adjacency plus the node
and edge attributes emitted by GraphVAE's three decoder paths. It is not the
topology-only F1-PR in `stat_rnn.py`. Generated attributes are grouped-argmax
decoded from the model logits and are never replaced, repaired from reference
data, endpoint-filtered, or substituted with degree features.

## Requirements and isolation

Install the BO client stack with `pip install -r requirements-bo-py38.txt`.
Distributed mode requires exactly Optuna 4.2.1 and psycopg2-binary 2.9.10;
the ordinary requirements also pin Optuna 4.2.1 so the broad historical range
cannot select a different native state machine. The base YAML must use
`split_mode: paper_70_10_20` with a positive
validation fraction. The driver rejects the legacy split because that code
uses a prefix of the training set as "validation". It also rejects
`ideal_Evalaution`, disabled dataset caching, sanity-check-only runs, and
tiny-overfit mode.

All trials use the same base configuration, training budget, cache/split seed,
training seed, generation seed, evaluator seeds, evaluator repeat count, and
number of accepted validation graphs. By default only `alpha_node_feat` and
`alpha_edge_feat` are sampled log-uniformly from `1e-3` through `1e2` by a
seeded Optuna TPE sampler. `--tune-alpha-motif` additionally samples the
repository parameter `alpha_motif_loss`; it requires `motif_loss: true` in the
base YAML. No topology, KL, or other motif setting is changed.

Training runs with `skip_final_evaluation: true`, which saves the final-epoch
checkpoint without invoking `main.py`'s automatic held-out test evaluation.
The optimizer then calls the attributed evaluator with exactly
`--split validation --modes decoded_node_edge`. Missing node/edge decoder heads,
missing edge attributes, non-finite metrics, or failed subprocesses mark only
that Optuna trial failed; logs and a failure record remain in its trial folder.

## Smoke test

The mock path exercises Optuna, SQLite persistence, trial directories, output
parsing, and best-result artifacts without training a model:

```bash
python scripts/tune_graphvae_attribute_weights.py \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --study-name qm9_attr_f1pr_smoke \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr_smoke \
  --trials 2 \
  --device cpu \
  --mock
```

Re-running the same command resumes `study.sqlite3`; `--trials` is the target
number of finished trials, rather than an additional count.

## Real study

```bash
python scripts/tune_graphvae_attribute_weights.py \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --trials 30 \
  --split-seed 123 \
  --training-seed 0 \
  --generation-seed 123 \
  --evaluator-seed 0 \
  --evaluator-repeats 5 \
  --device cuda
```

Useful controls include `--alpha-node-feat-min/max`,
`--alpha-edge-feat-min/max`, `--max-graphs` (`0` means every retained
validation graph), `--generation-batch-size`, `--nearest-k`,
`--adjacency-threshold`, and subprocess timeouts. Keep the output directory and
study name unchanged to resume.

Each trial stores its sampled weights, resolved YAML, seeds, checkpoint and
SHA-256, validation Attr-F1PR/precision/recall, elapsed times, logs, and any
failure. Study-level outputs are `best_config.yaml`, `best_trial.json`,
`trials.csv`, `study.sqlite3`, `study_definition.json`, and `SUMMARY.md`.
Resumption checks the recorded study definition and refuses to mix trials made
with different base-config content, ranges, seeds, evaluator settings, device,
or training budget.

## Explicit final test evaluation

Test data is never evaluated by the optimization command. After the best
hyperparameters have been selected, opt in separately:

```bash
python scripts/tune_graphvae_attribute_weights.py \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --evaluate-best-on-test \
  --generation-seed 123 \
  --evaluator-seed 0 \
  --evaluator-repeats 5 \
  --device cuda
```

This verifies the selected checkpoint hash, evaluates only that checkpoint in
`decoded_node_edge` mode against the held-out test split, and writes results
under `final_test/`. It does not start or continue optimization.
When the generation/evaluator seed or repeat flags are omitted, this command
inherits their values from `best_trial.json`.

## Distributed PostgreSQL workflow

Distributed mode uses native Optuna `RDBStorage` and accepts PostgreSQL only.
It rejects SQLite and shared/copy-based storage. Put the URL in a protected
environment variable; never pass a URL or password on a command line. The
normal production form uses hostname-verifying TLS and a protected libpq
password file:

```bash
export GRAPHVAE_BO_STORAGE_URL='postgresql+psycopg2://graphvae_bo@DB_HOST:5432/graphvae_attr_bo?sslmode=verify-full'
export PGPASSFILE=/protected/outside/repository/graphvae_bo.pgpass
```

The password file and its parent directory must be private, the file must be
mode `0600`, and neither file is staged by the repository scripts. Diagnostics,
manifests, worker commands, CSV, Markdown, and JSON contain only the storage
environment-variable name or a redacted host/database identity.

Prepare one already-created cache. This fails if it is missing and never
regenerates it:

```bash
python scripts/prepare_graphvae_attr_bo_cache.py \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --cache-path cache_datasets/EXACT_CACHE.pkl \
  --output dataset_cache_manifest.json \
  --max-graphs 0
```

Initialize exactly `N` empty `WAITING` reservations. Initialization is
idempotent for an identical definition/count and fills only missing budget
indexes after an interrupted `INITIALIZING` operation:

```bash
python scripts/run_distributed_graphvae_attr_bo.py init \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --dataset-cache-manifest dataset_cache_manifest.json \
  --trials 30 --sampler-seed 0 --max-parallel 3
```

Preflight and dry-run rendering are separate from execution:

```bash
python scripts/run_distributed_graphvae_attr_bo.py preflight \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --repo-paths CLUSTER_REPO_PATHS.txt \
  --python-paths CLUSTER_MICRO_PYTHON_PATHS.txt \
  --slots CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt --dry-run

python scripts/run_distributed_graphvae_attr_bo.py run \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr.yaml \
  --repo-paths CLUSTER_REPO_PATHS.txt \
  --python-paths CLUSTER_MICRO_PYTHON_PATHS.txt \
  --slots CLUSTER_GRAPHVAE_ATTR_BO_SLOTS_SAMPLE.txt \
  --max-parallel 3 --dry-run
```

Actual SSH/tmux dispatch additionally requires `--execute-remote`. Do not use
that acknowledgement until Gates 2 and 3 pass and the intended worker slots
are explicitly approved. Workers use `CUDA_VISIBLE_DEVICES=<physical>` and
logical `--device cuda:0`, run exactly one `study.optimize(n_trials=1)`, and
exit. Each dispatch seed is derived from the study seed and immutable dispatch
sequence by the roadmap's SHA-256 formula, and TPE always uses
`constant_liar=True`.

Status, local staged collection, and freeze/finalization are explicit:

```bash
python scripts/run_distributed_graphvae_attr_bo.py status \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr \
  --json runs/bayesian_optimization/qm9_attr_f1pr/status.json

python scripts/run_distributed_graphvae_attr_bo.py finalize \
  --study-name qm9_attr_f1pr \
  --output-dir runs/bayesian_optimization/qm9_attr_f1pr
```

Finalization refuses any reserved `WAITING`/`RUNNING` trial, audits every
selectable result against the exact validation JSON path and all cache/schema/
checkpoint hashes, records parameter-free unreserved guard rows separately,
and writes an atomically verified `study_snapshot.sqlite3`. A failed reserved
slot consumes budget and is never silently replaced. Parallel studies provide
trial reproducibility within the frozen tolerance, but do not promise an
identical TPE proposal order; use `--max-parallel 1` for study-path replay.

### Gate 1 and local Gate 2 tests

```bash
MICRO=/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python
$MICRO -m pytest -m unit

# Use a dedicated test database/role or test study namespace. The suite deletes
# only its own UUID-named Optuna studies; it never drops the database/schema.
export GRAPHVAE_BO_TEST_STORAGE_URL='postgresql+psycopg2://TEST_USER@localhost:5432/TEST_DB'
$MICRO -m pytest -m postgres tests/test_distributed_graphvae_attr_bo_postgres.py
```

The localhost-only `--allow-insecure-local-postgres` switch exists strictly for
a disposable Gate 2 endpoint. Production commands require
`sslmode=verify-full`. PostgreSQL tests skip only when the explicit test URL is
absent; a Gate 2 acceptance run must supply it and have no skips.
