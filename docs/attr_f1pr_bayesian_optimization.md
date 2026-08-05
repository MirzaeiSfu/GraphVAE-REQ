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

Install the repository environment with the added `optuna>=3.6,<5`
dependency. The base YAML must use `split_mode: paper_70_10_20` with a positive
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
