# Frozen GraphVAE benchmark adapter

This branch adds a generator-side adapter for the frozen comparison package in
GraphVAE-REQ. It reads exact, pre-normalized train/validation/test PyG tensor
collections and never recreates a split from a random seed.

The adapter is selected with `dataset=frozen_graphvae`. Required overrides are
the logical dataset identity, artifact directory, and the three collection
digests. The GraphVAE-REQ `graph_evaluation/src` directory must be importable.

For benchmark training, set:

```text
general.wandb=disabled
general.validation_selection=loss
train.seed=<0|1|2>
```

This monitors `val_loss`, retains one best-validation checkpoint plus `last`,
and does not sample or inspect the held-out test collection during model
selection.

For final inference, load the selected checkpoint and set:

```text
general.test_only=/absolute/path/to/best.ckpt
general.generation_seed=12345
general.strict_generation=true
general.defog_commit=<this-branch-commit>
general.final_model_samples_to_generate=<accepted-reference-count>
```

The generated `generated_graphs.pt` sidecar records the training and generation
seeds, checkpoint digest, DeFoG commit, and accepted/rejected/attempt counts.
The GraphVAE-REQ campaign verifier must pass before Random-GIN evaluation.

