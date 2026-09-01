# Add DeFoG Baseline Roadmap

## Goal

Add DeFoG as a reproducible baseline without mixing its dependencies or large
artifacts into the GraphVAE-REQ source tree. Use the repository's existing
`graph_evaluation` PyG contract so DeFoG and GraphVAE are evaluated on the same
data split, features, graph count, and metrics.

## Current repository state

- `main` was cleaned, fully merged, tested, and pushed to `origin/main`.
- Merge commit `bc5c577` records the already-integrated `new_loss` branch.
- The focused matrix-motif tests passed: `22 passed`.
- One stash remains intentionally unpopped because it deletes two Grid configs.
- DeFoG source is intentionally not imported into GraphVAE-REQ.
- The source archive is `/local-scratch2/mirzaei/Abdolreza/defog.zip` with
  SHA-256 `84a8e774c49357c0c668b3a82f6e64b852033f3502c57b12c64b777a8bfa8d64`.
- The archive is valid and contains a complete DeFoG Git repository with 73
  commits. Its `main` is at upstream commit `365bda9` from
  `https://github.com/manuelmlmadeira/DeFoG.git`.
- The archive originally had 17 modified tracked files and 28 untracked files;
  its non-ignored source/configuration changes are now committed in the fork.
- The official fork is now `https://github.com/MirzaeiSfu/defog`. Commit
  `474f9405bdcdddc2d96cfedc3305172dffbe8fbd` contains the ZIP's non-ignored
  source/configuration snapshot on top of all 73 upstream commits and is
  published as both `main` and `graphvae-evaluation`.

## Integration design

Because DeFoG is needed only as an evaluation baseline, keep it as an
independent sibling Git repository. This preserves its exact 73-commit history,
keeps its dependencies isolated, and avoids adding unrelated upstream commits
and duplicate code to GraphVAE-REQ.

The archive is restored as `/local-scratch2/mirzaei/Abdolreza/DeFoG/`. Its
`graphvae-evaluation` branch adds the ZIP changes above `365bda9` and is
published to the permanent fork; no local-path submodule is used.

The resulting layout is:

```text
/local-scratch2/mirzaei/Abdolreza/
├── DeFoG/                   # independent source, history, and environment
└── GraphVAE-REQ/
    ├── baselines/defog/
    │   ├── README.md        # evaluation integration instructions
    │   ├── protocol.yaml    # one model-independent comparison contract
    │   ├── artifact_manifest.example.yaml
    │   └── verify_protocol.py
    ├── graph_evaluation/    # existing shared evaluator
    ├── tests/test_defog_protocol.py
    ├── reports/defog/       # small, versioned evaluation summaries
    └── runs/defog/          # ignored checkpoints and generated graphs
```

Record the DeFoG upstream URL, base commit, integration-branch commit, license,
and environment in `baselines/defog/README.md`. Do not unpack the ZIP into the
GraphVAE-REQ root.

## Minimal fairness contract

Do not try to make GraphVAE and DeFoG use one model-specific training config;
their configuration schemas and model hyperparameters are different. Instead,
store one small `baselines/defog/protocol.yaml` as the source of truth for every
condition that must be identical:

- dataset source and preprocessing, including graph filtering;
- exact train/validation/test indices and their digests;
- split fractions and seed;
- node/edge feature schema and category meanings;
- training and generation seeds used for each reported replicate;
- held-out reference collection and generated graph count;
- connected-component and invalid-sample policy;
- checkpoint-selection rule based only on validation data;
- evaluator type, evaluator checkpoint digests, limits, and metric settings.

The protocol should point to the existing GraphVAE config and list the DeFoG
Hydra overrides needed for the same common conditions. A small verifier must
fail before evaluation if either repository commit is wrong, either worktree is
dirty, the split indices differ, collection metadata/digests differ, or graph
counts and feature schemas do not match.

Model-specific settings such as architecture, learning rate, batch size, and
number of epochs remain in each model's native config and are recorded in the
artifact manifest. They should not be forced to identical values because they
do not represent equivalent computation across different model families.

The audited DeFoG changes already export `generated_graphs.pt` through
`ggm_eval`, so do not add another exporter unless validation shows that one is
needed. Use GraphVAE-REQ's frozen `real_test_graphs.pt` as the shared held-out
reference for final scoring.

## Work plan

- [x] Receive and integrity-check the DeFoG ZIP and its embedded Git history.
- [ ] Receive or locate one representative checkpoint, one generated-graph
      file, the target dataset name, and the current generation command.
- [x] Create `feat/defog-baseline` from an up-to-date, clean `main`.
- [x] Extract the ZIP into a temporary audit directory and inspect its layout,
      license, dependencies, embedded Git metadata, and ignored generated files.
- [x] Run `git fsck --full --strict`; the embedded object graph is healthy.
- [x] Review the 17 modified and 28 untracked DeFoG files for correctness,
      secrets, generated content, and appropriate commit grouping.
- [x] Restore DeFoG as an independent sibling repository and create a
      `graphvae-evaluation` branch at `365bda9`.
- [x] Commit the reviewed local DeFoG changes on that branch without rewriting
      the upstream commits.
- [x] Verify the resulting DeFoG repository is clean and retains the original
      73 commits. Publish the branch or make an external Git bundle if needed
      for portability.
- [x] Keep DeFoG source and checkpoints out of GraphVAE-REQ; add only the
      evaluation adapter, configs, manifests, tests, and small reports here.
- [x] Record the exact DeFoG branch commit in the integration protocol.
- [ ] Record SHA-256 digests for supplied checkpoints and graph files.
- [ ] Create a separate DeFoG environment using the versions required by its
      dependency files. Do not modify the working GraphVAE environment.
- [x] Add `baselines/defog/protocol.yaml` with the shared dataset, split,
      feature, seed, sample-count, checkpoint-selection, and evaluator rules.
- [x] Add a small fail-closed protocol verifier; avoid changing either model's
      training code unless a verified mismatch requires it.
- [ ] Export generated graphs from the independent DeFoG checkout into the
      ignored `GraphVAE-REQ/runs/defog/` artifact area.
- [ ] Export `real_train_graphs.pt`, `real_test_graphs.pt`, and
      `generated_graphs.pt` with matching dataset and feature-schema metadata.
- [ ] Validate all three files with `ggm-eval validate`.
- [x] Add focused protocol and artifact-manifest unit tests.
- [ ] Run a bounded export smoke test after a checkpoint is supplied.
- [ ] Evaluate generated graphs against the frozen held-out test graphs using
      `ggm-eval evaluate-trained` when the dataset has bundled encoders;
      otherwise train matched encoders only on `real_train_graphs.pt`.
- [ ] Save small JSON/CSV/Markdown summaries under `reports/defog/`.
- [x] Confirm that checkpoints, generated graphs, caches, and raw ZIP files are
      not staged in Git. Commit code, configs, tests, documentation, manifests,
      and small reports only.
- [x] Run relevant tests, review the diff, commit, push the feature branch, and
      merge it through the normal review workflow.

## Required graph contract

Each exported graph should be an individual PyG `Data` object with:

- floating-point `x` node features;
- bidirectional `edge_index` for undirected edges;
- aligned floating-point `edge_attr` when edge features exist;
- no self-loops or duplicate directed edges;
- the same feature dimensions, channel order, and meanings in generated and
  reference collections.

Use a single constant node-feature channel for topology-only graphs. Generated
and held-out reference collections must use the same frozen split and equal
graph counts. Never train an evaluator on the held-out test collection.

## Artifact policy

Keep large files under `runs/defog/` or in external artifact storage. Do not add
a global `*.pt` ignore rule because this repository intentionally tracks some
evaluator checkpoints. Every artifact manifest should record source revision,
license, dataset, split, preprocessing, feature schema, seeds, generation
count, command, environment versions, file size, and SHA-256 digest.

Treat checkpoints as trusted executable inputs. Inspect their source before
loading them, and prefer restricted or weights-only loading where supported.

## Resume instructions for another chat

Read this file, `graph_evaluation/README.md`, and
`graph_evaluation/examples/export_pyg_generator.py`. Then run `git status` and
inspect the supplied files without executing them. Continue from the first
unchecked item above, preserving the separate-environment and artifact rules.
