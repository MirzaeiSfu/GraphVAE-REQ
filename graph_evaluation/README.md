# Reusable Graph-Generator Evaluation

This folder is an independently installable, PyTorch Geometric-first
evaluation layer for GraphVAE, DeFoG, and other graph generative models. It
keeps generators, graph interchange, and research evaluator implementations
separate.

The primary comparison is deliberately matched:

```text
same PyG graphs + same upstream GConv
    ├── random initialization
    ├── GraphCL training
    └── InfoGraph training
```

The existing DGL Random-GIN remains available through an optional adapter for
historical comparison. It is not treated as the control for GraphCL because
its backend, GIN implementation, and preprocessing differ.

## Design boundaries

The package owns:

- the strict PyG interchange contract;
- deterministic largest-component normalization;
- DGL/PyG conversion without NetworkX;
- explicit training/evaluation split inputs at the command boundary;
- safe tensor-only graph artifacts and checkpoint manifests;
- process isolation and unified reports.

It does **not** copy or modify the contrastive research repository. A checkout
is supplied at runtime. It also does not import `main.py`, GraphVAE model
classes, DeFoG classes, or generator checkpoints.

Both research repositories contain a top-level module named `evaluation`.
Loading them into one Python process can resolve imports from the wrong
repository. Every evaluator invocation therefore runs in a fresh subprocess.

## Installation

Install the core package into the environment used to manipulate PyG graphs:

```bash
pip install -e ./graph_evaluation
```

The restricted `.pt` loader requires PyTorch 2.0 or newer. The implementation
was integration-tested with Python 3.8.20, PyTorch 2.1.2, PyG 2.6.1, and
PyGCL 0.1.2; each run records its actual installed versions.

GraphCL and InfoGraph additionally require the dependencies imported by the
released implementation:

```bash
pip install -e './graph_evaluation[contrastive]'
```

PyGCL's random-walk augmentation also imports `torch-scatter` and
`torch-sparse`. Install wheels matching the evaluator environment's exact
PyTorch and CUDA/CPU build, following the
[PyG extension installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Do not let pip replace an existing PyTorch/CUDA stack with an incompatible
build.

Install DGL separately only when converting old DGL artifacts or running the
legacy evaluator. Select the DGL wheel appropriate for the environment's
PyTorch and CUDA versions.

The released contrastive code is expected at commit:

```text
fb6bc26237eb21d7617fd41b22b4bb26ab29bf95
```

Prepare a checkout outside this package:

```bash
git clone https://github.com/hamed1375/Self-Supervised-Models-for-GGM-Evaluation.git
git -C Self-Supervised-Models-for-GGM-Evaluation checkout \
  fb6bc26237eb21d7617fd41b22b4bb26ab29bf95

ggm-eval inspect-upstream \
  --upstream-repo Self-Supervised-Models-for-GGM-Evaluation
```

See [UPSTREAMS.md](UPSTREAMS.md) before redistributing that checkout.

## Quick evaluation with bundled trained GNNs

The package includes three independently trained GraphCL-GIN checkpoints for
each of `AIDS`, `ENZYMES`, `MUTAG`, `PROTEINS`, `PTC`, `QM9`, and
`ogbg-molbbbp`. This convenience path is entirely inside `graph_evaluation`;
it does not call or import the repository's root `main.py`.

From Python:

```python
from ggm_eval import evaluate_with_trained_gnns

result = evaluate_with_trained_gnns(
    dataset="PROTEINS",
    generated="run/generated_graphs.pt",
    reference="run/real_test_graphs.pt",
    output_dir="run/trained_gnn_evaluation",
    device="auto",
)

print(result["summary"])
```

Or use the equivalent command:

```bash
ggm-eval evaluate-trained \
  --dataset PROTEINS \
  --generated run/generated_graphs.pt \
  --reference run/real_test_graphs.pt \
  --output-dir run/trained_gnn_evaluation \
  --device auto
```

The wrapper accepts common aliases such as `protein` and `ogbg`, selects seeds
`0 1 2`, verifies every checkpoint against its manifest size and SHA-256
digest, and delegates to the same isolated evaluation runner described below.
Use `seeds=[0]` in Python or `--seeds 0` on the command line only when a
single-encoder diagnostic is intentional.

The trained weights are bundled, but the research source and its runtime
dependencies are not. Point to the pinned checkout explicitly:

```bash
export GGM_EVAL_CONTRASTIVE_REPO=\
../Self-Supervised-Models-for-GGM-Evaluation
```

Alternatively, pass `upstream_repository=...` in Python or
`--upstream-repo ...` on the command line. Without an explicit setting, the
wrapper searches common sibling checkout locations. The runner still rejects
a checkout at the wrong revision or with local modifications unless
`allow_unpinned_upstream=True`/`--allow-unpinned-upstream` is deliberately
selected.

The generated and reference files must use the same dataset and feature
schema as the selected trained encoder. For example, the bundled PROTEINS
encoders expect `default|export=decoded_node`, while the AIDS, QM9, and
`ogbg-molbbbp` encoders expect their documented node-and-edge schemas. The
worker checks these identities before computing metrics.

The checkpoint registry, complete training provenance, and integrity digests
are in
`src/ggm_eval/trained_models/manifest.json`; the binary layout and usage
constraints are documented in `src/ggm_eval/trained_models/README.md`.

## PyG interchange contract

Each graph is an individual homogeneous `torch_geometric.data.Data`:

```python
Data(
    x=node_features.float(),          # [N, D_node]
    edge_index=edge_index.long(),     # [2, E]
    edge_attr=edge_features.float(),  # [E, D_edge], optional
    source_node_ids=original_ids,     # [N] int64, optional provenance
    num_nodes=N,
)
```

For an undirected graph:

1. Every edge must be represented in both directions.
2. Reverse directions must carry identical edge features.
3. Self-loops and duplicate directed edges are forbidden.
4. Categorical IDs must be converted to one-hot floats by the producer.
5. Generated and reference collections must use identical feature dimensions,
   channel ordering, and meanings.
6. Pass individual `Data` objects, not a PyG `Batch`.

`x` remains mandatory for a topology-only producer; use a float column of
ones. Graphs with no non-self-loop edge are rejected instead of silently
dropping failed generator samples.

The package keeps the deterministic largest connected component and records a
SHA-256 digest over the normalized topology and feature tensors. Optional
`source_node_ids` survive normalization and serialization so a producer can
trace retained rows back to its decoded tensors; local `0..N-1` IDs are used
when the producer does not supply them.

Dimensions alone cannot prove that channel meanings agree. Put stable
`dataset` and `feature_schema` identities in every training, generated, and
reference artifact. If either identity is declared by one collection, the
worker requires the same value from all compared collections.

### Why the package still writes a `.pt` artifact

PyG `Data` is the interchange contract; `.pt` is only its durable transport
between generator jobs and isolated evaluator processes. There is no `.npz`
layer. An NPZ file would still need a custom ragged-graph schema, dtype rules,
and feature-alignment rules, so it would not remove the need for a contract.

The saved `.pt` does not pickle PyG classes. It contains only tensors and
primitive dictionaries that map one-to-one back to individual `Data` objects.
That gives the worker a restricted-loading path while retaining PyG tensor
dtypes and avoiding a second graph representation.

### Saving graphs from DeFoG or another PyG model

```python
from ggm_eval import save_pyg_collection

save_pyg_collection(
    "defog_generated.pt",
    generated_data_list,
    metadata={
        "generator": "DeFoG",
        "dataset": "PROTEINS",
        "feature_schema": "proteins-node-edge-onehot-v1",
        "split": "generated",
    },
)
```

The `.pt` payload contains tensors and primitive metadata rather than pickled
`Data` objects. An adjacent `.pt.json` file records dimensions, versions, and
the collection digest. The digest is also stored inside the payload and
verified whenever the collection is loaded.

Raw `torch.save(list_of_data)` files are accepted only with
`--trusted-input`. Loading such a file uses Python pickle semantics and should
never be enabled for an untrusted artifact.

Validate an artifact before training or evaluation:

```bash
ggm-eval validate --graphs defog_generated.pt
```

### GraphVAE exports from `main.py`

A completed GraphVAE graph-generation run writes the contract directly:

```text
<run-dir>/real_train_graphs.pt
<run-dir>/real_test_graphs.pt
<run-dir>/generated_graphs.pt
```

The generated collection is captured during the same sampling pass used by
the final structural evaluation. Adjacency, node attributes, and edge
attributes therefore come from the same latent vector. The real training and
test collections are projected into the same feature mode as the generator:
models without a node or edge decoder do not receive unavailable real
attributes on only one side of the comparison.

Use `real_train_graphs.pt` only for encoder training. Use
`real_test_graphs.pt` and `generated_graphs.pt` only for the frozen-encoder
comparison. The main workflow creates no DGL graph artifact; the DGL adapters
below are retained for old files and the historical evaluator.

## Existing DGL artifacts

Convert a file created with `dgl.save_graphs`:

```bash
ggm-eval dgl-to-pyg \
  --input generated_attributed_graphs.bin \
  --output generated.pt \
  --dataset PROTEINS \
  --feature-schema proteins-node-edge-onehot-v1
```

The adapter reads DGL edges in edge-ID order, merges matching directions,
rejects conflicting edge attributes, removes input self-loops, and writes the
strict bidirectional PyG representation.

The reverse adapter exists for the legacy evaluator and external tools:

```bash
ggm-eval pyg-to-dgl \
  --input generated.pt \
  --output generated.bin
```

No adapter passes through NetworkX.

## Training encoder checkpoints

The graph collection passed to `train` must contain only the real training
split used to train the generator. Generated, validation, and held-out test
graphs must not be included.

Create matched random checkpoints:

```bash
ggm-eval train \
  --graphs real_train.pt \
  --encoder gin-random \
  --feature-mode decoded_node_edge \
  --seeds 0 1 2 \
  --upstream-repo ../Self-Supervised-Models-for-GGM-Evaluation \
  --output-dir encoders/pyg_random
```

Train GraphCL:

```bash
ggm-eval train \
  --graphs real_train.pt \
  --encoder graphcl \
  --feature-mode decoded_node_edge \
  --seeds 0 1 2 \
  --epochs 100 \
  --upstream-repo ../Self-Supervised-Models-for-GGM-Evaluation \
  --output-dir encoders/graphcl
```

Train InfoGraph by replacing `--encoder graphcl` with
`--encoder infograph`.

`train` uses the upstream `GConv`, GraphCL/InfoGraph training functions, and
augmentations without editing the checkout. The adapter saves an additional
state-dict checkpoint and provenance manifest. The upstream whole-model
artifact remains under each run's `runtime/saved_models/` for auditing.

The upstream `data_utils.py` imports DGL and Ray even for its native-PyG,
non-parallel path. The isolated worker supplies minimal import shims when
those packages are absent and marks the adapter-selected features as already
prepared. This prevents an unused DGL/NetworkX conversion and does not replace
any model, augmentation, training, embedding, or metric implementation.

For edge features, current PyG releases ask the upstream custom MLP for an
`in_features` property when constructing `GINEConv`. The worker exposes that
property from the already-existing first linear layer. This is a runtime API
compatibility bridge only: model computation and checkpoint parameter names
remain unchanged.

PyGCL 0.1.2 imports scikit-learn 0.23, which still refers to scalar aliases
removed by NumPy 1.24. The worker restores those five historical aliases
before loading PyGCL. This compatibility shim does not change graph tensors,
augmentations, losses, or model code in the research checkout.

The matching `torch-sparse` CUDA random-walk kernel can fail with an illegal
memory access on the worker GPU stack. For GraphCL CUDA runs, the adapter
therefore executes only that sparse walk on its CPU backend and transfers the
sampled node indices back to the graph device. Start-node sampling, random
augmentation selection, walk length, subgraph extraction, and all learned
operations retain the released implementation.

The worker also imports `sklearn.metrics` explicitly before calling the
released precision/recall and density/coverage code. Some supported
scikit-learn releases do not populate that namespace from `import sklearn`
alone.

By default the upstream checkout must have both the pinned commit and a clean
worktree. `--allow-unpinned-upstream` permits a different or modified checkout
only for an intentional experiment, and the resulting provenance remains in
reports.

### Reusable real-split export

Encoder pre-training does not require running a graph generator. Export the
real splits directly through the same loader, categorical one-hot builder,
BFS normalization, and safe PyG serializer used by `main.py`:

```bash
python scripts/export_real_pyg_splits.py \
  --dataset MUTAG \
  --data-dir /scratch/graph-data \
  --split-mode legacy_80_20 \
  --output-dir graph_evaluation_inputs/MUTAG
```

The command writes `real_train_graphs.pt` and `real_test_graphs.pt`; a
paper-style split also writes `real_validation_graphs.pt`. Training consumes
only the training artifact. The adjacent manifests record the dataset,
feature schema, split and selection seeds, optional source cap, normalization
strategy, tensor dimensions, and collection digest.

The exporter accepts `PROTEINS`/`protein`, `MUTAG`, `PTC`, `AIDS`, `ENZYMES`,
`QM9`, and `ogbg`/`ogbg-molbbbp`. Its defaults are deterministic:
`selection_seed=0`, `split_seed=123`, and largest-component BFS ordering.
Choose the split mode used by the generator experiment. The last 20% is the
held-out test set in both built-in modes, so a legacy 80/20 encoder split does
not consume the paper 70/10/20 test partition.

### Distributed seven-dataset GraphCL campaign

The checked-in campaign files make the current run reproducible:

- `CLUSTER_GRAPHCL_GIN_20260725.txt` assigns one dataset to each CUDA device;
- `CLUSTER_REPO_PATHS_GRAPHCL_20260725.txt` includes only workers with adequate
  free scratch space;
- `scripts/cluster_run_graphcl_schedule.sh` validates inputs and launches one
  isolated `tmux` session per dataset;
- `scripts/summarize_graphcl_training.py` verifies and hashes every collected
  checkpoint, then writes JSON, CSV, and Markdown summaries.

The schedule trains the untouched upstream GraphCL-GIN implementation for
100 epochs with seeds `0 1 2`, three GIN layers, hidden width 32, orthogonal
initialization, and the released Lipschitz limiter. Node-only datasets use
`decoded_node`; molecular datasets with bond features use
`decoded_node_edge`.

Each worker needs:

1. this repository;
2. a clean sibling checkout named
   `Self-Supervised-Models-for-GGM-Evaluation` at the pinned commit;
3. `.graphcl_deps/` containing PyGCL 0.1.2 plus `torch-scatter` and
   `torch-sparse` wheels built for that worker's PyTorch/CUDA combination;
4. `graph_evaluation_inputs/20260725/<dataset>/real_train_graphs.pt`.

After those inputs are synchronized, launch:

```bash
scripts/cluster_run_graphcl_schedule.sh \
  --repo-paths CLUSTER_REPO_PATHS_GRAPHCL_20260725.txt
```

Collect the completed run roots and validate the aggregate:

```bash
scripts/cluster_collect_results.sh \
  --repo-paths CLUSTER_REPO_PATHS_GRAPHCL_20260725.txt \
  --remote-run-root runs/graphcl_gin_20260725 \
  --date-prefix 20260725

python scripts/summarize_graphcl_training.py \
  --collected-root collected_runs/20260725/graphcl_gin_20260725 \
  --schedule CLUSTER_GRAPHCL_GIN_20260725.txt \
  --seeds 0 1 2 \
  --output-dir collected_runs/20260725/graphcl_gin_20260725_summary
```

For this campaign, PROTEINS and `ogbg-molbbbp` use the paper 70/10/20 split;
MUTAG, PTC, AIDS, ENZYMES, and QM9 use 80/20. QM9 is deterministically capped
at 10,000 source molecules before splitting (8,000 training graphs) to keep
the released 100-epoch, three-seed loop computationally practical. That cap
is part of its artifact metadata and must be reported when using the
checkpoint. All other datasets use every graph retained by their repository
loader.

### Feature modes

- `topology_control`: one constant node channel, no edge attributes;
- `decoded_node`: original node attributes, no edge attributes;
- `decoded_edge`: one constant node channel and original edge attributes;
- `decoded_node_edge`: original node and edge attributes.

A trained encoder must be evaluated with the same mode used during training.
Train separate checkpoints for separate modes. Zeroing attributes only at
evaluation would place the trained encoder out of distribution.

## Evaluating frozen PyG encoders

Use `evaluate-trained` above for the bundled GraphCL-GINs. The lower-level
`evaluate` command remains available for newly trained GraphCL, InfoGraph, or
matched random-GIN checkpoints:

```bash
ggm-eval evaluate \
  --generated defog_generated.pt \
  --reference real_test.pt \
  --checkpoint encoders/graphcl/seed_0/checkpoint.pt \
  --checkpoint encoders/graphcl/seed_1/checkpoint.pt \
  --checkpoint encoders/graphcl/seed_2/checkpoint.pt \
  --upstream-repo ../Self-Supervised-Models-for-GGM-Evaluation \
  --output-dir reports/defog_graphcl
```

The worker uses the upstream feature extraction and implementations of
Fréchet distance, precision/recall, density/coverage, MMD-RBF, and linear MMD.
It embeds each graph collection once per frozen checkpoint and aggregates
uncertainty across independent encoder seeds.

Generated and reference sets are compared at equal counts. By default every
reference graph is used and the generated collection must contain at least as
many graphs. `--max-graphs N` deterministically selects the first `N`
reference graphs and the first `N` generated graphs.

Do not repeat one trained checkpoint and interpret the repeated number as
encoder uncertainty.

## Legacy DGL Random-GIN

```bash
ggm-eval evaluate-legacy \
  --generated defog_generated.pt \
  --reference real_test.pt \
  --legacy-repo .. \
  --repeats 10 \
  --output-dir reports/defog_legacy_random_gin
```

The command converts strict PyG graphs directly to DGL and invokes the
existing `eval.attributed_gin.evaluate_dgl_feature_modes` in an isolated
process. No legacy metric code is duplicated in this package.

## Output layout

Training:

```text
encoders/graphcl/
├── training_summary.json
├── seed_0/
│   ├── checkpoint.pt
│   ├── training.json
│   └── runtime/
│       ├── saved_models/       # artifact written by untouched upstream code
│       ├── stdout.log
│       └── stderr.log
└── seed_1/...
```

Evaluation:

```text
reports/defog_graphcl/
├── evaluation.json
├── evaluation.md
├── checkpoint_000/
│   ├── evaluation.json
│   └── runtime/{stdout.log,stderr.log}
└── checkpoint_001/...
```

Both `evaluate` and `evaluate-trained` use this layout. With the bundled
three-seed evaluators, directories `checkpoint_000` through
`checkpoint_002` are present.

The JSON preserves per-checkpoint values, input digests, upstream revision,
upstream dirty-state, feature mode, dimensions, and nearest-neighbour
configuration. The evaluator rejects mixed encoder types, feature modes, or
model/training configurations, repeated checkpoint paths, or input digests
rather than aggregating incomparable checkpoints.

## Recommended reporting

Report all of the following when compute permits:

1. `gin-random` from the PyG contrastive repository;
2. `graphcl`;
3. optionally `infograph`;
4. legacy DGL Random-GIN only for historical continuity.

The valid training-effect comparison is PyG `gin-random` versus PyG
`graphcl`/`infograph`. Do not attribute differences between legacy DGL
Random-GIN and PyG GraphCL solely to pretraining.

## Current limitations

- The released GraphCL training loop has no validation/early-stopping path;
  the adapter preserves that behavior and records the requested epoch count.
- The upstream repository has no tests and targets an older dependency stack.
- GPU GraphCL training relies on the adapter moving input `Data` objects to
  the configured device because the released GraphCL loop does not do it.
- Distributed Ray preprocessing is intentionally disabled; this package uses
  the upstream non-parallel path because inputs already satisfy the contract.
- The upstream Lipschitz limiter runs during train-mode forwards. Consequently,
  its released default constrains trained encoders but does not transform a
  never-trained Random-GIN checkpoint. Use `--no-limit-lipschitz` for every
  compared encoder if that distinction is undesirable, and report the choice.
- The contrastive checkout currently has no explicit software license; it is
  not vendored here.
- Framework-specific `.pt` and `.bin` artifacts are not ideal permanent
  archival formats. Their manifests record versions and hashes to make
  compatibility failures visible.
