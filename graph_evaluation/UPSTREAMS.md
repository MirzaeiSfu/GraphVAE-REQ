# Upstream evaluator policy

## Random-GIN legacy engine

GraphVAE-REQ already vendors the required subset of:

```text
uoguelph-mlrg/GGM-metrics
```

The upstream project declares the MIT License. Local compatibility changes
and provenance are described in:

```text
../third_party/ggmeval/UPSTREAM_README.md
```

This package calls the existing public DGL boundary in
`eval/attributed_gin.py`; it does not carry a second copy.

## Contrastive PyG engine

Repository:

```text
https://github.com/hamed1375/Self-Supervised-Models-for-GGM-Evaluation
```

Pinned revision:

```text
fb6bc26237eb21d7617fd41b22b4bb26ab29bf95
```

Files used through imports:

```text
GIN_train_pyg.py
data_utils.py
evaluation/gin_evaluation.py
evaluation/models/gin/gin_pyg.py
```

The checkout is loaded only in an isolated worker process. Inputs are already
PyG `Data` objects, so the upstream DGL/NetworkX conversion is bypassed.
`data_utils.py` nevertheless imports DGL and Ray eagerly. If either package is
absent, the adapter installs worker-local compatibility shims for those unused
imports; any attempt to enter the DGL branch fails explicitly.

PyG 2.6 also changed how `GINEConv` discovers the input width of a custom MLP.
The worker exposes `MLP.in_features` from the upstream first linear layer at
runtime. No upstream file, tensor operation, or state-dict key is modified.
It likewise loads `sklearn.metrics` explicitly for versions where
`import sklearn` does not expose that submodule eagerly.

At the time this adapter was added, the public repository did not contain a
`LICENSE` file. Public visibility alone does not grant redistribution rights.
For that reason:

- no contrastive source file is copied into GraphVAE-REQ;
- the user supplies the checkout explicitly;
- the exact revision and dirty state are recorded in every checkpoint and
  evaluation report;
- unpinned or locally modified revisions require an explicit CLI flag.

Teams distributing a combined environment should obtain appropriate license
clarification from the upstream authors.

## Why subprocess isolation is mandatory

Both evaluator repositories define the top-level Python package:

```python
import evaluation
```

Python caches imported modules in `sys.modules`. Altering `sys.path` after one
engine has been imported does not reliably switch the package implementation.
The package therefore never imports both engines in one interpreter. The
parent process exchanges only file paths and JSON with workers.
