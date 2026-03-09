# GraphVAE-REQ

GraphVAE-REQ is a GraphVAE / GraphVAE-MM training codebase with:
- graph generation and reconstruction
- optional node/edge feature decoding heads
- optional relational motif counting pipeline (MySQL -> pickle -> batched counting)

The main entry point is `main.py`.

## What The Current Code Trains

Important: in the current training loop, the final loss is hard-coded as:
- `alpha_kernel_cost = 0`
- `alpha_node_feat = 1.0`
- `alpha_edge_feat = 0.0`

So by default it optimizes **node feature loss only** (`node_feat_loss`), not graph reconstruction/kernel loss.

## Environment (Python 3.8.20)

Use your micro/micromamba env with Python 3.8.20:

```bash
micromamba create -n graphvae-req python=3.8.20 -y
micromamba activate graphvae-req
pip install -r requirements.txt
```

Notes:
- `torch-geometric` may require extra wheels (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) depending on OS/CUDA.
- `dgl==0.4.3.post2` is kept to match this codebase.

## Project Flow (`main.py`)

1. Parse CLI arguments and set experiment config.
2. Load cache from `dataset_cached/<dataset>.pkl` if present.
3. Otherwise:
   - load raw graphs (`data.py:list_graph_loader`)
   - apply BFS ordering
   - build node/edge one-hot features (`util.py:build_onehot_features`)
   - build train/test `Datasets`
   - save all artifacts to cache
4. Optional motif pipeline (`motif_store.py`, `motif_counter.py`).
5. Build model (`model.py`) and run training.
6. Save logs, generated graphs, and checkpoints.

## Cache Keys (Current)

`dataset_cached/<dataset>.pkl` stores at least:
- `list_adj`, `list_x`, `list_label`
- node/edge raw features and metadata
- node/edge one-hot tensors and metadata
- `test_list_adj`, `val_adj`
- `list_graphs`, `list_test_graphs`
- split tensors for multi-graph datasets:
  - `list_x_train/test`, `list_label_train/test`
  - `list_noh_train/test`, `list_eoh_train/test`

The load path expects these keys to exist (new-cache format).

## Motif Pipeline Requirements

Motif counting uses:
- `RuleBasedMotifStore` (`motif_store.py`)
- `RelationalMotifCounter` (`motif_counter.py`)

It writes/reads motif pickle files under `./db/<database_name>.pkl`.

If no motif pickle exists, it connects to MySQL with defaults:
- host: `localhost`
- user: `fbuser`
- password: `''`

Required databases:
- `<database_name>`
- `<database_name>_setup`
- `<database_name>_BN`

## Running

Typical run:

```bash
python main.py -dataset QM9 -model GraphVAE-MM -e 20000 -batchSize 200 --device cuda
```

### Important defaults in current code

- `--tiny_overfit` is declared with `default=True`, so tiny-overfit mode is active unless you change code.
- In tiny-overfit mode, code forces:
  - `motif_loss=False`
  - `task='debug'`
  - reduced epoch/visualization settings

If you want full training/motif flow, set tiny-overfit default to `False` in `main.py`.

## Outputs

During/after runs you will see artifacts in `graph_save_path`:
- `log.log`
- `_MMD.log` (validation MMD logging)
- generated graph `.npy` dumps
- model checkpoints (`model_<epoch>_<batch>`)
- training plot images

Dataset cache files are under:
- `dataset_cached/`

Motif cache files are under:
- `db/`

## Key Files

- `main.py`: full training/eval pipeline and cache orchestration
- `model.py`: encoder/decoder and `kernelGVAE`
- `data.py`: dataset loading, preprocessing, `Datasets`, merge wrapper
- `util.py`: kernels, one-hot feature builders, utility layers
- `motif_store.py`: DB -> motif pickle builder
- `motif_counter.py`: batched motif counting
- `stat_rnn.py`, `mmd_rnn.py`, `eval/`: MMD/statistics utilities

## Practical Notes

- `accu` printed in training is adjacency reconstruction accuracy, not node-feature accuracy.
- Node feature loss is currently masked by true node counts to ignore padded nodes.
- If you only optimize node-feature loss, adjacency metrics may stay near random.
