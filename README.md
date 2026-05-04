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
2. Load cache from `cache_datasets/<dataset>.pkl` by default if present.
3. Otherwise:
   - load raw graphs (`data.py:list_graph_loader`)
   - apply BFS ordering
   - build node/edge one-hot features (`util.py:build_onehot_features`)
   - build train/test `Datasets`
   - save all artifacts to cache
4. Optional motif pipeline (`motif_counting/motif_store.py`, `motif_counting/motif_counter.py`).
5. Build model (`model.py`) and run training.
6. Save logs, generated graphs, and checkpoints.

## Cache Keys (Current)

By default, `cache_datasets/<dataset>.pkl` stores at least:
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
- `RuleBasedMotifStore` (`motif_counting/motif_store.py`)
- `RelationalMotifCounter` (`motif_counting/motif_counter.py`)

It writes/reads motif pickle files under `./cache_motifs/<database_name>.pkl` by default.

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
python main.py --config configs/default.yaml
```

### Important defaults in current code

- `configs/default.yaml` is now a QM9 baseline preset:
  - `dataset=QM9`
  - `model=GraphVAE`
  - `motif_loss=false`
  - `tiny_overfit=false`
- CLI aliases now accept:
  - `-model GraphVAE` for the baseline (`kipf` internally)
  - `-batchSize` as a legacy alias for `--train_batch_size`
  - `--no-tiny_overfit` to disable the debug preset explicitly

## Outputs

During/after runs you will see artifacts in `graph_save_path`.
By default, auto-created run directories now live under `runs/<run_name>/`.

Run logs now live inside the run directory:
- `train.log`
- `mmd.log` (validation MMD logging)

Run artifact directories contain:
- generated graph `.npy` dumps
- model checkpoints (`model_<epoch>_<batch>`)
- training plot images
- `generated_graph_train/` samples from intermediate training snapshots

Raw datasets are under `data_raw/` by default. OGB downloads/caches should live under `data_raw/ogb/`.

Dataset cache files are under `cache_datasets/` by default, but can be redirected with `DATASET_CACHE_DIR` or `--dataset_cache_dir`.

Motif cache files are under `cache_motifs/` by default, but can be redirected with `MOTIF_CACHE_DIR` or `--motif_cache_dir`.

## Key Files

- `main.py`: full training/eval pipeline and cache orchestration
- `model.py`: encoder/decoder and `kernelGVAE`
- `data.py`: dataset loading, preprocessing, `Datasets`, merge wrapper
- `util.py`: kernels, one-hot feature builders, utility layers
- `motif_counting/motif_store.py`: DB -> motif pickle builder
- `motif_counting/motif_counter.py`: batched motif counting
- `stat_rnn.py`, `mmd_rnn.py`, `eval/`: MMD/statistics utilities

## Practical Notes

- `accu` printed in training is adjacency reconstruction accuracy, not node-feature accuracy.
- Node feature loss is currently masked by true node counts to ignore padded nodes.
- If you only optimize node-feature loss, adjacency metrics may stay near random.
