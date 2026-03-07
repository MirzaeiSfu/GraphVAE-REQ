# GraphVAE-REQ

This repository contains a GraphVAE training pipeline with:
- GraphVAE / GraphVAE-MM generation and reconstruction training (`main.py`)
- Dataset loading, padding, BFS ordering, and feature preprocessing (`data.py`, `util.py`)
- Relational motif extraction and batched motif counting (`motif_store.py`, `motif_counter.py`)

## Current Pipeline (as implemented)

`main.py` runs the following steps:
1. Parse training + motif arguments.
2. Load dataset (or load a cached preprocessed dataset from `dataset_cached/<dataset>.pkl`).
3. Run BFS-based node reordering and build node/edge one-hot features.
4. Build `Datasets` objects for train/test splits.
5. If motif loss is enabled, initialize motif store/counter and run batched motif counting.
6. Train GraphVAE/GraphVAE-MM model.

## Key Files

- `main.py`: end-to-end run script (data pipeline, optional motif counting, model training)
- `data.py`: dataset loaders + `Datasets` class + `DataWrapper` for motif batching
- `util.py`: one-hot feature building and graph utilities
- `motif_store.py`: reads relational/BN rules from MySQL and caches them as pickle
- `motif_counter.py`: batched differentiable motif counting on GPU/CPU
- `model.py`: encoder/decoder and VAE model blocks
- `ReportedResult/`: generated outputs and logs from prior runs

## Datasets in Code

`data.py` includes loaders for multiple datasets, including:
- `QM9` (PyG-based branch with node/edge feature extraction)
- `IMDBBINARY`, `NCI1`, `MUTAG`, `COLLAB`, `PTC`, `PROTEINS`
- Synthetic families (`grid`, `triangular_grid`, `community`, `lobster`, etc.)

## Run

Typical run:

```bash
python main.py -dataset QM9 -model GraphVAE-MM -e 20000 -batchSize 200
```

Motif-related flags in `main.py`:

```bash
--database_name qm9
--graph_type homogeneous|heterogeneous
--rule_prune <bool>
--interactive
--graph_index_start <int>
--graph_index_end <int>
--batch_size <int>
--sanity_check_local_mults
--device cuda|cpu
```

## Motif Store / Counter Notes

When motif counting is active:
- `RuleBasedMotifStore` creates/uses a pickle cache for motif rules and metadata.
- `RelationalMotifCounter` loads that pickle and computes motif counts in batches.

Important current path assumption in code:
- Both `motif_store.py` and `motif_counter.py` currently use:
  - `/localhome/mirzaei/ali/gvae/GraphVAE-MC/db`

If your environment differs, update that path in both files.

## Cache

Preprocessed data is cached under:
- `dataset_cached/<dataset>.pkl`

Cached payload includes:
- adjacency/features/labels
- node and edge categorical features
- node and edge one-hot tensors + metadata
- train/test split artifacts

## Dependencies

`requirements.txt` and `requirements.yml` are legacy snapshots from the original codebase.
Current code paths additionally rely on packages used by motif and QM9 branches, such as:
- `torch_geometric`
- `pymysql`
- `pandas`

Install missing packages according to your environment.

## Project Status

This README reflects the repository state where motif counting modules (`motif_store.py`, `motif_counter.py`) are integrated into the training script and data pipeline.
