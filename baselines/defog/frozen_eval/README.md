# Frozen DeFoG/GraphVAE comparison

This package is the source of truth for the fair DeFoG comparison. It does
not recreate dataset splits from a seed. Instead, it exports the exact
serialized GraphVAE train, validation, and test identities into the shared
safe PyG tensor format and pins their hashes in `manifest.yaml`.

The primary campaign contains MUTAG, PROTEINS, GRID, LOBSTER, and
TRIANGULAR_GRID. PTC, QM9, and ENZYMES are excluded. MUTAG and PROTEINS use
categorical node features without edge features. The synthetic datasets are
topology-only. Original and LinkCorr FactorBase universes are evaluation
views of the same synthetic graph collections; they are not separate DeFoG
training runs.

## Fixed protocol

- exact serialized GraphVAE 70/10/20 splits, split seed 123;
- training seeds 0, 1, and 2;
- validation-only checkpoint selection;
- generation seed 12345;
- unconditional node-count sampling from train plus validation only;
- threshold 0.5 followed by symmetric, loop-free, deterministic-largest-
  component normalization;
- generated count equal to the accepted reference count;
- third-party Random-GIN evaluator seeds 0 through 9 with k=5;
- evaluator mean within each training seed, followed by mean and sample SD
  across the three training seeds.

Large caches, split artifacts, checkpoints, and generated graphs belong under
the ignored `runs/defog/frozen_eval/` tree. Only code, manifests, provenance,
and compact reports are committed.

## Commands

Export a pinned source cache:

```bash
PYTHONPATH=graph_evaluation/src \
python baselines/defog/frozen_eval/export_graphvae_cache.py \
  --manifest baselines/defog/frozen_eval/manifest.yaml \
  --dataset MUTAG \
  --output-root runs/defog/frozen_eval
```

Verify frozen references before training or evaluation:

```bash
PYTHONPATH=graph_evaluation/src \
python baselines/defog/frozen_eval/verify_campaign.py \
  --manifest baselines/defog/frozen_eval/manifest.yaml \
  --artifact-root runs/defog/frozen_eval \
  --references-only
```

Run one complete training-seed job (training, best-validation checkpoint
selection, generation, provenance verification):

```bash
CUDA_VISIBLE_DEVICES=0 \
python baselines/defog/frozen_eval/run_defog_job.py \
  --manifest baselines/defog/frozen_eval/manifest.yaml \
  --campaign baselines/defog/frozen_eval/campaign.yaml \
  --defog-root runs/defog/source \
  --artifact-root runs/defog/frozen_eval \
  --run-root runs/defog/frozen_eval/jobs \
  --dataset MUTAG --seed 0 \
  --python /path/to/defog-environment/bin/python
```

Each worker writes `job_record.json`. A job is usable only when that record
has `status: complete` and the generated collection passes the strict
campaign verifier. Worker allocation and GPU identity do not change any
scientific parameter.

Final generated collections must additionally provide a sidecar recording
training seed, generation seed, best-validation checkpoint hash, generation
attempts, accepted/rejected counts, and the DeFoG commit. The full verifier
rejects missing or mismatched provenance.
