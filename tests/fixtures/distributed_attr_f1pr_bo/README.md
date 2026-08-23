# Distributed Attr-F1PR cache fixture

`qm9_tiny_cache.pkl` is a deterministic, synthetic Gate 3 transport and
fingerprint fixture. It contains 21 training, 3 validation, and 6 held-out test
graph records with explicit atom and bond channel meanings. It is deliberately
small and is not a substitute for the real QM9 cache required by Gate 4.

Rebuild the cache and its canonical manifest in the qualified Python 3.8
environment:

```bash
python tests/fixtures/distributed_attr_f1pr_bo/build_tiny_cache.py
python scripts/prepare_graphvae_attr_bo_cache.py \
  --base-config configs/bayesian_optimization/qm9_graphvae_attr_f1pr_smoke.yaml \
  --cache-path tests/fixtures/distributed_attr_f1pr_bo/qm9_tiny_cache.pkl \
  --output tests/fixtures/distributed_attr_f1pr_bo/dataset_cache_manifest.json \
  --max-graphs 3
```

The builder writes the cache atomically with pickle protocol 4. The preparer
validates `dataset-cache-v4` metadata, fingerprints all three splits and both
feature schemas, verifies that the source cache did not change, and publishes
the JSON manifest atomically.
