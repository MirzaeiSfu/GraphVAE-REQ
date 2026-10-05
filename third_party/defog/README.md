# DeFoG (baseline)

This is the DeFoG code used for every DeFoG result in this project. It is a snapshot; the code with full history
is in the fork.

- **Upstream:** DeFoG by Manuel Madeira, Yiming Qin et al. MIT License (see `LICENSE`). The upstream README is `UPSTREAM_README.md`.
- **Fork:** [`MirzaeiSfu/defog`](https://github.com/MirzaeiSfu/defog), branch `feat/frozen-graphvae-benchmark`.
- **Base commit:** [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e)
  (2026-09-01, "Add frozen GraphVAE benchmark adapter"). The frozen-benchmark runs record this commit as
  `defog_commit` in their `job_record.json`.

## Local changes on top of `c631697`

`qm9_defog_lab24_20260914`, `QM9_FULL_TEST_20260925` and `GRID_DEFOG_SEED6_20260924` ran with these three files
modified; `qm9_defog_campaign_20260914` ran with only `src/datasets/qm9_dataset.py` modified (identical content).
None of these changes were committed to the fork. The modified versions are the ones included here:

- `src/main.py`
- `src/datasets/qm9_dataset.py`
- `src/analysis/ggmeval/evaluation/models/gin/gin.py`

All other campaigns (GRID, TRIANGULAR_GRID, LOBSTER, PTC, PROTEINS, MUTAG, AIDS, OGB) ran unmodified `c631697`.
To see the local changes:

```bash
git clone https://github.com/MirzaeiSfu/defog.git && cd defog && git checkout c631697
diff -r . ../GraphVAE-REQ/third_party/defog --exclude=.git
```

Which DeFoG run used which code is listed in
[`reports/EXPERIMENT_CODE_PROVENANCE.md`](../../reports/EXPERIMENT_CODE_PROVENANCE.md).
