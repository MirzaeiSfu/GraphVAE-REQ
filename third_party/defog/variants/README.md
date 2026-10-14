# DeFoG versions used in the paper

`third_party/defog/` is DeFoG commit [`c631697`](https://github.com/MirzaeiSfu/defog/commit/c631697b9cd5a2474d22ba12de33943c6b49e53e)
(fork `MirzaeiSfu/defog`, branch `feat/frozen-graphvae-benchmark`), unchanged. Two datasets ran a modified version.
Their changes are stored here instead of as full copies:

| Version | Used for | Made from `c631697` by |
|---|---|---|
| `c631697` | GRID, TRIANGULAR_GRID, LOBSTER, PTC | nothing; run `third_party/defog/` as it is |
| `qm9` | QM9 | `qm9.patch` (3 files) |
| `proteins` | PROTEINS | `proteins.patch` (9 files changed, 2 removed) + `proteins_files/` (the data split and the compiled `orca` program) |

What the changes do:

- **`qm9.patch`**: seed fixes. `gin.py` no longer resets every random generator to 0 when it is imported (that made all
  training seeds start from the same model); `main.py` sets the training seed again just before building the model
  and prints a checksum of the initial weights; `qm9_dataset.py` uses the same 70/10/20 split as GraphVAE-REQ. The
  first two change model initialization, so the other datasets must not use this version.
- **`proteins.patch`**: the PROTEINS campaign's copy, which branched off before the frozen-split support was added to
  `c631697`. It also sets the training seed again before building the model, keeps only the final checkpoint, and
  changes the MUTAG/PROTEINS dataset code, spectral metrics and plots.

`scripts/retrain/run_experiment.py` builds a version automatically: it copies `third_party/defog/` into the run's
output folder, applies the patch, adds the extra files, and checks every file against `<version>.sha256`. The
`.sha256` lists were made from the folders the paper runs were launched from (docs, images and `__pycache__` left
out), so a run cannot start with code that differs from the paper's.

To build one by hand:

```bash
cp -r third_party/defog /tmp/defog_qm9 && cd /tmp/defog_qm9
git apply variants/qm9.patch                      # PROTEINS: git apply variants/proteins.patch && cp -r variants/proteins_files/. .
```
