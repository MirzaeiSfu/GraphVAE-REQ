# ENZYMES results

Collected from the completed three-seed ENZYMES runs on the Solar/Venus cluster.

- Dataset: `ENZYMES`
- Database: `enzymes_top4_undir_feat`
- Features: selected source features 3 through 6
- BFS strategy: `legacy_first_component`
- Epochs: `20000`
- Training batch size: `200`
- Best-checkpoint metric: `table3_priority`
- Final evaluation uses the saved `best_validation_mmd_model` on the test split.

## Settings

| Setting | Meaning | Motif loss | Retained motif combinations | alpha motif | Motif mode |
|---|---|---:|---:|---:|---|
| 01 | GraphVAE baseline | False | none | 0.00 | `calibrated_gaussian` |
| 03 | GraphVAE + motif, no temperature annealing | True | globally top 100 | 0.10 | `calibrated_gaussian` |

Both settings use deterministic execution and the same `paper_70_10_20` data split.

## Aggregate final test metrics across three seeds

Values are mean +/- sample standard deviation. Higher F1/precision/recall is better; lower MMD is better.

| Setting | Test F1-PR | Test MMD RBF | Precision | Recall | Degree MMD | Clustering MMD | Orbit MMD | Spectral MMD | Diameter MMD | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.9046 +/- 0.0210 | 0.0212 +/- 0.0319 | 0.8647 +/- 0.0367 | 0.9503 +/- 0.0089 | 0.0339 +/- 0.0047 | 0.0224 +/- 0.0022 | 0.0189 +/- 0.0107 | 0.0144 +/- 0.0028 | 0.0497 +/- 0.0279 | 0.9133 +/- 0.0167 | 0.0183 +/- 0.0259 | 13666203.0800 +/- 19973223.8000 |
| 03 | 0.9206 +/- 0.0502 | 0.0008 +/- 0.0005 | 0.9047 +/- 0.0658 | 0.9386 +/- 0.0423 | 0.0307 +/- 0.0043 | 0.0230 +/- 0.0021 | 0.0168 +/- 0.0054 | 0.0137 +/- 0.0010 | 0.0189 +/- 0.0038 | 0.9214 +/- 0.0171 | 0.0013 +/- 0.0002 | 5652769.1220 +/- 4039717.3210 |

## Per-seed final test metrics

| Setting | Seed | Test F1-PR | Test MMD RBF | Precision | Recall | Degree | Clustering | Orbit | Spectral | Diameter | 3rd-party F1-PR | 3rd-party MMD RBF | 3rd-party linear MMD trimmed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 0.8869 | 0.0044 | 0.8292 | 0.9550 | 0.0285 | 0.0213 | 0.0224 | 0.0122 | 0.0367 | 0.8966 | 0.0048 | 36588317.5000 |
| 01 | 1 | 0.8992 | 0.0011 | 0.8625 | 0.9400 | 0.0366 | 0.0248 | 0.0273 | 0.0135 | 0.0817 | 0.9132 | 0.0019 | 4410179.1250 |
| 01 | 2 | 0.9278 | 0.0580 | 0.9025 | 0.9558 | 0.0367 | 0.0209 | 0.0068 | 0.0176 | 0.0306 | 0.9301 | 0.0481 | 112.6014 |
| 03 | 0 | 0.8733 | 0.0012 | 0.8583 | 0.8900 | 0.0345 | 0.0226 | 0.0214 | 0.0129 | 0.0196 | 0.9244 | 0.0013 | 9203368.8750 |
| 03 | 1 | 0.9153 | 0.0010 | 0.8758 | 0.9592 | 0.0317 | 0.0253 | 0.0181 | 0.0148 | 0.0223 | 0.9030 | 0.0015 | 6497440.0620 |
| 03 | 2 | 0.9732 | 0.0003 | 0.9800 | 0.9667 | 0.0260 | 0.0212 | 0.0108 | 0.0135 | 0.0148 | 0.9368 | 0.0012 | 1257498.4300 |

## Interpretation

- Setting 03 improves mean local F1-PR from `0.9046` to `0.9206` and mean third-party F1-PR from `0.9133` to `0.9214`.
- The strongest improvement is RBF MMD: local MMD decreases from `0.0212` to `0.0008`, and third-party MMD decreases from `0.0183` to `0.0013`.
- Mean precision improves, while mean recall is slightly lower. Setting 03 also improves degree, orbit, spectral, and diameter MMD, while clustering MMD is slightly worse.
- Random-GIN linear MMD has extreme outliers in both settings; the values are preserved but should not be treated as the primary conclusion.

## New top-150, alpha-0.01 experiment status

This experiment is separate from the completed results above. As of `2026-07-15 07:20 PDT`, Slurm array job `232413` is running all three seeds on `cs-venus-03`:

| Seed | Latest epoch | Target | Final evaluation available? |
|---:|---:|---:|---|
| 0 | 10294 | 20000 | No |
| 1 | 11505 | 20000 | No |
| 2 | 11331 | 20000 | No |

Its final metrics are intentionally not mixed into the completed Setting 03 aggregate.

## Source files

- Setting 01: `solar:/project/cs-schulte-lab/ali/GraphVAE-REQ-aids-enzymes-top100-20260714/runs/solar_20260714/enzymes/setting_01/seed_*/final_metrics_summary.json`
- Setting 03: `solar:/project/cs-schulte-lab/ali/GraphVAE-REQ-aids-enzymes-top100-20260714/runs/solar_20260714/enzymes/setting_03/seed_*/final_metrics_summary.json`
- Top-150 run: `solar:/project/cs-schulte-lab/ali/GraphVAE-REQ-aids-enzymes-top100-20260714/runs/solar_20260715/enzymes/setting_03_top150_alpha001/seed_*/`
