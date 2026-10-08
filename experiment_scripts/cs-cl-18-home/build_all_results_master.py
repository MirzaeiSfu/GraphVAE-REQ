#!/usr/bin/env python3
"""Combine the two verified GraphVAE result regimes into one master report."""

from pathlib import Path
import re


ROOT = Path("/local-scratch/localhome/mirzaei")
OLD = ROOT / "FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md"
NEW = ROOT / "NEW_HYPERPARAMETER_EXPERIMENTS_RESULTS_20260902.md"
TARGET = ROOT / "ALL_GRAPHVAE_MOTIF_RESULTS_AVAILABLE_20260902.md"


def nested_report(path: Path) -> str:
    text = path.read_text(encoding="utf-8").rstrip()
    return re.sub(r"^(#{1,4})(?=\s)", r"##\1", text, flags=re.MULTILINE)


header = """# All GraphVAE motif results currently available

Generated: 2026-09-02 (America/Vancouver).

This is the master report for the currently verified motif=False versus motif=True results. It preserves the complete tables, metric definitions, training rules, count-distance variants, motif-correlation results, node-feature evaluation, hyperparameters, model paths, recovery details, and current restart status from the two valid experiment regimes.

## Read this before comparing numbers

The report contains two separate experiment regimes. They must not be pooled into a single mean:

| Regime | Training seeds | Motif coefficient | Other loss weights | Included datasets | Current completeness |
| --- | ---: | ---: | --- | --- | --- |
| Earlier controlled comparison | 3 | 0.1 | Earlier weights, generally 1 for non-motif terms as documented below | MUTAG, PROTEINS, GRID, LOBSTER, TRIANGULAR_GRID; synthetic results both without and with LinkCorrelations=1 | Complete for the reported metrics |
| New code-weight comparison | 2 | 0.1 × dataset BCE (0.4–5) | Dataset-specific BCE/KL/node/edge weights | MUTAG, PROTEINS, GRID, LOBSTER, TRIANGULAR_GRID with LinkCorrelations=1 synthetics | 27/30 training runs finished; all 11 failed evaluations recovered; 3 GRID motif=True restarts running |

Within each regime, compare motif=False, total-count, and full-matrix columns side by side. Across regimes, treat differences as a hyperparameter ablation rather than extra seeds: the loss scales and some evaluator artifacts differ.

## Coverage at a glance

| Dataset | Earlier 3-seed motif comparison | New 2-seed code-weight comparison | Notes |
| --- | --- | --- | --- |
| MUTAG | Complete | Complete | Node and edge features enabled in training |
| PROTEINS | Complete for local/count/correlation metrics | Training complete; repaired third-party evaluation complete | Node features enabled; edge-feature loss disabled |
| GRID | Complete, including LinkCorrelations=1 | Motif=False complete; full-matrix seed 1 complete; three motif=True restarts running | Topology-only |
| LOBSTER | Complete, including LinkCorrelations=1 | Complete after one recovered evaluation | Topology-only |
| TRIANGULAR_GRID | Complete, including LinkCorrelations=1 | Complete after three recovered/original mixed evaluations | Topology-only |

PTC is intentionally absent from the main comparison because the requested pruned-rule count analysis was left out, and it does not belong to the new code-weight run set. AIDS, ENZYMES, and ogbg-molbbbp have baseline/gather artifacts but do not currently have a verified matched motif=True versus motif=False package under these two regimes. QM9 has no current verified matched result report in the gathered artifacts. They are not silently treated as completed comparisons.

## Source-of-truth files

- Earlier complete comparison: `/local-scratch2/mirzaei/FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md`
- New code-weight comparison: `/local-scratch2/mirzaei/NEW_HYPERPARAMETER_EXPERIMENTS_RESULTS_20260902.md`
- Node-feature evaluation artifacts: `/local-scratch2/mirzaei/node_feature_evaluation_20260901/`
- Baseline/gather archive: `/local-scratch2/new/gather/`
- Large baseline-only terminal-chain reports: `/local-scratch2/new/gather/MATCHED_ALL_TERMINAL_CHAIN_RESULTS.md` and `/local-scratch2/new/gather/PER_CHAIN_TERMINAL_MOTIF_MMD_RESULTS.md`

The two complete source reports are reproduced below so this file is self-contained.

---

## Part I — Earlier three-seed motif comparison

"""

middle = """

---

## Part II — New dataset-specific code-weight comparison

"""

footer = """

---

## Current live work

At the time this master report was created, the three restarted GRID motif=True runs were running from epoch 0 under the seven-day `cs-gpu-research` partition with the `cs-schulte` account:

- GRID total-count seed 0: job `264183_0`, `cs-venus-09` (Schulte-priority node).
- GRID total-count seed 1: job `264183_1`, `cs-venus-13` (Schulte-priority node).
- GRID full-matrix seed 0: job `264186_2`, `cs-venus-17` (normal shared Venus fallback).

Their result/model root is `/project/cs-schulte-lab/ali/GraphVAE-REQ-kia-motif-20260831/runs/solar_kia_bce_motif01_2seed_restarts_20260902`. Do not add these runs to a final two-seed aggregate until training, best-model selection, graph generation, and evaluation all finish.
"""

TARGET.write_text(
    header + nested_report(OLD) + middle + nested_report(NEW) + footer,
    encoding="utf-8",
)
print(TARGET)
