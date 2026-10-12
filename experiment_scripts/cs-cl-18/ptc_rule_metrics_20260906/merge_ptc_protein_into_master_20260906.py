#!/usr/bin/env python3
"""Merge the verified PTC/PROTEINS addendum into the all-results report."""

from pathlib import Path


root = Path("/local-scratch2/mirzaei")
master = root / "ALL_GRAPHVAE_MOTIF_RESULTS_AVAILABLE_20260902.md"
addendum = root / "PTC_3SEED_AND_PROTEINS_REPAIRED_METRICS_20260906.md"
dated = root / "ALL_GRAPHVAE_MOTIF_RESULTS_AVAILABLE_UPDATED_20260906.md"
legacy_final = root / "FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md"

text = master.read_text()
text = text.replace(
    "Generated: 2026-09-02 (America/Vancouver).",
    "Generated: 2026-09-02; updated with PTC and repaired PROTEINS evaluation on 2026-09-06 (America/Vancouver).",
)
text = text.replace(
    "| New code-weight comparison | 2 | 0.1 × dataset BCE (0.4–5) | Dataset-specific BCE/KL/node/edge weights | MUTAG, PROTEINS, GRID, LOBSTER, TRIANGULAR_GRID with LinkCorrelations=1 synthetics | 27/30 training runs finished; all 11 failed evaluations recovered; 3 GRID motif=True restarts running |",
    "| New code-weight comparison | 2 | 0.1 × dataset BCE (0.4–5) | Dataset-specific BCE/KL/node/edge weights | MUTAG, PROTEINS, GRID, LOBSTER, TRIANGULAR_GRID with LinkCorrelations=1 synthetics | Historical 2026-09-02 snapshot; see its detailed section |\n| PTC available-results comparison | 3 | 0.1 | Motif=False and motif=True architecture/preprocessing differ; see warning | PTC | All 6 motif=True runs complete; final, pruned-rule, and complete-correlation metrics verified |",
)
text = text.replace(
    "| PROTEINS | Complete for local/count/correlation metrics | Training complete; repaired third-party evaluation complete | Node features enabled; edge-feature loss disabled |",
    "| PROTEINS | Complete, including repaired topology third-party and decoded-node evaluation | Training complete; repaired third-party evaluation complete | Node features enabled; edge-feature loss disabled |",
)
text = text.replace(
    "| TRIANGULAR_GRID | Complete, including LinkCorrelations=1 | Complete after three recovered/original mixed evaluations | Topology-only |\n\nPTC is intentionally absent from the main comparison because the requested pruned-rule count analysis was left out, and it does not belong to the new code-weight run set. AIDS, ENZYMES, and ogbg-molbbbp have baseline/gather artifacts but do not currently have a verified matched motif=True versus motif=False package under these two regimes.",
    "| TRIANGULAR_GRID | Complete, including LinkCorrelations=1 | Complete after three recovered/original mixed evaluations | Topology-only |\n| PTC | Complete three-seed final, pruned-rule, and complete-correlation results | Not part of this regime | Available comparison is not a motif-only controlled ablation; see Part III |\n\nPTC is now included in Part III with explicit comparability qualifications. AIDS, ENZYMES, and ogbg-molbbbp have baseline/gather artifacts but do not currently have a verified matched motif=True versus motif=False package under these two regimes.",
)
text = text.replace(
    "- Node-feature evaluation artifacts: `/local-scratch2/mirzaei/node_feature_evaluation_20260901/`",
    "- Node-feature evaluation artifacts: `/local-scratch2/mirzaei/node_feature_evaluation_20260901/`\n- PTC three-seed and repaired PROTEINS addendum: `/local-scratch2/mirzaei/PTC_3SEED_AND_PROTEINS_REPAIRED_METRICS_20260906.md`\n- PTC pruned-rule artifacts: `/local-scratch2/mirzaei/ptc_rule_metrics_20260906/`\n- PTC complete-state correlation artifacts: `/local-scratch2/mirzaei/ptc_full_state_correlation_20260906/`",
)
text = text.replace(
    "| PROTEINS | 3 | 3 | 3 | Training complete; local metrics recovered from logs, new third-party export unavailable |",
    "| PROTEINS | 3 | 3 | 3 | Complete; topology third-party evaluation repaired to matched 209/209 collections |",
)
text = text.replace(
    "The motif=False values come from the three-seed gather baseline. The newer total/full runs reached epoch 20,000 and produced local evaluation values, but their final PyG export stopped at `210 generated vs. 209 reference` graphs. Therefore third-party Random-GIN values are intentionally omitted for the new motif=True columns.",
    "The motif=False values come from the three-seed gather baseline. The newer total/full runs reached epoch 20,000. Their original third-party export encountered an unequal postprocessing count; the repaired evaluator removes empty graphs and deterministically equalizes both sides to 209/209 before running the same ten-seed structural-feature Random-GIN. Complete repaired results are in Part III.",
)
text = text.replace(
    "These local results are usable, but the missing new third-party evaluation must be repaired before claiming a complete final package.",
    "These local results are usable, and the repaired three-seed topology third-party evaluation is now reported in Part III.",
)
text = text.replace(
    "- The original PROTEINS topology-oriented third-party export failed at 210\n  generated versus 209 reference graphs. The node-feature-aware post-hoc\n  evaluation above is available and uses equal generated/reference collection\n  sizes within each run; its motif=False and motif=True held-out splits still\n  differ by one graph.",
    "- The original PROTEINS topology-oriented third-party export failed at 210\n  generated versus 209 reference graphs. It has now been repaired by removing\n  empty graphs and deterministically matching collection sizes; the complete\n  topology-oriented results are in Part III. The node-feature-aware post-hoc\n  evaluation above remains available; its motif=False and motif=True held-out\n  splits still differ by one graph.",
)

marker = "\n---\n\n## Part III — PTC and repaired PROTEINS update (2026-09-06)\n"
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n"
addition = addendum.read_text()
if addition.startswith("# "):
    addition = addition.split("\n", 1)[1]
text = text.rstrip() + marker + addition.lstrip()

dated.write_text(text)
master.write_text(text)

legacy_text = legacy_final.read_text()
legacy_text = legacy_text.replace(
    "# Motif=False vs. Motif=True: MUTAG, PROTEINS, and Synthetic Datasets",
    "# Motif=False vs. Motif=True: MUTAG, PROTEINS, PTC, and Synthetic Datasets",
    1,
)
legacy_marker = "\n---\n\n## PTC and repaired PROTEINS update (2026-09-06)\n"
if legacy_marker in legacy_text:
    legacy_text = legacy_text.split(legacy_marker, 1)[0].rstrip() + "\n"
legacy_final.write_text(legacy_text.rstrip() + legacy_marker + addition.lstrip())
print(master)
print(dated)
print(legacy_final)
