#!/usr/bin/env python3
"""Append a transparent DeFoG outlier-sensitivity analysis to the synthetic report."""

import json
import statistics
from pathlib import Path


REPORT = Path("/local-scratch/localhome/mirzaei/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.md")
SOURCE = Path("/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/DEFOG_VS_GRAPHVAE_MOTIF_TRUE_SYNTHETIC_20260906.json")
MARKER = "## DeFoG outlier sensitivity analysis"

DS_DISPLAY = {"grid": "GRID", "lobster": "LOBSTER", "triangular_grid": "TRIANGULAR_GRID"}
KEEP = {"grid": [0], "lobster": [0, 1, 2], "triangular_grid": [0, 1]}
EXCLUDED = {"grid": "seed 1", "lobster": "none", "triangular_grid": "seed 2"}

GIN = {
    "F1-PR": "f1_pr", "Precision": "precision", "Recall": "recall", "MMD-RBF": "mmd_rbf",
    "MMD-linear mean": "mmd_linear_mean", "MMD-linear median": "mmd_linear_median",
    "MMD-linear 10%-trimmed": "mmd_linear_trimmed",
}
DIRECT = {
    "Degree MMD": "degree", "Clustering MMD": "clustering", "Orbit MMD": "orbit", "Spectral MMD": "spectral",
    "Diameter MMD": "diameter", "Triangle MMD": "triangle", "Sparsity MMD": "sparsity",
    "Edge-count absolute error": "edge_count_absolute_error",
}


def mean_cell(values):
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.6g} (n=1; no SD)", mean
    return f"{mean:.6g} ± {statistics.stdev(values):.6g} (n={len(values)})", mean


def cell_mean(cell):
    return float(cell.split("±")[0].split("(")[0].strip())


def advantage(base, candidate, better):
    signed = candidate - base if better == "↑" else base - candidate
    return f"{100 * signed / abs(base):+.2f}%" if base else "N/A"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def existing_metric_rows(text, heading):
    lines = text.splitlines()
    start = lines.index(heading)
    while start < len(lines) and not lines[start].startswith("| Metric |"):
        start += 1
    start += 2
    rows = {}
    while start < len(lines) and lines[start].startswith("|"):
        c = [x.strip() for x in lines[start].strip().strip("|").split("|")]
        rows[c[0]] = c
        start += 1
    return rows


text = REPORT.read_text(encoding="utf-8")
if MARKER in text:
    raise SystemExit("sensitivity section already present")
raw = json.loads(SOURCE.read_text(encoding="utf-8"))["datasets"]

parts = [MARKER, """
This section tests the user's outlier hypothesis while preserving the original all-seed tables above as the primary results. A seed is excluded only when it is anomalous across several independent metric families, not merely because it is unfavorable on one metric.

This is an **exploratory sensitivity analysis**. With only two or three available seeds, formal statistical outlier detection is not reliable. In particular, filtered GRID has only one remaining DeFoG seed, so it has no training-seed uncertainty estimate and must not replace the provisional `n=2` result in a paper.
""".strip()]

evidence = [
    ["GRID", "1", "0.108469 vs 0.825252", "0.413420 vs 0.017966", "1.135641 vs 0.023481", "1.090499 vs 0.033686", "230.05 vs 19.70", "Exclude for sensitivity only"],
    ["LOBSTER", "None", "0.944–0.992", "0.000415–0.000710", "0.000505–0.006028", "0.006997–0.023822", "4.00–6.40", "Keep all seeds"],
    ["TRIANGULAR_GRID", "2", "0.000019 vs 0.577–0.600", "0.636044 vs 0.003697–0.020211", "0.995633 vs 0.844–0.937", "1.037100 vs 0.006144–0.185585", "487.55 vs 6.85–67.30", "Exclude for sensitivity only"],
]
parts += ["\n### Outlier evidence", table(["Dataset", "Flagged seed", "F1-PR", "Degree MMD", "Clustering MMD", "Orbit MMD", "Edge error", "Decision"], evidence)]

for ds, display in DS_DISPLAY.items():
    parts.append(f"\n### {display}: comparison after outlier exclusion")
    parts.append(f"DeFoG seeds retained: `{', '.join(map(str, KEEP[ds]))}`; excluded: `{EXCLUDED[ds]}`. GraphVAE false/total/full remain unchanged at three seeds.")
    gin_old = existing_metric_rows(text, f"### {display} — third-party Random-GIN")
    rows = []
    for label, key in GIN.items():
        vals = [raw[ds]["per_seed"]["defog"][s]["random_gin"][key] for s in KEEP[ds]]
        dcell, dmean = mean_cell(vals); old = gin_old[label]; better = old[1]
        false, total, full = old[3], old[5], old[7]
        rows.append([label, better, dcell, false, advantage(dmean, cell_mean(false), better), total,
                     advantage(dmean, cell_mean(total), better), full, advantage(dmean, cell_mean(full), better)])
    parts += ["\n#### Third-party structural-feature Random-GIN", table(
        ["Metric", "Better", "DeFoG filtered", "GraphVAE false", "False advantage", "GraphVAE total", "Total advantage", "GraphVAE full", "Full advantage"], rows)]

    direct_old = existing_metric_rows(text, f"### {display} — direct structural metrics")
    rows = []
    for label, key in DIRECT.items():
        vals = [raw[ds]["per_seed"]["defog"][s]["structural"][key] for s in KEEP[ds]]
        dcell, dmean = mean_cell(vals); old = direct_old[label]; better = old[1]
        false, total, full = old[3], old[5], old[7]
        rows.append([label, better, dcell, false, advantage(dmean, cell_mean(false), better), total,
                     advantage(dmean, cell_mean(total), better), full, advantage(dmean, cell_mean(full), better)])
    parts += ["\n#### Direct structural MMD", table(
        ["Metric", "Better", "DeFoG filtered", "GraphVAE false", "False advantage", "GraphVAE total", "Total advantage", "GraphVAE full", "Full advantage"], rows)]

    tv_vals = [raw[ds]["per_seed"]["defog"][s]["motif_correlation"]["total_variation"] for s in KEEP[ds]]
    tv_cell, tv_mean = mean_cell(tv_vals)
    # The historical false value is intentionally not assigned an advantage due to the 70-vs-20 graph protocol difference.
    tv_false = {"grid": "0.012910 ± 0.003124", "lobster": "0.032247 ± 0.002789", "triangular_grid": "0.018245 ± 0.003490"}[ds]
    a = raw[ds]["aggregate"]
    total = a["total_count"]["motif_correlation"]["total_variation"]
    full = a["full_matrix"]["motif_correlation"]["total_variation"]
    total_cell = f"{total['mean']:.6g} ± {total['sample_sd']:.6g} (n=3)"
    full_cell = f"{full['mean']:.6g} ± {full['sample_sd']:.6g} (n=3)"
    parts += ["\n#### Normalized full-state LinkCorrelation TV", table(
        ["DeFoG filtered", "GraphVAE false (historical 70-graph)", "False comparison", "GraphVAE total", "Total advantage", "GraphVAE full", "Full advantage"],
        [[tv_cell, tv_false, "N/C: different protocol", total_cell, advantage(tv_mean, total["mean"], "↓"), full_cell, advantage(tv_mean, full["mean"], "↓")]])]

summary = """
### Sensitivity conclusion

- GRID's poor two-seed DeFoG aggregate is heavily driven by seed 1. After excluding it, seed 0 is substantially stronger, but `n=1` is too weak for a final claim; GRID seed 2 should finish before drawing conclusions.
- TRIANGULAR_GRID seed 2 is a broad failure/outlier. The filtered two-seed DeFoG result improves sharply, but GraphVAE remains much stronger on most F1, structural, and motif-TV comparisons.
- LOBSTER does not show a broad failed seed. Seed 2 has a larger motif TV, but excluding it only for that metric would be selective, so all three seeds remain included.
"""
parts.append(summary.strip())

section = "\n\n".join(parts) + "\n\n"
text = text.replace("## Main findings", section + "## Main findings")
REPORT.write_text(text, encoding="utf-8")
print(REPORT)
