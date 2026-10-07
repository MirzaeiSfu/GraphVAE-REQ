#!/usr/bin/env python3
import json
import math
import statistics
from pathlib import Path

ROOT = Path("/local-scratch2/mirzaei/qm9_common_eval_20260917")
OUT = ROOT / "QM9_TRUE_FALSE_DEFOG_COMMON_512_REPORT.md"

METHODS = {
    "Motif=True (full)": ("graphvae/true_full", [0, 1, 2]),
    "Motif=False": ("graphvae/false", [0, 1, 2]),
    "DeFoG": ("defog", [1, 2]),
}


def read_json(path):
    return json.loads(path.read_text())


def source(method, rel, seed):
    if method == "DeFoG":
        return ROOT / "evaluations" / rel / f"seed_{seed}" / "common_metrics.json"
    return ROOT / "evaluations" / rel / f"seed_{seed}" / "attributed_random_gin.json"


def structural_source(method, rel, seed):
    name = "common_metrics.json" if method == "DeFoG" else "common_structural.json"
    return ROOT / "evaluations" / rel / f"seed_{seed}" / name


def vals_for(extractor):
    result = {}
    for method, (rel, seeds) in METHODS.items():
        result[method] = [(seed, extractor(method, rel, seed)) for seed in seeds]
    return result


def fmt(value):
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def aggregate(pairs):
    values = [float(value) for _, value in pairs]
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    return mean, sd


def agg_fmt(pairs):
    mean, sd = aggregate(pairs)
    return f"{fmt(mean)} ± {fmt(sd)}"


def table(title, metrics, extractor, directions):
    lines = [f"### {title}", "", "Aggregate = mean ± sample SD across generator training seeds.", ""]
    lines.append("| Metric | Direction | Motif=True full (3 seeds) | Motif=False (3 seeds) | DeFoG (2 seeds: 1, 2) |")
    lines.append("|---|:---:|---:|---:|---:|")
    cached = {}
    for key, label in metrics:
        values = vals_for(lambda m, r, s: extractor(m, r, s, key))
        cached[key] = values
        lines.append(f"| {label} | {directions[key]} | " + " | ".join(agg_fmt(values[m]) for m in METHODS) + " |")
    lines += ["", "Per-generator-seed values:", ""]
    lines.append("| Metric | Method | Seed values |")
    lines.append("|---|---|---|")
    for key, label in metrics:
        for method in METHODS:
            pairs = cached[key][method]
            text = ", ".join(f"s{seed}={fmt(value)}" for seed, value in pairs)
            lines.append(f"| {label} | {method} | {text} |")
    lines.append("")
    return lines


def gin_extract(mode):
    def extract(method, rel, seed, metric):
        payload = read_json(source(method, rel, seed))
        if method == "DeFoG":
            return payload["random_gin"]["modes"][mode]["summary"][metric]["mean"]
        return payload["evaluation"]["modes"][mode]["summary"][metric]["mean"]
    return extract


def struct_extract(method, rel, seed, metric):
    return read_json(structural_source(method, rel, seed))["structural_mmd"][metric]


def motif_extract(method, rel, seed, metric):
    path = ROOT / "evaluations" / rel / f"seed_{seed}" / "motif_tv.json"
    return read_json(path)[metric]


lines = [
    "# QM9: motif=True full vs motif=False vs DeFoG",
    "",
    "Generated: 2026-09-17. This is a common-protocol comparison, not a merge of historical evaluator outputs.",
    "",
    "## Common evaluation protocol",
    "",
    "- Generator seeds: motif=True full = 0, 1, 2; motif=False = 0, 1, 2; DeFoG = corrected independent seeds 1 and 2.",
    "- Exactly 512 generated graphs per seed and the same first 512 graphs from the frozen held-out QM9 test split as reference.",
    "- Split: paper 70/10/20, split seed 123; the reference bundle is shared by all methods.",
    "- RandomGIN: 10 evaluator seeds (0–9) per generator seed. Each table entry first averages evaluator repeats within a generator seed, then reports mean ± sample SD across generator seeds.",
    "- `without node features` uses topology only. `with node features` uses the aligned 9-dimensional QM9 representation: atom type (5) plus hydrogen-count category (4).",
    "- Structural metrics use the same generated/reference graph bundles. MMD metrics are lower-is-better.",
    "- Motif correlation is hard aggregate normalized full-state total variation (TV), lower-is-better, on the exact loaded `_CP_smoothed` rule-state universe used by the GraphVAE runs: no rule pruning, 10 loaded rules / 53 loaded states; 3 multi-atom rules are eligible for this correlation summary.",
    "- DeFoG has only two corrected seeds, so its uncertainty is less stable and remains provisional relative to the three-seed GraphVAE results.",
    "",
]

gin_metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1_pr", "F1-PR"), ("mmd_rbf", "Embedding MMD-RBF"), ("mmd_linear", "Embedding MMD-linear")]
gin_dirs = {"precision": "↑", "recall": "↑", "f1_pr": "↑", "mmd_rbf": "↓", "mmd_linear": "↓"}
lines += table("RandomGIN without node features (topology only)", gin_metrics, gin_extract("topology_control"), gin_dirs)
lines += table("RandomGIN with node features", gin_metrics, gin_extract("decoded_node"), gin_dirs)

struct_metrics = [
    ("degree", "Degree MMD"), ("clustering", "Clustering MMD"), ("orbit", "Orbit MMD"),
    ("spectral", "Spectral MMD"), ("diameter", "Diameter MMD"), ("triangle", "Triangle MMD"),
    ("sparsity", "Sparsity MMD"), ("edge_count_absolute_error", "Absolute error in mean edge count"),
]
struct_dirs = {key: "↓" for key, _ in struct_metrics}
lines += table("Structural metrics", struct_metrics, struct_extract, struct_dirs)
lines += table("Motif correlation", [("mean_rule_total_variation", "Mean rule-state TV")], motif_extract, {"mean_rule_total_variation": "↓"})

lines += [
    "## Result locations",
    "",
    f"- Evaluation root: `{ROOT}`",
    f"- Motif=True full per-seed results: `{ROOT / 'evaluations/graphvae/true_full/seed_<0|1|2>'}`",
    f"- Motif=False per-seed results: `{ROOT / 'evaluations/graphvae/false/seed_<0|1|2>'}`",
    f"- DeFoG per-seed results: `{ROOT / 'evaluations/defog/seed_<1|2>'}`",
    f"- GraphVAE checkpoints/configs: `{ROOT / 'graphvae'}`",
    "",
    "## Interpretation caution",
    "",
    "The motif TV above is a state-distribution comparison on the shared rule universe. It is not the Gaussian training loss itself, and only three loaded rules are multi-atom rules eligible for this particular correlation calculation. Accordingly, it should be reported alongside—rather than substituted for—the structural and RandomGIN results.",
    "",
]

OUT.write_text("\n".join(lines))
print(OUT)
