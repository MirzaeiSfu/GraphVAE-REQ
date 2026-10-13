import glob
import json
import re
import statistics

ROOT = "/tmp/all_results.wSpUxE"

METRICS = [
    ("Degree MMD", "table2.metrics.degree", "lower"),
    ("Clustering MMD", "table2.metrics.clustering", "lower"),
    ("Orbit MMD", "table2.metrics.orbit", "lower"),
    ("Spectral MMD", "table2.metrics.spectral", "lower"),
    ("Diameter MMD", "table2.metrics.diameter", "lower"),
    ("Sparsity MMD", "table2.extra_metrics.sparsity", "lower"),
    ("Triangle MMD", "table2.extra_metrics.triangle", "lower"),
    ("Generated edge count", "table2.extra_metrics.generated_edge_count", "neutral"),
    ("Reference edge count", "table2.extra_metrics.reference_edge_count", "neutral"),
    ("Local MMD-RBF", "table3.local_eval_metrics.mmd_rbf", "lower"),
    ("Local precision", "table3.local_eval_metrics.precision", "higher"),
    ("Local recall", "table3.local_eval_metrics.recall", "higher"),
    ("Local F1-PR", "table3.local_eval_metrics.f1_pr", "higher"),
    ("Third-party MMD-RBF", "table3.third_party_eval_metrics.metrics.mmd_rbf.mean", "lower"),
    ("Third-party MMD-linear", "table3.third_party_eval_metrics.metrics.mmd_linear.mean", "lower"),
    ("Third-party MMD-linear trimmed mean", "table3.third_party_eval_metrics.metrics.mmd_linear.trimmed_mean", "lower"),
    ("Third-party precision", "table3.third_party_eval_metrics.metrics.precision.mean", "higher"),
    ("Third-party recall", "table3.third_party_eval_metrics.metrics.recall.mean", "higher"),
    ("Third-party F1-PR", "table3.third_party_eval_metrics.metrics.f1_pr.mean", "higher"),
]


def nested(data, path):
    for key in path.split("."):
        data = data[key]
    return float(data)


def load(pattern):
    return [json.load(open(path)) for path in sorted(glob.glob(pattern))]


def fmt(value):
    if value != 0 and abs(value) < 0.0001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def stats(values):
    return statistics.mean(values), statistics.stdev(values)


def stat_cell(values):
    mean, std = stats(values)
    return f"{fmt(mean)} ± {fmt(std)}"


def values(docs, path):
    return [nested(doc, path) for doc in docs]


def edge_errors(docs):
    return [
        abs(nested(doc, "table2.extra_metrics.generated_edge_count") - nested(doc, "table2.extra_metrics.reference_edge_count"))
        for doc in docs
    ]


def print_seed_table(title, docs):
    print(f"### {title}\n")
    print("| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |")
    print("|---|---:|---:|---:|---:|")
    for name, path, _ in METRICS:
        vals = values(docs, path)
        print(f"| {name} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {stat_cell(vals)} |")
    vals = edge_errors(docs)
    print(f"| Edge-count absolute error | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {stat_cell(vals)} |")
    print()


def parse_protein_log(path):
    text = open(path).read()
    matches = re.findall(
        r"degree ([\d.eE+-]+) clustering ([\d.eE+-]+) sparsity ([\d.eE+-]+) "
        r"orbits ([\d.eE+-]+) Spec: ([\d.eE+-]+) Tri ([\d.eE+-]+)\s+diameter: ([\d.eE+-]+)\s+"
        r"GGM GNN: \{'mmd_rbf': ([\d.eE+-]+), 'mmd_rbf_std': ([\d.eE+-]+), "
        r"'precision': ([\d.eE+-]+), 'precision_std': ([\d.eE+-]+), "
        r"'recall': ([\d.eE+-]+), 'recall_std': ([\d.eE+-]+), "
        r"'f1_pr': ([\d.eE+-]+)",
        text,
    )
    if not matches:
        raise RuntimeError(f"No final metrics found in {path}")
    row = [float(x) for x in matches[-1]]
    keys = ["degree", "clustering", "sparsity", "orbit", "spectral", "triangle", "diameter", "mmd_rbf", "mmd_rbf_std", "precision", "precision_std", "recall", "recall_std", "f1_pr"]
    return dict(zip(keys, row))


true_groups = {}
for dataset in ["lobster", "grid", "triangular_grid"]:
    for mode in ["total_count", "full_matrix"]:
        true_groups[(dataset, mode)] = load(f"{ROOT}/true/{dataset}/{mode}/*.json")

false_proteins = load(f"{ROOT}/false/proteins/*.json")
false_enzymes = load(f"{ROOT}/false/enzymes/*.json")
protein_true = {
    mode: [parse_protein_log(path) for path in sorted(glob.glob(f"{ROOT}/proteins_logs/{mode}_s*.log"))]
    for mode in ["total_count", "full_matrix"]
}

print("# Motif=True Experiment Results and Gather Motif=False Comparison\n")
print("Generated from the campaign state and result files available on 2026-08-28. All aggregate values are mean ± sample standard deviation across seeds 0, 1, and 2.\n")
print("## Result availability\n")
print("| Dataset | Motif=True status | Motif=False in `/local-scratch2/new/gather` | Comparison status |")
print("|---|---|---|---|")
print("| LOBSTER | 6/6 complete | Missing | Motif=True results only |")
print("| GRID | 6/6 complete | Missing | Motif=True results only |")
print("| TRIANGULAR_GRID | 6/6 complete | Missing | Motif=True results only |")
print("| PROTEINS | 6/6 trained; final export failed | Available, 3 seeds | Partial comparison using metrics computed before export failure |")
print("| ENZYMES | 0/6 complete | Available, 3 seeds | Baseline reported; comparison pending |")
print("| QM9 | 0/6 complete | Missing | No results available yet |\n")
print("The gather directory contains only `aids`, `enzymes`, `mutag`, `ogb`, `proteins`, and `ptc`. It has no LOBSTER, GRID, TRIANGULAR_GRID, or QM9 directory. No substitute baseline from outside gather is silently used here.\n")
print("For MMD and distance metrics, lower is better. For precision, recall, and F1, higher is better. Edge-count absolute error is `abs(generated_edge_count - reference_edge_count)` and is not the FactorBase motif count-distance loss.\n")

print("## Completed motif=True aggregate overview\n")
print("| Dataset | Mode | Degree | Clustering | Orbit | Spectral | Diameter | Local MMD-RBF | Local F1-PR | Third-party MMD-RBF | Third-party F1-PR | Edge abs. error |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (dataset, mode), docs in true_groups.items():
    paths = ["table2.metrics.degree", "table2.metrics.clustering", "table2.metrics.orbit", "table2.metrics.spectral", "table2.metrics.diameter", "table3.local_eval_metrics.mmd_rbf", "table3.local_eval_metrics.f1_pr", "table3.third_party_eval_metrics.metrics.mmd_rbf.mean", "table3.third_party_eval_metrics.metrics.f1_pr.mean"]
    cells = [stat_cell(values(docs, path)) for path in paths] + [stat_cell(edge_errors(docs))]
    print("| " + " | ".join([dataset.upper(), mode, *cells]) + " |")
print()

print("## Complete motif=True per-seed results\n")
for (dataset, mode), docs in true_groups.items():
    print_seed_table(f"{dataset.upper()} — {mode}", docs)

print("## PROTEINS: gather Motif=False versus partial Motif=True\n")
print("All six motif=True PROTEINS jobs reached epoch 20,000 and computed structural/local GNN metrics. They then failed during PyG export because the generated and reference collections had sizes 210 and 209. Consequently, the motif=True numbers below are recoverable computed metrics, but they are marked partial and have no third-party JSON results.\n")
print("| Metric | Better | Gather False | True total_count (partial) | Improvement | True full_matrix (partial) | Improvement |")
print("|---|---|---:|---:|---:|---:|---:|")
protein_map = [
    ("Degree MMD", "table2.metrics.degree", "degree", "lower"),
    ("Clustering MMD", "table2.metrics.clustering", "clustering", "lower"),
    ("Orbit MMD", "table2.metrics.orbit", "orbit", "lower"),
    ("Spectral MMD", "table2.metrics.spectral", "spectral", "lower"),
    ("Diameter MMD", "table2.metrics.diameter", "diameter", "lower"),
    ("Sparsity MMD", "table2.extra_metrics.sparsity", "sparsity", "lower"),
    ("Triangle MMD", "table2.extra_metrics.triangle", "triangle", "lower"),
    ("Local MMD-RBF", "table3.local_eval_metrics.mmd_rbf", "mmd_rbf", "lower"),
    ("Local precision", "table3.local_eval_metrics.precision", "precision", "higher"),
    ("Local recall", "table3.local_eval_metrics.recall", "recall", "higher"),
    ("Local F1-PR", "table3.local_eval_metrics.f1_pr", "f1_pr", "higher"),
]
for name, false_path, true_key, direction in protein_map:
    baseline = values(false_proteins, false_path)
    base_mean = statistics.mean(baseline)
    row = [name, direction.title(), stat_cell(baseline)]
    for mode in ["total_count", "full_matrix"]:
        vals = [item[true_key] for item in protein_true[mode]]
        mean = statistics.mean(vals)
        improvement = ((base_mean - mean) if direction == "lower" else (mean - base_mean)) / abs(base_mean) * 100
        row.extend([stat_cell(vals), f"{improvement:+.2f}%"])
    print("| " + " | ".join(row) + " |")
print()

for mode in ["total_count", "full_matrix"]:
    print(f"### PROTEINS motif=True {mode}: partial per-seed metrics\n")
    print("| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |")
    print("|---|---:|---:|---:|---:|")
    for name, _, key, _ in protein_map:
        vals = [item[key] for item in protein_true[mode]]
        print(f"| {name} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {stat_cell(vals)} |")
    print()

print_seed_table("PROTEINS gather motif=False baseline", false_proteins)

print("## ENZYMES gather Motif=False baseline\n")
print("The three-seed gather baseline is available, but no motif=True ENZYMES run is complete, so no true/false difference is reported yet.\n")
print_seed_table("ENZYMES gather motif=False baseline", false_enzymes)

print("## Interpretation limits and next update\n")
print("- Synthetic datasets cannot be compared against motif=False from gather because those baselines are absent there.")
print("- PROTEINS motif=True results must remain partial until the 210-versus-209 PyG export mismatch is fixed and final evaluation JSON files are produced.")
print("- ENZYMES comparison should be added after its motif=True queues complete.")
print("- QM9 has neither a completed motif=True result nor a matching gather baseline.")
print("- Percentage improvements are only shown for PROTEINS, the sole dataset with both gather motif=False values and available motif=True metrics. Even there, final comparability should be confirmed from identical database, split, preprocessing, and evaluation artifacts.")
