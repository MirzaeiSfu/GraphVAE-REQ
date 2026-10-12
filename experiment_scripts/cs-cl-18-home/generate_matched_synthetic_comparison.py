import glob
import json
import statistics

ROOT = "/tmp/matched_synth_compare.HuPZVI"
OUT = "/local-scratch/localhome/mirzaei/LOBSTER_TRIANGULAR_MATCHED_MOTIF_COMPARISON.md"

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


def get(data, path):
    for key in path.split("."):
        data = data[key]
    return float(data)


def load(dataset, variant):
    return [json.load(open(path)) for path in sorted(glob.glob(f"{ROOT}/{dataset}/{variant}/*.json"))]


def vals(docs, path):
    return [get(doc, path) for doc in docs]


def edge_error(docs):
    return [abs(get(doc, "table2.extra_metrics.generated_edge_count") - get(doc, "table2.extra_metrics.reference_edge_count")) for doc in docs]


def fmt(value):
    if value and abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def summary(values):
    return f"{fmt(statistics.mean(values))} ± {fmt(statistics.stdev(values))}"


def improvement(base, candidate, direction):
    b = statistics.mean(base)
    c = statistics.mean(candidate)
    if direction == "neutral" or b == 0:
        return "—"
    favorable = b - c if direction == "lower" else c - b
    return f"{favorable / abs(b) * 100:+.2f}%"


lines = []
add = lines.append
add("# Matched Synthetic Motif Comparison: LOBSTER and TRIANGULAR_GRID")
add("")
add("This report compares the newly completed motif=False baselines against the completed motif=True `total_count` and `full_matrix` experiments. Every setting uses seeds 0, 1, and 2. Aggregate values are mean ± sample standard deviation across the three training seeds.")
add("")
add("The runs are matched on dataset/database, featureless input, 70/10/20 split, split seed 123, loader seed 123, model architecture, 20,000 epochs, learning rate, training batch size, model-selection criterion, and evaluation protocol. The intended training difference is the topology-based motif loss.")
add("")
add("For MMD and distance metrics, lower is better. For precision, recall, and F1, higher is better. A positive improvement percentage favors motif=True; a negative percentage favors motif=False. Edge counts are descriptive.")
add("")

all_docs = {}
for dataset in ("lobster", "triangular_grid"):
    all_docs[dataset] = {variant: load(dataset, variant) for variant in ("false", "total_count", "full_matrix")}
    refs = {variant: vals(docs, "table2.extra_metrics.reference_edge_count") for variant, docs in all_docs[dataset].items()}
    aligned = len({tuple(round(x, 10) for x in value) for value in refs.values()}) == 1
    add(f"## {dataset.upper()}")
    add("")
    add(f"Reference edge-count alignment check: **{'passed' if aligned else 'failed'}**. Motif=False, total-count, and full-matrix reference means are {statistics.mean(refs['false']):.6f}, {statistics.mean(refs['total_count']):.6f}, and {statistics.mean(refs['full_matrix']):.6f}.")
    add("")
    add("| Metric | Better | Motif=False | True total_count | Δ mean | Improvement | True full_matrix | Δ mean | Improvement |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    base_docs = all_docs[dataset]["false"]
    for name, path, direction in METRICS + [("Edge-count absolute error", None, "lower")]:
        base = edge_error(base_docs) if path is None else vals(base_docs, path)
        row = [name, direction.title() if direction != "neutral" else "Descriptive", summary(base)]
        for variant in ("total_count", "full_matrix"):
            candidate_docs = all_docs[dataset][variant]
            candidate = edge_error(candidate_docs) if path is None else vals(candidate_docs, path)
            row.extend([summary(candidate), fmt(statistics.mean(candidate) - statistics.mean(base)), improvement(base, candidate, direction)])
        add("| " + " | ".join(row) + " |")
    add("")
    add("### Per-seed overview")
    add("")
    add("| Setting | Seed | Degree | Clustering | Orbit | Spectral | Diameter | Local MMD-RBF | Local F1-PR | Third-party MMD-RBF | Third-party F1-PR | Edge abs. error |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    key_paths = ["table2.metrics.degree", "table2.metrics.clustering", "table2.metrics.orbit", "table2.metrics.spectral", "table2.metrics.diameter", "table3.local_eval_metrics.mmd_rbf", "table3.local_eval_metrics.f1_pr", "table3.third_party_eval_metrics.metrics.mmd_rbf.mean", "table3.third_party_eval_metrics.metrics.f1_pr.mean"]
    labels = {"false": "Motif=False", "total_count": "True total_count", "full_matrix": "True full_matrix"}
    for variant in ("false", "total_count", "full_matrix"):
        docs = all_docs[dataset][variant]
        errors = edge_error(docs)
        for seed, doc in enumerate(docs):
            row = [labels[variant], str(seed), *[fmt(get(doc, path)) for path in key_paths], fmt(errors[seed])]
            add("| " + " | ".join(row) + " |")
    add("")

add("## Summary")
add("")
for dataset in ("lobster", "triangular_grid"):
    docs = all_docs[dataset]
    winners = {}
    for name, path, direction in METRICS:
        if direction == "neutral":
            continue
        means = {variant: statistics.mean(vals(group, path)) for variant, group in docs.items()}
        winners[name] = min(means, key=means.get) if direction == "lower" else max(means, key=means.get)
    counts = {variant: list(winners.values()).count(variant) for variant in docs}
    add(f"- **{dataset.upper()}**: across the {len(winners)} directionally scored reported metrics, motif=False is best on {counts['false']}, motif=True total-count on {counts['total_count']}, and motif=True full-matrix on {counts['full_matrix']} metrics.")
add("")
add("The edge-count absolute error is `abs(generated_edge_count - reference_edge_count)`. It measures generated edge-count mismatch and is not the FactorBase motif count-distance objective.")
add("")
add("## Source directories")
add("")
add("- LOBSTER motif=False: `/localhome/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/lobster/seed_<N>` on cs-cl-16")
add("- LOBSTER motif=True: `/localhome/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/lobster/{total_count,full_matrix}/seed_<N>` on cs-cl-16")
add("- TRIANGULAR_GRID motif=False: `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/motif_false_3seed/triangular_grid/seed_<N>` on cs-cl-19")
add("- TRIANGULAR_GRID motif=True: `/local-scratch2/mirzaei/fb/GraphVAE-REQ/runs/multihop_smoothed_3seed/triangular_grid/{total_count,full_matrix}/seed_<N>` on cs-cl-19")

open(OUT, "w").write("\n".join(lines) + "\n")
