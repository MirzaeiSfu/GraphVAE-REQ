#!/usr/bin/env python3
"""Build the complete PTC GraphVAE/DeFoG comparison report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path("/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907")
OLD = Path("/local-scratch2/mirzaei/PTC_3SEED_AND_PROTEINS_REPAIRED_METRICS_20260906.json")
RULE_ROOT = Path("/local-scratch2/mirzaei/ptc_rule_metrics_20260906/results")
DEFOG_AGG = Path("/local-scratch2/mirzaei/defog_ptc_frozen_20260906/aggregate/ptc/aggregate.json")
OUT_JSON = ROOT / "PTC_DEFOG_VS_MOTIF_TRUE_FALSE_ALL_METRICS_20260907.json"
OUT_MD = ROOT / "PTC_DEFOG_VS_MOTIF_TRUE_FALSE_ALL_METRICS_20260907.md"
LOCAL_JSON = Path("/local-scratch/localhome/mirzaei") / OUT_JSON.name
LOCAL_MD = Path("/local-scratch/localhome/mirzaei") / OUT_MD.name

MODEL_KEYS = ["false", "total_count", "full_matrix", "defog"]
MODEL_NAMES = {
    "false": "GraphVAE motif=False",
    "total_count": "GraphVAE True total",
    "full_matrix": "GraphVAE True full",
    "defog": "DeFoG",
}


def load(path):
    return json.loads(path.read_text())


def summary(values):
    vals = [float(x) for x in values]
    return {"mean": fmean(vals), "sample_sd": stdev(vals), "n": len(vals), "values": vals}


def fmt(obj):
    if obj is None:
        return "N/A"
    mean = obj["mean"]
    sd = obj["sample_sd"]
    if abs(mean) < 1e-4 or abs(sd) < 1e-5:
        return f"{mean:.6e} ± {sd:.6e}"
    return f"{mean:.6f} ± {sd:.6f}"


def percent_better(candidate, baseline, higher):
    if candidate is None or baseline is None or baseline["mean"] == 0:
        return "N/A"
    gain = candidate["mean"] - baseline["mean"] if higher else baseline["mean"] - candidate["mean"]
    return f"{100.0 * gain / abs(baseline['mean']):+.2f}%"


def table(metrics, data, include_gain=True):
    header = "| Metric | Better | " + " | ".join(MODEL_NAMES[k] for k in MODEL_KEYS)
    if include_gain:
        header += " | DeFoG vs False | DeFoG vs best True |"
    else:
        header += " |"
    separator = "| --- | :---: | " + " | ".join("---:" for _ in MODEL_KEYS)
    separator += " | ---: | ---: |" if include_gain else " |"
    lines = [header, separator]
    for label, key, higher in metrics:
        values = [data.get(model, {}).get(key) for model in MODEL_KEYS]
        row = f"| {label} | {'↑' if higher else '↓'} | " + " | ".join(fmt(x) for x in values)
        if include_gain:
            true_candidates = [data.get(k, {}).get(key) for k in ("total_count", "full_matrix")]
            true_candidates = [x for x in true_candidates if x is not None]
            best_true = (max if higher else min)(true_candidates, key=lambda x: x["mean"])
            row += " | " + percent_better(values[3], values[0], higher)
            row += " | " + percent_better(values[3], best_true, higher) + " |"
        else:
            row += " |"
        lines.append(row)
    return "\n".join(lines)


old = load(OLD)["ptc"]

# Final held-out structural and structural-feature Random-GIN metrics.
final = {k: dict(old["final"][k]) for k in ("false", "total_count", "full_matrix")}
defog_structural_raw = [load(ROOT / "structural" / f"seed_{seed}.json") for seed in range(3)]
defog_final = {}
for key in defog_structural_raw[0]["structural_mmd"]:
    defog_final[key] = summary([item["structural_mmd"][key] for item in defog_structural_raw])
gin_map = {
    "third_party_f1_pr": ("f1_pr", "mean"),
    "third_party_precision": ("precision", "mean"),
    "third_party_recall": ("recall", "mean"),
    "third_party_mmd_rbf": ("mmd_rbf", "mean"),
    "third_party_mmd_linear": ("mmd_linear", "mean"),
    "third_party_mmd_linear_median": ("mmd_linear", "median"),
    "third_party_mmd_linear_trimmed": ("mmd_linear", "trimmed_mean"),
}
for target, (metric, field) in gin_map.items():
    defog_final[target] = summary([
        item["third_party_random_gin_structural_features"]["metrics"][metric][field]
        for item in defog_structural_raw
    ])
final["defog"] = defog_final

# Attributed (decoded PTC node labels) DeFoG Random-GIN; matching GraphVAE values were not run.
attributed = {k: {} for k in MODEL_KEYS}
defog_random_gin = load(DEFOG_AGG)["aggregate"]
decoded = defog_random_gin["decoded_node"]
for key in ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear"):
    attributed["defog"][key] = decoded[key]

# Preserve the older topology-only/local Random-GIN values as a secondary table.
topology_gin = {k: {} for k in MODEL_KEYS}
for model in ("false", "total_count", "full_matrix"):
    for source, target in {
        "local_f1_pr": "f1_pr",
        "local_precision": "precision",
        "local_recall": "recall",
        "local_mmd_rbf": "mmd_rbf",
    }.items():
        topology_gin[model][target] = final[model][source]
for key in ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear"):
    topology_gin["defog"][key] = defog_random_gin["topology_control"][key]

# Exact 182 retained-rule hard metrics. Preserve all shared scalar diagnostics.
raw_rule = {k: [] for k in MODEL_KEYS}
for seed in range(3):
    for key, prefix in (("false", "false"), ("total_count", "total_count"), ("full_matrix", "full_matrix")):
        raw_rule[key].append(load(RULE_ROOT / f"{prefix}_seed{seed}.json"))
    raw_rule["defog"].append(load(ROOT / "rules" / f"seed_{seed}.json"))

rule_alias = {
    "aggregate_count_distance": "hard_aggregate_rmse",
    "aggregate_mean_absolute_count_difference": "hard_aggregate_mae",
    "mean_vector_count_distance": "hard_mean_vector_rmse",
    "mean_vector_absolute_count_difference": "hard_mean_vector_mae",
    "relative_count_distance": "hard_relative_rmse",
    "standardized_mean_vector_count_distance": "hard_standardized_mean_rmse",
    "log1p_mean_vector_count_distance": "hard_log1p_mean_rmse",
    "robust_count_distance": "hard_robust_wasserstein",
    "log1p_wasserstein_mean": "hard_wasserstein_mean",
    "log1p_wasserstein_median": "hard_wasserstein_median",
    "gaussian_aligned_score": "hard_gaussian",
    "paired_graph_count_distance_mean": "hard_paired_graph_rmse_mean",
    "paired_graph_count_distance_median": "hard_paired_graph_rmse_median",
    "paired_graph_count_distance_std": "hard_paired_graph_rmse_std",
}
pruned = {k: {} for k in MODEL_KEYS}
for model, records in raw_rule.items():
    blocks = [r["hard"] if model != "defog" else r["hard_pruned_rule_metrics"] for r in records]
    for source, target in rule_alias.items():
        pruned[model][target] = summary([b[source] for b in blocks])

# Soft diagnostics only exist for GraphVAE, because DeFoG exports discrete graphs.
for model in ("false", "total_count", "full_matrix"):
    blocks = [r["soft"] for r in raw_rule[model]]
    for source, target in {
        "aggregate_count_distance": "soft_aggregate_rmse",
        "mean_vector_count_distance": "soft_mean_vector_rmse",
        "robust_count_distance": "soft_robust_wasserstein",
        "gaussian_aligned_score": "soft_gaussian",
    }.items():
        pruned[model][target] = summary([b[source] for b in blocks])

# Complete 6,899-state normalized rule correlation.
correlation = {k: {} for k in MODEL_KEYS}
for model in ("false", "total_count", "full_matrix"):
    correlation[model]["full_state_tv"] = old["full_state"][model]["aggregate_state_distribution_tv"]
for model in ("defog",):
    correlation[model]["full_state_tv"] = summary([
        r["normalized_complete_state_correlation"]["mean_per_rule_aggregate_state_tv"]
        for r in raw_rule[model]
    ])

simple_edge_path = {"defog": {"tv": summary([
    item["linkcorr_full_state_distribution"]["total_variation"] for item in defog_structural_raw
])}}

payload = {
    "schema_version": "ptc-defog-graphvae-comparison-v1",
    "dataset": "PTC",
    "training_seeds": [0, 1, 2],
    "held_out_final_metrics": final,
    "decoded_node_random_gin": attributed,
    "topology_control_random_gin": topology_gin,
    "exact_pruned_rule_metrics": pruned,
    "normalized_complete_state_correlation": correlation,
    "simple_edge_path_distribution_tv": simple_edge_path,
    "artifacts": {
        "graphvae_source": str(OLD),
        "defog_campaign": "/local-scratch2/mirzaei/defog_ptc_frozen_20260906",
        "defog_metric_root": str(ROOT),
    },
}

structural_metrics = [
    ("Degree MMD", "degree", False),
    ("Clustering MMD", "clustering", False),
    ("Orbit MMD", "orbit", False),
    ("Spectral MMD", "spectral", False),
    ("Diameter MMD", "diameter", False),
    ("Sparsity error", "sparsity", False),
    ("Triangle metric", "triangle", False),
    ("Mean-edge-count absolute error", "edge_count_absolute_error", False),
]
third_party_metrics = [
    ("MMD-RBF", "third_party_mmd_rbf", False),
    ("Linear MMD, arithmetic mean", "third_party_mmd_linear", False),
    ("Linear MMD, median", "third_party_mmd_linear_median", False),
    ("Linear MMD, 10%-trimmed mean", "third_party_mmd_linear_trimmed", False),
    ("Precision", "third_party_precision", True),
    ("Recall", "third_party_recall", True),
    ("F1-PR", "third_party_f1_pr", True),
]
soft_metrics = [
    ("Soft Gaussian-aligned NLL", "soft_gaussian", False),
    ("Soft aggregate raw-count RMSE", "soft_aggregate_rmse", False),
    ("Soft mean-vector RMSE", "soft_mean_vector_rmse", False),
    ("Soft robust log1p-Wasserstein", "soft_robust_wasserstein", False),
]
hard_metrics = [
    ("Hard Gaussian-aligned NLL", "hard_gaussian", False),
    ("Aggregate raw-count RMSE", "hard_aggregate_rmse", False),
    ("Aggregate raw-count MAE", "hard_aggregate_mae", False),
    ("Mean-vector RMSE", "hard_mean_vector_rmse", False),
    ("Mean-vector MAE", "hard_mean_vector_mae", False),
    ("Relative mean-count RMSE", "hard_relative_rmse", False),
    ("Standardized mean-count RMSE", "hard_standardized_mean_rmse", False),
    ("Log1p mean-vector RMSE", "hard_log1p_mean_rmse", False),
    ("Robust upper-10%-trimmed log1p-Wasserstein", "hard_robust_wasserstein", False),
    ("Mean log1p-Wasserstein", "hard_wasserstein_mean", False),
    ("Median log1p-Wasserstein", "hard_wasserstein_median", False),
    ("Paired graph RMSE, mean", "hard_paired_graph_rmse_mean", False),
    ("Paired graph RMSE, median", "hard_paired_graph_rmse_median", False),
    ("Paired graph RMSE, SD", "hard_paired_graph_rmse_std", False),
]

best_structural = {}
for label, key, higher in structural_metrics:
    candidates = [(model, final[model][key]["mean"]) for model in MODEL_KEYS]
    best_structural[label] = (max if higher else min)(candidates, key=lambda x: x[1])[0]

lines = [
    "# PTC: GraphVAE motif=False, motif=True, and DeFoG — complete metric comparison",
    "",
    "Generated 2026-09-07. Values are **mean ± sample SD across training seeds 0, 1, and 2**. "
    "All arrows state metric direction. DeFoG percentage columns report direction-aware change relative to GraphVAE motif=False and to the better of the two motif=True variants; positive means DeFoG is better.",
    "",
    "## Executive result",
    "",
    "Across the three seeds, GraphVAE motif=True full matrix is strongest on most held-out structural and structural-feature Random-GIN metrics. "
    "DeFoG is best on sparsity error, but its seed 2 is substantially weaker than seeds 0 and 1, producing large held-out-metric variance. "
    "On the motif-specific evaluation, however, DeFoG is strongest: its exact aggregate rule-count RMSE is 52.3% lower than the best GraphVAE baseline, and its complete-state TV error is 28.3% lower than GraphVAE motif=True full matrix. "
    "GraphVAE motif=True full matrix remains slightly better on hard Gaussian-aligned NLL and median log1p-Wasserstein.",
    "",
    "## Fairness and protocol",
    "",
    "| Property | GraphVAE motif=False | GraphVAE motif=True total/full | DeFoG |",
    "| --- | --- | --- | --- |",
    "| Training seeds | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 |",
    "| Split | paper 70/10/20, seed 123 | paper 70/10/20, seed 123 | same frozen paper 70/10/20, seed 123 |",
    "| Epochs | 20,000 | 20,000 | 1,000 |",
    "| Batch size | 240 | 256 | 12 |",
    "| Latent dimension | 1,024 | 128 | N/A (discrete diffusion) |",
    "| Loss weights | kernel/BCE 1; KL 1; node 1; edge 0; motif 0 | kernel/BCE 1; KL 1; node 1; edge 1; motif 0.1 | DeFoG objective; no GraphVAE motif term |",
    "| PTC features | 19-way categorical node label; no edge attributes | same | same decoded 19-way node label; no edge attributes |",
    "| Final held-out metrics | about 70 generated vs 70 test graphs | 70 generated vs 70 test graphs | exactly 70 generated vs 70 frozen test graphs |",
    "| Exact rule counts | 240 generated vs 240 train graphs | same | newly generated 240 vs the identical frozen 240 train graphs |",
    "",
    "> **Important:** this is a model comparison, not a motif-only controlled ablation. The GraphVAE motif=False and motif=True runs differ in latent size, batch size, and edge-loss weight as well as motif loss. DeFoG is a different architecture and training schedule. The legacy motif=False final evaluator also reports seed-dependent reference edge means, whereas motif=True and DeFoG use the frozen 70-graph test reference; exact rule metrics below all use the identical frozen 240-graph training reference.",
    "",
    "Coverage note: PTC has node labels but no edge attributes. These graph-generation outputs do not provide link-prediction AUC/AP, so no AUC/AP values are invented or substituted.",
    "",
    "## Held-out structural metrics (70 test graphs)",
    "",
    table(structural_metrics, final),
    "",
    "Descriptive mean edge counts:",
    "",
    table([
        ("Generated edges", "generated_edge_count", False),
        ("Reference edges", "reference_edge_count", False),
    ], final, include_gain=False),
    "",
    "The generated/reference edge-count rows are descriptive means, not optimization metrics; do not rank models by the reference row.",
    "",
    "## Third-party Random-GIN using structural node features (70 test graphs)",
    "",
    "Each training-seed result is the mean of ten evaluator initializations; the displayed SD is across the three independently trained generator seeds.",
    "",
    table(third_party_metrics, final),
    "",
    "## Topology-only/local Random-GIN (secondary)",
    "",
    "These values are retained for completeness. DeFoG uses the frozen third-party constant-node topology-control protocol, while the GraphVAE columns are the older local Random-GIN results. Because the implementations are not the same frozen evaluator path, this table is descriptive and should not be used as the primary cross-model ranking.",
    "",
    table([
        ("MMD-RBF", "mmd_rbf", False),
        ("Linear MMD", "mmd_linear", False),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-PR", "f1_pr", True),
    ], topology_gin, include_gain=False),
    "",
    "## DeFoG Random-GIN using decoded PTC node labels",
    "",
    "This attributed evaluation is available for DeFoG, but an exactly matching decoded-node GraphVAE evaluation is not present, so the GraphVAE cells remain N/A rather than mixing evaluator modes.",
    "",
    table([
        ("MMD-RBF", "mmd_rbf", False),
        ("Linear MMD", "mmd_linear", False),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-PR", "f1_pr", True),
    ], attributed, include_gain=False),
    "",
    "## Exact retained-rule metrics (182 of 6,899 combinations; 240 graphs)",
    "",
    "The retained set is exactly the PTC CP-smoothed training selection: five rule structures and 182 nonzero-weight combinations after data-driven pruning (full universe 6,899). Counts are computed on hard graphs, summed over all 240 graphs for aggregate metrics, and compared with the same 240 training graphs.",
    "",
    "### Soft GraphVAE diagnostics",
    "",
    table(soft_metrics, pruned),
    "",
    "DeFoG is N/A here because it exports discrete graphs and has no GraphVAE soft adjacency/reconstruction tensor.",
    "",
    "### Hard/discrete diagnostics",
    "",
    table(hard_metrics, pruned),
    "",
    "**Gaussian pairing caveat:** GraphVAE Gaussian metrics use reconstruction-aligned/node-count-matched pairs. DeFoG is unconditional, so its 240 generated and reference graphs are independently sorted by node count before computing graph-wise Gaussian residuals. Therefore the DeFoG hard Gaussian and paired-graph rows are size-matched diagnostics, not exact reconstruction-pair equivalents. The aggregate, mean-vector, Wasserstein, and normalized-correlation metrics do not depend on graph ordering.",
    "",
    "## Normalized complete full-state motif correlation (all 6,899 states)",
    "",
    table([("Aggregate-state total-variation error", "full_state_tv", False)], correlation),
    "",
    "For PTC, exactly **one** eligible multi-atom rule contains all 6,859 categorical path states: `node_feature(nodes0) AND edges(nodes0,nodes1) AND edges(nodes1,nodes2) AND node_feature(nodes1) AND node_feature(nodes2)`. Counts are summed over graphs, normalized once within that rule, and compared by total variation. The remaining 40 entries belong to two unit-edge rows and two 19-state unary node-label rules; they are not eligible for the multi-atom correlation average.",
    "",
    "A simpler topology-only edge-path state check on DeFoG gives TV " + fmt(simple_edge_path["defog"]["tv"]) + ". It is reported only as an auxiliary check and is not a replacement for the 6,899-state PTC metric.",
    "",
    "## PTC rule structures used by motif=True training",
    "",
    "1. `edges(nodes0,nodes1)`",
    "2. `edges(nodes1,nodes2)`",
    "3. `node_feature(nodes0) AND edges(nodes0,nodes1) AND edges(nodes1,nodes2) AND node_feature(nodes1) AND node_feature(nodes2)`",
    "4. `node_feature(nodes1)`",
    "5. `node_feature(nodes2)`",
    "",
    "## Metric definitions",
    "",
    "- **Structural MMD/error:** standard degree, clustering, orbit, spectral, diameter, sparsity, triangle, and edge-count distribution/comparison metrics on held-out graphs.",
    "- **Random-GIN:** precision/recall/F1-PR and embedding MMD from ten random GIN evaluator initializations per generator seed.",
    "- **Aggregate raw-count RMSE:** `sqrt(mean_j((sum_g C_gen[g,j] - sum_g C_ref[g,j])^2))` over the 182 retained rule/value rows. It is explicitly summed, not averaged, over graphs.",
    "- **Mean-vector RMSE:** the same vector error after dividing each aggregate by 240; numerically aggregate RMSE / 240 here.",
    "- **Gaussian-aligned NLL:** calibrated Gaussian negative log likelihood over graph-wise residuals with the training-compatible smooth `min_log_sigma=-6` floor.",
    "- **Robust distance:** upper-10%-trimmed mean across rule/value rows of one-dimensional Wasserstein distances between `log1p` per-graph counts.",
    "- **Complete-state correlation:** total variation between aggregate normalized distributions across all states of the eligible multi-atom rule; lower is better.",
    "",
    "## Artifacts and saved models",
    "",
    "- GraphVAE motif=False results/models: `/local-scratch2/new/gather/datasets/ptc/setting_01/seed_{0,1,2}`",
    "- GraphVAE motif=True results/models: `/local-scratch2/mirzaei/motif_true_clean_20260906/ptc/{total_count,full_matrix}/seed_{0,1,2}`",
    "- Each GraphVAE directory contains `best_validation_mmd_model`.",
    "- DeFoG checkpoints/results: `/local-scratch2/mirzaei/defog_ptc_frozen_20260906/jobs/ptc/seed_{0,1,2}`",
    "- DeFoG checkpoints are selected by minimum validation loss: seed 0 epoch 249 (`0.846336`), seed 1 epoch 699 (`0.838646`), seed 2 epoch 49 (`0.994130`).",
    "- Newly computed DeFoG structural metrics: `/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/structural`",
    "- Newly computed DeFoG 240-graph exact-rule metrics: `/local-scratch2/mirzaei/defog_ptc_full_metrics_20260907/rules`",
    "- Existing GraphVAE exact-rule metrics: `/local-scratch2/mirzaei/ptc_rule_metrics_20260906/results`",
    "- Existing GraphVAE complete-state metrics: `/local-scratch2/mirzaei/ptc_full_state_correlation_20260906/results`",
    "",
    "## Reproducibility notes",
    "",
    "- Frozen split seed: 123; generation seed: 12345; evaluator seeds: 0–9.",
    "- DeFoG commit: `c631697b9cd5a2474d22ba12de33943c6b49e53e`.",
    "- Frozen PTC train/validation/test collection SHA-256: `0750c12d...`, `48142546...`, `aa0fa76e...` (full hashes are in the campaign manifest and machine-readable output).",
    "- Adjacency threshold 0.5; undirected; self-loops rejected; isolated nodes removed; deterministic largest connected component retained.",
]

ROOT.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
OUT_MD.write_text("\n".join(lines) + "\n")
LOCAL_JSON.write_text(OUT_JSON.read_text())
LOCAL_MD.write_text(OUT_MD.read_text())
print(OUT_MD)
print(OUT_JSON)
print(LOCAL_MD)
print(LOCAL_JSON)
