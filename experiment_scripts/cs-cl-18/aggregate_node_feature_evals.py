#!/usr/bin/env python3
"""Aggregate attributed Random-GIN evaluations across training seeds."""

import json
import statistics
from pathlib import Path


ROOT = Path("/local-scratch2/mirzaei/node_feature_evaluation_20260901")
METRICS = ("f1_pr", "precision", "recall", "mmd_rbf", "mmd_linear")
HIGHER_IS_BETTER = {"f1_pr", "precision", "recall"}
LABELS = {"false": "Motif=False", "total": "True total", "full": "True full"}


def values(dataset: str, setting: str, mode: str, metric: str) -> list[float]:
    result = []
    for seed in range(3):
        path = ROOT / dataset / f"{setting}_s{seed}" / "attributed_random_gin.json"
        payload = json.loads(path.read_text())
        result.append(payload["evaluation"]["modes"][mode]["summary"][metric]["mean"])
    return result


def fmt(samples: list[float]) -> str:
    return f"{statistics.mean(samples):.6f} ± {statistics.stdev(samples):.6f}"


def improvement(baseline: list[float], candidate: list[float], metric: str) -> str:
    base = statistics.mean(baseline)
    other = statistics.mean(candidate)
    signed = other - base if metric in HIGHER_IS_BETTER else base - other
    return f"{100.0 * signed / abs(base):+.2f}%"


lines = [
    "# Node-feature-aware third-party Random-GIN evaluation",
    "",
    "Generated 2026-09-01 for the experiments summarized in "
    "`FINAL_ALL_MOTIF_RESULTS_PRUNED_COUNT_DISTANCE.md`.",
    "",
    "- All non-motif loss weights: `1`; motif loss weight: `0.1` for motif=True.",
    "- Three training seeds per setting; values are mean ± sample SD across those seeds.",
    "- Each training-seed value is the mean over 10 matched Random-GIN initializations.",
    "- `decoded_node` consumes adjacency and the model-decoded categorical node feature.",
    "- `topology_control` consumes adjacency only and is included as an ablation.",
    "- Higher is better for F1-PR, precision, and recall; lower is better for MMD.",
    "- Positive improvement means motif=True is better than motif=False.",
    "- No edge features are evaluated because these checkpoints have no edge-feature decoder.",
    "",
]

for dataset in ("mutag", "proteins"):
    display = dataset.upper()
    first = json.loads(
        (ROOT / dataset / "false_s0" / "attributed_random_gin.json").read_text()
    )
    true_first = json.loads(
        (ROOT / dataset / "total_s0" / "attributed_random_gin.json").read_text()
    )
    lines.extend(
        [
            f"## {display}",
            "",
            f"Node-feature dimension: `{first['evaluation']['feature_dimensions']['node']}`. "
            f"Accepted graphs per collection: `{first['graph_counts']['accepted_per_collection']}` "
            f"for motif=False and `{true_first['graph_counts']['accepted_per_collection']}` "
            "for motif=True.",
            "",
            "### Adjacency plus decoded node features",
            "",
            "| Metric | Better | Motif=False | True total | Total improvement | True full | Full improvement |",
            "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        base = values(dataset, "false", "decoded_node", metric)
        total = values(dataset, "total", "decoded_node", metric)
        full = values(dataset, "full", "decoded_node", metric)
        better = "↑" if metric in HIGHER_IS_BETTER else "↓"
        lines.append(
            f"| {metric} | {better} | {fmt(base)} | {fmt(total)} | "
            f"{improvement(base, total, metric)} | {fmt(full)} | "
            f"{improvement(base, full, metric)} |"
        )
    lines.extend(
        [
            "",
            "### Topology-only control",
            "",
            "| Metric | Better | Motif=False | True total | True full |",
            "| --- | :---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        better = "↑" if metric in HIGHER_IS_BETTER else "↓"
        cells = [fmt(values(dataset, setting, "topology_control", metric)) for setting in ("false", "total", "full")]
        lines.append(f"| {metric} | {better} | " + " | ".join(cells) + " |")
    lines.append("")

(ROOT / "NODE_FEATURE_RANDOM_GIN_RESULTS.md").write_text("\n".join(lines) + "\n")
print(ROOT / "NODE_FEATURE_RANDOM_GIN_RESULTS.md")
