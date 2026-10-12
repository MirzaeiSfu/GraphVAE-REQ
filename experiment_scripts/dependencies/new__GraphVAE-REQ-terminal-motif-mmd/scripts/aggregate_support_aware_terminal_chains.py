#!/usr/bin/env python3
"""Aggregate support-aware terminal-chain metrics without post-hoc chain choice.

The comparison set is fixed independently of generated/test metric quality:
chains must have the same canonical identity in every setting and occur in at
least ``--min-training-support`` training graphs.  Random-GIN uses the stricter
subset that also has adequate real/generated nonempty samples in every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


EVALUATION_NAME = "count_aligned_per_chain.json"
ALL_GRAPH_METRICS = [
    "generated_occurrence_rate",
    "reference_occurrence_rate",
    "occurrence_rate_absolute_error",
    "generated_binary_motif_count_mean_all_graphs",
    "reference_binary_motif_count_mean_all_graphs",
    "binary_motif_count_mean_absolute_error",
    "binary_motif_count_wasserstein",
    "generated_active_nodes_mean_all_graphs",
    "reference_active_nodes_mean_all_graphs",
    "active_nodes_mean_absolute_error",
    "active_nodes_wasserstein",
    "generated_edges_mean_all_graphs",
    "reference_edges_mean_all_graphs",
    "edges_mean_absolute_error",
    "edges_wasserstein",
    "generated_components_mean_all_graphs",
    "reference_components_mean_all_graphs",
    "components_mean_absolute_error",
    "components_wasserstein",
]
ALL_GRAPH_ERROR_METRICS = [
    "occurrence_rate_absolute_error",
    "binary_motif_count_mean_absolute_error",
    "binary_motif_count_wasserstein",
    "active_nodes_mean_absolute_error",
    "active_nodes_wasserstein",
    "edges_mean_absolute_error",
    "edges_wasserstein",
    "components_mean_absolute_error",
    "components_wasserstein",
]
STRUCTURAL_METRICS = [
    "degree",
    "clustering",
    "sparsity",
    "orbit",
    "spectral",
    "triangle",
    "diameter",
]
GIN_METRICS = ["mmd_rbf", "precision", "recall", "f1_pr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        type=Path,
        help="Evaluation root. May be supplied more than once.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-training-support", default=20, type=int)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_label(root: Path) -> str:
    lowered = str(root).lower()
    return "last" if "last_model" in lowered else "best"


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_std(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def metric_value(chain: Mapping[str, Any], family: str, metric: str) -> Any:
    if family == "all_graph":
        return chain["distribution_metrics"].get(metric)
    if family == "structural":
        return chain[
            "conditional_largest_component_structural_mmd"
        ].get(metric)
    if family == "gin":
        gin = chain.get("random_gin")
        if not gin:
            return None
        return gin.get("metrics", {}).get(metric)
    raise KeyError(family)


def macro_for_run(
    chains: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
    family: str,
    metric: str,
) -> float | None:
    values = [
        float(metric_value(chains[key], family, metric))
        for key in keys
        if key in chains and finite(metric_value(chains[key], family, metric))
    ]
    return float(np.mean(values)) if values else None


def discover_roots(
    roots: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    runs: List[Dict[str, Any]] = []
    labels: Dict[str, str] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        label = checkpoint_label(root)
        if label in labels:
            raise ValueError(
                f"Duplicate checkpoint label {label!r}: "
                f"{labels[label]} and {root}"
            )
        labels[label] = str(root)
        paths = sorted(root.rglob(EVALUATION_NAME))
        for path in paths:
            payload = load_json(path)
            payload["_path"] = str(path)
            payload["_checkpoint_label"] = label
            payload["_chains_by_key"] = {
                chain["canonical_chain_key"]: chain
                for chain in payload["chains"]
            }
            runs.append(payload)
    return runs, labels


def fixed_sets(
    dataset_runs: Sequence[Mapping[str, Any]],
    min_training_support: int,
) -> Dict[str, Any]:
    key_sets = [set(run["_chains_by_key"]) for run in dataset_runs]
    common = set.intersection(*key_sets) if key_sets else set()
    training = {
        key
        for key in common
        if all(
            int(run["_chains_by_key"][key]["training_nonempty_count"])
            >= min_training_support
            for run in dataset_runs
        )
    }
    stable_reference = {
        key
        for key in training
        if all(
            bool(run["_chains_by_key"][key].get("reference_eligible"))
            for run in dataset_runs
        )
    }
    stable_bilateral = {
        key
        for key in stable_reference
        if all(
            bool(run["_chains_by_key"][key].get("gin_eligible"))
            for run in dataset_runs
        )
    }
    structural: Dict[str, List[str]] = {}
    for metric in STRUCTURAL_METRICS:
        structural[metric] = sorted(
            key
            for key in stable_bilateral
            if all(
                finite(
                    metric_value(
                        run["_chains_by_key"][key], "structural", metric
                    )
                )
                for run in dataset_runs
            )
        )
    strict_gin = {
        key
        for key in stable_bilateral
        if all(
            all(
                finite(metric_value(run["_chains_by_key"][key], "gin", metric))
                for metric in GIN_METRICS
            )
            for run in dataset_runs
        )
    }
    return {
        "common_canonical_keys": sorted(common),
        "training_eligible_keys": sorted(training),
        "stable_reference_keys": sorted(stable_reference),
        "stable_bilateral_keys": sorted(stable_bilateral),
        "strict_common_gin_keys": sorted(strict_gin),
        "structural_fixed_keys": structural,
    }


def aggregate(
    runs: Sequence[Mapping[str, Any]], min_training_support: int
) -> Dict[str, Any]:
    all_by_dataset: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_checkpoint_dataset: Dict[
        Tuple[str, str], List[Mapping[str, Any]]
    ] = defaultdict(list)
    for run in runs:
        all_by_dataset[str(run["dataset"])].append(run)
        by_checkpoint_dataset[
            (run["_checkpoint_label"], str(run["dataset"]))
        ].append(run)

    chain_sets: Dict[str, Any] = {
        dataset: {
            "dataset": dataset,
            "scope": "all compared settings and both checkpoint types",
            **fixed_sets(dataset_runs, min_training_support),
        }
        for dataset, dataset_runs in sorted(all_by_dataset.items())
    }
    for value in chain_sets.values():
        value["common_chain_count"] = len(value["common_canonical_keys"])
        value["training_eligible_chain_count"] = len(
            value["training_eligible_keys"]
        )
        value["stable_reference_chain_count"] = len(
            value["stable_reference_keys"]
        )
        value["stable_bilateral_chain_count"] = len(
            value["stable_bilateral_keys"]
        )
        value["strict_common_gin_chain_count"] = len(
            value["strict_common_gin_keys"]
        )
    rows: List[Dict[str, Any]] = []
    for (checkpoint, dataset), dataset_runs in sorted(
        by_checkpoint_dataset.items()
    ):
        sets = chain_sets[dataset]
        by_setting: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for run in dataset_runs:
            by_setting[str(run["setting"])].append(run)

        for setting, setting_runs in sorted(by_setting.items()):
            row: Dict[str, Any] = {
                "checkpoint": checkpoint,
                "dataset": dataset,
                "setting": setting,
                "training_seeds": sorted(
                    int(run["training_seed"]) for run in setting_runs
                ),
                "generation_seeds": sorted(
                    {int(run["generation_seed"]) for run in setting_runs}
                ),
                "seed_count": len(setting_runs),
                "common_chain_count": len(sets["common_canonical_keys"]),
                "training_eligible_chain_count": len(
                    sets["training_eligible_keys"]
                ),
                "stable_reference_chain_count": len(
                    sets["stable_reference_keys"]
                ),
                "stable_bilateral_chain_count": len(
                    sets["stable_bilateral_keys"]
                ),
                "strict_common_gin_chain_count": len(
                    sets["strict_common_gin_keys"]
                ),
                "metrics": {},
            }
            for metric in ALL_GRAPH_METRICS:
                values = [
                    value
                    for run in setting_runs
                    if (
                        value := macro_for_run(
                            run["_chains_by_key"],
                            sets["training_eligible_keys"],
                            "all_graph",
                            metric,
                        )
                    )
                    is not None
                ]
                row["metrics"][metric] = {
                    **mean_std(values),
                    "chain_count": len(sets["training_eligible_keys"]),
                    "chain_set": "training_eligible_common",
                }
            for metric in STRUCTURAL_METRICS:
                keys = sets["structural_fixed_keys"][metric]
                values = [
                    value
                    for run in setting_runs
                    if (
                        value := macro_for_run(
                            run["_chains_by_key"],
                            keys,
                            "structural",
                            metric,
                        )
                    )
                    is not None
                ]
                row["metrics"][metric] = {
                    **mean_std(values),
                    "chain_count": len(keys),
                    "chain_set": "metric_specific_fixed_common",
                }
            for metric in GIN_METRICS:
                keys = sets["strict_common_gin_keys"]
                values = [
                    value
                    for run in setting_runs
                    if (
                        value := macro_for_run(
                            run["_chains_by_key"],
                            keys,
                            "gin",
                            metric,
                        )
                    )
                    is not None
                ]
                row["metrics"][metric] = {
                    **mean_std(values),
                    "chain_count": len(keys),
                    "chain_set": "strict_common_gin",
                }
            rows.append(row)
    return {
        "protocol": {
            "unit_of_analysis": (
                "per-seed macro average over a fixed canonical chain set; "
                "then mean and sample SD across training seeds"
            ),
            "chain_identity": "canonical rule/states/value/source SHA256",
            "all_graph_chain_set": (
                "intersection across all compared settings and both checkpoint "
                "types for a dataset, training occurrence >= threshold"
            ),
            "all_graph_empty_handling": (
                "motif count/occurrence include every valid binarized terminal-"
                "matrix cell, including diagonal; empty chains are retained as "
                "zero count/occurrence; node/edge/component summaries use a "
                "separate diagonal-cleared structural projection"
            ),
            "conditional_structural_chain_set": (
                "fixed common subset with at least the configured nonempty "
                "reference and generated support in every compared run and "
                "for which that metric is defined"
            ),
            "random_gin_chain_set": (
                "fixed common subset with training, reference, and generated "
                "nonempty support >= thresholds in every compared run"
            ),
            "random_gin_graph_handling": (
                "full nonempty graph retained; disconnected components are "
                "not discarded; 10 deterministic repetitions per chain"
            ),
            "minimum_training_support": min_training_support,
            "lower_is_better": (
                ALL_GRAPH_ERROR_METRICS
                + STRUCTURAL_METRICS
                + ["mmd_rbf"]
            ),
            "higher_is_better": ["precision", "recall", "f1_pr"],
        },
        "chain_sets": chain_sets,
        "rows": rows,
    }


def fmt(stat: Mapping[str, Any], digits: int = 4) -> str:
    if stat.get("mean") is None:
        return "N/A"
    return (
        f"{float(stat['mean']):.{digits}f} ± "
        f"{float(stat['std']):.{digits}f}"
    )


def write_summary_csv(path: Path, payload: Mapping[str, Any]) -> None:
    metric_names = ALL_GRAPH_METRICS + STRUCTURAL_METRICS + GIN_METRICS
    fields = [
        "checkpoint",
        "dataset",
        "setting",
        "training_seeds",
        "generation_seeds",
        "seed_count",
        "common_chain_count",
        "training_eligible_chain_count",
        "stable_reference_chain_count",
        "stable_bilateral_chain_count",
        "strict_common_gin_chain_count",
    ]
    for metric in metric_names:
        fields += [
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_seed_n",
            f"{metric}_chain_count",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in payload["rows"]:
            row = {key: source.get(key) for key in fields}
            row["training_seeds"] = ",".join(
                str(seed) for seed in source["training_seeds"]
            )
            row["generation_seeds"] = ",".join(
                str(seed) for seed in source["generation_seeds"]
            )
            for metric in metric_names:
                stat = source["metrics"][metric]
                row[f"{metric}_mean"] = stat["mean"]
                row[f"{metric}_std"] = stat["std"]
                row[f"{metric}_seed_n"] = stat["n"]
                row[f"{metric}_chain_count"] = stat["chain_count"]
            writer.writerow(row)


def write_per_chain_csv(
    path: Path, runs: Sequence[Mapping[str, Any]]
) -> None:
    metric_names = ALL_GRAPH_METRICS + STRUCTURAL_METRICS + GIN_METRICS
    fields = [
        "checkpoint",
        "dataset",
        "setting",
        "training_seed",
        "generation_seed",
        "run_dir",
        "chain_index",
        "canonical_chain_key",
        "canonical_chain_json",
        "rule_index",
        "value_index",
        "training_graph_count",
        "training_nonempty_count",
        "reference_input_count",
        "reference_nonempty_count",
        "generated_input_count",
        "generated_nonempty_count",
        "training_eligible",
        "reference_eligible",
        "gin_eligible",
    ] + metric_names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(
            runs,
            key=lambda item: (
                item["_checkpoint_label"],
                item["dataset"],
                item["setting"],
                int(item["training_seed"]),
            ),
        ):
            for chain in run["chains"]:
                row = {
                    "checkpoint": run["_checkpoint_label"],
                    "dataset": run["dataset"],
                    "setting": run["setting"],
                    "training_seed": run["training_seed"],
                    "generation_seed": run["generation_seed"],
                    "run_dir": run["run_dir"],
                    "canonical_chain_json": json.dumps(
                        chain["canonical_chain"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for key in fields:
                    if key in chain:
                        row[key] = chain[key]
                for metric in ALL_GRAPH_METRICS:
                    row[metric] = metric_value(chain, "all_graph", metric)
                for metric in STRUCTURAL_METRICS:
                    row[metric] = metric_value(chain, "structural", metric)
                for metric in GIN_METRICS:
                    row[metric] = metric_value(chain, "gin", metric)
                writer.writerow(row)


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Tuple[str, str]],
) -> List[str]:
    headers = ["Checkpoint", "Dataset", "Setting", "Seeds", "Chains"] + [
        label for _, label in metrics
    ]
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [
            str(row["checkpoint"]),
            str(row["dataset"]),
            str(row["setting"]),
            ",".join(str(seed) for seed in row["training_seeds"]),
            (
                f"{row['training_eligible_chain_count']} train / "
                f"{row['strict_common_gin_chain_count']} GIN"
            ),
        ]
        cells += [fmt(row["metrics"][metric]) for metric, _ in metrics]
        output.append("| " + " | ".join(cells) + " |")
    return output


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Count-aligned support-aware terminal motif evaluation",
        "",
        (
            "Values are mean ± sample SD across training seeds. Each seed is "
            "first macro-averaged over the same canonical, training-supported "
            "chains. Motif count/occurrence sum all valid binarized terminal-"
            "matrix cells including the diagonal, matching count-mode "
            "reduction. Structural graph metrics use a separate diagonal-"
            "cleared projection."
        ),
        "",
        (
            "Lower is better for occurrence error, Wasserstein distances, "
            "structural MMDs, and MMD-RBF. Higher is better for F1-PR, "
            "precision, and recall."
        ),
        "",
        "## Fixed chain-set coverage",
        "",
        (
            "| Dataset | Common | Training-supported | Stable real | "
            "Stable both | Strict GIN | Degree | Clustering | Sparsity | Orbit | "
            "Spectral | Triangle | Diameter |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, sets in sorted(payload["chain_sets"].items()):
        structural = sets["structural_fixed_keys"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    str(sets["common_chain_count"]),
                    str(sets["training_eligible_chain_count"]),
                    str(sets["stable_reference_chain_count"]),
                    str(sets["stable_bilateral_chain_count"]),
                    str(sets["strict_common_gin_chain_count"]),
                    str(len(structural["degree"])),
                    str(len(structural["clustering"])),
                    str(len(structural["sparsity"])),
                    str(len(structural["orbit"])),
                    str(len(structural["spectral"])),
                    str(len(structural["triangle"])),
                    str(len(structural["diameter"])),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## All-graph occurrence/distribution and Random-GIN",
        "",
    ]
    lines += markdown_table(
        rows,
        [
            ("occurrence_rate_absolute_error", "Occurrence AE ↓"),
            ("binary_motif_count_wasserstein", "Count W1 ↓"),
            ("active_nodes_wasserstein", "Nodes W1 ↓"),
            ("edges_wasserstein", "Edges W1 ↓"),
            ("components_wasserstein", "Components W1 ↓"),
            ("mmd_rbf", "MMD-RBF ↓"),
            ("f1_pr", "F1-PR ↑"),
            ("precision", "Precision ↑"),
            ("recall", "Recall ↑"),
        ],
    )
    lines += ["", "## Conditional structural MMD (fixed comparable chains)", ""]
    lines += markdown_table(
        rows,
        [
            ("degree", "Degree ↓"),
            ("clustering", "Clustering ↓"),
            ("sparsity", "Sparsity ↓"),
            ("orbit", "Orbit ↓"),
            ("spectral", "Spectral ↓"),
            ("triangle", "Triangle ↓"),
            ("diameter", "Diameter ↓"),
        ],
    )
    lines += [
        "",
        "## Important interpretation",
        "",
        (
            "- `Chains` is `training-supported / strict GIN`. A zero strict "
            "GIN count is reported as N/A, not estimated from tiny samples."
        ),
        (
            "- Random-GIN uses complete nonempty motif graphs, including all "
            "connected components. It is no longer run on only the largest "
            "component."
        ),
        (
            "- Count/occurrence and graph structure are intentionally separate: "
            "diagonal entries contribute to the motif count but are removed as "
            "self-loops before structural and GIN metrics."
        ),
        (
            "- The full raw result for every seed and chain is in "
            "`count_aligned_per_chain_full.csv`; exact aggregate values are "
            "in `count_aligned_aggregate.json` and "
            "`count_aligned_summary.csv`."
        ),
        (
            "- All currently archived generations use generation seed 0; "
            "training-seed variation is shown separately."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: clean_for_json(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    return value


def main() -> int:
    cli = parse_args()
    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, roots = discover_roots(cli.root)
    if not runs:
        raise RuntimeError("No support-aware result files found.")
    payload = aggregate(runs, int(cli.min_training_support))
    payload["roots"] = roots
    payload["run_count"] = len(runs)
    with (output_dir / "count_aligned_aggregate.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(clean_for_json(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_summary_csv(output_dir / "count_aligned_summary.csv", payload)
    write_per_chain_csv(
        output_dir / "count_aligned_per_chain_full.csv", runs
    )
    write_markdown(output_dir / "COUNT_ALIGNED_RESULTS.md", payload)
    print(
        f"[done] {len(runs)} runs -> "
        f"{output_dir / 'COUNT_ALIGNED_RESULTS.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
