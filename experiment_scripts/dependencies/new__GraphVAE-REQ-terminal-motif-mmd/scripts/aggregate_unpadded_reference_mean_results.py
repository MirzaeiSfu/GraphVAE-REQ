#!/usr/bin/env python3
"""Aggregate reference-mean unpadded terminal-rule metrics over seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from per_chain_terminal_motif_evaluation import METRIC_SPECS
from terminal_motif_evaluation import write_json


METRICS = (*tuple(METRIC_SPECS), "mmd_rbf", "precision", "recall", "f1_pr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary(values: List[float]) -> Dict[str, Any]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "sample_std": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ) if values else None,
        "seed_count": len(values),
    }


def fmt(metric: Dict[str, Any]) -> str:
    if metric["mean"] is None:
        return "N/A"
    return f"{metric['mean']:.4f} ± {metric['sample_std']:.4f}"


def main() -> int:
    cli = parse_args()
    root = cli.root.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    paths = sorted(root.rglob("per_rule_metrics.json"))
    if not paths:
        raise RuntimeError(f"No per_rule_metrics.json files under {root}.")

    runs = [load_json(path) for path in paths]
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["setting"])].append(run)

    rows: List[Dict[str, Any]] = []
    for (dataset, setting), group in sorted(groups.items()):
        group.sort(key=lambda item: int(item["training_seed"]))
        metric_payload: Dict[str, Any] = {}
        for metric in METRICS:
            values = [
                float(run["macro_average_over_rules"][metric]["mean"])
                for run in group
                if run["macro_average_over_rules"][metric]["mean"] is not None
            ]
            metric_payload[metric] = summary(values)
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "seeds": [int(run["training_seed"]) for run in group],
                "rule_counts": [
                    int(run["selected_rule_count"]) for run in group
                ],
                "metric_rule_coverage_per_seed": {
                    metric: [
                        {
                            "seed": int(run["training_seed"]),
                            **run["macro_average_over_rules"][metric],
                        }
                        for run in group
                    ]
                    for metric in METRICS
                },
                "metrics": metric_payload,
            }
        )

    payload = {
        "protocol_version": "reference_mean_per_rule_unpadded_v1",
        "run_count": len(runs),
        "setting_count": len(rows),
        "occurrence_metric_computed": False,
        "padding_handling": "all padded rows/columns excluded before motif counting",
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "aggregated_results.json", payload)

    csv_fields = [
        "dataset",
        "setting",
        "seeds",
        "rule_counts",
        *[
            suffix
            for metric in METRICS
            for suffix in (f"{metric}_mean", f"{metric}_sample_std")
        ],
    ]
    with (output_dir / "aggregated_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row: Dict[str, Any] = {
                "dataset": row["dataset"],
                "setting": row["setting"],
                "seeds": ",".join(map(str, row["seeds"])),
                "rule_counts": ",".join(map(str, row["rule_counts"])),
            }
            for metric in METRICS:
                csv_row[f"{metric}_mean"] = row["metrics"][metric]["mean"]
                csv_row[f"{metric}_sample_std"] = row["metrics"][metric][
                    "sample_std"
                ]
            writer.writerow(csv_row)

    lines = [
        "# Reference-mean unpadded terminal-rule evaluation",
        "",
        "Every rule is evaluated independently against the matching test rule. "
        "The table reports the mean across rules for each seed, followed by "
        "mean ± sample SD across training seeds. Padded nodes are removed before "
        "motif counting. Occurrence is not computed.",
        "",
        "| Dataset | Setting | Seeds | Rules | Degree ↓ | Clustering ↓ | "
        "Sparsity ↓ | Orbit ↓ | Spectral ↓ | Triangle ↓ | Diameter ↓ | "
        "MMD-RBF ↓ | F1-PR ↑ | Precision ↑ | Recall ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {dataset} | {setting} | {seeds} | {rules} | {degree} | "
            "{clustering} | {sparsity} | {orbit} | {spectral} | "
            "{triangle} | {diameter} | {mmd_rbf} | {f1_pr} | "
            "{precision} | {recall} |".format(
                dataset=row["dataset"],
                setting=row["setting"],
                seeds=",".join(map(str, row["seeds"])),
                rules=",".join(map(str, row["rule_counts"])),
                **{metric: fmt(metrics[metric]) for metric in METRICS},
            )
        )
    lines.extend(
        [
            "",
            "Lower is better for structural MMD and MMD-RBF. Higher is better "
            "for F1-PR, precision, and recall. Per-rule values and rule coverage "
            "are saved in the JSON/CSV outputs.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        f"[done] runs={len(runs)} settings={len(rows)} "
        f"output={output_dir / 'RESULTS.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
