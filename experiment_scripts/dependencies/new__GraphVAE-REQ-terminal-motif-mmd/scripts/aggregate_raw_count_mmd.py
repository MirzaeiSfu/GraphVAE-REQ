#!/usr/bin/env python3
"""Aggregate raw training-vs-generated motif-count MMD results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


METRICS = (
    "raw_count_mmd_rbf",
    "raw_count_mmd_rbf_sigma1",
    "raw_count_mmd_linear",
    "count_wasserstein",
    "mean_count_absolute_error",
    "train_count_mean",
    "generated_count_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def sample_std(values: List[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> int:
    cli = parse_args()
    root = cli.root.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("raw_count_mmd.json"))
    if not paths:
        raise FileNotFoundError(f"No raw_count_mmd.json files under {root}.")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    run_rows: List[Dict[str, Any]] = []
    rule_rows: List[Dict[str, Any]] = []
    for path, payload in zip(paths, payloads):
        base = {
            "dataset": payload["dataset"],
            "setting": payload["setting"],
            "training_seed": payload["training_seed"],
            "generation_seed": payload["generation_seed"],
            "checkpoint": payload["checkpoint"],
            "training_graph_count": payload["training_graph_count"],
            "generated_graph_count": payload["generated_graph_count"],
            "selected_rule_count": payload["selected_rule_count"],
            "run_result": str(path),
        }
        run_rows.append({**base, **payload["macro_average_over_rules"]})
        for rule in payload["rules"]:
            rule_rows.append({**base, **rule})

    run_rows.sort(key=lambda row: (row["dataset"], row["setting"], row["training_seed"]))
    rule_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["setting"],
            row["training_seed"],
            row["rule_position"],
        )
    )
    run_fields = [
        "dataset",
        "setting",
        "training_seed",
        "generation_seed",
        "checkpoint",
        "training_graph_count",
        "generated_graph_count",
        "selected_rule_count",
        *METRICS,
        "run_result",
    ]
    write_csv(output_dir / "raw_count_mmd_all_runs.csv", run_rows, run_fields)

    rule_fields = [
        "dataset",
        "setting",
        "training_seed",
        "generation_seed",
        "checkpoint",
        "training_graph_count",
        "generated_graph_count",
        "selected_rule_count",
        "rule_position",
        "chain_index",
        "canonical_chain_key",
        "rule_index",
        "value_index",
        "train_count_mean",
        "train_count_std",
        "train_count_min",
        "train_count_max",
        "generated_count_mean",
        "generated_count_std",
        "generated_count_min",
        "generated_count_max",
        "mean_count_absolute_error",
        "count_wasserstein",
        "reference_rbf_sigma",
        "raw_count_mmd_rbf",
        "raw_count_mmd_rbf_sigma1",
        "raw_count_mmd_linear",
        "run_result",
    ]
    write_csv(output_dir / "raw_count_mmd_per_rule.csv", rule_rows, rule_fields)

    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["setting"])].append(row)
    aggregate_rows: List[Dict[str, Any]] = []
    for (dataset, setting), rows in sorted(grouped.items()):
        aggregate: Dict[str, Any] = {
            "dataset": dataset,
            "setting": setting,
            "training_seeds": ",".join(
                str(row["training_seed"]) for row in rows
            ),
            "seed_count": len(rows),
            "training_graph_counts": ",".join(
                str(row["training_graph_count"]) for row in rows
            ),
            "selected_rule_counts": ",".join(
                str(row["selected_rule_count"]) for row in rows
            ),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_sample_std"] = sample_std(values)
        aggregate_rows.append(aggregate)

    aggregate_fields = [
        "dataset",
        "setting",
        "training_seeds",
        "seed_count",
        "training_graph_counts",
        "selected_rule_counts",
    ]
    for metric in METRICS:
        aggregate_fields.extend(
            [f"{metric}_mean", f"{metric}_sample_std"]
        )
    write_csv(
        output_dir / "raw_count_mmd_aggregated.csv",
        aggregate_rows,
        aggregate_fields,
    )

    aggregate_payload = {
        "protocol": "raw_training_count_mmd_v1",
        "run_count": len(run_rows),
        "setting_count": len(aggregate_rows),
        "count_definition": (
            "sum of every valid unbinarized terminal-matrix cell per graph "
            "and rule; diagonal included; padded nodes excluded before motif counting"
        ),
        "mmd_definition": (
            "biased squared scalar RBF MMD independently per rule; training-only "
            "median-positive-distance bandwidth; linear-kernel squared MMD; "
            "arithmetic mean across rules"
        ),
        "occurrence_metric_computed": False,
        "runs": run_rows,
        "aggregated": aggregate_rows,
    }
    (output_dir / "raw_count_mmd_aggregated.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Raw training-count motif MMD",
        "",
        "Generated graph count equals the exact training graph count. Counts are raw sums of valid terminal-matrix cells; no terminal threshold or binarization is used. Padded nodes are excluded before motif counting. MMD is computed independently for every rule and then macro-averaged.",
        "",
        "| Dataset | Setting | Seeds | Train graphs | Rules | Raw count MMD-RBF ↓ | Linear MMD ↓ | Count W1 ↓ | Mean count AE ↓ |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {dataset} | {setting} | {training_seeds} | "
            "{training_graph_counts} | {selected_rule_counts} | "
            "{mmd:.6f} ± {mmd_std:.6f} | {linear:.6f} ± {linear_std:.6f} | "
            "{w1:.6f} ± {w1_std:.6f} | "
            "{ae:.6f} ± {ae_std:.6f} |".format(
                **row,
                mmd=row["raw_count_mmd_rbf_mean"],
                mmd_std=row["raw_count_mmd_rbf_sample_std"],
                linear=row["raw_count_mmd_linear_mean"],
                linear_std=row["raw_count_mmd_linear_sample_std"],
                w1=row["count_wasserstein_mean"],
                w1_std=row["count_wasserstein_sample_std"],
                ae=row["mean_count_absolute_error_mean"],
                ae_std=row["mean_count_absolute_error_sample_std"],
            )
        )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"[done] runs={len(run_rows)} settings={len(aggregate_rows)} "
        f"output={output_dir / 'raw_count_mmd_all_runs.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
