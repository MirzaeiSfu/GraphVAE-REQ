#!/usr/bin/env python3
"""Aggregate per-run terminal motif structural-MMD JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

import yaml


PRIMARY_METRICS = [
    "degree",
    "clustering",
    "orbit",
    "spectral",
    "diameter",
    "mmd_rbf",
    "f1_pr",
]
EXTRA_METRICS = [
    "sparsity",
    "triangle",
    "precision",
    "recall",
    "reference_edge_count",
    "generated_edge_count",
]
ALL_METRICS = PRIMARY_METRICS + EXTRA_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gather-root", required=True, type=Path)
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include motif_loss=false runs that have post-hoc results.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_motif_training_run(run_dir: Path) -> bool:
    config_path = run_dir / "run_config_used.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return bool(config.get("motif", {}).get("motif_loss"))


def seed_from_run_dir(run_dir: Path, fallback: int) -> int:
    match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    return int(match.group(1)) if match else int(fallback)


def collect_rows(root: Path, include_baselines: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_path in sorted(
        root.rglob("terminal_motif_mmd/terminal_motif_mmd.json")
    ):
        run_dir = result_path.parent.parent
        if not include_baselines and not is_motif_training_run(run_dir):
            continue
        payload = load_json(result_path)
        metrics = payload["evaluation"]["metrics"]
        binarization = payload["matrix_binarization"]
        if not binarization.get("applied_before_graph_construction_and_metrics"):
            raise RuntimeError(f"Binarization audit flag is false in {result_path}.")
        if any(metrics.get(key) is None for key in PRIMARY_METRICS):
            raise RuntimeError(f"Incomplete primary metrics in {result_path}.")

        row = {
            "dataset": payload["dataset"],
            "setting": payload["setting"],
            # The copied YAML/result metadata in some historical seed_1 and
            # seed_2 directories still contains the seed-0 config template.
            # The run-directory identity and checkpoint hashes are correct.
            "seed": seed_from_run_dir(run_dir, payload["training_seed"]),
            "motif_used_during_training": is_motif_training_run(run_dir),
            "generation_seed": int(payload["generation_seed"]),
            "sample_count": int(payload["sample_count"]),
            "selected_chain_count": int(
                payload["terminal_graph_definition"]["selected_chain_count"]
            ),
            "total_chain_count": int(
                payload["terminal_graph_definition"]["total_chain_count"]
            ),
            "binarization_threshold": float(binarization["threshold"]),
            "motif_definition_config": payload.get(
                "motif_definition_config", payload["config"]
            ),
            "result_path": str(result_path),
        }
        row.update({key: metrics.get(key) for key in ALL_METRICS})
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["setting"])].append(row)

    results = []
    for (dataset, setting), group in sorted(grouped.items()):
        result: Dict[str, Any] = {
            "dataset": dataset,
            "setting": setting,
            "run_count": len(group),
            "seeds": sorted(row["seed"] for row in group),
            "sample_count_per_seed": sorted(
                set(row["sample_count"] for row in group)
            ),
            "selected_chain_count": sorted(
                set(row["selected_chain_count"] for row in group)
            ),
            "binarization_threshold": sorted(
                set(row["binarization_threshold"] for row in group)
            ),
            "metrics": {},
        }
        for metric in ALL_METRICS:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) is not None
            ]
            result["metrics"][metric] = {
                "mean": mean(values) if values else None,
                "std": pstdev(values) if values else None,
                "values": values,
            }
        results.append(result)
    return results


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "setting",
        "seed",
        "generation_seed",
        "sample_count",
        "selected_chain_count",
        "total_chain_count",
        "binarization_threshold",
        "motif_used_during_training",
        "motif_definition_config",
        *ALL_METRICS,
        "result_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def metric_text(entry: Dict[str, float | None]) -> str:
    if entry["mean"] is None:
        return "N/A"
    return f'{entry["mean"]:.6f} ± {entry["std"]:.6f}'


def write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    aggregates: List[Dict[str, Any]],
    *,
    include_baselines: bool,
) -> None:
    lines = [
        "# Binarized Terminal Motif-Chain Structural MMD",
        "",
        "Every terminal matrix was binarized before graph construction and "
        "metric computation: `value >= 0.5 -> 1`, otherwise `0`. Only square "
        "`N x N` terminal motif-chain matrices were retained; the one-atom raw "
        "adjacency chain was excluded. Retained matrices were combined with a "
        "binary elementwise OR, then self-loops and isolates were removed and "
        "the largest connected component was evaluated exactly through the "
        "repository's `stat_rnn.mmd_eval(..., diam=True)` path.",
        "",
        "Lower is better for the five structural MMD columns and random-GIN "
        "MMD. Higher is better for F1-PR.",
    ]
    if include_baselines:
        lines.extend(
            [
                "",
                "Setting 01 rows are baseline checkpoints evaluated post hoc. "
                "For AIDS, ENZYMES, and OGB, evaluation explicitly reuses the "
                "corresponding setting-03 motif definition so baseline and "
                "motif-trained rows have the same terminal chains. MUTAG and "
                "PTC setting-01 configs already select the same pruned chains "
                "as setting 03.",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregated by setting",
            "",
            "| Dataset | Setting | Seeds | Chains | Degree | Clustering | Orbit | "
            "Spectral | Diameter | Random-GIN MMD | F1-PR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregates:
        metrics = item["metrics"]
        lines.append(
            "| {dataset} | {setting} | {seeds} | {chains} | {degree} | "
            "{clustering} | {orbit} | {spectral} | {diameter} | {mmd_rbf} | "
            "{f1_pr} |".format(
                dataset=item["dataset"],
                setting=item["setting"],
                seeds=", ".join(str(seed) for seed in item["seeds"]),
                chains=", ".join(str(value) for value in item["selected_chain_count"]),
                **{
                    metric: metric_text(metrics[metric])
                    for metric in PRIMARY_METRICS
                },
            )
        )

    lines.extend(
        [
            "",
            "## Individual seeds",
            "",
            "| Dataset | Setting | Seed | Samples | Chains | Degree | Clustering | "
            "Orbit | Spectral | Diameter | Random-GIN MMD | F1-PR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (
        item["dataset"], item["setting"], item["seed"]
    )):
        lines.append(
            "| {dataset} | {setting} | {seed} | {sample_count} | "
            "{selected_chain_count} | {degree:.6f} | {clustering:.6f} | "
            "{orbit:.6f} | {spectral:.6f} | {diameter:.6f} | {mmd_rbf:.6f} | "
            "{f1_pr:.6f} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.gather_root.expanduser().resolve()
    rows = collect_rows(root, args.include_baselines)
    aggregates = aggregate(rows)
    if not rows:
        raise RuntimeError("No complete terminal motif MMD results found.")

    csv_path = root / "terminal_motif_mmd_results.csv"
    json_path = root / "terminal_motif_mmd_aggregated.json"
    markdown_path = root / "TERMINAL_MOTIF_MMD_RESULTS.md"
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_count": len(rows),
                "setting_count": len(aggregates),
                "motif_training_runs_only": not args.include_baselines,
                "aggregation": "population mean and population standard deviation",
                "settings": aggregates,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    write_markdown(
        markdown_path,
        rows,
        aggregates,
        include_baselines=args.include_baselines,
    )
    print(
        f"Wrote {len(rows)} runs / {len(aggregates)} settings:\n"
        f"  {csv_path}\n  {json_path}\n  {markdown_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
