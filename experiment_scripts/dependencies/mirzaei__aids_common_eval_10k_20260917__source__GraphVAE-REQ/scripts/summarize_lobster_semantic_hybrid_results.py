#!/usr/bin/env python3
"""Combine structural, Random-GIN, and pretrained-GIN LOBSTER results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


STRUCTURAL_METRICS = (
    "degree",
    "clustering",
    "orbit",
    "spectral",
    "diameter",
)
RANDOM_GIN_METRICS = ("f1_pr", "mmd_rbf")
PRETRAINED_METRICS = (
    "fid",
    "f1_pr",
    "mmd_rbf",
    "precision",
    "recall",
    "density",
    "coverage",
    "f1_dc",
    "mmd_linear",
)
CONDITIONS = (
    (
        "lobster_graphvae_mm_fixed_split_matched1_legacy",
        "Manual Kiarash control",
    ),
    (
        "lobster_kiarash_parity_kia40_2000_legacy",
        "Motif Kiarash bundle only",
    ),
    (
        "lobster_semantic_hybrid_r001_legacy",
        "Bundle + relational 0.01",
    ),
    (
        "lobster_semantic_hybrid_r001_edgecount01_legacy",
        "Bundle + relational 0.01 + edge count 0.1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-json",
        type=Path,
        action="append",
        required=True,
        help="Held-out rollout JSON; repeat for control and hybrid campaigns.",
    )
    parser.add_argument(
        "--random-gin-csv",
        type=Path,
        action="append",
        required=True,
        help="Per-run Random-GIN CSV; repeat for control and hybrid campaigns.",
    )
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def load_structural(paths: list[Path]) -> dict:
    available = {}
    reference_hashes = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        reference_hashes.add(payload["heldout_reference_sha256"])
        for condition, values in payload["condition_summary"].items():
            available.setdefault(condition, values)
    if len(reference_hashes) != 1:
        raise ValueError(
            "Structural campaigns use different held-out references: "
            f"{sorted(reference_hashes)}"
        )
    return {
        "reference_sha256": next(iter(reference_hashes)),
        "conditions": available,
    }


def condition_and_seed(run_dir: str) -> tuple[str, int]:
    path = Path(run_dir)
    condition = path.parent.name.split("__", maxsplit=1)[0]
    if not path.name.startswith("seed_"):
        raise ValueError(f"Cannot parse training seed from {path}.")
    return condition, int(path.name[len("seed_"):])


def load_random_gin(paths: list[Path]) -> dict:
    values = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition, seed = condition_and_seed(row["run_dir"])
                for metric in RANDOM_GIN_METRICS:
                    values[condition][metric][seed] = float(
                        row[f"{metric}_mean"]
                    )
    return values


def load_pretrained(root: Path) -> tuple[dict, dict]:
    seed_means = defaultdict(lambda: defaultdict(dict))
    pair_values = defaultdict(lambda: defaultdict(list))
    provenance = {
        "encoders": set(),
        "feature_modes": set(),
        "checkpoint_seeds": set(),
        "reference_sha256": set(),
    }
    for path in sorted(root.glob("*__seed_*/evaluation.json")):
        condition, seed_text = path.parent.name.rsplit("__seed_", maxsplit=1)
        seed = int(seed_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance["encoders"].add(payload["encoder"])
        provenance["feature_modes"].add(payload["feature_mode"])
        for metric in PRETRAINED_METRICS:
            seed_means[condition][metric][seed] = float(
                payload["summary"][metric]["mean"]
            )
        for checkpoint_result in payload["per_checkpoint"]:
            provenance["checkpoint_seeds"].add(
                int(checkpoint_result["checkpoint_seed"])
            )
            provenance["reference_sha256"].add(
                checkpoint_result["reference_sha256"]
            )
            for metric in PRETRAINED_METRICS:
                pair_values[condition][metric].append(
                    float(checkpoint_result["metrics"][metric])
                )
    return seed_means, {
        "pair_values": pair_values,
        "provenance": {
            key: sorted(values) for key, values in provenance.items()
        },
    }


def require_three_seeds(values: dict, *, source: str, condition: str) -> None:
    for metric, by_seed in values.items():
        seeds = sorted(by_seed)
        if seeds != [0, 1, 2]:
            raise ValueError(
                f"{source} {condition} {metric} has seeds {seeds}, "
                "expected [0, 1, 2]."
            )


def aggregate(args: argparse.Namespace) -> dict:
    structural = load_structural(args.structural_json)
    random_gin = load_random_gin(args.random_gin_csv)
    pretrained, pretrained_details = load_pretrained(args.pretrained_dir)
    conditions = {}
    for condition, label in CONDITIONS:
        if condition not in structural["conditions"]:
            raise KeyError(f"Missing structural condition {condition}.")
        if condition not in random_gin:
            raise KeyError(f"Missing Random-GIN condition {condition}.")
        if condition not in pretrained:
            raise KeyError(f"Missing pretrained-GIN condition {condition}.")
        require_three_seeds(
            random_gin[condition],
            source="Random-GIN",
            condition=condition,
        )
        require_three_seeds(
            pretrained[condition],
            source="pretrained-GIN",
            condition=condition,
        )
        structural_values = structural["conditions"][condition]
        conditions[condition] = {
            "label": label,
            "structural": {
                metric: structural_values["metrics_across_seed_means"][metric]
                for metric in STRUCTURAL_METRICS
            },
            "lcc_nodes": structural_values[
                "lcc_nodes_across_seed_means"
            ],
            "raw_nodes": structural_values[
                "raw_nodes_across_seed_means"
            ],
            "random_gin_across_training_seed_means": {
                metric: summary(list(sorted(by_seed.values())))
                for metric, by_seed in random_gin[condition].items()
            },
            "pretrained_graphcl_gin_across_training_seed_means": {
                metric: summary(list(sorted(by_seed.values())))
                for metric, by_seed in pretrained[condition].items()
            },
            "pretrained_graphcl_gin_across_9_seed_encoder_pairs": {
                metric: summary(values)
                for metric, values in pretrained_details["pair_values"][
                    condition
                ].items()
            },
        }
    return {
        "conditions": conditions,
        "condition_order": [condition for condition, _ in CONDITIONS],
        "structural_reference_sha256": structural["reference_sha256"],
        "pretrained_graphcl_gin": {
            "training_graphs": 70,
            "training_seeds": [0, 1, 2],
            "epochs": 100,
            "architecture": "3-layer GIN, hidden dimension 32",
            "feature_mode": "topology_control",
            "aggregation": (
                "Each training-seed mean averages three independently "
                "pretrained GraphCL-GIN encoders; reported standard "
                "deviations are across the three generator training seeds."
            ),
            **pretrained_details["provenance"],
        },
    }


def format_value(item: dict) -> str:
    return f"{item['mean']:.5f} ± {item['std']:.5f}"


def write_csv(path: Path, payload: dict) -> None:
    fields = ["condition", "label"]
    sections = (
        ("structural", STRUCTURAL_METRICS),
        ("", ("lcc_nodes", "raw_nodes")),
        ("random_gin_across_training_seed_means", RANDOM_GIN_METRICS),
        (
            "pretrained_graphcl_gin_across_training_seed_means",
            PRETRAINED_METRICS,
        ),
    )
    for section, metrics in sections:
        prefix = f"{section}_" if section else ""
        for metric in metrics:
            fields.extend((f"{prefix}{metric}_mean", f"{prefix}{metric}_std"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in payload["condition_order"]:
            values = payload["conditions"][condition]
            row = {"condition": condition, "label": values["label"]}
            for section, metrics in sections:
                source = values if not section else values[section]
                prefix = f"{section}_" if section else ""
                for metric in metrics:
                    row[f"{prefix}{metric}_mean"] = source[metric]["mean"]
                    row[f"{prefix}{metric}_std"] = source[metric]["std"]
            writer.writerow(row)


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Semantically grouped hybrid: frozen held-out results",
        "",
        (
            "Checkpoint selection used validation graphs only. Structural "
            "statistics use ten held-out prior rollouts per generator seed."
        ),
        "",
        "## Structural statistics",
        "",
        (
            "| Method | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | "
            "Diameter ↓ | LCC nodes | Raw nodes |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["condition_order"]:
        item = payload["conditions"][condition]
        cells = [
            format_value(item["structural"][metric])
            for metric in STRUCTURAL_METRICS
        ]
        cells += [
            format_value(item["lcc_nodes"]),
            format_value(item["raw_nodes"]),
        ]
        lines.append(f"| {item['label']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Random-GIN",
        "",
        (
            "Each generator-seed mean uses ten independently initialized "
            "Random-GIN encoders."
        ),
        "",
        "| Method | F1-PR ↑ | RBF MMD ↓ |",
        "|---|---:|---:|",
    ]
    for condition in payload["condition_order"]:
        item = payload["conditions"][condition]
        metrics = item["random_gin_across_training_seed_means"]
        lines.append(
            f"| {item['label']} | {format_value(metrics['f1_pr'])} | "
            f"{format_value(metrics['mmd_rbf'])} |"
        )

    lines += [
        "",
        "## LOBSTER-pretrained GraphCL-GIN",
        "",
        (
            "Three independent topology-control GraphCL-GIN encoders were "
            "trained for 100 epochs on the exact 70-graph real training "
            "split. Each generator-seed value averages the three encoders; "
            "the displayed standard deviation is across generator seeds."
        ),
        "",
        (
            "| Method | FID ↓ | F1-PR ↑ | RBF MMD ↓ | Density ↑ | "
            "Coverage ↑ | F1-DC ↑ |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["condition_order"]:
        item = payload["conditions"][condition]
        metrics = item[
            "pretrained_graphcl_gin_across_training_seed_means"
        ]
        fields = ("fid", "f1_pr", "mmd_rbf", "density", "coverage", "f1_dc")
        lines.append(
            f"| {item['label']} | "
            + " | ".join(format_value(metrics[metric]) for metric in fields)
            + " |"
        )

    lines += [
        "",
        "## Decision",
        "",
        (
            "The relational 0.01 hybrid does not beat the matched motif-bundle "
            "control: its mean degree, clustering, orbit, spectral, diameter, "
            "and graph-size errors all move in the wrong direction. The "
            "additional edge-count loss degrades them further and has the "
            "worst pretrained-GIN FID/coverage. Do not expand this hybrid "
            "configuration."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = aggregate(args)
    (output_dir / "combined_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "combined_results.csv", payload)
    write_markdown(output_dir / "analysis_full.md", payload)


if __name__ == "__main__":
    main()
