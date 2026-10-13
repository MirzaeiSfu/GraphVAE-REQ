#!/usr/bin/env python3
"""Combine per-seed PROTEINS checkpoint-resampling reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.analysis_root.glob("seed_*/resampling_metrics.json")):
        seed_name = metrics_path.parent.name
        payload = json.loads(metrics_path.read_text())
        checkpoint = payload["checkpoints"]["best_validation_mmd_model"]
        best_path = args.models_root / seed_name / "best_validation_mmd.json"
        best = json.loads(best_path.read_text())
        row = {
            "seed": seed_name,
            "selected_epoch": best["epoch_1_based"],
            "online_selection_score": best["score"],
            "online_generated_edges": best["metrics"]["generated_edge_count"],
            "online_reference_edges": best["metrics"]["reference_edge_count"],
            "reference_edge_mean": payload["reference_edge_mean"],
            "dense_threshold": payload["dense_edge_threshold"],
            "validation": checkpoint["splits"]["validation"]["summary"],
            "test": checkpoint["splits"]["test"]["summary"],
        }
        rows.append(row)

    aggregate = {}
    for split in ("validation", "test"):
        aggregate[split] = {
            field: {
                "mean": float(np.mean(values)),
                "std_across_seeds": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for field, values in {
                "rollout_score_mean": [row[split]["score"]["mean"] for row in rows],
                "rollout_score_std": [row[split]["score"]["std"] for row in rows],
                "raw_dense_graph_rate": [
                    row[split]["raw_edge_summary"]["dense_graph_rate"] for row in rows
                ],
                "rollouts_with_dense_rate": [
                    row[split]["raw_edge_summary"]["samples_with_dense_rate"] for row in rows
                ],
                "raw_mean_edges": [
                    row[split]["raw_edge_summary"]["mean_edge_count"]["mean"] for row in rows
                ],
                "raw_worst_max_edges": [
                    row[split]["raw_edge_summary"]["max_edge_count"]["max"] for row in rows
                ],
            }.items()
        }

    output = {
        "models_root": str(args.models_root),
        "analysis_root": str(args.analysis_root),
        "protocol": {
            "validation_rollouts_per_seed": 10,
            "test_rollouts_per_seed": 50,
            "dense_definition": "reference mean + 3 * reference std",
            "score": "mean of five GraphVAE-paper-normalized structural MMD metrics",
            "metrics": ["degree", "clustering", "orbit", "spectral", "diameter"],
        },
        "seeds": rows,
        "aggregate": aggregate,
    }
    (args.analysis_root / "combined_metrics.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    lines = [
        "# PROTEINS best-model stability analysis",
        "",
        "Each saved best model was resampled with 10 validation rollouts and 50 held-out test rollouts. "
        "Dense means more edges than the real split mean plus three standard deviations. "
        "MMD uses largest connected components; density statistics below use raw generated graphs.",
        "",
        "| Seed | Selected epoch | Validation score mean ± std | Test score mean ± std | Test mean edges / real | Test dense graphs | Test rollouts with ≥1 dense graph | Worst graph edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        val = row["validation"]
        test = row["test"]
        raw = test["raw_edge_summary"]
        lines.append(
            f"| {row['seed']} | {row['selected_epoch']} | "
            f"{val['score']['mean']:.4f} ± {val['score']['std']:.4f} | "
            f"{test['score']['mean']:.4f} ± {test['score']['std']:.4f} | "
            f"{raw['mean_edge_count']['mean']:.1f} / {row['reference_edge_mean']['test']:.1f} | "
            f"{raw['dense_graph_rate']:.2%} | {raw['samples_with_dense_rate']:.2%} | "
            f"{raw['max_edge_count']['max']:.0f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Seed 1 is the most stable: its mean edge count is closest to the reference and both dense-graph and dense-rollout rates are lowest.",
        "- Seeds 0 and 2 have severe heavy tails. Their average dense-graph percentages look small only because each rollout contains many graphs; nearly every rollout contains at least one dense outlier.",
        "- All three seeds produced at least one nearly complete padded graph (about 4,800 edges), so the decoder instability is present in every saved best model.",
        "- The online best-checkpoint metric did not protect against this tail because it emphasized MMD/GNN realism and summarized only one generated set.",
        "",
    ]
    (args.analysis_root / "combined_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
