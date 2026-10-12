#!/usr/bin/env python3
"""Compute DeFoG full-state LinkCorrelation TV against real training graphs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluate_defog_synthetic_20260906 import (
    canonical_graph_digest,
    load_safe_pyg,
    state_distribution_comparison,
)


ARTIFACT_ROOT = Path(
    "/local-scratch2/mirzaei/defog_synthetic_evaluation_20260906/artifacts"
)
OUTPUT_ROOT = Path(
    "/local-scratch2/mirzaei/defog_synthetic_train_reference_tv_20260909"
)
DATASETS = ("grid", "lobster", "triangular_grid")
OUTLIER_FILTERED_SEEDS = {
    "grid": {0},
    "lobster": {0, 1, 2},
    "triangular_grid": {0, 1},
}


def aggregate(rows: list[dict]) -> dict:
    values = np.asarray([row["total_variation"] for row in rows], dtype=float)
    return {
        "seeds": [row["seed"] for row in rows],
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else None,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": "normalized full-state LinkCorrelation total-variation distance",
        "direction": "lower is better",
        "rule": ["edges(nodes0,nodes1)", "edges(nodes1,nodes2)"],
        "state_order": ["FF", "TT", "FT", "TF"],
        "reference": "real training-graph collection",
        "aggregation": "sum state counts over each collection, then normalize once",
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_root = ARTIFACT_ROOT / dataset
        reference_path = dataset_root / "real_train_graphs.pt"
        reference_graphs, reference_metadata, reference_serialized_digest = load_safe_pyg(
            reference_path
        )
        rows = []
        for generated_path in sorted(
            (dataset_root / "generated").glob("seed_*/generated_graphs.pt")
        ):
            seed = int(generated_path.parent.name.split("seed_", 1)[1])
            generated_graphs, generated_metadata, generated_serialized_digest = load_safe_pyg(
                generated_path
            )
            comparison = state_distribution_comparison(reference_graphs, generated_graphs)
            row = {
                "seed": seed,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "reference_graph_count": len(reference_graphs),
                "generated_graph_count": len(generated_graphs),
                "reference_metadata": reference_metadata,
                "generated_metadata": generated_metadata,
                "reference_serialized_sha256": reference_serialized_digest,
                "generated_serialized_sha256": generated_serialized_digest,
                "reference_canonical_graph_sha256": canonical_graph_digest(reference_graphs),
                "generated_canonical_graph_sha256": canonical_graph_digest(generated_graphs),
                **comparison,
            }
            rows.append(row)
            seed_output = OUTPUT_ROOT / dataset / f"seed_{seed}.json"
            seed_output.parent.mkdir(parents=True, exist_ok=True)
            seed_output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")

        keep = OUTLIER_FILTERED_SEEDS[dataset]
        filtered_rows = [row for row in rows if row["seed"] in keep]
        summary["datasets"][dataset] = {
            "per_seed": [
                {"seed": row["seed"], "total_variation": row["total_variation"]}
                for row in rows
            ],
            "all_seeds": aggregate(rows),
            "outlier_filtered": aggregate(filtered_rows),
            "excluded_seeds": sorted({row["seed"] for row in rows} - keep),
        }

    output = OUTPUT_ROOT / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
