#!/usr/bin/env python3
"""Materialize an exact cached test split in run directories missing its NPY."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terminal_motif_evaluation import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, action="append", type=Path)
    args = parser.parse_args()
    cache_path = args.dataset_cache.expanduser().resolve()
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    adjacencies = []
    for adjacency in cache["test_list_adj"]:
        array = (
            adjacency.toarray()
            if sp.issparse(adjacency)
            else np.asarray(adjacency)
        )
        array = np.asarray(array).copy()
        np.fill_diagonal(array, 0)
        adjacencies.append(array)

    for raw_run_dir in args.run_dir:
        run_dir = raw_run_dir.expanduser().resolve()
        output = run_dir / "testGraphs_adj_.npy"
        if output.exists():
            existing = np.load(output, allow_pickle=True)
            if len(existing) != len(adjacencies) or any(
                not np.array_equal(np.asarray(left), right)
                for left, right in zip(existing, adjacencies)
            ):
                raise ValueError(
                    f"Existing saved test split differs from cache: {output}"
                )
            print(f"[verified] {output}")
            continue
        np.save(output, np.asarray(adjacencies, dtype=object), allow_pickle=True)
        write_json(
            run_dir / "testGraphs_adj_.materialized.json",
            {
                "source_dataset_cache": str(cache_path),
                "test_graph_count": len(adjacencies),
                "reason": (
                    "exact deterministic test split was absent from copied "
                    "run artifacts and was materialized for post-hoc evaluation"
                ),
            },
        )
        print(f"[created] {output} graphs={len(adjacencies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
