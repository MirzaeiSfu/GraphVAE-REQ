#!/usr/bin/env python3
"""Compute raw terminal-matrix motif counts on an exact training split.

Each graph is cropped to its true node count before it enters the motif
counter.  For every selected adjacency-shaped terminal chain, the raw count is
the sum of every valid terminal-matrix cell, including the diagonal.  Nothing
is thresholded or binarized.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import build_chain_metadata, write_json
from compute_unpadded_reference_mean_archive import (
    count_group,
    flatten,
    grouped_indices,
    load_yaml,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    started = time.time()
    reference_path = cli.reference_archive.expanduser().resolve()
    reference_metadata_path = reference_path.with_suffix(".json")
    output_path = cli.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    os.chdir(REPO_ROOT)

    with reference_metadata_path.open("r", encoding="utf-8") as handle:
        reference_metadata = json.load(handle)
    with np.load(reference_path, allow_pickle=False) as archive:
        selected_indices = archive["selected_chain_indices"].astype(np.int64)

    config_path = Path(reference_metadata["config"]).resolve()
    cache_path = Path(reference_metadata["dataset_cache"]).resolve()
    args = flatten(load_yaml(config_path))
    args.device = str(cli.device)
    args.motif_cache_dir = str(
        (
            REPO_ROOT
            / (getattr(args, "motif_cache_dir", None) or "cache_motifs")
        ).resolve()
    )
    if not hasattr(args, "motif_prune_max_total_values"):
        args.motif_prune_max_total_values = None
    if not hasattr(args, "motif_prune_max_values_per_rule"):
        args.motif_prune_max_values_per_rule = None

    with cache_path.open("rb") as handle:
        dataset_cache: Dict[str, Any] = pickle.load(handle)
    training_dataset = dataset_cache["list_graphs"]
    original_adjacencies = dataset_cache["list_adj"]
    if len(original_adjacencies) != len(training_dataset.processed_adjs):
        raise ValueError("Original/processed training graph counts differ.")
    node_counts = np.asarray(
        [int(adjacency.shape[0]) for adjacency in original_adjacencies],
        dtype=np.int32,
    )

    counter = RelationalMotifCounter(
        database_name=args.database_name,
        args=args,
    )
    raw_counts = np.empty(
        (len(node_counts), len(selected_indices)), dtype=np.float64
    )
    observed_metadata: List[Dict[str, Any]] | None = None

    for node_count, indices in grouped_indices(node_counts):
        matrices, valid_mask = count_group(
            training_dataset,
            indices,
            node_count,
            counter,
            dataset_cache,
            str(cli.device),
            int(cli.batch_size),
        )
        if observed_metadata is None:
            observed_metadata, observed_indices = build_chain_metadata(
                counter,
                valid_mask,
                matrix_size=node_count,
                exclude_raw_relation=True,
            )
            if not np.array_equal(
                np.asarray(observed_indices, dtype=np.int64), selected_indices
            ):
                raise ValueError(
                    "Training/reference selected chain indices differ."
                )
        index = torch.as_tensor(selected_indices, dtype=torch.long)
        selected = matrices.index_select(1, index)
        selected_mask = valid_mask.index_select(0, index)
        counts = (
            selected.to(torch.float64)
            * selected_mask.unsqueeze(0).to(torch.float64)
        ).sum(dim=(2, 3))
        raw_counts[indices] = counts.numpy()
        print(
            f"[training-raw-count] {args.dataset}: n={node_count} "
            f"graphs={len(indices)}"
        )

    if not np.isfinite(raw_counts).all():
        raise ValueError("Non-finite raw training motif count detected.")
    integer_deviation = float(
        np.max(np.abs(raw_counts - np.rint(raw_counts)))
    )
    if integer_deviation > 1e-5:
        raise ValueError(
            "Discrete training graphs produced non-integer raw counts: "
            f"max deviation={integer_deviation}."
        )

    selected_metadata = reference_metadata["selected_chains"]
    if len(selected_metadata) != len(selected_indices):
        raise ValueError("Reference chain metadata length differs from indices.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_motif_counts=raw_counts,
        selected_chain_indices=selected_indices,
        node_counts=node_counts,
    )
    motif_cache_path = Path(args.motif_cache_dir) / f"{args.database_name}.pkl"
    write_json(
        metadata_path,
        {
            "protocol": "raw_training_count_mmd_v1",
            "archive_role": "training_reference",
            "dataset": args.dataset,
            "database_name": args.database_name,
            "reference_archive": str(reference_path),
            "reference_archive_sha256": sha256(reference_path),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "dataset_cache": str(cache_path),
            "dataset_cache_sha256": sha256(cache_path),
            "motif_cache": str(motif_cache_path),
            "motif_cache_sha256": sha256(motif_cache_path),
            "training_graph_count": int(len(node_counts)),
            "selected_rule_count": int(len(selected_indices)),
            "true_node_count_min": int(node_counts.min()),
            "true_node_count_max": int(node_counts.max()),
            "count_definition": (
                "sum of every valid raw terminal-matrix cell for each rule; "
                "diagonal included; no threshold and no binarization"
            ),
            "padded_nodes_enter_motif_algorithm": False,
            "occurrence_metric_computed": False,
            "max_integer_deviation": integer_deviation,
            "selected_chains": selected_metadata,
            "elapsed_seconds": time.time() - started,
            "output": str(output_path),
        },
    )
    print(
        f"[done] {args.dataset}: graphs={len(node_counts)} "
        f"rules={len(selected_indices)} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
