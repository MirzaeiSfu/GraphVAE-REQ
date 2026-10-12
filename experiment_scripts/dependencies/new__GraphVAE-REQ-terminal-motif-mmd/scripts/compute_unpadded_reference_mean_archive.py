#!/usr/bin/env python3
"""Build reference terminal-chain archives without ever counting padded nodes.

For every adjacency-shaped terminal rule, the threshold is the mean of all
valid terminal-matrix entries across the complete attributed test split.  The
same per-rule threshold is later applied to generated terminal matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
import scipy.sparse as sp
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataWrapper
from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import build_chain_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return loaded


def flatten(config: Dict[str, Any]) -> SimpleNamespace:
    values: Dict[str, Any] = {}
    for section in config.values():
        if isinstance(section, dict):
            values.update(section)
    return SimpleNamespace(**values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense(adjacency: Any) -> np.ndarray:
    return (
        adjacency.toarray()
        if sp.issparse(adjacency)
        else np.asarray(adjacency)
    )


def verify_saved_test_split(saved_path: Path, cached_adjacencies: list) -> None:
    saved = np.load(saved_path, allow_pickle=True)
    if len(saved) != len(cached_adjacencies):
        raise ValueError(
            f"Saved/cached test lengths differ: {len(saved)} vs "
            f"{len(cached_adjacencies)}."
        )
    for index, (saved_adj, cached_adj) in enumerate(
        zip(saved, cached_adjacencies)
    ):
        left = np.asarray(saved_adj).copy()
        right = dense(cached_adj).copy()
        if left.shape != right.shape:
            raise ValueError(
                f"Test graph {index} shapes differ: {left.shape} vs "
                f"{right.shape}."
            )
        np.fill_diagonal(left, 0)
        np.fill_diagonal(right, 0)
        if not np.array_equal(left, right):
            raise ValueError(
                f"Cached test graph {index} does not match saved test graph."
            )


def slice_square(value: Any, node_count: int) -> Any:
    if value is None:
        return None
    return value[:node_count, :node_count]


def slice_rows(value: Any, node_count: int) -> Any:
    if value is None:
        return None
    return value[:node_count]


def slice_edge(value: Any, node_count: int) -> Any:
    if value is None:
        return None
    return value[:, :node_count, :node_count]


def grouped_indices(node_counts: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    # Start with the largest graph so N x N and scalar 1 x 1 terminal shapes
    # cannot be confused when metadata is initialized.
    for node_count in sorted(np.unique(node_counts).tolist(), reverse=True):
        yield int(node_count), np.flatnonzero(node_counts == node_count)


def group_merged_dataset(
    dataset: Any,
    indices: np.ndarray,
    node_count: int,
) -> Dict[str, Any]:
    return {
        "processed_adjs": [
            slice_square(dataset.processed_adjs[int(index)], node_count)
            for index in indices
        ],
        "processed_Xs": [
            slice_rows(dataset.processed_Xs[int(index)], node_count)
            for index in indices
        ],
        "processed_node_onehot": [
            slice_rows(dataset.processed_node_onehot[int(index)], node_count)
            for index in indices
        ],
        "processed_edge_onehot": [
            slice_edge(dataset.processed_edge_onehot[int(index)], node_count)
            for index in indices
        ],
        "max_num_nodes": node_count,
    }


def build_wrapper(
    dataset: Any,
    indices: np.ndarray,
    node_count: int,
    counter: RelationalMotifCounter,
    dataset_cache: Dict[str, Any],
    device: str,
) -> DataWrapper:
    return DataWrapper(
        group_merged_dataset(dataset, indices, node_count),
        counter.relation_keys,
        node_onehot_info=dataset_cache["node_onehot_info"],
        edge_onehot_info=dataset_cache.get("edge_onehot_info"),
        edge_feature_info_mapping=counter.feature_info_mapping,
        device=device,
    )


def count_group(
    dataset: Any,
    indices: np.ndarray,
    node_count: int,
    counter: RelationalMotifCounter,
    dataset_cache: Dict[str, Any],
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    wrapper = build_wrapper(
        dataset, indices, node_count, counter, dataset_cache, device
    )
    return counter.count_batch(
        wrapper,
        batch_size=batch_size,
        output_mode="matrix",
        detach_to_cpu=True,
    )


def main() -> int:
    cli = parse_args()
    run_dir = cli.run_dir.expanduser().resolve()
    cache_path = cli.dataset_cache.expanduser().resolve()
    output_path = cli.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    os.chdir(REPO_ROOT)

    config_path = run_dir / "run_config_used.yaml"
    args = flatten(load_yaml(config_path))
    args.device = cli.device
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
        dataset_cache = pickle.load(handle)
    verify_saved_test_split(
        run_dir / "testGraphs_adj_.npy",
        dataset_cache["test_list_adj"],
    )
    test_dataset = dataset_cache["list_test_graphs"]
    node_counts = np.asarray(
        [int(adjacency.shape[0]) for adjacency in dataset_cache["test_list_adj"]],
        dtype=np.int32,
    )
    max_nodes = int(node_counts.max())
    counter = RelationalMotifCounter(
        database_name=args.database_name,
        args=args,
    )

    selected_indices: List[int] | None = None
    chain_metadata: List[Dict[str, Any]] | None = None
    sums: np.ndarray | None = None
    denominators: np.ndarray | None = None

    # Pass 1: calculate one reference-only mean threshold for every rule.
    for node_count, indices in grouped_indices(node_counts):
        matrices, valid_mask = count_group(
            test_dataset,
            indices,
            node_count,
            counter,
            dataset_cache,
            cli.device,
            cli.batch_size,
        )
        if selected_indices is None:
            chain_metadata, selected_indices = build_chain_metadata(
                counter,
                valid_mask,
                matrix_size=node_count,
                exclude_raw_relation=True,
            )
            sums = np.zeros(len(selected_indices), dtype=np.float64)
            denominators = np.zeros(len(selected_indices), dtype=np.int64)

        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        selected = matrices.index_select(1, index_tensor)
        selected_mask = valid_mask.index_select(0, index_tensor)
        masked = selected * selected_mask.unsqueeze(0)
        sums += masked.sum(dim=(0, 2, 3), dtype=torch.float64).numpy()
        denominators += (
            selected_mask.sum(dim=(1, 2)).numpy().astype(np.int64)
            * int(len(indices))
        )
        print(
            f"[reference-threshold] n={node_count} graphs={len(indices)}"
        )

    assert selected_indices is not None
    assert chain_metadata is not None
    assert sums is not None and denominators is not None
    if np.any(denominators == 0):
        raise RuntimeError("At least one selected terminal rule has no valid cells.")
    thresholds = sums / denominators

    # Pass 2: apply each rule's reference-derived threshold. A strict comparison
    # keeps an all-zero rule all-zero when its mean threshold is exactly zero.
    binary = np.zeros(
        (len(node_counts), len(selected_indices), max_nodes, max_nodes),
        dtype=np.uint8,
    )
    for node_count, indices in grouped_indices(node_counts):
        matrices, valid_mask = count_group(
            test_dataset,
            indices,
            node_count,
            counter,
            dataset_cache,
            cli.device,
            cli.batch_size,
        )
        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        selected = matrices.index_select(1, index_tensor)
        selected_mask = valid_mask.index_select(0, index_tensor)
        threshold_tensor = torch.as_tensor(
            thresholds, dtype=selected.dtype
        ).view(1, -1, 1, 1)
        current = (
            (selected > threshold_tensor) & selected_mask.unsqueeze(0)
        ).to(torch.uint8).numpy()
        binary[indices, :, :node_count, :node_count] = current
        print(f"[reference-binary] n={node_count} graphs={len(indices)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        binary_terminal_matrices=binary,
        selected_chain_indices=np.asarray(selected_indices, dtype=np.int64),
        node_counts=node_counts,
        rule_thresholds=thresholds.astype(np.float64),
    )
    selected_metadata = [
        chain_metadata[index] for index in selected_indices
    ]
    write_json(
        metadata_path,
        {
            "protocol": "reference_mean_per_rule_unpadded_v1",
            "dataset": args.dataset,
            "database_name": args.database_name,
            "representative_run_dir": str(run_dir),
            "config": str(config_path),
            "dataset_cache": str(cache_path),
            "dataset_cache_sha256": sha256(cache_path),
            "saved_test_adjacencies_verified_exact": True,
            "test_graph_count": int(len(node_counts)),
            "true_node_count_min": int(node_counts.min()),
            "true_node_count_max": int(node_counts.max()),
            "selected_chain_count": int(len(selected_indices)),
            "threshold_definition": (
                "For each rule, arithmetic mean of every valid terminal-matrix "
                "entry over all reference test graphs; padded rows/columns never "
                "enter motif counting or the mean."
            ),
            "threshold_comparison": "terminal_value > reference_rule_mean",
            "diagonal_included_in_threshold": True,
            "padded_nodes_counted_by_motif_algorithm": False,
            "occurrence_metric_computed": False,
            "selected_chains": selected_metadata,
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
