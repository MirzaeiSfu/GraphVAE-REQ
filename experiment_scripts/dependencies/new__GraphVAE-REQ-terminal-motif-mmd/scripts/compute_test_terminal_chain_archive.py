#!/usr/bin/env python3
"""Compute binarized terminal matrices for an exact attributed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataWrapper, merge_datasets
from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def verify_saved_test_split(
    saved_path: Path, cached_adjacencies: list
) -> None:
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
    config = load_yaml(config_path)
    args = flatten(config)
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
    counter = RelationalMotifCounter(
        database_name=args.database_name,
        args=args,
    )
    wrapper = DataWrapper(
        merge_datasets(test_dataset),
        counter.relation_keys,
        node_onehot_info=dataset_cache["node_onehot_info"],
        edge_onehot_info=dataset_cache.get("edge_onehot_info"),
        edge_feature_info_mapping=counter.feature_info_mapping,
        device=cli.device,
    )
    matrices, valid_mask = counter.count_batch(
        wrapper,
        batch_size=cli.batch_size,
        output_mode="matrix",
        detach_to_cpu=True,
    )

    generated_archive_path = (
        run_dir
        / "terminal_motif_mmd"
        / "terminal_chain_binary_matrices.npz"
    )
    with np.load(generated_archive_path, allow_pickle=False) as generated:
        selected_indices = generated["selected_chain_indices"].astype(np.int64)
    selected = matrices.index_select(
        1, torch.as_tensor(selected_indices, dtype=torch.long)
    )
    selected_mask = valid_mask.index_select(
        0, torch.as_tensor(selected_indices, dtype=torch.long)
    )
    binary = (
        (selected >= float(cli.threshold))
        & selected_mask.unsqueeze(0)
    ).to(torch.uint8).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        binary_terminal_matrices=binary,
        selected_chain_indices=selected_indices,
    )
    write_json(
        metadata_path,
        {
            "dataset": args.dataset,
            "database_name": args.database_name,
            "representative_run_dir": str(run_dir),
            "config": str(config_path),
            "dataset_cache": str(cache_path),
            "dataset_cache_sha256": sha256(cache_path),
            "saved_test_adjacencies_verified_exact": True,
            "test_graph_count": int(binary.shape[0]),
            "selected_chain_count": int(binary.shape[1]),
            "matrix_size": int(binary.shape[2]),
            "binarization_threshold": float(cli.threshold),
            "combined_across_chains": False,
            "output": str(output_path),
        },
    )
    print(
        f"[done] {args.dataset}: shape={binary.shape} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
