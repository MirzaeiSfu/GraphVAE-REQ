#!/usr/bin/env python3
"""Measure terminal motif-chain occurrence support on the training split.

The resulting support set is independent of generated graphs and test metric
values.  It can therefore be used to predefine stable chains for test-time
comparisons without selecting chains on test performance.
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
from typing import Any, Dict

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataWrapper
from motif_counting.motif_counter import RelationalMotifCounter
from terminal_motif_evaluation import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--generated-archive", required=True, type=Path)
    parser.add_argument("--chain-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
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


def canonical_chain_payload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule": metadata["rule"],
        "states": metadata["states"],
        "value": metadata["value"],
        "rule_source": metadata.get("rule_source"),
    }


def canonical_chain_key(metadata: Dict[str, Any]) -> str:
    serialized = json.dumps(
        canonical_chain_payload(metadata),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def dataset_slice(dataset: Any, start: int, end: int) -> Dict[str, Any]:
    """Return the small merged-dataset mapping consumed by DataWrapper."""

    count = end - start

    def values(attribute: str) -> list:
        loaded = getattr(dataset, attribute, None)
        if loaded:
            return list(loaded[start:end])
        return [None] * count

    return {
        "processed_adjs": list(dataset.processed_adjs[start:end]),
        "processed_Xs": list(dataset.processed_Xs[start:end]),
        "processed_node_onehot": values("processed_node_onehot"),
        "processed_edge_onehot": values("processed_edge_onehot"),
        "max_num_nodes": int(dataset.max_num_nodes),
    }


def main() -> int:
    cli = parse_args()
    run_dir = cli.run_dir.expanduser().resolve()
    cache_path = cli.dataset_cache.expanduser().resolve()
    generated_archive_path = cli.generated_archive.expanduser().resolve()
    chain_metadata_path = cli.chain_metadata.expanduser().resolve()
    output_path = cli.output.expanduser().resolve()
    if output_path.exists() and not cli.overwrite:
        print(f"[skip] {output_path}")
        return 0
    os.chdir(REPO_ROOT)

    config_path = run_dir / "run_config_used.yaml"
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
        dataset_cache = pickle.load(handle)
    training_dataset = dataset_cache["list_graphs"]
    training_count = len(training_dataset.processed_adjs)

    with np.load(generated_archive_path, allow_pickle=False) as archive:
        selected_indices = archive["selected_chain_indices"].astype(np.int64)
    with chain_metadata_path.open("r", encoding="utf-8") as handle:
        all_metadata = json.load(handle)
    metadata_by_index = {
        int(entry["chain_index"]): entry for entry in all_metadata
    }
    selected_metadata = [
        metadata_by_index[int(index)] for index in selected_indices.tolist()
    ]

    counter = RelationalMotifCounter(
        database_name=args.database_name,
        args=args,
    )
    support_counts = np.zeros(len(selected_indices), dtype=np.int64)
    binary_count_sums = np.zeros(len(selected_indices), dtype=np.int64)
    diagonal_included = True

    for start in range(0, training_count, int(cli.batch_size)):
        end = min(start + int(cli.batch_size), training_count)
        wrapper = DataWrapper(
            dataset_slice(training_dataset, start, end),
            counter.relation_keys,
            node_onehot_info=dataset_cache["node_onehot_info"],
            edge_onehot_info=dataset_cache.get("edge_onehot_info"),
            edge_feature_info_mapping=counter.feature_info_mapping,
            device=cli.device,
        )
        matrices, valid_mask = counter.count_batch(
            wrapper,
            batch_size=end - start,
            output_mode="matrix",
            detach_to_cpu=True,
        )
        index = torch.as_tensor(selected_indices, dtype=torch.long)
        selected = matrices.index_select(1, index)
        selected_mask = valid_mask.index_select(0, index)
        binary = (selected >= float(cli.threshold)) & selected_mask.unsqueeze(0)
        # Count mode sums every valid spatial element of the terminal result:
        #
        #   result.reshape(B, -1).sum(dim=1)
        #
        # Therefore count-aligned support must include diagonal cells.  The
        # diagonal is removed only later when a separate adjacency projection
        # is constructed for graph-structural metrics.
        binary_counts = binary.flatten(start_dim=2).sum(dim=2)
        present = binary_counts > 0
        support_counts += present.sum(dim=0).numpy().astype(np.int64)
        binary_count_sums += (
            binary_counts.sum(dim=0).numpy().astype(np.int64)
        )
        print(
            f"[training-support] {args.dataset}: {end}/{training_count} graphs"
        )

    chains = []
    for position, (chain_index, metadata, count) in enumerate(
        zip(selected_indices.tolist(), selected_metadata, support_counts.tolist())
    ):
        chains.append(
            {
                "archive_position": position,
                "chain_index": int(chain_index),
                "canonical_chain_key": canonical_chain_key(metadata),
                "canonical_chain": canonical_chain_payload(metadata),
                "rule_index": metadata["rule_index"],
                "value_index": metadata["value_index"],
                "training_graph_count": training_count,
                "training_nonempty_count": int(count),
                "training_nonempty_rate": float(count / training_count),
                "training_binary_motif_count_sum": int(
                    binary_count_sums[position]
                ),
                "training_binary_motif_count_mean": float(
                    binary_count_sums[position] / training_count
                ),
            }
        )

    motif_cache_path = (
        Path(args.motif_cache_dir) / f"{args.database_name}.pkl"
    )
    write_json(
        output_path,
        {
            "metric_family": "terminal_motif_chain_training_support",
            "dataset": args.dataset,
            "database_name": args.database_name,
            "run_dir": str(run_dir),
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "dataset_cache": str(cache_path),
            "dataset_cache_sha256": sha256(cache_path),
            "motif_cache": str(motif_cache_path),
            "motif_cache_sha256": sha256(motif_cache_path),
            "generated_archive_used_for_chain_indices": str(
                generated_archive_path
            ),
            "chain_metadata": str(chain_metadata_path),
            "training_graph_count": training_count,
            "selected_chain_count": len(chains),
            "binarization_threshold": float(cli.threshold),
            "protocol_version": "count_aligned_v2",
            "diagonal_included_in_count_and_occurrence": diagonal_included,
            "occurrence_definition": (
                "sum of all valid binarized terminal-matrix cells, including "
                "the diagonal, is greater than zero; this matches count mode's "
                "all-spatial-cell reduction after the requested binarization"
            ),
            "chains": chains,
        },
    )
    print(f"[done] {args.dataset}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
