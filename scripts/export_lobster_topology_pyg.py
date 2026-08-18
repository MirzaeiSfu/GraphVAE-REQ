#!/usr/bin/env python3
"""Export exact LOBSTER splits or adjacency NPY files for PyG GNN evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import BFS, data_split_three_way, list_graph_loader  # noqa: E402


FEATURE_SCHEMA = "lobster-topology-control-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ggm-eval-src",
        type=Path,
        default=ROOT.parent
        / "GraphVAE-REQ-main-evaluation"
        / "graph_evaluation"
        / "src",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    real = subparsers.add_parser(
        "real-splits",
        help="Reconstruct the exact paper 70/10/20 LOBSTER topology splits.",
    )
    real.add_argument("--output-dir", type=Path, required=True)
    real.add_argument("--validation-reference", type=Path)
    real.add_argument("--test-reference", type=Path)

    arrays = subparsers.add_parser(
        "npy",
        help="Convert one ragged adjacency NPY collection.",
    )
    arrays.add_argument("--input", type=Path, required=True)
    arrays.add_argument("--output", type=Path, required=True)
    arrays.add_argument("--split", required=True)
    arrays.add_argument("--generator")
    return parser.parse_args()


def adjacency_to_data(raw_adjacency, *, name: str) -> Data:
    adjacency = np.asarray(
        (
            raw_adjacency.toarray()
            if hasattr(raw_adjacency, "toarray")
            else raw_adjacency
        ),
        dtype=np.float32,
    )
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{name} must be square, got {adjacency.shape}.")
    if adjacency.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two nodes.")
    if not np.isfinite(adjacency).all():
        raise ValueError(f"{name} contains non-finite entries.")
    binary = adjacency > 0.5
    if np.any(np.diag(binary)):
        raise ValueError(f"{name} contains a self-loop.")
    if not np.array_equal(binary, binary.T):
        raise ValueError(f"{name} is not symmetric.")
    source, target = np.nonzero(binary)
    if source.size == 0:
        raise ValueError(f"{name} contains no edge.")
    return Data(
        x=torch.ones((adjacency.shape[0], 1), dtype=torch.float32),
        edge_index=torch.as_tensor(
            np.stack((source, target)),
            dtype=torch.int64,
        ),
        num_nodes=int(adjacency.shape[0]),
    )


def as_data_list(adjacencies, *, name: str) -> list[Data]:
    return [
        adjacency_to_data(adjacency, name=f"{name}[{index}]")
        for index, adjacency in enumerate(adjacencies)
    ]


def assert_matches_reference(adjacencies, path: Path, *, split: str) -> None:
    reference = np.load(path.expanduser().resolve(), allow_pickle=True)
    if len(adjacencies) != len(reference):
        raise RuntimeError(
            f"{split} count mismatch: reconstructed={len(adjacencies)}, "
            f"reference={len(reference)}."
        )
    for index, (actual, expected) in enumerate(zip(adjacencies, reference)):
        actual_array = (
            actual.toarray() if hasattr(actual, "toarray") else np.asarray(actual)
        )
        expected_array = np.asarray(expected)
        if not np.array_equal(actual_array, expected_array):
            raise RuntimeError(
                f"{split}[{index}] does not match the frozen run reference."
            )


def save_collection(path: Path, graphs: list[Data], metadata: dict) -> dict:
    from ggm_eval import save_pyg_collection

    return save_pyg_collection(
        path.expanduser().resolve(),
        graphs,
        metadata={
            "dataset": "LOBSTER",
            "feature_schema": FEATURE_SCHEMA,
            "feature_mode": "topology_control",
            **metadata,
        },
    )


def export_real_splits(args: argparse.Namespace) -> None:
    (
        adjacencies,
        _,
        _,
        node_features,
        edge_features,
        _,
        _,
    ) = list_graph_loader(
        "LOBSTER",
        return_labels=True,
        lobster_feature_schema="old_v1",
        shuffle_seed=0,
    )
    adjacencies, _, _ = BFS(
        adjacencies,
        node_features,
        edge_features,
    )
    train, validation, test, *_ = data_split_three_way(
        adjacencies,
        train_fraction=0.7,
        val_fraction=0.1,
        seed=123,
    )
    if args.validation_reference is not None:
        assert_matches_reference(
            validation,
            args.validation_reference,
            split="validation",
        )
    if args.test_reference is not None:
        assert_matches_reference(test, args.test_reference, split="test")

    output_dir = args.output_dir.expanduser().resolve()
    common = {
        "source": "GraphVAE-REQ deterministic LOBSTER loader",
        "lobster_feature_schema": "old_v1",
        "dataset_loader_seed": 0,
        "split_mode": "paper_70_10_20",
        "split_seed": 123,
        "bfs_strategy": "legacy_first_component",
    }
    for split, values in (
        ("train", train),
        ("validation", validation),
        ("heldout_test", test),
    ):
        manifest = save_collection(
            output_dir / f"real_{split}_graphs.pt",
            as_data_list(values, name=split),
            {**common, "split": split},
        )
        print(
            f"{split}: graphs={manifest['summary']['graph_count']} "
            f"sha256={manifest['collection_sha256']}"
        )


def export_npy(args: argparse.Namespace) -> None:
    source = args.input.expanduser().resolve()
    values = np.load(source, allow_pickle=True)
    metadata = {
        "source": str(source),
        "split": args.split,
    }
    if args.generator:
        metadata["generator"] = args.generator
    manifest = save_collection(
        args.output,
        as_data_list(values, name=source.name),
        metadata,
    )
    print(
        f"graphs={manifest['summary']['graph_count']} "
        f"sha256={manifest['collection_sha256']}"
    )


def main() -> None:
    args = parse_args()
    ggm_eval_src = args.ggm_eval_src.expanduser().resolve()
    if not (ggm_eval_src / "ggm_eval").is_dir():
        raise FileNotFoundError(
            f"ggm_eval source package not found under {ggm_eval_src}."
        )
    sys.path.insert(0, str(ggm_eval_src))
    if args.command == "real-splits":
        export_real_splits(args)
    else:
        export_npy(args)


if __name__ == "__main__":
    main()
