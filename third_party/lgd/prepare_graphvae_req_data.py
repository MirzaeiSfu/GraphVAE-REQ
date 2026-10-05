#!/usr/bin/env python3
"""Convert an exact GraphVAE-REQ cache into a LatentGraphDiffusion bundle.

This is the preferred preparation path for fair comparisons: it consumes the
already frozen GraphVAE-REQ train/validation/test lists instead of downloading
the dataset and independently reshuffling it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


SPLIT_KEYS = {
    "train": ("list_adj", "list_noh_train", "list_eoh_train"),
    "val": ("val_adj", "list_noh_val", "list_eoh_val"),
    "test": ("test_list_adj", "list_noh_test", "list_eoh_test"),
}


def _as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "toarray"):
        return value.toarray()
    return np.asarray(value)


def _feature_groups(info):
    """Return ordered one-hot channel groups and their original values."""

    groups = OrderedDict()
    for channel, metadata in sorted((info or {}).items(), key=lambda pair: int(pair[0])):
        name = str(metadata["feature_name"])
        # The original value can be numeric (current production caches) or a
        # string (older/fixture caches); only channel order is used internally.
        groups.setdefault(name, []).append((int(channel), metadata["value"]))
    return groups


def _decode_node_columns(onehot, groups, num_nodes, topology_only):
    if topology_only or onehot is None or not groups:
        return np.zeros((num_nodes, 1), dtype=np.int64), [1], ["constant"]
    onehot = _as_numpy(onehot)[:num_nodes]
    columns, dimensions, names = [], [], []
    for name, entries in groups.items():
        channels = [channel for channel, _ in entries]
        block = onehot[:, channels]
        if np.any(block.sum(axis=1) <= 0):
            raise ValueError(f"Node feature {name!r} has an unassigned category")
        columns.append(block.argmax(axis=1).astype(np.int64))
        dimensions.append(len(entries))
        names.append(name)
    return np.stack(columns, axis=1), dimensions, names


def _decode_edge_columns(onehot, groups, adjacency, topology_only):
    """Build sparse edge rows with adjacency first and shifted feature IDs."""

    sources, targets = np.nonzero(adjacency)
    keep = sources != targets
    sources, targets = sources[keep], targets[keep]
    columns = [np.ones(len(sources), dtype=np.int64)]
    dimensions, names = [2], ["adjacency"]
    if not topology_only and onehot is not None and groups:
        onehot = _as_numpy(onehot)
        for name, entries in groups.items():
            channels = [channel for channel, _ in entries]
            block = onehot[channels][:, sources, targets].T
            if np.any(block.sum(axis=1) <= 0):
                raise ValueError(
                    f"Present edges have an unassigned {name!r} category"
                )
            # Zero is reserved for absent dense pairs after preprocessing.
            columns.append(block.argmax(axis=1).astype(np.int64) + 1)
            dimensions.append(len(entries) + 1)
            names.append(name)
    edge_index = np.stack((sources, targets), axis=0)
    edge_attr = np.stack(columns, axis=1)
    return edge_index, edge_attr, dimensions, names


def _load_cache(path, graphvae_repo=None):
    if graphvae_repo is not None:
        sys.path.insert(0, str(Path(graphvae_repo).expanduser().resolve()))
    with Path(path).expanduser().open("rb") as stream:
        return pickle.load(stream)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_indexed_raw(raw_dir, prefix):
    """Load the indexed tensor layout used by the frozen DeFoG datasets.

    This second input path is useful when the corresponding GraphVAE run
    intentionally disabled its dataset cache (notably edge-aware MUTAG).
    It still preserves the same frozen split-index files.
    """

    raw_dir = Path(raw_dir).expanduser().resolve()
    adjacency = _torch_load(raw_dir / f"{prefix}_adj.pt")
    node_labels = _torch_load(raw_dir / f"{prefix}_node_labels.pt")
    edge_labels_path = raw_dir / f"{prefix}_edge_labels.pt"
    edge_labels = _torch_load(edge_labels_path) if edge_labels_path.is_file() else None
    node_cardinality = max(int(values.max()) for values in node_labels) + 1
    edge_cardinality = (
        max(int(values.max()) for _, values in edge_labels) + 1
        if edge_labels else 0
    )

    def node_onehot(values):
        return torch.nn.functional.one_hot(
            values.long(), num_classes=node_cardinality
        ).float().numpy()

    def edge_onehot(graph_index):
        if edge_labels is None:
            return None
        num_nodes = int(adjacency[graph_index].shape[0])
        edge_index, values = edge_labels[graph_index]
        result = np.zeros(
            (edge_cardinality, num_nodes, num_nodes), dtype=np.float32
        )
        result[values.numpy(), edge_index[0].numpy(), edge_index[1].numpy()] = 1
        return result

    payload = {
        "node_onehot_info": {
            index: {"feature_name": "node_label", "value": index}
            for index in range(node_cardinality)
        },
        "edge_onehot_info": (
            {
                index: {"feature_name": "edge_label", "value": index}
                for index in range(edge_cardinality)
            } if edge_cardinality else None
        ),
    }
    key_by_split = {
        "train": ("list_adj", "list_noh_train", "list_eoh_train"),
        "val": ("val_adj", "list_noh_val", "list_eoh_val"),
        "test": ("test_list_adj", "list_noh_test", "list_eoh_test"),
    }
    for split, keys in key_by_split.items():
        indices = _torch_load(raw_dir / f"{split}_indices.pt").tolist()
        payload[keys[0]] = [adjacency[index] for index in indices]
        payload[keys[1]] = [node_onehot(node_labels[index]) for index in indices]
        payload[keys[2]] = [edge_onehot(index) for index in indices]
    payload["source_indexed_raw"] = str(raw_dir)
    return payload


def convert(args):
    if args.cache is not None:
        payload = _load_cache(args.cache, args.graphvae_repo)
        source = str(args.cache.expanduser().resolve())
    else:
        payload = _load_indexed_raw(args.indexed_raw_dir, args.raw_prefix)
        source = str(args.indexed_raw_dir.expanduser().resolve())
    node_groups = _feature_groups(payload.get("node_onehot_info"))
    edge_groups = _feature_groups(payload.get("edge_onehot_info"))
    destination = args.output_dir.expanduser().resolve() / args.dataset
    destination.mkdir(parents=True, exist_ok=True)

    expected_schema = None
    split_counts = {}
    for split, (adj_key, node_key, edge_key) in SPLIT_KEYS.items():
        adjacencies = payload.get(adj_key)
        if adjacencies is None:
            raise KeyError(f"Cache has no {adj_key!r} split")
        node_values = payload.get(node_key)
        edge_values = payload.get(edge_key)
        if node_values is None:
            node_values = [None] * len(adjacencies)
        if edge_values is None:
            edge_values = [None] * len(adjacencies)
        if not (len(adjacencies) == len(node_values) == len(edge_values)):
            raise ValueError(f"Unaligned arrays in {split} split")

        graphs = []
        for adjacency_raw, node_raw, edge_raw in zip(
            adjacencies, node_values, edge_values
        ):
            adjacency = _as_numpy(adjacency_raw)
            adjacency = (adjacency > args.adjacency_threshold).astype(np.int64)
            np.fill_diagonal(adjacency, 0)
            if args.force_undirected:
                adjacency = np.maximum(adjacency, adjacency.T)
            num_nodes = int(adjacency.shape[0])
            node_columns, node_dims, node_names = _decode_node_columns(
                node_raw, node_groups, num_nodes, args.topology_only
            )
            edge_index, edge_columns, edge_dims, edge_names = _decode_edge_columns(
                edge_raw, edge_groups, adjacency, args.topology_only
            )
            schema = (node_dims, node_names, edge_dims, edge_names)
            if expected_schema is None:
                expected_schema = schema
            elif schema != expected_schema:
                raise ValueError(f"Feature schema changed within cache: {schema}")
            graphs.append(Data(
                x=torch.as_tensor(node_columns, dtype=torch.long),
                edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                edge_attr=torch.as_tensor(edge_columns, dtype=torch.long),
                y=torch.zeros(1, dtype=torch.float),
                num_nodes=num_nodes,
            ))
        torch.save(graphs, destination / f"{split}.pt")
        split_counts[split] = len(graphs)

    node_dims, node_names, edge_dims, edge_names = expected_schema
    metadata = {
        "format_version": 1,
        "dataset": args.dataset,
        "source": source,
        "exact_frozen_splits": True,
        "topology_only": bool(args.topology_only),
        "force_undirected": bool(args.force_undirected),
        "split_counts": split_counts,
        "node_feature_dims": node_dims,
        "node_feature_names": node_names,
        "edge_feature_dims": edge_dims,
        "edge_feature_names": edge_names,
        "edge_zero_means_absent": True,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path)
    source.add_argument(
        "--indexed-raw-dir", type=Path,
        help="Directory with *_adj.pt, *_node_labels.pt and split indices.",
    )
    parser.add_argument(
        "--raw-prefix", default="mutag",
        help="Filename prefix used with --indexed-raw-dir (default: mutag).",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/graphvae_req"))
    parser.add_argument(
        "--graphvae-repo", type=Path,
        help="GraphVAE-REQ source directory, needed only when pickle classes require it.",
    )
    parser.add_argument("--adjacency-threshold", type=float, default=0.5)
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--directed", dest="force_undirected", action="store_false")
    parser.set_defaults(force_undirected=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
