#!/usr/bin/env python3
"""Build the deterministic Gate 3 GraphVAE cache qualification fixture."""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from resample_grid_checkpoints import build_dataset_cache_metadata  # noqa: E402
from tune_graphvae_attribute_weights import flatten_config, load_yaml_mapping  # noqa: E402


OUTPUT = Path(__file__).with_name("qm9_tiny_cache.pkl")
CONFIG = REPO_ROOT / "configs/bayesian_optimization/qm9_graphvae_attr_f1pr_smoke.yaml"

NODE_SCHEMA = {
    0: {"feature_name": "atom_type", "value": "C"},
    1: {"feature_name": "atom_type", "value": "N"},
    2: {"feature_name": "atom_type", "value": "O"},
    3: {"feature_name": "atom_type", "value": "F"},
}
EDGE_SCHEMA = {
    0: {"feature_name": "bond_type", "value": "single"},
    1: {"feature_name": "bond_type", "value": "double"},
    2: {"feature_name": "bond_type", "value": "triple"},
}


def graph_arrays(index: int):
    """Return a small attributed path/ring graph with fixed numeric encodings."""

    node_count = 3 + index % 5
    adjacency = np.zeros((node_count, node_count), dtype=np.uint8)
    node_attributes = np.zeros((node_count, len(NODE_SCHEMA)), dtype=np.float32)
    edge_attributes = np.zeros(
        (len(EDGE_SCHEMA), node_count, node_count), dtype=np.float32
    )
    for node in range(node_count):
        node_attributes[node, (index + node) % len(NODE_SCHEMA)] = 1.0
    for source in range(node_count - 1):
        target = source + 1
        relation = (index + source) % len(EDGE_SCHEMA)
        adjacency[source, target] = adjacency[target, source] = 1
        edge_attributes[relation, source, target] = 1.0
        edge_attributes[relation, target, source] = 1.0
    if index % 2 and node_count > 3:
        relation = (index + node_count) % len(EDGE_SCHEMA)
        adjacency[0, -1] = adjacency[-1, 0] = 1
        edge_attributes[relation, 0, -1] = 1.0
        edge_attributes[relation, -1, 0] = 1.0
    return adjacency, node_attributes, edge_attributes


def build_payload():
    config = flatten_config(load_yaml_mapping(CONFIG))
    graphs = [graph_arrays(index) for index in range(30)]

    def select(start: int, stop: int, component: int):
        return [graphs[index][component] for index in range(start, stop)]

    return {
        "list_adj": select(0, 21, 0),
        "list_noh_train": select(0, 21, 1),
        "list_eoh_train": select(0, 21, 2),
        "val_adj": select(21, 24, 0),
        "list_noh_val": select(21, 24, 1),
        "list_eoh_val": select(21, 24, 2),
        "test_list_adj": select(24, 30, 0),
        "list_noh_test": select(24, 30, 1),
        "list_eoh_test": select(24, 30, 2),
        "node_onehot_info": NODE_SCHEMA,
        "edge_onehot_info": EDGE_SCHEMA,
        "split_mode": config["split_mode"],
        "cache_metadata": build_dataset_cache_metadata(config),
        "qualification_fixture": True,
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=str(OUTPUT.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(build_payload(), handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(OUTPUT))
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
