#!/usr/bin/env python3
"""Cross-check GraphVAE-REQ motif counts against FactorBase MUTAG local_mult."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from data import DataWrapper, Datasets, merge_datasets
from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_store import RuleBasedMotifStore
from motif_counting.sanity_check_compare import (
    compare_aggregated_counts_to_factorbase_detailed,
)
from util import build_onehot_features, remove_self_loops


DATABASE = "mutag_undir_edgefeat_multi"
RAW = Path(
    "/local-scratch2/mirzaei/miniconda3/envs/mmd/lib/python3.7/"
    "site-packages/grakel/tests/data/MUTAG"
)
OUT = Path(
    "/local-scratch2/mirzaei/factorbase_edge_features_20260910/"
    "mutag_undir_edgefeat_multi/sanity_check"
)
CONFIG = Path(
    "/local-scratch2/mirzaei/factorbase_edge_features_20260910/"
    "mutag_undir_edgefeat_multi/factorbase_config.cfg"
)


def read_ints(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def read_pairs(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            left, right = line.split(",")
            rows.append((int(left) - 1, int(right) - 1))
    return rows


def load_graph_tensors():
    indicators = read_ints(RAW / "MUTAG_graph_indicator.txt")
    node_labels = read_ints(RAW / "MUTAG_node_labels.txt")
    global_edges = read_pairs(RAW / "MUTAG_A.txt")
    edge_labels = read_ints(RAW / "MUTAG_edge_labels.txt")
    if len(global_edges) != len(edge_labels):
        raise ValueError("MUTAG edge rows and edge labels have different lengths")

    nodes_by_graph: dict[int, list[int]] = defaultdict(list)
    for node, graph_id in enumerate(indicators):
        nodes_by_graph[graph_id].append(node)

    edges_by_graph: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (source, target), label in zip(global_edges, edge_labels):
        source_graph = indicators[source]
        if source_graph != indicators[target]:
            raise ValueError(f"Cross-graph edge: {source}->{target}")
        edges_by_graph[source_graph].append((source, target, label))

    adjs, node_features, edge_features = [], [], []
    for graph_id in sorted(nodes_by_graph):
        global_nodes = nodes_by_graph[graph_id]
        local = {node: index for index, node in enumerate(global_nodes)}
        edge_rows = edges_by_graph[graph_id]
        sources = [local[source] for source, _target, _label in edge_rows]
        targets = [local[target] for _source, target, _label in edge_rows]
        n = len(global_nodes)
        adjs.append(
            sp.csr_matrix(
                (np.ones(len(edge_rows), dtype=np.int8), (sources, targets)),
                shape=(n, n),
            )
        )
        node_features.append(
            np.asarray([[node_labels[node] + 1] for node in global_nodes], dtype=np.int64)
        )
        edge_features.append(
            np.asarray(
                [
                    [local[source], local[target], label]
                    for source, target, label in edge_rows
                ],
                dtype=np.int64,
            )
        )
    return adjs, node_features, edge_features


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        device="cpu",
        motif_cache_dir=str(OUT / "motif_cache"),
        motif_cp_table_source="cp",
        rule_prune=False,
        motif_prune_max_values_per_rule=None,
        use_syntactic_literal_rules=False,
        syntactic_literal_rule_mode="original",
    )
    adjs, node_features, edge_features = load_graph_tensors()
    node_info = {0: {"feature_name": "node_feature", "unique_values": list(range(1, 8))}}
    edge_info = {0: {"feature_name": "edge_label", "unique_values": list(range(4))}}
    node_oh, edge_oh, node_oh_info, edge_oh_info = build_onehot_features(
        node_features, edge_features, adjs, node_info, edge_info
    )

    dataset = Datasets(
        adjs,
        True,
        [None] * len(adjs),
        graphlabels=None,
        list_node_onehot=node_oh,
        list_edge_onehot=edge_oh,
    )
    remove_self_loops(dataset)

    RuleBasedMotifStore(database_name=DATABASE, args=args)
    counter = RelationalMotifCounter(database_name=DATABASE, args=args)
    wrapper = DataWrapper(
        merge_datasets(dataset),
        counter.relation_keys,
        node_onehot_info=node_oh_info,
        edge_onehot_info=edge_oh_info,
        edge_feature_info_mapping=counter.feature_info_mapping,
        device="cpu",
    )
    counts = counter.count_batch(
        wrapper,
        batch_size=64,
        output_mode="total_count",
        detach_to_cpu=True,
    )
    if isinstance(counts, tuple):
        counts = counts[0]
    aggregated = counter.aggregate_motif_counts(counts)
    matches, mismatches = compare_aggregated_counts_to_factorbase_detailed(
        aggregated_counts=aggregated,
        motif_counter=counter,
        database_name=DATABASE,
        config_path=CONFIG,
        atol=1e-4,
    )
    report = {
        "database": DATABASE,
        "graphs": len(adjs),
        "nodes": int(sum(adj.shape[0] for adj in adjs)),
        # Datasets adds temporary diagonal entries to its internal adjacency
        # representation, so use the original labeled-edge rows here.
        "directed_edge_rows": int(sum(rows.shape[0] for rows in edge_features)),
        "rules": len(counter.rules),
        "active_rule_value_rows": int(sum(len(rows) for rows in counter.values)),
        "counts_shape": list(counts.shape),
        "aggregated_count_entries": int(aggregated.numel()),
        "absolute_tolerance": 1e-4,
        "matches_factorbase_local_mult": matches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:100],
        "rule_definitions": [list(rule) for rule in counter.rules],
    }
    output = OUT / "sanity_check_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
