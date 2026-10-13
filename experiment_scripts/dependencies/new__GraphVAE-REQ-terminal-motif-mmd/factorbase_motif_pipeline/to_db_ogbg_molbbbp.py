#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OGBG-MOLBBBP Dataset to MySQL Database Converter

This mirrors the PROTEINS importer style but keeps OGB molecular features:

- nodes: one row per atom with the 9 OGB atom categorical features
- edges: one row per directed bond edge with the 3 OGB bond categorical features

All molecules are flattened into one disconnected union graph, matching the
GraphVAE/FactorBase convention used by the other dataset importers.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data_raw"
DEFAULT_EDGE_MODE = "directed"

NODE_FEATURE_COLUMNS = [
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_h",
    "num_radical_e",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
]

EDGE_FEATURE_COLUMNS = [
    "bond_type",
    "bond_stereo",
    "is_conjugated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ogbg-molbbbp into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")
    parser.add_argument(
        "--ogb-root",
        type=Path,
        default=None,
        help=(
            "OGB root directory containing ogbg_molbbbp/. If omitted, uses "
            "OGB_DATA_ROOT, data_raw/ogb, dataset, then ../GraphVAE-MM/dataset."
        ),
    )

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store exactly the edge directions exposed by the source DGL graph",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store both directions for each edge pair",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the SQL schema and exit without loading data.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=60,
        help=(
            "Keep only OGB graphs with at most this many nodes. The default "
            "matches data.py's ogbg-molbbbp GraphVAE loader; pass 0 or a "
            "negative value to import every graph."
        ),
    )

    args = parser.parse_args()
    if not args.directed and not args.undirected:
        if DEFAULT_EDGE_MODE == "directed":
            args.directed = True
        elif DEFAULT_EDGE_MODE == "undirected":
            args.undirected = True
    return args


EDGE_MODE_LABELS = {
    "directed": "DIRECTED (source graph edge directions)",
    "undirected": "UNDIRECTED (A->B and B->A for each edge pair)",
}


@dataclass
class OgbMolGraph:
    node_features: List[Tuple[int, ...]]
    edges: List[Tuple[int, int, Tuple[int, ...]]]
    graph_label: int | None


def candidate_ogb_roots(args: argparse.Namespace) -> List[Path]:
    candidates: List[Path] = []
    if args.ogb_root is not None:
        candidates.append(args.ogb_root.expanduser())
    env_root = os.environ.get("OGB_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([
        DATA_ROOT / "ogb",
        REPO_ROOT / "dataset",
        REPO_ROOT.parent / "GraphVAE-MM" / "dataset",
    ])

    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def load_ogb_dataset(args: argparse.Namespace):
    from ogb.graphproppred import DglGraphPropPredDataset

    last_error = None
    for root in candidate_ogb_roots(args):
        dataset_dir = root / "ogbg_molbbbp"
        try:
            if dataset_dir.exists() or root == DATA_ROOT / "ogb":
                print(f"Trying OGB root: {root}")
                return DglGraphPropPredDataset(
                    name="ogbg-molbbbp",
                    root=str(root),
                ), root
        except Exception as exc:
            last_error = exc
            print(f"Warning: failed to load from {root}: {exc}")

    searched = "\n  - ".join(str(path) for path in candidate_ogb_roots(args))
    message = (
        "Could not load ogbg-molbbbp.\n"
        f"Tried:\n  - {searched}\n"
        "Set OGB_DATA_ROOT or pass --ogb-root to a directory containing ogbg_molbbbp/."
    )
    if last_error is not None:
        message += f"\nLast error: {last_error}"
    raise RuntimeError(message)


def _label_to_int(label) -> int | None:
    label_values = label.cpu().view(-1)
    if label_values.numel() == 0:
        return None
    value = label_values[0].item()
    try:
        return int(value)
    except Exception:
        return None


def _extract_graph(graph, label) -> OgbMolGraph:
    if "feat" not in graph.ndata:
        raise KeyError(
            "OGB graph has no node feature key 'feat'. "
            f"Available node keys: {list(graph.ndata.keys())}"
        )

    node_feat = graph.ndata["feat"].cpu().long()
    if node_feat.dim() == 1:
        node_feat = node_feat.view(-1, 1)
    node_features = [
        tuple(int(v) for v in node_feat[node_idx].tolist())
        for node_idx in range(graph.num_nodes())
    ]

    src_nodes, dst_nodes = graph.edges()
    edge_feat = graph.edata["feat"] if "feat" in graph.edata else None
    if edge_feat is None:
        edge_feat = torch.zeros((src_nodes.numel(), 0), dtype=torch.long)
    edge_feat = edge_feat.cpu().long()
    if edge_feat.dim() == 1:
        edge_feat = edge_feat.view(-1, 1)

    edges = []
    for row_idx, (src, dst) in enumerate(zip(src_nodes.tolist(), dst_nodes.tolist())):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        features = tuple(int(v) for v in edge_feat[row_idx].tolist())
        edges.append((src, dst, features))

    return OgbMolGraph(
        node_features=node_features,
        edges=edges,
        graph_label=_label_to_int(label),
    )


def load_ogbg_molbbbp(args: argparse.Namespace) -> Tuple[List[OgbMolGraph], str]:
    dataset, root = load_ogb_dataset(args)
    graphs: List[OgbMolGraph] = []
    skipped_large_graphs = 0
    max_nodes = int(args.max_nodes)

    for idx, (graph, label) in enumerate(dataset):
        if idx % 250 == 0:
            print(f"Loading OGB graph {idx}/{len(dataset)}")
        if max_nodes > 0 and graph.num_nodes() > max_nodes:
            skipped_large_graphs += 1
            continue
        graphs.append(_extract_graph(graph, label))

    if max_nodes > 0:
        print(
            "ogbg-molbbbp max-node filter: "
            f"kept {len(graphs)}/{len(dataset)} graphs, "
            f"skipped {skipped_large_graphs} graphs with num_nodes > {max_nodes}"
        )
    else:
        print("ogbg-molbbbp max-node filter disabled; imported every graph")

    return graphs, f"OGB DglGraphPropPredDataset(name='ogbg-molbbbp', root='{root}')"


def analyze_source_edge_direction(graphs: List[OgbMolGraph]) -> dict:
    stats = {
        "graphs": len(graphs),
        "graphs_with_edges": 0,
        "source_edge_rows": 0,
        "undirected_edge_pairs": 0,
        "missing_reverse_rows": 0,
        "graphs_with_missing_reverse": 0,
    }

    for graph in graphs:
        edge_rows = {
            (int(src), int(dst))
            for src, dst, _features in graph.edges
            if int(src) != int(dst)
        }
        if not edge_rows:
            continue

        missing_reverse_rows = sum(
            1 for src, dst in edge_rows if (dst, src) not in edge_rows
        )

        stats["graphs_with_edges"] += 1
        stats["source_edge_rows"] += len(edge_rows)
        stats["undirected_edge_pairs"] += len({tuple(sorted(edge)) for edge in edge_rows})
        stats["missing_reverse_rows"] += missing_reverse_rows
        if missing_reverse_rows:
            stats["graphs_with_missing_reverse"] += 1

    return stats


def print_source_edge_direction_analysis(stats: dict, edge_mode: str) -> None:
    print("=" * 60)
    print("SOURCE EDGE DIRECTION ANALYSIS")
    print("=" * 60)
    print("Dataset: ogbg-molbbbp")
    print(f"Graphs analyzed: {stats['graphs']:,}")
    print(f"Graphs with edges: {stats['graphs_with_edges']:,}")
    print(f"Source edge rows: {stats['source_edge_rows']:,}")
    print(f"Unique undirected edge pairs: {stats['undirected_edge_pairs']:,}")
    print(f"Rows missing reverse edge: {stats['missing_reverse_rows']:,}")

    has_edges = stats["source_edge_rows"] > 0
    source_is_bidirectional = has_edges and stats["missing_reverse_rows"] == 0
    if source_is_bidirectional:
        print("Source appears bidirectional/undirected: every edge row has a reverse row.")
    elif has_edges:
        print("Source contains one-way edge rows.")
    else:
        print("Source has no edge rows to analyze.")
    print(f"Selected DB edge mode: {EDGE_MODE_LABELS[edge_mode]}")
    print()


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def nodes_schema_sql() -> str:
    feature_defs = ",\n    ".join(
        f"{quote_identifier(name)} INT NOT NULL"
        for name in NODE_FEATURE_COLUMNS
    )
    feature_indexes = ",\n    ".join(
        f"INDEX idx_{name} ({quote_identifier(name)})"
        for name in NODE_FEATURE_COLUMNS
    )
    return f"""
CREATE TABLE IF NOT EXISTS nodes (
    node_id INT PRIMARY KEY,
    {feature_defs},
    {feature_indexes}
)
"""


def edges_schema_sql() -> str:
    feature_defs = ",\n    ".join(
        f"{quote_identifier(name)} INT NOT NULL"
        for name in EDGE_FEATURE_COLUMNS
    )
    feature_indexes = ",\n    ".join(
        f"INDEX idx_{name} ({quote_identifier(name)})"
        for name in EDGE_FEATURE_COLUMNS
    )
    return f"""
CREATE TABLE IF NOT EXISTS edges (
    source_node_id INT NOT NULL,
    target_node_id INT NOT NULL,
    {feature_defs},
    PRIMARY KEY (source_node_id, target_node_id),
    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
    {feature_indexes}
)
"""


def print_schema() -> None:
    print("NODES SQL:")
    print(nodes_schema_sql())
    print("EDGES SQL:")
    print(edges_schema_sql())


def pad_tuple(values: Tuple[int, ...], width: int) -> Tuple[int, ...]:
    if len(values) >= width:
        return tuple(int(v) for v in values[:width])
    return tuple(int(v) for v in values) + tuple(0 for _ in range(width - len(values)))


def main() -> None:
    args = parse_args()
    if args.print_schema:
        print_schema()
        return

    print("=" * 60)
    print("LOADING OGBG-MOLBBBP DATASET")
    print("=" * 60)
    graphs, load_source = load_ogbg_molbbbp(args)
    print(f"Loaded {len(graphs):,} OGB molecular graphs")
    print(f"Source: {load_source}\n")

    if args.directed:
        edge_mode = "directed"
        print("Selected: DIRECTED\n")
    elif args.undirected:
        edge_mode = "undirected"
        print("Selected: UNDIRECTED\n")
    else:
        raise RuntimeError("No edge mode selected.")

    source_edge_stats = analyze_source_edge_direction(graphs)
    print_source_edge_direction_analysis(source_edge_stats, edge_mode)

    print("=" * 60)
    print("ANALYZING DATA DISTRIBUTIONS")
    print("=" * 60)
    node_feature_dist: Dict[str, defaultdict[int, int]] = {
        name: defaultdict(int) for name in NODE_FEATURE_COLUMNS
    }
    edge_feature_dist: Dict[str, defaultdict[int, int]] = {
        name: defaultdict(int) for name in EDGE_FEATURE_COLUMNS
    }
    graph_label_dist = defaultdict(int)
    total_nodes = 0
    total_source_edges = 0

    for graph in graphs:
        if graph.graph_label is not None:
            graph_label_dist[graph.graph_label] += 1
        total_nodes += len(graph.node_features)
        total_source_edges += len(graph.edges)
        for features in graph.node_features:
            features = pad_tuple(features, len(NODE_FEATURE_COLUMNS))
            for name, value in zip(NODE_FEATURE_COLUMNS, features):
                node_feature_dist[name][value] += 1
        for _src, _dst, features in graph.edges:
            features = pad_tuple(features, len(EDGE_FEATURE_COLUMNS))
            for name, value in zip(EDGE_FEATURE_COLUMNS, features):
                edge_feature_dist[name][value] += 1

    print(f"Graphs: {len(graphs):,}")
    print(f"Nodes: {total_nodes:,}")
    print(f"Source edge rows: {total_source_edges:,}")
    print("\nNode feature distributions:")
    for name in NODE_FEATURE_COLUMNS:
        values = node_feature_dist[name]
        summary = ", ".join(f"{value}:{values[value]:,}" for value in sorted(values))
        print(f"  {name}: {summary}")
    print("\nEdge feature distributions:")
    for name in EDGE_FEATURE_COLUMNS:
        values = edge_feature_dist[name]
        summary = ", ".join(f"{value}:{values[value]:,}" for value in sorted(values))
        print(f"  {name}: {summary}")
    print("\nGraph label distribution:")
    for label in sorted(graph_label_dist):
        print(f"  Label {label}: {graph_label_dist[label]:,}")

    print("\n" + "=" * 60)
    print("DATABASE CONFIGURATION")
    print("=" * 60)
    db_name = args.db_name if args.db_name else input("Enter the database name: ").strip()

    from pymysql import connect

    db_params = {
        "host": "localhost",
        "user": "fbuser",
        "password": "",
    }

    print("\n" + "=" * 60)
    print("CONNECTING TO DATABASE")
    print("=" * 60)

    connection = connect(**db_params)
    cursor = connection.cursor()

    cursor.execute(f"DROP DATABASE IF EXISTS {quote_identifier(db_name)}")
    cursor.execute(
        f"CREATE DATABASE {quote_identifier(db_name)} "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.execute(f"USE {quote_identifier(db_name)}")
    cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
    cursor.execute("SET sql_mode='STRICT_TRANS_TABLES';")
    print(f"Connected to MySQL | Database: {db_name} created and selected\n")

    print("=" * 60)
    print("CREATING DATABASE SCHEMA")
    print("=" * 60)
    cursor.execute(nodes_schema_sql())
    print("NODES TABLE created")
    print("   - 9 OGB atom feature columns")
    cursor.execute(edges_schema_sql())
    print("EDGES TABLE created")
    print("   - 3 OGB bond feature columns")
    print(f"Edge mode: {EDGE_MODE_LABELS[edge_mode]}")
    print("All molecules will be flattened into one disconnected union with no inter-graph edges")

    print("\n" + "=" * 60)
    print("POPULATING DATABASE WITH OGBG-MOLBBBP DATA")
    print("=" * 60)

    global_node_id = 0
    inserted_edges = 0
    node_insert_sql = (
        "INSERT INTO nodes (node_id, "
        + ", ".join(quote_identifier(name) for name in NODE_FEATURE_COLUMNS)
        + ") VALUES ("
        + ", ".join(["%s"] * (1 + len(NODE_FEATURE_COLUMNS)))
        + ")"
    )
    edge_insert_sql = (
        "INSERT INTO edges (source_node_id, target_node_id, "
        + ", ".join(quote_identifier(name) for name in EDGE_FEATURE_COLUMNS)
        + ") VALUES ("
        + ", ".join(["%s"] * (2 + len(EDGE_FEATURE_COLUMNS)))
        + ")"
    )

    for graph_id, graph in enumerate(graphs):
        if graph_id % 100 == 0:
            pct = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({pct:.1f}%)")

        graph_node_offset = global_node_id
        node_rows = []
        for features in graph.node_features:
            node_rows.append((global_node_id, *pad_tuple(features, len(NODE_FEATURE_COLUMNS))))
            global_node_id += 1

        if node_rows:
            cursor.executemany(node_insert_sql, node_rows)

        edge_rows = []
        seen_edges = set()
        for src_local, dst_local, features in graph.edges:
            src_global = graph_node_offset + src_local
            dst_global = graph_node_offset + dst_local
            feature_tuple = pad_tuple(features, len(EDGE_FEATURE_COLUMNS))

            if edge_mode == "directed":
                edge_key = (src_global, dst_global)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_rows.append((src_global, dst_global, *feature_tuple))
                continue

            for src_out, dst_out in ((src_global, dst_global), (dst_global, src_global)):
                edge_key = (src_out, dst_out)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_rows.append((src_out, dst_out, *feature_tuple))

        if edge_rows:
            cursor.executemany(edge_insert_sql, edge_rows)
            inserted_edges += len(edge_rows)

        if (graph_id + 1) % 100 == 0:
            connection.commit()

    connection.commit()

    print("\n" + "=" * 60)
    print("DATABASE POPULATION COMPLETE!")
    print("=" * 60)

    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]
    print(f"NODES: {node_count:,}")
    print(f"EDGES: {edge_count:,}")
    print(f"Original source edge rows: {total_source_edges:,}")
    print(f"Inserted edge rows: {inserted_edges:,}")

    print("\nSAMPLE DATA (First 5 nodes):")
    cursor.execute("SELECT * FROM nodes LIMIT 5")
    node_rows = cursor.fetchall()
    print("  " + " | ".join(["node_id", *NODE_FEATURE_COLUMNS]))
    for row in node_rows:
        print("  " + " | ".join(str(value) for value in row))

    print("\nSAMPLE DATA (First 5 edges):")
    cursor.execute("SELECT * FROM edges LIMIT 5")
    edge_rows = cursor.fetchall()
    print("  " + " | ".join(["source_node_id", "target_node_id", *EDGE_FEATURE_COLUMNS]))
    for row in edge_rows:
        print("  " + " | ".join(str(value) for value in row))

    print("\nSCHEMA VERIFICATION:")
    cursor.execute("SHOW CREATE TABLE edges")
    result = cursor.fetchone()
    create_table_sql = result[1].lower()
    checks = [
        ("PRIMARY KEY (`source_node_id`,`target_node_id`)", "Composite primary key"),
        ("FOREIGN KEY (`source_node_id`) REFERENCES `nodes`", "source_node_id foreign key"),
        ("FOREIGN KEY (`target_node_id`) REFERENCES `nodes`", "target_node_id foreign key"),
        ("`bond_type` int", "bond_type feature column"),
        ("`bond_stereo` int", "bond_stereo feature column"),
        ("`is_conjugated` int", "is_conjugated feature column"),
    ]
    for pattern, label in checks:
        status = "OK" if pattern.lower() in create_table_sql else "MISSING"
        print(f"  [{status}] {label}")

    cursor.close()
    connection.close()

    print("\n" + "=" * 60)
    print("DATABASE READY!")
    print("=" * 60)
    print(f"  Database : {db_name}")
    print("  Tables   : nodes, edges")


if __name__ == "__main__":
    main()
