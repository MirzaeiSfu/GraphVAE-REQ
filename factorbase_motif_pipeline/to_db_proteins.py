#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROTEINS Dataset to MySQL Database Converter
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - only used when DGL is unavailable
    torch = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data_raw"
DEFAULT_EDGE_MODE = "directed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the PROTEINS dataset into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store exactly the edge directions exposed by the source graph",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store both directions for each edge pair",
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
class ProteinGraph:
    node_features: List[int]
    edges: List[Tuple[int, int]]
    graph_label: int


def analyze_source_edge_direction(graphs: List[ProteinGraph]) -> dict:
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
            for src, dst in graph.edges
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


def print_source_edge_direction_analysis(
    dataset_name: str,
    stats: dict,
    edge_mode: str,
) -> None:
    print("=" * 60)
    print("SOURCE EDGE DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Graphs analyzed: {stats['graphs']:,}")
    print(f"Graphs with edges: {stats['graphs_with_edges']:,}")
    print(f"Source edge rows: {stats['source_edge_rows']:,}")
    print(f"Unique undirected edge pairs: {stats['undirected_edge_pairs']:,}")
    print(f"Rows missing reverse edge: {stats['missing_reverse_rows']:,}")

    has_edges = stats["source_edge_rows"] > 0
    source_is_bidirectional = has_edges and stats["missing_reverse_rows"] == 0

    if source_is_bidirectional:
        print("Source appears bidirectional/undirected: every edge row has a reverse row.")
        if edge_mode == "directed":
            print(
                "WARNING: --directed preserves source rows, but this source is already "
                "bidirectional; the edge table should match --undirected."
            )
        else:
            print(
                "NOTE: --undirected will ensure reverse rows, but the source already has "
                "them; no extra edge rows are expected."
            )
    elif has_edges:
        print("Source contains one-way edge rows.")
        if edge_mode == "directed":
            print("NOTE: --directed will preserve those one-way source rows.")
        else:
            print("NOTE: --undirected will add reverse rows for one-way source edges.")
    else:
        print("Source has no edge rows to analyze.")
    print()


def _to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _extract_node_feature(raw_feature) -> int:
    if hasattr(raw_feature, "numel"):
        if raw_feature.numel() == 1:
            return _to_int(raw_feature) + 1
        if torch is None:
            raw_feature = raw_feature.tolist()
        else:
            return int(torch.argmax(raw_feature).item()) + 1

    if isinstance(raw_feature, (list, tuple)):
        if len(raw_feature) == 1:
            return int(raw_feature[0]) + 1
        return max(range(len(raw_feature)), key=lambda idx: raw_feature[idx]) + 1

    return int(raw_feature) + 1


def _extract_dgl_node_features(graph) -> List[int]:
    for key in ("feat", "attr", "label"):
        if key in graph.ndata:
            data = graph.ndata[key]
            return [_extract_node_feature(data[node_id]) for node_id in range(graph.num_nodes())]
    raise KeyError(
        f"No supported node feature key found. Available keys: {list(graph.ndata.keys())}"
    )


def _extract_dgl_edges(graph) -> List[Tuple[int, int]]:
    src_nodes, dst_nodes = graph.edges()
    edges = []

    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        edges.append((src, dst))

    return edges


def load_proteins_from_dgl() -> Tuple[List[ProteinGraph], str]:
    import dgl

    dgl_root = DATA_ROOT / "dgl"
    try:
        dataset = dgl.data.GINDataset(
            name="PROTEINS",
            self_loop=False,
            raw_dir=str(dgl_root),
        )
        dataset_source = f"DGL GINDataset(name='PROTEINS', raw_dir='{dgl_root}')"
    except TypeError:  # pragma: no cover - depends on installed DGL version
        dataset = dgl.data.GINDataset(name="PROTEINS", self_loop=False)
        dataset_source = "DGL GINDataset(name='PROTEINS')"
    graphs = []

    for graph, label in zip(dataset.graphs, dataset.labels):
        graphs.append(
            ProteinGraph(
                node_features=_extract_dgl_node_features(graph),
                edges=_extract_dgl_edges(graph),
                graph_label=_to_int(label),
            )
        )

    return graphs, dataset_source


def load_proteins_from_text(dataset_path: Path) -> Tuple[List[ProteinGraph], str]:
    with dataset_path.open("r", encoding="utf-8") as infile:
        lines = [line.strip() for line in infile if line.strip()]

    if not lines:
        raise ValueError(f"Dataset file is empty: {dataset_path}")

    num_graphs = int(lines[0])
    line_idx = 1
    graphs: List[ProteinGraph] = []

    for _ in range(num_graphs):
        num_nodes, graph_label = map(int, lines[line_idx].split()[:2])
        line_idx += 1

        node_features: List[int] = []
        edges: List[Tuple[int, int]] = []

        for node_id in range(num_nodes):
            tokens = lines[line_idx].split()
            line_idx += 1

            node_label = int(tokens[0])
            degree = int(tokens[1])
            neighbors = [int(token) for token in tokens[2:2 + degree]]

            node_features.append(node_label + 1)
            for neighbor in neighbors:
                if node_id == neighbor:
                    continue
                edges.append((node_id, neighbor))

        graphs.append(
            ProteinGraph(
                node_features=node_features,
                edges=edges,
                graph_label=graph_label,
            )
        )

    if len(graphs) != num_graphs:
        raise ValueError(f"Expected {num_graphs} graphs, parsed {len(graphs)}")

    return graphs, f"raw text file ({dataset_path})"


def load_proteins_dataset() -> Tuple[List[ProteinGraph], str]:
    errors = []

    try:
        return load_proteins_from_dgl()
    except Exception as exc:  # pragma: no cover - depends on local environment
        errors.append(f"DGL load failed: {exc}")

    candidate_paths = [
        Path.home() / ".dgl" / "GINDataset" / "dataset" / "PROTEINS" / "PROTEINS.txt",
        DATA_ROOT / "Kernel_dataset" / "PROTEINS" / "PROTEINS.txt",
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return load_proteins_from_text(candidate_path)

    searched = "\n  - ".join(str(path) for path in candidate_paths)
    error_text = "\n  - ".join(errors) if errors else "No loader errors captured."
    raise RuntimeError(
        "Could not load the PROTEINS dataset.\n"
        f"Tried:\n  - {searched}\n"
        f"Loader errors:\n  - {error_text}"
    )


args = parse_args()

print("=" * 60)
print("LOADING PROTEINS DATASET")
print("=" * 60)
graphs, load_source = load_proteins_dataset()
print(f"Loaded {len(graphs):,} protein graphs")
print(f"Source: {load_source}\n")

print("=" * 60)
print("GRAPH DIRECTION CONFIGURATION")
print("=" * 60)
if args.directed:
    edge_mode = "directed"
    print("Selected: DIRECTED\n")
elif args.undirected:
    edge_mode = "undirected"
    print("Selected: UNDIRECTED\n")
else:
    while True:
        choice = input(
            "Edge storage mode?\n"
            "  1 - DIRECTED (store exactly the source edge directions)\n"
            "  2 - UNDIRECTED (store A->B and B->A for each edge pair)\n"
            "Choice: "
        ).strip()
        if choice == "1":
            edge_mode = "directed"
            print("Selected: DIRECTED\n")
            break
        if choice == "2":
            edge_mode = "undirected"
            print("Selected: UNDIRECTED\n")
            break
        print("Please enter 1 or 2.")

source_edge_stats = analyze_source_edge_direction(graphs)
print_source_edge_direction_analysis(
    "PROTEINS",
    source_edge_stats,
    edge_mode,
)

print("=" * 60)
print("ANALYZING DATA DISTRIBUTIONS")
print("=" * 60)

node_feature_dist = defaultdict(int)
graph_label_dist = defaultdict(int)
sample_nodes = 0
sample_source_edges = 0
sample_size = min(1000, len(graphs))

for graph in graphs[:sample_size]:
    graph_label_dist[graph.graph_label] += 1
    sample_nodes += len(graph.node_features)
    sample_source_edges += len(graph.edges)
    for node_feature in graph.node_features:
        node_feature_dist[node_feature] += 1

print(f"Analyzed {sample_size:,} graphs")
print(f"Sample nodes: {sample_nodes:,}")
print(f"Sample source edge rows: {sample_source_edges:,}")
print("\nNode feature distribution (1-based labels):")
for node_feature in sorted(node_feature_dist.keys()):
    print(f"  Feature {node_feature}: {node_feature_dist[node_feature]:,}")

print("\nGraph label distribution (sample):")
for graph_label in sorted(graph_label_dist.keys()):
    print(f"  Label {graph_label}: {graph_label_dist[graph_label]:,}")

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

cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
cursor.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
cursor.execute(f"USE `{db_name}`")
cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
cursor.execute("SET sql_mode='STRICT_TRANS_TABLES';")
print(f"Connected to MySQL | Database: {db_name} created and selected\n")

print("=" * 60)
print("CREATING DATABASE SCHEMA")
print("=" * 60)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS nodes (
    node_id INT PRIMARY KEY,
    node_feature INT NOT NULL,
    INDEX idx_node_feature (node_feature)
)
"""
)
print("NODES TABLE created")
print("   - node_feature: INT (1-based node label from PROTEINS)")

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS edges (
    source_node_id INT NOT NULL,
    target_node_id INT NOT NULL,
    PRIMARY KEY (source_node_id, target_node_id),
    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id)
)
"""
)
print("EDGES TABLE created")
print(f"Edge mode: {EDGE_MODE_LABELS[edge_mode]}")
print("All protein graphs will be flattened into one disconnected union with no inter-graph edges")

print("\n" + "=" * 60)
print("POPULATING DATABASE WITH PROTEINS DATA")
print("=" * 60)
print("This will take a little while...\n")

global_node_id = 0
node_feature_counts = defaultdict(int)
total_graph_label_counts = defaultdict(int)

for graph_id, graph in enumerate(graphs):
    if graph_id % 100 == 0:
        pct = graph_id / len(graphs) * 100
        print(f"Progress: {graph_id}/{len(graphs)} graphs ({pct:.1f}%)")

    total_graph_label_counts[graph.graph_label] += 1
    graph_node_offset = global_node_id

    node_rows = []
    for node_feature in graph.node_features:
        node_rows.append((global_node_id, node_feature))
        node_feature_counts[node_feature] += 1
        global_node_id += 1

    cursor.executemany(
        """
        INSERT INTO nodes (node_id, node_feature)
        VALUES (%s, %s)
        """,
        node_rows,
    )

    edge_rows = []
    seen_edges = set()
    for src_local, dst_local in graph.edges:
        src_global = graph_node_offset + src_local
        dst_global = graph_node_offset + dst_local
        if edge_mode == "directed":
            edge_key = (src_global, dst_global)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_rows.append(edge_key)
            continue

        for edge_key in ((src_global, dst_global), (dst_global, src_global)):
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_rows.append(edge_key)

    if edge_rows:
        cursor.executemany(
            """
            INSERT INTO edges (source_node_id, target_node_id)
            VALUES (%s, %s)
            """,
            edge_rows,
        )

    if (graph_id + 1) % 100 == 0:
        connection.commit()

connection.commit()

print("\n" + "=" * 60)
print("DATABASE POPULATION COMPLETE!")
print("=" * 60)

cursor.execute("SELECT COUNT(*) FROM nodes")
node_count = cursor.fetchone()[0]
print(f"\nNODES: {node_count:,} total")
print("\nNode feature distribution:")
for node_feature in sorted(node_feature_counts.keys()):
    count = node_feature_counts[node_feature]
    pct = count / node_count * 100
    print(f"  Feature {node_feature}: {count:,} ({pct:.1f}%)")

print("\nGraph label distribution:")
for graph_label in sorted(total_graph_label_counts.keys()):
    count = total_graph_label_counts[graph_label]
    pct = count / len(graphs) * 100
    print(f"  Label {graph_label}: {count:,} ({pct:.1f}%)")

cursor.execute("SELECT COUNT(*) FROM edges")
edge_count = cursor.fetchone()[0]
print(f"\nEDGES: {edge_count:,} total")
if edge_mode == "directed":
    print(f"  (DIRECTED: exact source edge directions: {edge_count:,})")
else:
    print(f"  (UNDIRECTED: 2x edge pairs | Edge pairs: {edge_count // 2:,})")

print(f"  Inserted without cross-graph connections across {len(graphs):,} original protein graphs")

print("\nSAMPLE DATA (First 10 nodes):")
cursor.execute("SELECT * FROM nodes LIMIT 10")
rows = cursor.fetchall()
print("  node_id | node_feature")
print("  " + "-" * 26)
for row in rows:
    print(f"  {row[0]:7d} | {row[1]:12d}")

print("\nSAMPLE DATA (First 10 edges):")
cursor.execute("SELECT * FROM edges LIMIT 10")
rows = cursor.fetchall()
print("  source_node_id | target_node_id")
print("  " + "-" * 35)
for row in rows:
    print(f"  {row[0]:14d} | {row[1]:14d}")

print("\nSCHEMA VERIFICATION:")
cursor.execute("SHOW CREATE TABLE edges")
result = cursor.fetchone()
checks = [
    ("PRIMARY KEY (`source_node_id`,`target_node_id`)", "Composite primary key"),
    ("FOREIGN KEY (`source_node_id`) REFERENCES `nodes`", "source_node_id foreign key"),
    ("FOREIGN KEY (`target_node_id`) REFERENCES `nodes`", "target_node_id foreign key"),
]
for pattern, label in checks:
    status = "OK" if pattern in result[1] else "MISSING"
    print(f"  [{status}] {label}")

if edge_mode == "undirected":
    print("\nBIDIRECTIONAL EDGE VERIFICATION:")
    cursor.execute(
        """
        SELECT COUNT(*) FROM edges e1
        WHERE EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
              AND e2.target_node_id = e1.source_node_id
        )
        """
    )
    bidirectional_count = cursor.fetchone()[0]
    print(f"  Edges with reverse direction: {bidirectional_count:,} / {edge_count:,}")
    if bidirectional_count == edge_count:
        print("  ALL edges are bidirectional")
    else:
        print("  WARNING: Some edges are missing the reverse direction")

cursor.close()
connection.close()

print("\n" + "=" * 60)
print("DATABASE READY!")
print("=" * 60)
print(f"  Database : {db_name}")
print(f"  Source   : {load_source}")
print(f"  Mode     : {edge_mode.upper()}")
print(f"  Graphs   : {len(graphs):,}")
print(f"  Nodes    : {node_count:,}")
print(f"  Edges    : {edge_count:,}")
