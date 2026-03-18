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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the PROTEINS dataset into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store both directions for each edge",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store one canonical edge per undirected pair",
    )
    return parser.parse_args()


@dataclass
class ProteinGraph:
    node_features: List[int]
    edges: List[Tuple[int, int]]
    graph_label: int


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
    undirected_edges = set()

    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        edge = (src, dst) if src < dst else (dst, src)
        undirected_edges.add(edge)

    return sorted(undirected_edges)


def load_proteins_from_dgl() -> Tuple[List[ProteinGraph], str]:
    import dgl

    dataset = dgl.data.GINDataset(name="PROTEINS", self_loop=False)
    graphs = []

    for graph, label in zip(dataset.graphs, dataset.labels):
        graphs.append(
            ProteinGraph(
                node_features=_extract_dgl_node_features(graph),
                edges=_extract_dgl_edges(graph),
                graph_label=_to_int(label),
            )
        )

    return graphs, "DGL GINDataset(name='PROTEINS')"


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
        undirected_edges = set()

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
                edge = (node_id, neighbor) if node_id < neighbor else (neighbor, node_id)
                undirected_edges.add(edge)

        graphs.append(
            ProteinGraph(
                node_features=node_features,
                edges=sorted(undirected_edges),
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
        Path(__file__).resolve().parent / "data" / "Kernel_dataset" / "PROTEINS" / "PROTEINS.txt",
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
    directed = True
    print("Selected: DIRECTED\n")
elif args.undirected:
    directed = False
    print("Selected: UNDIRECTED\n")
else:
    while True:
        choice = input(
            "Edge storage mode?\n"
            "  1 - DIRECTED (A->B and B->A)\n"
            "  2 - UNDIRECTED (only one stored edge per pair)\n"
            "Choice: "
        ).strip()
        if choice == "1":
            directed = True
            print("Selected: DIRECTED\n")
            break
        if choice == "2":
            directed = False
            print("Selected: UNDIRECTED\n")
            break
        print("Please enter 1 or 2.")

print("=" * 60)
print("ANALYZING DATA DISTRIBUTIONS")
print("=" * 60)

node_feature_dist = defaultdict(int)
graph_label_dist = defaultdict(int)
sample_nodes = 0
sample_undirected_edges = 0
sample_size = min(1000, len(graphs))

for graph in graphs[:sample_size]:
    graph_label_dist[graph.graph_label] += 1
    sample_nodes += len(graph.node_features)
    sample_undirected_edges += len(graph.edges)
    for node_feature in graph.node_features:
        node_feature_dist[node_feature] += 1

print(f"Analyzed {sample_size:,} graphs")
print(f"Sample nodes: {sample_nodes:,}")
print(f"Sample undirected edges: {sample_undirected_edges:,}")
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
print(f"Edge mode: {'DIRECTED (A->B and B->A)' if directed else 'UNDIRECTED (only one stored edge per pair)'}")
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
    for src_local, dst_local in graph.edges:
        src_global = graph_node_offset + src_local
        dst_global = graph_node_offset + dst_local
        edge_rows.append((src_global, dst_global))
        if directed:
            edge_rows.append((dst_global, src_global))

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
if directed:
    print(f"  (DIRECTED: 2x undirected edges | Undirected edges: {edge_count // 2:,})")
else:
    print(f"  (UNDIRECTED: 1x undirected edges: {edge_count:,})")

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

if directed:
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
print(f"  Mode     : {'DIRECTED' if directed else 'UNDIRECTED'}")
print(f"  Graphs   : {len(graphs):,}")
print(f"  Nodes    : {node_count:,}")
print(f"  Edges    : {edge_count:,}")
