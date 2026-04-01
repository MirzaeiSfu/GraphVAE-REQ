#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lobster graphs to MySQL database converter for FactorBase.

This script matches the CLI contract used by `run_factorbase_pipeline.py`:
- `--db-name` selects the MySQL database name to create
- `--directed` stores both directions for each edge
- `--undirected` stores one canonical edge per pair
- `--feature-mode` selects whether to create a schema with or without features
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
from pymysql import connect

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_feature_utils.lobster_features import (
    DISTANCE_TO_SPINE_LABELS,
    ECCENTRICITY_BUCKET_LABELS,
    EDGE_TYPE_LABELS,
    NODE_DEGREE_LABELS,
    SUBTREE_SIZE_BUCKET_LABELS,
    compute_branch_component_sizes,
    compute_distance_to_spine_labels,
    compute_eccentricity,
    compute_edge_type,
    compute_node_degree,
    find_spine_path,
)


DEFAULT_DB_NAME = "lobster"
DEFAULT_FEATURE_MODE = "with-features"
DB_HOST = "127.0.0.1"
DB_USER = "fbuser"
DB_PASSWORD = ""

LOBSTER_P1 = 0.7
LOBSTER_P2 = 0.7
LOBSTER_MIN_NODES = 10
LOBSTER_MAX_NODES = 100
LOBSTER_MEAN_NODES = 80
LOBSTER_NUM_GRAPHS = 100
LOBSTER_RANDOM_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated LOBSTER graphs into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the LOBSTER schema with or without node/edge features",
    )

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


def default_db_name_for_mode(feature_mode: str) -> str:
    if feature_mode == "without-features":
        return f"{DEFAULT_DB_NAME}_no_feature"
    return DEFAULT_DB_NAME


def resolve_edge_mode(args: argparse.Namespace) -> bool:
    print("=" * 60)
    print("GRAPH DIRECTION CONFIGURATION")
    print("=" * 60)

    if args.directed:
        print("Selected: DIRECTED\n")
        return True
    if args.undirected:
        print("Selected: UNDIRECTED\n")
        return False

    while True:
        choice = input(
            "Edge storage mode?\n"
            "  1 - DIRECTED (A->B and B->A)\n"
            "  2 - UNDIRECTED (only one stored edge per pair)\n"
            "Choice: "
        ).strip()
        if choice == "1":
            print("Selected: DIRECTED\n")
            return True
        if choice == "2":
            print("Selected: UNDIRECTED\n")
            return False
        print("Please enter 1 or 2.")


def build_lobster_graphs() -> list[nx.Graph]:
    print("\n" + "=" * 70)
    print("GENERATING LOBSTER GRAPHS")
    print("=" * 70)
    print("Config:")
    print(f"  p1 = {LOBSTER_P1}, p2 = {LOBSTER_P2}")
    print(f"  mean_node = {LOBSTER_MEAN_NODES}")
    print(f"  min_node = {LOBSTER_MIN_NODES}, max_node = {LOBSTER_MAX_NODES}")
    print(f"  num_graphs = {LOBSTER_NUM_GRAPHS}")

    graphs: list[nx.Graph] = []
    seed_value = LOBSTER_RANDOM_SEED
    while len(graphs) < LOBSTER_NUM_GRAPHS:
        graph = nx.random_lobster(
            LOBSTER_MEAN_NODES,
            LOBSTER_P1,
            LOBSTER_P2,
            seed=seed_value,
        )
        if LOBSTER_MIN_NODES <= graph.number_of_nodes() <= LOBSTER_MAX_NODES:
            graphs.append(graph)
            if len(graphs) % 10 == 0:
                print(f"  Generated {len(graphs)}/{LOBSTER_NUM_GRAPHS} lobster graphs...")
        seed_value += 1

    print(f"Created {len(graphs)} lobster graphs")
    print(
        f"  Node range: {min(graph.number_of_nodes() for graph in graphs)} "
        f"to {max(graph.number_of_nodes() for graph in graphs)}"
    )
    print(
        f"  Edge range: {min(graph.number_of_edges() for graph in graphs)} "
        f"to {max(graph.number_of_edges() for graph in graphs)}"
    )
    return graphs


def add_edge_rows(
    edge_rows: list[tuple[int, int, int]],
    source_node_id: int,
    target_node_id: int,
    edge_type: int,
    directed: bool,
) -> int:
    if directed:
        edge_rows.append((source_node_id, target_node_id, edge_type))
        edge_rows.append((target_node_id, source_node_id, edge_type))
        return 2

    src = min(source_node_id, target_node_id)
    dst = max(source_node_id, target_node_id)
    edge_rows.append((src, dst, edge_type))
    return 1

def create_lobster_database_with_features(
    db_name: str,
    graphs: list[nx.Graph],
    directed: bool,
) -> None:
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (LOBSTER WITH ROTATION-INVARIANT FEATURES)")
    print("=" * 70)

    connection = connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
    cursor = connection.cursor()
    cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
    cursor.execute("SET sql_mode='STRICT_TRANS_TABLES';")

    cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    cursor.execute(f"CREATE DATABASE `{db_name}`")
    cursor.execute(f"USE `{db_name}`")
    print(f"Database '{db_name}' created")

    cursor.execute(
        """
        CREATE TABLE nodes (
            node_id INT PRIMARY KEY,
            node_degree INT NOT NULL,
            distance_to_spine INT NOT NULL,
            subtree_size INT NOT NULL,
            eccentricity INT NOT NULL,
            INDEX idx_degree (node_degree),
            INDEX idx_spine (distance_to_spine),
            INDEX idx_subtree (subtree_size),
            INDEX idx_eccentricity (eccentricity)
        )
        """
    )
    print("\nNODES table created")
    print("  - node_degree: INT (1=Leaf, 2=Branch, 3=Hub, 4=SuperHub)")
    print("  - distance_to_spine: INT (1=On-Spine, 2=Near-Spine, 3=Mid-Spine, 4=Far-Spine)")
    print("  - subtree_size: INT (1=1-5, 2=6-20, 3=21-40, 4=41+)")
    print("  - eccentricity: INT (1=1-5, 2=6-10, 3=11-15, 4=16+)")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            edge_type INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
            INDEX idx_edge_type (edge_type)
        )
        """
    )
    print("\nEDGES table created")
    print("  - edge_type: INT (1=Spine-Edge, 2=Branch-Edge, 3=Leaf-Edge)")
    print(
        f"  - edge mode: {'DIRECTED (A->B and B->A)' if directed else 'UNDIRECTED (one stored edge per pair)'}"
    )

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    degree_counts = defaultdict(int)
    spine_counts = defaultdict(int)
    subtree_counts = defaultdict(int)
    eccentricity_counts = defaultdict(int)
    edge_type_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 10 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        spine_path = find_spine_path(graph)
        spine_nodes = set(spine_path)
        distance_labels = compute_distance_to_spine_labels(graph, spine_path)
        subtree_sizes = compute_branch_component_sizes(graph, spine_path)

        local_to_global: dict[int, int] = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id

            node_degree = compute_node_degree(graph, node)
            distance_to_spine = distance_labels[node]
            subtree_size = subtree_sizes[node]
            eccentricity = compute_eccentricity(graph, node)

            degree_counts[node_degree] += 1
            spine_counts[distance_to_spine] += 1
            subtree_counts[subtree_size] += 1
            eccentricity_counts[eccentricity] += 1
            node_rows.append(
                (
                    global_id,
                    node_degree,
                    distance_to_spine,
                    subtree_size,
                    eccentricity,
                )
            )
            global_node_id += 1

        cursor.executemany(
            """
            INSERT INTO nodes (node_id, node_degree, distance_to_spine, subtree_size, eccentricity)
            VALUES (%s, %s, %s, %s, %s)
            """,
            node_rows,
        )

        edge_rows = []
        for source_node, target_node in graph.edges():
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_type = compute_edge_type(source_node, target_node, spine_nodes)
            inserted_rows = add_edge_rows(
                edge_rows,
                source_node_id,
                target_node_id,
                edge_type,
                directed,
            )
            edge_type_counts[edge_type] += inserted_rows

        cursor.executemany(
            """
            INSERT INTO edges (source_node_id, target_node_id, edge_type)
            VALUES (%s, %s, %s)
            """,
            edge_rows,
        )

        if (graph_id + 1) % 10 == 0:
            connection.commit()

    connection.commit()

    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATABASE STATISTICS - LOBSTER FEATURES")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]

    stored_degree = edge_count / node_count if directed else (2 * edge_count) / node_count

    print("\nDATASET SUMMARY: Lobster Tree (Rotation-Invariant)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    print(f"  Average stored degree: {stored_degree:.2f}")

    print("\nNODE FEATURE 1: NODE_DEGREE (Degree-Based)")
    print("  " + "=" * 70)
    degree_labels = {
        1: "Degree 1",
        2: "Degree 2-3",
        3: "Degree 4-5",
        4: "Degree 6+",
    }
    cumulative = 0.0
    for value in sorted(degree_counts):
        count = degree_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = degree_labels.get(value, "")
        name = NODE_DEGREE_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:8s}, {label:10s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 2: DISTANCE_TO_SPINE (Spine Proximity)")
    print("  " + "=" * 70)
    spine_labels = {
        1: "Directly on spine path",
        2: "1 step from spine",
        3: "2-3 steps from spine",
        4: "4+ steps from spine",
    }
    cumulative = 0.0
    for value in sorted(spine_counts):
        count = spine_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = spine_labels.get(value, "")
        name = DISTANCE_TO_SPINE_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:11s}, {label:24s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 3: SUBTREE_SIZE (Branch Component Size)")
    print("  " + "=" * 70)
    subtree_labels = {
        1: "Small local branch",
        2: "Medium local branch",
        3: "Large local branch",
        4: "Very large local branch",
    }
    cumulative = 0.0
    for value in sorted(subtree_counts):
        count = subtree_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        name = SUBTREE_SIZE_BUCKET_LABELS.get(value, str(value))
        description = subtree_labels.get(value, "")
        print(
            f"  {value:2d} ({name:5s}, {description:24s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 4: ECCENTRICITY (Tree Position)")
    print("  " + "=" * 70)
    eccentricity_labels = {
        1: "Peripheral node",
        2: "Near-center node",
        3: "Center node",
        4: "Very central node",
    }
    cumulative = 0.0
    for value in sorted(eccentricity_counts):
        count = eccentricity_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        name = ECCENTRICITY_BUCKET_LABELS.get(value, str(value))
        description = eccentricity_labels.get(value, "")
        print(
            f"  {value:2d} ({name:5s}, {description:18s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nEDGE FEATURE 1: EDGE_TYPE (Edge Position)")
    print("  " + "=" * 70)
    cumulative = 0.0
    for value in sorted(edge_type_counts):
        count = edge_type_counts[value]
        pct = (count / edge_count) * 100 if edge_count > 0 else 0.0
        cumulative += pct
        name = EDGE_TYPE_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:11s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {edge_count:,} edges (100.00%)")

    print("\nSAMPLE DATA")
    print("=" * 70)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | node_degree | distance_to_spine | subtree_size | eccentricity")
    print("  " + "-" * 80)
    for row in cursor.fetchall():
        degree_name = NODE_DEGREE_LABELS.get(row[1], str(row[1]))
        distance_name = DISTANCE_TO_SPINE_LABELS.get(row[2], str(row[2]))
        subtree_name = SUBTREE_SIZE_BUCKET_LABELS.get(row[3], str(row[3]))
        eccentricity_name = ECCENTRICITY_BUCKET_LABELS.get(row[4], str(row[4]))
        print(
            f"  {row[0]:7d} | {row[1]:2d} ({degree_name:8s}) | {row[2]:2d} ({distance_name:11s}) | "
            f"{row[3]:2d} ({subtree_name:5s}) | {row[4]:2d} ({eccentricity_name:5s})"
        )

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_type")
    print("  " + "-" * 42)
    for row in cursor.fetchall():
        edge_name = EDGE_TYPE_LABELS.get(row[2], str(row[2]))
        print(f"  {row[0]:6d} | {row[1]:6d} | {row[2]:2d} ({edge_name:11s})")

    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


def create_database_no_features(
    db_name: str,
    graphs: list[nx.Graph],
    graph_type_name: str,
    directed: bool,
) -> None:
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} ({graph_type_name} STRUCTURE ONLY)")
    print("=" * 70)

    connection = connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
    cursor = connection.cursor()
    cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
    cursor.execute("SET sql_mode='STRICT_TRANS_TABLES';")

    cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    cursor.execute(f"CREATE DATABASE `{db_name}`")
    cursor.execute(f"USE `{db_name}`")
    print(f"Database '{db_name}' created")

    cursor.execute("CREATE TABLE nodes (node_id INT PRIMARY KEY)")
    print("\nNODES table created (no features)")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id)
        )
        """
    )
    print("\nEDGES table created (no features)")

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    for graph_id, graph in enumerate(graphs):
        if graph_id % 10 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        local_to_global: dict[int, int] = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id
            node_rows.append((global_id,))
            global_node_id += 1

        cursor.executemany("INSERT INTO nodes (node_id) VALUES (%s)", node_rows)

        edge_rows = []
        for source_node, target_node in graph.edges():
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            if directed:
                edge_rows.append((source_node_id, target_node_id))
                edge_rows.append((target_node_id, source_node_id))
            else:
                src = min(source_node_id, target_node_id)
                dst = max(source_node_id, target_node_id)
                edge_rows.append((src, dst))

        cursor.executemany(
            "INSERT INTO edges (source_node_id, target_node_id) VALUES (%s, %s)",
            edge_rows,
        )

        if (graph_id + 1) % 10 == 0:
            connection.commit()

    connection.commit()

    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]

    print(f"\nDATASET SUMMARY: {graph_type_name} (structure only)")
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")

    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("LOBSTER DATASET GENERATOR (v1.2)")
    print("=" * 70)
    print("Supports 2 LOBSTER schema modes:")
    print("  1. with-features    - node_degree, distance_to_spine, subtree_size, eccentricity, edge_type")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    directed = resolve_edge_mode(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    db_name = args.db_name if args.db_name else input("Enter the database name: ").strip()
    if not db_name:
        db_name = default_db_name_for_mode(args.feature_mode)

    lobster_graphs = build_lobster_graphs()
    if args.feature_mode == "with-features":
        create_lobster_database_with_features(db_name, lobster_graphs, directed)
    else:
        create_database_no_features(db_name, lobster_graphs, "Lobster Tree", directed)

    print("\n" + "=" * 70)
    print("ALL DATABASES CREATED SUCCESSFULLY!")
    print("=" * 70)
    if args.feature_mode == "with-features":
        print(f"  1. {db_name} (4 node + 1 edge features) [LOBSTER]")
        print("\nLOBSTER FEATURES (Rotation-Invariant v1.2):")
        print("  Node features:")
        print("    1. node_degree        (1=Leaf, 2=Branch, 3=Hub, 4=SuperHub)")
        print("    2. distance_to_spine  (1=On-Spine, 2=Near-Spine, 3=Mid-Spine, 4=Far-Spine)")
        print("    3. subtree_size       (1=1-5, 2=6-20, 3=21-40, 4=41+)")
        print("    4. eccentricity       (1=1-5, 2=6-10, 3=11-15, 4=16+)")
        print("  Edge features:")
        print("    1. edge_type          (1=Spine-Edge, 2=Branch-Edge, 3=Leaf-Edge)")
    else:
        print(f"  1. {db_name} (structure only, no features) [LOBSTER]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
