#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangular grid graphs to MySQL database converter for FactorBase.

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

from dataset_feature_utils.triangular_grid_features import (
    DISTANCE_TO_BOUNDARY_LABELS,
    EDGE_ORBIT_LABELS,
    NUM_6CYCLES_LABELS,
    STRUCT_TYPE_LABELS,
    compute_distance_to_boundary,
    compute_edge_orbit,
    compute_num_3cycles,
    compute_num_6cycles,
    compute_struct_type,
    decode_num_3cycles,
    get_lattice_bounds,
)


DEFAULT_DB_NAME = "triangular_grid"
DEFAULT_FEATURE_MODE = "with-features"
DB_HOST = "127.0.0.1"
DB_USER = "fbuser"
DB_PASSWORD = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated TRIANGULAR_GRID graphs into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the TRIANGULAR_GRID schema with or without node/edge features",
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


def build_triangular_grid_graphs() -> list[nx.Graph]:
    print("\n" + "=" * 70)
    print("GENERATING TRIANGULAR GRID GRAPHS")
    print("=" * 70)

    graphs: list[nx.Graph] = []
    for width in range(10, 20):
        for height in range(10, 20):
            graph = nx.triangular_lattice_graph(width, height)
            graphs.append(graph)
            if len(graphs) % 20 == 0:
                print(f"  Generated {len(graphs)}/100 triangular grid graphs...")

    print(f"Created {len(graphs)} triangular grid graphs")
    return graphs


def add_edge_rows(
    edge_rows: list[tuple[int, int, int]],
    source_node_id: int,
    target_node_id: int,
    edge_orbit: int,
    directed: bool,
) -> int:
    if directed:
        edge_rows.append((source_node_id, target_node_id, edge_orbit))
        edge_rows.append((target_node_id, source_node_id, edge_orbit))
        return 2

    src = min(source_node_id, target_node_id)
    dst = max(source_node_id, target_node_id)
    edge_rows.append((src, dst, edge_orbit))
    return 1


def create_triangular_grid_database_with_features(
    db_name: str,
    graphs: list[nx.Graph],
    directed: bool,
) -> None:
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (TRIANGULAR GRID WITH ROTATION-INVARIANT FEATURES)")
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
            struct_type INT NOT NULL,
            distance_to_boundary INT NOT NULL,
            num_3cycles INT NOT NULL,
            num_6cycles INT NOT NULL,
            INDEX idx_struct (struct_type),
            INDEX idx_distance (distance_to_boundary),
            INDEX idx_3cycles (num_3cycles),
            INDEX idx_6cycles (num_6cycles)
        )
        """
    )
    print("\nNODES table created")
    print("  - struct_type: INT (1=Vertex, 2=Boundary, 3=Edge-Corner, 4=Edge-Transition, 5=Interior)")
    print("  - distance_to_boundary: INT (1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center)")
    print("  - num_3cycles: INT (1-based categorical triangle count)")
    print("  - num_6cycles: INT (1=No hexagon, 2=Has hexagon)")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            edge_orbit INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
            INDEX idx_edge_orbit (edge_orbit)
        )
        """
    )
    print("\nEDGES table created")
    print("  - edge_orbit: INT (1=Boundary, 2=Interior)")
    print(
        f"  - edge mode: {'DIRECTED (A->B and B->A)' if directed else 'UNDIRECTED (one stored edge per pair)'}"
    )

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    struct_counts = defaultdict(int)
    distance_counts = defaultdict(int)
    cycle3_counts = defaultdict(int)
    cycle6_counts = defaultdict(int)
    edge_orbit_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        bounds = get_lattice_bounds(graph)

        local_to_global: dict[tuple[int, int], int] = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id

            struct_type = compute_struct_type(graph, node)
            distance_to_boundary = compute_distance_to_boundary(node, bounds)
            num_3cycles = compute_num_3cycles(graph, node)
            num_6cycles = compute_num_6cycles(graph, node)

            struct_counts[struct_type] += 1
            distance_counts[distance_to_boundary] += 1
            cycle3_counts[num_3cycles] += 1
            cycle6_counts[num_6cycles] += 1
            node_rows.append(
                (
                    global_id,
                    struct_type,
                    distance_to_boundary,
                    num_3cycles,
                    num_6cycles,
                )
            )
            global_node_id += 1

        cursor.executemany(
            """
            INSERT INTO nodes (node_id, struct_type, distance_to_boundary, num_3cycles, num_6cycles)
            VALUES (%s, %s, %s, %s, %s)
            """,
            node_rows,
        )

        edge_rows = []
        for source_node, target_node in graph.edges():
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_orbit = compute_edge_orbit(source_node, target_node, bounds)
            inserted_rows = add_edge_rows(
                edge_rows,
                source_node_id,
                target_node_id,
                edge_orbit,
                directed,
            )
            edge_orbit_counts[edge_orbit] += inserted_rows

        cursor.executemany(
            """
            INSERT INTO edges (source_node_id, target_node_id, edge_orbit)
            VALUES (%s, %s, %s)
            """,
            edge_rows,
        )

        if (graph_id + 1) % 20 == 0:
            connection.commit()

    connection.commit()

    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATABASE STATISTICS - TRIANGULAR GRID FEATURES")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]

    stored_degree = edge_count / node_count if directed else (2 * edge_count) / node_count

    print("\nDATASET SUMMARY: Triangular Grid (Rotation-Invariant)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    print(f"  Average stored degree: {stored_degree:.2f}")

    print("\nNODE FEATURE 1: STRUCT_TYPE (Degree-Based)")
    print("  " + "=" * 70)
    struct_labels = {
        1: "Degree 2",
        2: "Degree 3",
        3: "Degree 4",
        4: "Degree 5",
        5: "Degree 6",
    }
    cumulative = 0.0
    for value in sorted(struct_counts):
        count = struct_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = struct_labels.get(value, "")
        name = STRUCT_TYPE_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:15s}, {label:10s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 2: DISTANCE_TO_BOUNDARY (Position Depth)")
    print("  " + "=" * 70)
    distance_labels = {
        1: "On lattice boundary",
        2: "1 step from boundary",
        3: "2-3 steps from boundary",
        4: "4-5 steps from boundary",
        5: "6+ steps from boundary",
    }
    cumulative = 0.0
    for value in sorted(distance_counts):
        count = distance_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = distance_labels.get(value, "")
        name = DISTANCE_TO_BOUNDARY_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:15s}, {label:25s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 3: NUM_3CYCLES (Triangle Participation)")
    print("  " + "=" * 70)
    cumulative = 0.0
    for value in sorted(cycle3_counts):
        count = cycle3_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        print(
            f"  Value {value:2d} ({decode_num_3cycles(value):2d} triangles): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 4: NUM_6CYCLES (Hexagon Participation)")
    print("  " + "=" * 70)
    cumulative = 0.0
    for value in sorted(cycle6_counts):
        count = cycle6_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = NUM_6CYCLES_LABELS.get(value, "")
        print(
            f"  Value {value:2d} ({label:12s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nEDGE FEATURE 1: EDGE_ORBIT (Position-Based)")
    print("  " + "=" * 70)
    cumulative = 0.0
    for value in sorted(edge_orbit_counts):
        count = edge_orbit_counts[value]
        pct = (count / edge_count) * 100 if edge_count > 0 else 0.0
        cumulative += pct
        name = EDGE_ORBIT_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:8s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {edge_count:,} edges (100.00%)")

    print("\nSAMPLE DATA")
    print("=" * 70)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | struct_type | distance_to_boundary | num_3cycles | num_6cycles")
    print("  " + "-" * 84)
    for row in cursor.fetchall():
        struct_name = STRUCT_TYPE_LABELS.get(row[1], str(row[1]))
        distance_name = DISTANCE_TO_BOUNDARY_LABELS.get(row[2], str(row[2]))
        cycle6_name = NUM_6CYCLES_LABELS.get(row[4], str(row[4]))
        print(
            f"  {row[0]:7d} | {row[1]:2d} ({struct_name:15s}) | "
            f"{row[2]:2d} ({distance_name:15s}) | "
            f"{row[3]:2d} ({decode_num_3cycles(row[3]):2d} tri) | "
            f"{row[4]:2d} ({cycle6_name:10s})"
        )

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_orbit")
    print("  " + "-" * 42)
    for row in cursor.fetchall():
        edge_name = EDGE_ORBIT_LABELS.get(row[2], str(row[2]))
        print(f"  {row[0]:6d} | {row[1]:6d} | {row[2]:2d} ({edge_name:8s})")

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
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        local_to_global: dict[tuple[int, int], int] = {}
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

        if (graph_id + 1) % 20 == 0:
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
    print("TRIANGULAR GRID DATASET GENERATOR (v1.2)")
    print("=" * 70)
    print("Supports 2 TRIANGULAR_GRID schema modes:")
    print("  1. with-features    - struct_type, distance_to_boundary, num_3cycles, num_6cycles, edge_orbit")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    directed = resolve_edge_mode(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    db_name = args.db_name if args.db_name else input("Enter the database name: ").strip()
    if not db_name:
        db_name = default_db_name_for_mode(args.feature_mode)

    triangular_graphs = build_triangular_grid_graphs()
    if args.feature_mode == "with-features":
        create_triangular_grid_database_with_features(db_name, triangular_graphs, directed)
    else:
        create_database_no_features(db_name, triangular_graphs, "Triangular Grid", directed)

    print("\n" + "=" * 70)
    print("ALL DATABASES CREATED SUCCESSFULLY!")
    print("=" * 70)
    if args.feature_mode == "with-features":
        print(f"  1. {db_name} (4 node + 1 edge features) [TRIANGULAR_GRID]")
        print("     struct_type: 1=Vertex, 2=Boundary, 3=Edge-Corner, 4=Edge-Transition, 5=Interior")
        print("     distance_to_boundary: 1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center")
        print("     num_3cycles: 1-based categorical triangle count")
        print("     num_6cycles: 1=No hexagon, 2=Has hexagon")
        print("     edge_orbit: 1=Boundary, 2=Interior")
    else:
        print(f"  1. {db_name} (structure only, no features) [TRIANGULAR_GRID]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
