#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lobster graphs to MySQL database converter for FactorBase.

This script matches the CLI contract used by `run_factorbase_pipeline.py`:
- `--db-name` selects the MySQL database name to create
- `--undirected` stores both directions for each source edge
- `--directed` is rejected because it mismatches main.py's symmetric adjacency
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
DEFAULT_EDGE_MODE = "undirected"
DB_HOST = "127.0.0.1"
DB_USER = "fbuser"
DB_PASSWORD = ""

# Edge-mode semantics:
# QM9/PROTEINS --directed:
#   source gives A->B and B->A
#   DB stores both
# Synthetic --directed:
#   source gives A-B once
#   DB stores one row
SYNTHETIC_DIRECTED_MISMATCH_ERROR = (
    "Synthetic datasets cannot be imported with --directed: using --directed, "
    "DB will store only one row per undirected edge, while main.py uses symmetric "
    "adjacency. That would be a mismatch. Use --undirected."
)

EDGE_MODE_LABELS = {
    "directed": "DIRECTED (preserve source NetworkX edge rows)",
    "undirected": "UNDIRECTED (A->B and B->A for each source edge)",
}

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
    parser.add_argument(
        "--debug-edges",
        action="store_true",
        help="Print source-to-database edge mappings for the first few graphs",
    )
    parser.add_argument(
        "--debug-all-edges",
        action="store_true",
        help="Print every source-to-database edge mapping for every generated graph",
    )
    parser.add_argument(
        "--debug-graph-limit",
        type=int,
        default=2,
        help="When --debug-edges is set, print edge mappings for this many graphs",
    )
    parser.add_argument(
        "--debug-edge-limit",
        type=int,
        default=20,
        help="When --debug-edges is set, print this many source edges per debugged graph",
    )

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Rejected for synthetic datasets because main.py uses symmetric adjacency",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store both directions for each NetworkX edge",
    )
    args = parser.parse_args()
    if not args.directed and not args.undirected:
        if DEFAULT_EDGE_MODE == "directed":
            args.directed = True
        elif DEFAULT_EDGE_MODE == "undirected":
            args.undirected = True
    return args


def default_db_name_for_mode(feature_mode: str) -> str:
    if feature_mode == "without-features":
        return f"{DEFAULT_DB_NAME}_no_feature"
    return DEFAULT_DB_NAME


def resolve_edge_mode(args: argparse.Namespace) -> str:
    print("=" * 60)
    print("GRAPH DIRECTION CONFIGURATION")
    print("=" * 60)
    if args.directed:
        raise ValueError(SYNTHETIC_DIRECTED_MISMATCH_ERROR)
    elif args.undirected:
        print("Selected: UNDIRECTED (store both directions)\n")
        return "undirected"

    raise ValueError("No edge mode selected")


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
    seen_edges: set[tuple[int, int]],
    source_node_id: int,
    target_node_id: int,
    edge_type: int,
    edge_mode: str,
) -> int:
    edge_candidates = [(source_node_id, target_node_id)]
    if edge_mode == "undirected":
        edge_candidates.append((target_node_id, source_node_id))

    inserted_rows = 0
    for source_id, target_id in edge_candidates:
        edge_key = (source_id, target_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append((source_id, target_id, edge_type))
        inserted_rows += 1
    return inserted_rows


def add_plain_edge_rows(
    edge_rows: list[tuple[int, int]],
    seen_edges: set[tuple[int, int]],
    source_node_id: int,
    target_node_id: int,
    edge_mode: str,
) -> None:
    edge_candidates = [(source_node_id, target_node_id)]
    if edge_mode == "undirected":
        edge_candidates.append((target_node_id, source_node_id))

    for edge_key in edge_candidates:
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append(edge_key)


def should_debug_graph(debug_edges: bool, graph_id: int, debug_graph_limit: int | None) -> bool:
    return debug_edges and (debug_graph_limit is None or graph_id < debug_graph_limit)


def should_debug_edge(edge_index: int, debug_edge_limit: int | None) -> bool:
    return debug_edge_limit is None or edge_index < debug_edge_limit


def print_edge_debug(
    graph_id: int,
    edge_index: int,
    source_node,
    target_node,
    source_node_id: int,
    target_node_id: int,
    inserted_rows: int,
    edge_rows,
) -> None:
    emitted_rows = edge_rows[-inserted_rows:] if inserted_rows else []
    print(
        f"[DEBUG edges] graph={graph_id} edge={edge_index} "
        f"source_local={source_node}->{target_node} source_global={source_node_id}->{target_node_id} "
        f"emitted_rows={emitted_rows}"
    )


def expected_edge_rows(source_edge_count: int, edge_mode: str) -> int:
    if edge_mode == "directed":
        return source_edge_count
    return source_edge_count * 2


def edge_rule_for_mode(edge_mode: str) -> str:
    if edge_mode == "directed":
        return "preserve each source NetworkX edge row"
    return "every NetworkX edge creates A->B and B->A"


def print_graph_edge_check(
    graph_id: int,
    source_edge_count: int,
    inserted_edge_count: int,
    edge_mode: str,
) -> None:
    expected_rows = expected_edge_rows(source_edge_count, edge_mode)
    status = "OK" if inserted_edge_count == expected_rows else "MISMATCH"
    print(
        f"[CHECK graph] graph={graph_id} source_edges={source_edge_count} "
        f"expected_db_rows={expected_rows} actual_db_rows={inserted_edge_count} "
        f"status={status}"
    )


def compute_expected_dataset_counts(graphs: list[nx.Graph], edge_mode: str) -> tuple[int, int, int]:
    expected_nodes = sum(graph.number_of_nodes() for graph in graphs)
    expected_source_edges = sum(graph.number_of_edges() for graph in graphs)
    expected_db_edge_rows = expected_edge_rows(expected_source_edges, edge_mode)
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def print_expected_dataset_counts(graphs: list[nx.Graph], edge_mode: str) -> tuple[int, int, int]:
    expected_nodes, expected_source_edges, expected_db_edge_rows = (
        compute_expected_dataset_counts(graphs, edge_mode)
    )
    print("\n" + "=" * 70)
    print("EXPECTED LOBSTER DATABASE COUNTS")
    print("=" * 70)
    print(f"Expected graphs: {len(graphs):,}")
    print(f"Expected nodes: {expected_nodes:,}")
    print(f"Expected source NetworkX edges: {expected_source_edges:,}")
    print(f"Expected DB edge rows: {expected_db_edge_rows:,}")
    print(f"Expected DB edge rule: {edge_rule_for_mode(edge_mode)}")
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def analyze_source_edge_direction(graphs: list[nx.Graph]) -> dict:
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
            (source_node, target_node)
            for source_node, target_node in graph.edges()
            if source_node != target_node
        }
        if not edge_rows:
            continue

        missing_reverse_rows = sum(
            1 for source_node, target_node in edge_rows
            if (target_node, source_node) not in edge_rows
        )

        stats["graphs_with_edges"] += 1
        stats["source_edge_rows"] += len(edge_rows)
        stats["undirected_edge_pairs"] += len({frozenset(edge) for edge in edge_rows})
        stats["missing_reverse_rows"] += missing_reverse_rows
        if missing_reverse_rows:
            stats["graphs_with_missing_reverse"] += 1

    return stats


def print_source_edge_direction_analysis(dataset_name: str, stats: dict, edge_mode: str) -> None:
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
        print("Source edge rows are not bidirectional as exposed by NetworkX.")
        if edge_mode == "directed":
            print("NOTE: --directed will preserve one row per source NetworkX edge.")
        else:
            print("NOTE: --undirected will add reverse rows for each source NetworkX edge.")
    else:
        print("Source has no edge rows to analyze.")
    print()


def print_database_total_check(
    expected_node_count: int,
    expected_source_edge_count: int,
    expected_edge_row_count: int,
    actual_node_count: int,
    actual_edge_count: int,
) -> None:
    node_status = "OK" if actual_node_count == expected_node_count else "MISMATCH"
    edge_status = "OK" if actual_edge_count == expected_edge_row_count else "MISMATCH"
    print("\n" + "=" * 70)
    print("FINAL DATABASE COUNT CHECK")
    print("=" * 70)
    print(
        f"[CHECK total nodes] expected={expected_node_count:,} "
        f"actual={actual_node_count:,} status={node_status}"
    )
    print(
        f"[CHECK total source_edges] source_edges={expected_source_edge_count:,} "
        f"expected_db_rows={expected_edge_row_count:,}"
    )
    print(
        f"[CHECK total edges] expected={expected_edge_row_count:,} "
        f"actual={actual_edge_count:,} status={edge_status}"
    )


def verify_bidirectional_edges_with_features(cursor) -> None:
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL EDGE VERIFICATION")
    print("=" * 70)
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]
    cursor.execute(
        """
        SELECT COUNT(*) FROM edges e1
        WHERE EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
              AND e2.target_node_id = e1.source_node_id
              AND e2.edge_type = e1.edge_type
        )
        """
    )
    reverse_count = cursor.fetchone()[0]
    print(f"Edges with matching reverse row and edge_type: {reverse_count:,} / {edge_count:,}")
    print(f"Missing reverse rows: {edge_count - reverse_count:,}")
    cursor.execute(
        """
        SELECT e1.source_node_id, e1.target_node_id, e1.edge_type
        FROM edges e1
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
              AND e2.target_node_id = e1.source_node_id
              AND e2.edge_type = e1.edge_type
        )
        LIMIT 10
        """
    )
    missing_rows = cursor.fetchall()
    if missing_rows:
        print("Sample missing reverse rows:")
        for row in missing_rows:
            print(f"  source={row[0]} target={row[1]} edge_type={row[2]}")
    else:
        print("All feature edge rows have matching reverse rows.")


def verify_bidirectional_edges_plain(cursor) -> None:
    print("\n" + "=" * 70)
    print("BIDIRECTIONAL EDGE VERIFICATION")
    print("=" * 70)
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]
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
    reverse_count = cursor.fetchone()[0]
    print(f"Edges with matching reverse row: {reverse_count:,} / {edge_count:,}")
    print(f"Missing reverse rows: {edge_count - reverse_count:,}")
    cursor.execute(
        """
        SELECT e1.source_node_id, e1.target_node_id
        FROM edges e1
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
              AND e2.target_node_id = e1.source_node_id
        )
        LIMIT 10
        """
    )
    missing_rows = cursor.fetchall()
    if missing_rows:
        print("Sample missing reverse rows:")
        for row in missing_rows:
            print(f"  source={row[0]} target={row[1]}")
    else:
        print("All plain edge rows have matching reverse rows.")


def create_lobster_database_with_features(
    db_name: str,
    graphs: list[nx.Graph],
    edge_mode: str,
    debug_edges: bool = False,
    debug_graph_limit: int | None = 2,
    debug_edge_limit: int | None = 20,
) -> None:
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (LOBSTER WITH ROTATION-INVARIANT FEATURES)")
    print("=" * 70)
    expected_node_count, expected_source_edge_count, expected_edge_row_count = (
        print_expected_dataset_counts(graphs, edge_mode)
    )

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
        f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}"
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
        seen_edges: set[tuple[int, int]] = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        if debug_this_graph:
            print(
                f"\n[DEBUG edges] graph={graph_id} mode={edge_mode} "
                f"nodes={graph.number_of_nodes()} source_edges={graph.number_of_edges()}"
            )
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_type = compute_edge_type(source_node, target_node, spine_nodes)
            edge_count_before = len(edge_rows)
            inserted_rows = add_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_type,
                edge_mode,
            )
            edge_type_counts[edge_type] += inserted_rows
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                print_edge_debug(
                    graph_id,
                    edge_index,
                    source_node,
                    target_node,
                    source_node_id,
                    target_node_id,
                    len(edge_rows) - edge_count_before,
                    edge_rows,
                )

        if debug_this_graph:
            print(f"[DEBUG edges] graph={graph_id} total_db_edge_rows={len(edge_rows)}")
        print_graph_edge_check(graph_id, graph.number_of_edges(), len(edge_rows), edge_mode)

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
    print_database_total_check(
        expected_node_count,
        expected_source_edge_count,
        expected_edge_row_count,
        node_count,
        edge_count,
    )

    stored_degree = edge_count / node_count

    print("\nDATASET SUMMARY: Lobster Tree (Rotation-Invariant)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    print(f"  Average stored degree: {stored_degree:.2f}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_with_features(cursor)

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
    edge_mode: str,
    debug_edges: bool = False,
    debug_graph_limit: int | None = 2,
    debug_edge_limit: int | None = 20,
) -> None:
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} ({graph_type_name} STRUCTURE ONLY)")
    print("=" * 70)
    expected_node_count, expected_source_edge_count, expected_edge_row_count = (
        print_expected_dataset_counts(graphs, edge_mode)
    )

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
    print(f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}")

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
        seen_edges: set[tuple[int, int]] = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        if debug_this_graph:
            print(
                f"\n[DEBUG edges] graph={graph_id} mode={edge_mode} "
                f"nodes={graph.number_of_nodes()} source_edges={graph.number_of_edges()}"
            )
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_count_before = len(edge_rows)
            add_plain_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_mode,
            )
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                print_edge_debug(
                    graph_id,
                    edge_index,
                    source_node,
                    target_node,
                    source_node_id,
                    target_node_id,
                    len(edge_rows) - edge_count_before,
                    edge_rows,
                )

        if debug_this_graph:
            print(f"[DEBUG edges] graph={graph_id} total_db_edge_rows={len(edge_rows)}")
        print_graph_edge_check(graph_id, graph.number_of_edges(), len(edge_rows), edge_mode)

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
    print_database_total_check(
        expected_node_count,
        expected_source_edge_count,
        expected_edge_row_count,
        node_count,
        edge_count,
    )

    print(f"\nDATASET SUMMARY: {graph_type_name} (structure only)")
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_plain(cursor)

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

    edge_mode = resolve_edge_mode(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    db_name = args.db_name if args.db_name else input("Enter the database name: ").strip()
    if not db_name:
        db_name = default_db_name_for_mode(args.feature_mode)

    lobster_graphs = build_lobster_graphs()
    source_edge_stats = analyze_source_edge_direction(lobster_graphs)
    print_source_edge_direction_analysis("LOBSTER", source_edge_stats, edge_mode)

    debug_edges = args.debug_edges or args.debug_all_edges
    debug_graph_limit = None if args.debug_all_edges else args.debug_graph_limit
    debug_edge_limit = None if args.debug_all_edges else args.debug_edge_limit

    if args.feature_mode == "with-features":
        create_lobster_database_with_features(
            db_name,
            lobster_graphs,
            edge_mode,
            debug_edges,
            debug_graph_limit,
            debug_edge_limit,
        )
    else:
        create_database_no_features(
            db_name,
            lobster_graphs,
            "Lobster Tree",
            edge_mode,
            debug_edges,
            debug_graph_limit,
            debug_edge_limit,
        )

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
