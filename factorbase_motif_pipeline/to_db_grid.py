#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid graphs to MySQL database converter for FactorBase.

This script now matches the CLI contract used by `run_factorbase_pipeline.py`:
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

from dataset_feature_utils.grid_features import (
    DISTANCE_TO_BOUNDARY_LABELS,
    EDGE_ORBIT_LABELS,
    STRUCT_TYPE_LABELS,
    compute_distance_to_boundary,
    compute_edge_orbit,
    compute_struct_type,
    get_grid_dimensions,
)


DEFAULT_DB_NAME = "grid"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated GRID graphs into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the GRID schema with or without node/edge features",
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


def build_grid_graphs():
    print("\n" + "=" * 70)
    print("GENERATING SQUARE GRID GRAPHS")
    print("=" * 70)

    grid_graphs = []
    for width in range(10, 20):
        for height in range(10, 20):
            grid_graph = nx.grid_2d_graph(width, height)
            grid_graphs.append(grid_graph)
            if len(grid_graphs) % 20 == 0:
                print(f"  Generated {len(grid_graphs)}/100 grid graphs...")

    print(f"Created {len(grid_graphs)} square grid graphs")
    return grid_graphs


def add_edge_rows(edge_rows, seen_edges, source_node_id, target_node_id, edge_orbit, edge_mode):
    edge_candidates = [(source_node_id, target_node_id)]
    if edge_mode == "undirected":
        edge_candidates.append((target_node_id, source_node_id))

    inserted_rows = 0
    for source_id, target_id in edge_candidates:
        edge_key = (source_id, target_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append((source_id, target_id, edge_orbit))
        inserted_rows += 1
    return inserted_rows


def add_plain_edge_rows(edge_rows, seen_edges, source_node_id, target_node_id, edge_mode):
    edge_candidates = [(source_node_id, target_node_id)]
    if edge_mode == "undirected":
        edge_candidates.append((target_node_id, source_node_id))

    for edge_key in edge_candidates:
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append(edge_key)


def should_debug_graph(debug_edges, graph_id, debug_graph_limit):
    return debug_edges and (debug_graph_limit is None or graph_id < debug_graph_limit)


def should_debug_edge(edge_index, debug_edge_limit):
    if debug_edge_limit is None:
        return True
    return edge_index < debug_edge_limit


def print_feature_edge_debug(
    graph_id,
    edge_index,
    node_u,
    node_v,
    source_node_id,
    target_node_id,
    inserted_rows,
    edge_rows,
):
    emitted_rows = edge_rows[-inserted_rows:] if inserted_rows else []
    print(
        f"[DEBUG edges] graph={graph_id} edge={edge_index} "
        f"source_local={node_u}->{node_v} source_global={source_node_id}->{target_node_id} "
        f"emitted_rows={emitted_rows}"
    )


def print_plain_edge_debug(
    graph_id,
    edge_index,
    node_u,
    node_v,
    source_node_id,
    target_node_id,
    inserted_rows,
    edge_rows,
):
    emitted_rows = edge_rows[-inserted_rows:] if inserted_rows else []
    print(
        f"[DEBUG edges] graph={graph_id} edge={edge_index} "
        f"source_local={node_u}->{node_v} source_global={source_node_id}->{target_node_id} "
        f"emitted_rows={emitted_rows}"
    )


def expected_edge_rows_for_mode(source_edge_count, edge_mode):
    if edge_mode == "directed":
        return source_edge_count
    return source_edge_count * 2


def edge_rule_for_mode(edge_mode):
    if edge_mode == "directed":
        return "preserve each source NetworkX edge row"
    return "every NetworkX edge creates A->B and B->A"


def analyze_source_edge_direction(graphs):
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


def print_source_edge_direction_analysis(dataset_name, stats, edge_mode):
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


def print_graph_edge_check(graph_id, source_edge_count, inserted_edge_count, edge_mode):
    expected_edge_rows = expected_edge_rows_for_mode(source_edge_count, edge_mode)
    status = "OK" if inserted_edge_count == expected_edge_rows else "MISMATCH"
    print(
        f"[CHECK graph] graph={graph_id} source_edges={source_edge_count} "
        f"expected_db_rows={expected_edge_rows} actual_db_rows={inserted_edge_count} "
        f"status={status}"
    )


def compute_expected_dataset_counts(graphs, edge_mode):
    expected_nodes = sum(graph.number_of_nodes() for graph in graphs)
    expected_source_edges = sum(graph.number_of_edges() for graph in graphs)
    expected_db_edge_rows = expected_edge_rows_for_mode(expected_source_edges, edge_mode)
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def print_expected_dataset_counts(graphs, edge_mode):
    expected_nodes, expected_source_edges, expected_db_edge_rows = (
        compute_expected_dataset_counts(graphs, edge_mode)
    )
    print("\n" + "=" * 70)
    print("EXPECTED GRID DATABASE COUNTS")
    print("=" * 70)
    print(f"Expected graphs: {len(graphs):,}")
    print(f"Expected nodes: {expected_nodes:,}")
    print(f"Expected source NetworkX edges: {expected_source_edges:,}")
    print(f"Expected DB edge rows: {expected_db_edge_rows:,}")
    print(f"Expected DB edge rule: {edge_rule_for_mode(edge_mode)}")
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def print_database_total_check(
    expected_node_count,
    expected_source_edge_count,
    expected_edge_row_count,
    actual_node_count,
    actual_edge_count,
):
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


def verify_bidirectional_edges_with_features(cursor):
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
              AND e2.edge_orbit = e1.edge_orbit
        )
        """
    )
    reverse_count = cursor.fetchone()[0]
    print(f"Edges with matching reverse row and edge_orbit: {reverse_count:,} / {edge_count:,}")
    print(f"Missing reverse rows: {edge_count - reverse_count:,}")

    cursor.execute(
        """
        SELECT e1.source_node_id, e1.target_node_id, e1.edge_orbit
        FROM edges e1
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
              AND e2.target_node_id = e1.source_node_id
              AND e2.edge_orbit = e1.edge_orbit
        )
        LIMIT 10
        """
    )
    missing_rows = cursor.fetchall()
    if missing_rows:
        print("Sample missing reverse rows:")
        for row in missing_rows:
            print(f"  source={row[0]} target={row[1]} edge_orbit={row[2]}")
    else:
        print("All feature edge rows have matching reverse rows.")

    cursor.execute(
        """
        SELECT e1.source_node_id, e1.target_node_id, e1.edge_orbit,
               e2.source_node_id, e2.target_node_id, e2.edge_orbit
        FROM edges e1
        JOIN edges e2
          ON e2.source_node_id = e1.target_node_id
         AND e2.target_node_id = e1.source_node_id
         AND e2.edge_orbit = e1.edge_orbit
        WHERE e1.source_node_id < e1.target_node_id
        LIMIT 10
        """
    )
    print("\nSample edge/reverse pairs:")
    for row in cursor.fetchall():
        print(
            f"  ({row[0]} -> {row[1]}, edge_orbit={row[2]}) "
            f"<-> ({row[3]} -> {row[4]}, edge_orbit={row[5]})"
        )


def verify_bidirectional_edges_plain(cursor):
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

    cursor.execute(
        """
        SELECT e1.source_node_id, e1.target_node_id,
               e2.source_node_id, e2.target_node_id
        FROM edges e1
        JOIN edges e2
          ON e2.source_node_id = e1.target_node_id
         AND e2.target_node_id = e1.source_node_id
        WHERE e1.source_node_id < e1.target_node_id
        LIMIT 10
        """
    )
    print("\nSample edge/reverse pairs:")
    for row in cursor.fetchall():
        print(f"  ({row[0]} -> {row[1]}) <-> ({row[2]} -> {row[3]})")


# ============================================
# DATABASE CREATION - SQUARE GRID WITH FEATURES
# ============================================
def create_square_grid_database_with_features(
    db_name,
    graphs,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    """Create database for square grids with rotation-invariant features."""
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (GRID WITH ROTATION-INVARIANT FEATURES)")
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
            struct_type INT NOT NULL,
            distance_to_boundary INT NOT NULL,
            INDEX idx_struct (struct_type),
            INDEX idx_distance (distance_to_boundary)
        )
        """
    )
    print("\nNODES table created")
    print("  - struct_type: INT (1=Corner, 2=Edge, 3=Interior)")
    print("  - distance_to_boundary: INT (1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center)")

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
        f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}"
    )

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    struct_counts = defaultdict(int)
    distance_counts = defaultdict(int)
    edge_orbit_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        width, height = get_grid_dimensions(graph)
        grid_size = max(width, height)

        local_to_global = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id

            struct = compute_struct_type(graph, node)
            distance = compute_distance_to_boundary(node, grid_size)

            struct_counts[struct] += 1
            distance_counts[distance] += 1
            node_rows.append((global_id, struct, distance))
            global_node_id += 1

        cursor.executemany(
            """
            INSERT INTO nodes (node_id, struct_type, distance_to_boundary)
            VALUES (%s, %s, %s)
            """,
            node_rows,
        )

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        if debug_this_graph:
            print(
                f"\n[DEBUG edges] graph={graph_id} mode={edge_mode} "
                f"nodes={graph.number_of_nodes()} source_edges={graph.number_of_edges()} "
                f"grid_size={grid_size}"
            )

        for edge_index, (node_u, node_v) in enumerate(graph.edges()):
            source_node_id = local_to_global[node_u]
            target_node_id = local_to_global[node_v]
            edge_orbit = compute_edge_orbit(node_u, node_v, grid_size)
            edge_count_before = len(edge_rows)
            inserted_rows = add_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_orbit,
                edge_mode,
            )
            edge_orbit_counts[edge_orbit] += inserted_rows
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                print_feature_edge_debug(
                    graph_id,
                    edge_index,
                    node_u,
                    node_v,
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
            INSERT INTO edges (source_node_id, target_node_id, edge_orbit)
            VALUES (%s, %s, %s)
            """,
            edge_rows,
        )

        if (graph_id + 1) % 20 == 0:
            connection.commit()

    connection.commit()

    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATABASE STATISTICS - ROTATION-INVARIANT FEATURES")
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

    print("\nDATASET SUMMARY: Square Grid (Rotation-Invariant)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    print(f"  Average stored degree: {stored_degree:.2f}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_with_features(cursor)

    print("\nNODE FEATURE 1: STRUCT_TYPE (Degree-Based)")
    print("  " + "=" * 70)
    struct_labels = {
        1: "Degree 2",
        2: "Degree 3",
        3: "Degree 4",
    }
    cumulative = 0.0
    for value in sorted(struct_counts):
        count = struct_counts[value]
        pct = (count / node_count) * 100 if node_count > 0 else 0.0
        cumulative += pct
        label = struct_labels.get(value, "")
        name = STRUCT_TYPE_LABELS.get(value, str(value))
        print(
            f"  {value:2d} ({name:12s}, {label:10s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {node_count:,} nodes (100.00%)")

    print("\nNODE FEATURE 2: DISTANCE_TO_BOUNDARY (Position Depth)")
    print("  " + "=" * 70)
    distance_labels = {
        1: "On grid boundary",
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
            f"  {value:2d} ({name:14s}, {label:25s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
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
        print(f"  {value:2d} ({name:8s}): {count:8,} ({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]")
    print(f"  TOTAL: {edge_count:,} edges (100.00%)")

    print("\nFEATURE NOTE:")
    print("  - struct_type and distance_to_boundary have low correlation")
    print("  - struct_type captures degree (2/3/4), distance captures depth (position)")
    print("  - Use both for comprehensive structural motif detection")

    print("\n" + "=" * 70)
    print("SAMPLE DATA")
    print("=" * 70)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | struct_type | distance_to_boundary")
    print("  " + "-" * 55)
    for row in cursor.fetchall():
        struct_name = STRUCT_TYPE_LABELS.get(row[1], str(row[1]))
        dist_name = DISTANCE_TO_BOUNDARY_LABELS.get(row[2], str(row[2]))
        print(f"  {row[0]:7d} | {row[1]:2d} ({struct_name:8s}) | {row[2]:2d} ({dist_name:14s})")

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_orbit")
    print("  " + "-" * 40)
    for row in cursor.fetchall():
        edge_name = EDGE_ORBIT_LABELS.get(row[2], str(row[2]))
        print(f"  {row[0]:6d} | {row[1]:6d} | {row[2]:2d} ({edge_name:8s})")

    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


# ============================================
# DATABASE CREATION - NO FEATURES (Structure Only)
# ============================================
def create_database_no_features(
    db_name,
    graphs,
    graph_type_name,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    """Create database without features (structure only)."""
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

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0

    for graph_id, graph in enumerate(graphs):
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        node_list = sorted(graph.nodes())
        local_to_global = {}
        node_rows = []
        for node in node_list:
            global_id = global_node_id
            local_to_global[node] = global_id
            node_rows.append((global_id,))
            global_node_id += 1

        cursor.executemany("INSERT INTO nodes (node_id) VALUES (%s)", node_rows)

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        if debug_this_graph:
            print(
                f"\n[DEBUG edges] graph={graph_id} mode={edge_mode} "
                f"nodes={graph.number_of_nodes()} source_edges={graph.number_of_edges()}"
            )

        for edge_index, (node_u, node_v) in enumerate(graph.edges()):
            source_node_id = local_to_global[node_u]
            target_node_id = local_to_global[node_v]
            edge_count_before = len(edge_rows)
            add_plain_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_mode,
            )
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                print_plain_edge_debug(
                    graph_id,
                    edge_index,
                    node_u,
                    node_v,
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

        if (graph_id + 1) % 20 == 0:
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


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("GRID DATASET GENERATOR (v1.2)")
    print("=" * 70)
    print("Supports 2 GRID schema modes:")
    print("  1. with-features    - struct_type, distance_to_boundary, edge_orbit")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    edge_mode = resolve_edge_mode(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    db_name = args.db_name or default_db_name_for_mode(args.feature_mode)

    grid_graphs = build_grid_graphs()
    source_edge_stats = analyze_source_edge_direction(grid_graphs)
    print_source_edge_direction_analysis("GRID", source_edge_stats, edge_mode)

    debug_edges = args.debug_edges or args.debug_all_edges
    debug_graph_limit = None if args.debug_all_edges else args.debug_graph_limit
    debug_edge_limit = None if args.debug_all_edges else args.debug_edge_limit

    if args.feature_mode == "with-features":
        create_square_grid_database_with_features(
            db_name,
            grid_graphs,
            edge_mode,
            debug_edges,
            debug_graph_limit,
            debug_edge_limit,
        )
    else:
        create_database_no_features(
            db_name,
            grid_graphs,
            "Square Grid",
            edge_mode,
            debug_edges,
            debug_graph_limit,
            debug_edge_limit,
        )

    print("\n" + "=" * 70)
    print("ALL DATABASES CREATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nDATABASES CREATED:")
    if args.feature_mode == "with-features":
        print(f"  1. {db_name} (2 node + 1 edge features) [GRID]")
        print("\nGRID FEATURES (Rotation-Invariant v1.2):")
        print("  Node features:")
        print("    1. struct_type          (1=Corner, 2=Edge, 3=Interior) - degree-based")
        print(
            "    2. distance_to_boundary (1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center)"
        )
        print("  Edge features:")
        print("    1. edge_orbit           (1=Boundary, 2=Interior) - position-based")
    else:
        print(f"  1. {db_name} (structure only, no features) [GRID]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
