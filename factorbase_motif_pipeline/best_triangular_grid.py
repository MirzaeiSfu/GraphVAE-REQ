#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangular grid graphs to MySQL database converter for FactorBase, using the
evidence-grounded "optimal" motif-aware feature schema.

This schema was derived by inspecting (a) the FactorBase GES/BDeu learning
mechanism (ca.sfu.cs.factorbase.jbn.BayesNet_Learning_main.java: GesCT with
samplePrior=10.0, structurePrior=1.0) and (b) the actual learned Bayesian
network and empirical category counts for the existing `to_db_triangular_grid.py`
schema (`triangular_grid_undir_feat_snap_*` / `..._BN`).

Hard evidence that drove this schema (verified by direct SQL query against the
live `triangular_grid_undir_feat_snap_ce92ed` database):

    SELECT COUNT(DISTINCT struct_type), COUNT(DISTINCT num_3cycles),
           COUNT(DISTINCT num_6cycles),
           COUNT(DISTINCT CONCAT(struct_type,'-',num_3cycles,'-',num_6cycles))
    FROM nodes;
    -- 5, 5, 2, 5  <-- only 5 distinct combos for 3 "different" features

`struct_type`, `num_3cycles`, and `num_6cycles` in the old schema are a
deterministic 1:1 relabeling of node degree -- every struct_type value maps
to exactly one (num_3cycles, num_6cycles) pair. The learned BN
(Final_Path_BayesNets_view) confirms FactorBase spent real learned edges on
this redundancy (`num_3cycles(nodes0)->struct_type(nodes0)`,
`num_6cycles(nodes0)->struct_type(nodes0)`, ...) -- three "motifs" that are
really just degree relabeling degree, adding near-zero new gradient signal
to the GraphVAE motif loss beyond what the existing degree-histogram kernel
term already provides. Separately, `compute_num_6cycles` in the old schema
was also a `degree>=4` proxy, not a real 6-cycle/hexagon count.

This schema therefore:
- DROPS `struct_type` entirely (pure degree relabeling).
- Replaces the fake `num_6cycles` proxy with a REAL induced-6-cycle (hexagon)
  participation count (exhaustive cycle detection), for both nodes and edges.
- KEEPS `num_3cycles` (real triangle count -- ties to a motif already tracked
  in graph_statistics.py / eval/stats.py's 'TotalNumberOfTriangles').
- KEEPS `distance_to_boundary` from the old schema: unlike struct_type, it is
  NOT a function of local degree (same-degree interior nodes differ in true
  position), so it is the one genuinely non-redundant *positional* signal --
  dropping it (as the earlier new_tri.py experiment did) loses real
  information, so it is added back here.
- Adds `edge_direction` (lattice axis) and `edge_triangle_count` (literal
  edge-level triangle participation, 0/1/2) -- low cardinality, ties to a
  real, well-known motif, and not derivable from degree alone.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
import sys

import networkx as nx
from pymysql import connect

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DB_NAME = "triangular_grid_optimal"
DEFAULT_FEATURE_MODE = "with-features"
DEFAULT_EDGE_MODE = "undirected"
DB_HOST = "127.0.0.1"
DB_USER = "fbuser"
DB_PASSWORD = ""

SYNTHETIC_DIRECTED_WARNING = (
    "WARNING: --directed stores one row per undirected NetworkX edge. Use this "
    "only for FactorBase comparison databases; GraphVAE synthetic training uses "
    "symmetric adjacency and should normally use --undirected."
)

EDGE_MODE_LABELS = {
    "directed": "DIRECTED (preserve source NetworkX edge rows)",
    "undirected": "UNDIRECTED (A->B and B->A for each source edge)",
}

DISTANCE_TO_BOUNDARY_LABELS = {
    1: "Boundary",
    2: "Near-Boundary",
    3: "Near-Center",
    4: "Center",
    5: "Deep-Center",
}

EDGE_DIRECTION_LABELS = {
    1: "Horizontal",
    2: "Positive-60",
    3: "Negative-60",
}

EDGE_TRIANGLE_COUNT_LABELS = {
    0: "Zero-Triangles",
    1: "One-Triangle",
    2: "Two-Triangles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated TRIANGULAR_GRID graphs into MySQL with the optimal motif feature schema."
    )
    parser.add_argument(
        "--db-name",
        help="MySQL database base name (edge-mode suffix, e.g. '_dir'/'_undir', is appended automatically)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the TRIANGULAR_GRID schema with or without features",
    )
    parser.add_argument("--debug-edges", action="store_true")
    parser.add_argument("--debug-all-edges", action="store_true")
    parser.add_argument("--debug-graph-limit", type=int, default=2)
    parser.add_argument("--debug-edge-limit", type=int, default=20)

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
    edge_group.add_argument(
        "--both",
        action="store_true",
        help="Create both a directed and an undirected database in this run",
    )
    return parser.parse_args()


EDGE_MODE_ALIASES = {"directed": "dir", "undirected": "undir"}


def prompt_edge_modes(args: argparse.Namespace) -> list[str]:
    """
    Returns the edge mode(s) to build. Explicit CLI flags always win (this is
    the path run_factorbase_pipeline.py uses, so it never blocks on stdin).
    With no flags and an interactive terminal, asks the user. With no flags
    and no terminal (e.g. invoked as a subprocess), falls back to the
    previous default (undirected) instead of hanging on stdin.
    """
    if args.directed:
        return ["directed"]
    if args.undirected:
        return ["undirected"]
    if args.both:
        return ["directed", "undirected"]

    if not sys.stdin.isatty():
        print(
            "No --directed/--undirected/--both given and no interactive terminal "
            f"detected; defaulting to {DEFAULT_EDGE_MODE}.\n"
        )
        return [DEFAULT_EDGE_MODE]

    print("=" * 60)
    print("GRAPH DIRECTION CONFIGURATION")
    print("=" * 60)
    while True:
        choice = input(
            "Which edge mode(s) should be created?\n"
            "  1 - Directed only\n"
            "  2 - Undirected only (recommended: matches main.py's symmetric adjacency)\n"
            "  3 - Both directed and undirected\n"
            "Choice: "
        ).strip()
        if choice == "1":
            return ["directed"]
        if choice == "2":
            return ["undirected"]
        if choice == "3":
            return ["directed", "undirected"]
        print("Please enter 1, 2, or 3.")


def announce_edge_mode(edge_mode: str) -> None:
    print("=" * 60)
    print("GRAPH DIRECTION CONFIGURATION")
    print("=" * 60)
    if edge_mode == "directed":
        print("Selected: DIRECTED (preserve source NetworkX edge rows)")
        print(SYNTHETIC_DIRECTED_WARNING + "\n")
    else:
        print("Selected: UNDIRECTED (store both directions)\n")


def build_db_name(db_name_override: str | None, feature_mode: str, edge_mode: str) -> str:
    base = db_name_override or DEFAULT_DB_NAME
    name = f"{base}_{EDGE_MODE_ALIASES[edge_mode]}"
    if feature_mode == "without-features":
        name += "_no_feature"
    return name


def build_triangular_grid_graphs():
    print("\n" + "=" * 70)
    print("GENERATING TRIANGULAR GRID GRAPHS")
    print("=" * 70)

    graphs = []
    for width in range(10, 20):
        for height in range(10, 20):
            graphs.append(nx.triangular_lattice_graph(width, height))
            if len(graphs) % 20 == 0:
                print(f"  Generated {len(graphs)}/100 triangular grid graphs...")
    print(f"Created {len(graphs)} triangular grid graphs")
    return graphs


def get_lattice_bounds(graph):
    rows = [node[0] for node in graph.nodes()]
    cols = [node[1] for node in graph.nodes()]
    return min(rows), max(rows), min(cols), max(cols)


def compute_distance_to_boundary(node, bounds):
    row, col = node
    min_row, max_row, min_col, max_col = bounds
    distance = min(row - min_row, max_row - row, col - min_col, max_col - col)
    if distance == 0:
        return 1
    if distance == 1:
        return 2
    if distance <= 3:
        return 3
    if distance <= 5:
        return 4
    return 5


def get_node_position(graph, node):
    position = graph.nodes[node].get("pos")
    if position is not None:
        return position
    col, row = node
    return 0.5 * (row % 2) + col, (sqrt(3) / 2) * row


def canonical_cycle(cycle):
    rotations = []
    cycle = list(cycle)
    for sequence in (cycle, list(reversed(cycle))):
        for index in range(len(sequence)):
            rotations.append(tuple(sequence[index:] + sequence[:index]))
    return min(rotations)


def induced_cycles_len_k(graph, cycle_length=6):
    cycles = set()
    for start_node in sorted(graph.nodes()):
        stack = [(start_node, [start_node], {start_node})]
        while stack:
            current_node, path, seen_nodes = stack.pop()
            if len(path) == cycle_length:
                if graph.has_edge(current_node, start_node):
                    cycle = canonical_cycle(path)
                    if cycle[0] == start_node and graph.subgraph(cycle).number_of_edges() == cycle_length:
                        cycles.add(cycle)
                continue
            for neighbor in graph.neighbors(current_node):
                if neighbor in seen_nodes:
                    continue
                if neighbor < start_node:
                    continue
                stack.append((neighbor, path + [neighbor], seen_nodes | {neighbor}))
    return cycles


def compute_induced_hexagon_participation(graph):
    node_counts = Counter({node: 0 for node in graph.nodes()})
    edge_counts = Counter({tuple(sorted(edge)): 0 for edge in graph.edges()})
    for cycle in induced_cycles_len_k(graph, 6):
        for node in cycle:
            node_counts[node] += 1
        for index, source_node in enumerate(cycle):
            target_node = cycle[(index + 1) % len(cycle)]
            edge_counts[tuple(sorted((source_node, target_node)))] += 1
    return node_counts, edge_counts


def compute_num_3cycles(graph, node) -> int:
    neighbors = list(graph.neighbors(node))
    triangle_count = 0
    for index, left in enumerate(neighbors):
        for right in neighbors[index + 1:]:
            if graph.has_edge(left, right):
                triangle_count += 1
    return triangle_count + 1


def decode_one_based_count(value: int) -> int:
    return int(value) - 1


def compute_edge_direction(graph, source_node, target_node) -> int:
    source_x, source_y = get_node_position(graph, source_node)
    target_x, target_y = get_node_position(graph, target_node)
    delta_x = target_x - source_x
    delta_y = target_y - source_y

    # Fold opposite directions together because the DB stores undirected edges
    # in both source/target orders but the feature must remain symmetric.
    if delta_y < -1e-9 or (abs(delta_y) <= 1e-9 and delta_x < 0):
        delta_x = -delta_x
        delta_y = -delta_y

    axis_vectors = {
        1: (1.0, 0.0),
        2: (0.5, sqrt(3) / 2),
        3: (-0.5, sqrt(3) / 2),
    }
    return min(
        axis_vectors,
        key=lambda axis: (delta_x - axis_vectors[axis][0]) ** 2
        + (delta_y - axis_vectors[axis][1]) ** 2,
    )


def compute_edge_triangle_count(graph, source_node, target_node) -> int:
    return len(set(graph.neighbors(source_node)).intersection(graph.neighbors(target_node)))


def add_feature_edge_rows(
    edge_rows,
    seen_edges,
    source_node_id,
    target_node_id,
    edge_direction,
    edge_hexagons,
    edge_triangle_count,
    edge_mode,
):
    edge_candidates = [(source_node_id, target_node_id)]
    if edge_mode == "undirected":
        edge_candidates.append((target_node_id, source_node_id))

    inserted_rows = 0
    for source_id, target_id in edge_candidates:
        edge_key = (source_id, target_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_rows.append(
            (source_id, target_id, edge_direction, edge_hexagons, edge_triangle_count)
        )
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
    return debug_edge_limit is None or edge_index < debug_edge_limit


def expected_edge_rows(source_edge_count, edge_mode):
    if edge_mode == "directed":
        return source_edge_count
    return source_edge_count * 2


def edge_rule_for_mode(edge_mode):
    if edge_mode == "directed":
        return "preserve each source NetworkX edge row"
    return "every NetworkX edge creates A->B and B->A"


def compute_expected_dataset_counts(graphs, edge_mode):
    expected_nodes = sum(graph.number_of_nodes() for graph in graphs)
    expected_source_edges = sum(graph.number_of_edges() for graph in graphs)
    expected_db_edge_rows = expected_edge_rows(expected_source_edges, edge_mode)
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def print_expected_dataset_counts(graphs, edge_mode):
    expected_nodes, expected_source_edges, expected_db_edge_rows = (
        compute_expected_dataset_counts(graphs, edge_mode)
    )
    print("\n" + "=" * 70)
    print("EXPECTED TRIANGULAR_GRID DATABASE COUNTS")
    print("=" * 70)
    print(f"Expected graphs: {len(graphs):,}")
    print(f"Expected nodes: {expected_nodes:,}")
    print(f"Expected source NetworkX edges: {expected_source_edges:,}")
    print(f"Expected DB edge rows: {expected_db_edge_rows:,}")
    print(f"Expected DB edge rule: {edge_rule_for_mode(edge_mode)}")
    return expected_nodes, expected_source_edges, expected_db_edge_rows


def print_graph_edge_check(graph_id, source_edge_count, inserted_edge_count, edge_mode):
    expected_rows = expected_edge_rows(source_edge_count, edge_mode)
    status = "OK" if inserted_edge_count == expected_rows else "MISMATCH"
    print(
        f"[CHECK graph] graph={graph_id} source_edges={source_edge_count} "
        f"expected_db_rows={expected_rows} actual_db_rows={inserted_edge_count} "
        f"status={status}"
    )


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
    if stats["source_edge_rows"] > 0 and stats["missing_reverse_rows"] > 0:
        print("Source edge rows are not bidirectional as exposed by NetworkX.")
    print()


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
              AND e2.edge_direction = e1.edge_direction
              AND e2.edge_hexagons = e1.edge_hexagons
              AND e2.edge_triangle_count = e1.edge_triangle_count
        )
        """
    )
    reverse_count = cursor.fetchone()[0]
    print(f"Edges with matching reverse row and features: {reverse_count:,} / {edge_count:,}")
    print(f"Missing reverse rows: {edge_count - reverse_count:,}")


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


def print_counter(title, counts, total, labels):
    print(f"\n{title}")
    print("  " + "=" * 70)
    cumulative = 0.0
    for value in sorted(counts):
        count = counts[value]
        pct = (count / total) * 100 if total else 0.0
        cumulative += pct
        name = labels.get(value, str(value))
        print(
            f"  {value:2d} ({name:18s}): {count:8,} "
            f"({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {total:,} rows (100.00%)")


def count_label_map(max_raw_count, suffix):
    return {value + 1: f"{value} {suffix}" for value in range(max_raw_count + 1)}


def create_triangular_grid_database_with_features(
    db_name,
    graphs,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (TRIANGULAR GRID WITH OPTIMAL MOTIF FEATURES)")
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
            distance_to_boundary INT NOT NULL,
            num_3cycles INT NOT NULL,
            num_hexagons INT NOT NULL,
            INDEX idx_distance_to_boundary (distance_to_boundary),
            INDEX idx_3cycles (num_3cycles),
            INDEX idx_hexagons (num_hexagons)
        )
        """
    )
    print("\nNODES table created")
    print("  - distance_to_boundary: INT (1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center)")
    print("  - num_3cycles: INT (1-based categorical REAL triangle count)")
    print("  - num_hexagons: INT (1-based categorical REAL induced 6-cycle count)")
    print("    (struct_type intentionally OMITTED -- verified via SQL to be a pure degree")
    print("     relabeling 100% determined by num_3cycles/num_6cycles in the old schema;")
    print("     num_6cycles replaced with a REAL hexagon detector, not a degree proxy.)")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            edge_direction INT NOT NULL,
            edge_hexagons INT NOT NULL,
            edge_triangle_count INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
            INDEX idx_edge_direction (edge_direction),
            INDEX idx_edge_hexagons (edge_hexagons),
            INDEX idx_edge_triangle_count (edge_triangle_count)
        )
        """
    )
    print("\nEDGES table created")
    print("  - edge_direction: INT (1=Horizontal, 2=Positive-60, 3=Negative-60)")
    print("  - edge_hexagons: INT (1-based categorical REAL induced 6-cycle count)")
    print("  - edge_triangle_count: INT (number of triangles containing the edge)")
    print(f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}")

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    boundary_counts = defaultdict(int)
    cycle3_counts = defaultdict(int)
    node_hexagon_counts = defaultdict(int)
    edge_direction_counts = defaultdict(int)
    edge_hexagon_counts = defaultdict(int)
    edge_triangle_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        bounds = get_lattice_bounds(graph)
        node_hexagons_raw, edge_hexagons_raw = compute_induced_hexagon_participation(graph)

        local_to_global = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id
            distance_to_boundary = compute_distance_to_boundary(node, bounds)
            num_3cycles = compute_num_3cycles(graph, node)
            num_hexagons = node_hexagons_raw[node] + 1
            boundary_counts[distance_to_boundary] += 1
            cycle3_counts[num_3cycles] += 1
            node_hexagon_counts[num_hexagons] += 1
            node_rows.append((global_id, distance_to_boundary, num_3cycles, num_hexagons))
            global_node_id += 1

        cursor.executemany(
            """
            INSERT INTO nodes (node_id, distance_to_boundary, num_3cycles, num_hexagons)
            VALUES (%s, %s, %s, %s)
            """,
            node_rows,
        )

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_key = tuple(sorted((source_node, target_node)))
            edge_direction = compute_edge_direction(graph, source_node, target_node)
            edge_hexagons = edge_hexagons_raw[edge_key] + 1
            edge_triangle_count = compute_edge_triangle_count(graph, source_node, target_node)
            before = len(edge_rows)
            inserted_rows = add_feature_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_direction,
                edge_hexagons,
                edge_triangle_count,
                edge_mode,
            )
            edge_direction_counts[edge_direction] += inserted_rows
            edge_hexagon_counts[edge_hexagons] += inserted_rows
            edge_triangle_counts[edge_triangle_count] += inserted_rows
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                emitted_rows = edge_rows[-(len(edge_rows) - before):] if len(edge_rows) > before else []
                print(
                    f"[DEBUG edges] graph={graph_id} edge={edge_index} "
                    f"source_local={source_node}->{target_node} emitted_rows={emitted_rows}"
                )

        if debug_this_graph:
            print(f"[DEBUG edges] graph={graph_id} total_db_edge_rows={len(edge_rows)}")
        print_graph_edge_check(graph_id, graph.number_of_edges(), len(edge_rows), edge_mode)

        cursor.executemany(
            """
            INSERT INTO edges (
                source_node_id,
                target_node_id,
                edge_direction,
                edge_hexagons,
                edge_triangle_count
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
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

    print("\nDATASET SUMMARY: Triangular Grid (Optimal Motif Features)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_with_features(cursor)

    print_counter("NODE FEATURE: DISTANCE_TO_BOUNDARY", boundary_counts, node_count, DISTANCE_TO_BOUNDARY_LABELS)
    print_counter("NODE FEATURE: NUM_3CYCLES", cycle3_counts, node_count, count_label_map(6, "triangles"))
    print_counter("NODE FEATURE: NUM_HEXAGONS", node_hexagon_counts, node_count, count_label_map(6, "hexagons"))
    print_counter("EDGE FEATURE: EDGE_DIRECTION", edge_direction_counts, edge_count, EDGE_DIRECTION_LABELS)
    print_counter("EDGE FEATURE: EDGE_HEXAGONS", edge_hexagon_counts, edge_count, count_label_map(2, "hexagons"))
    print_counter("EDGE FEATURE: EDGE_TRIANGLE_COUNT", edge_triangle_counts, edge_count, EDGE_TRIANGLE_COUNT_LABELS)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | distance_to_boundary | num_3cycles | num_hexagons")
    print("  " + "-" * 68)
    for row in cursor.fetchall():
        print(
            f"  {row[0]:7d} | {row[1]:2d} ({DISTANCE_TO_BOUNDARY_LABELS.get(row[1], str(row[1])):11s}) | "
            f"{row[2]:2d} ({decode_one_based_count(row[2])} tri) | "
            f"{row[3]:2d} ({decode_one_based_count(row[3])} hex)"
        )

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_direction | edge_hexagons | edge_triangle_count")
    print("  " + "-" * 84)
    for row in cursor.fetchall():
        print(
            f"  {row[0]:6d} | {row[1]:6d} | "
            f"{row[2]:2d} ({EDGE_DIRECTION_LABELS.get(row[2], str(row[2])):11s}) | "
            f"{row[3]:2d} ({decode_one_based_count(row[3])} hex) | "
            f"{row[4]:2d} ({EDGE_TRIANGLE_COUNT_LABELS.get(row[4], str(row[4]))})"
        )

    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


def create_database_no_features(
    db_name,
    graphs,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (TRIANGULAR GRID STRUCTURE ONLY)")
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
    cursor.execute("CREATE TABLE nodes (node_id INT PRIMARY KEY)")
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

    global_node_id = 0
    for graph_id, graph in enumerate(graphs):
        local_to_global = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            local_to_global[node] = global_node_id
            node_rows.append((global_node_id,))
            global_node_id += 1
        cursor.executemany("INSERT INTO nodes (node_id) VALUES (%s)", node_rows)

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            before = len(edge_rows)
            add_plain_edge_rows(
                edge_rows,
                seen_edges,
                local_to_global[source_node],
                local_to_global[target_node],
                edge_mode,
            )
            if debug_this_graph and should_debug_edge(edge_index, debug_edge_limit):
                emitted_rows = edge_rows[-(len(edge_rows) - before):] if len(edge_rows) > before else []
                print(f"[DEBUG edges] graph={graph_id} edge={edge_index} emitted_rows={emitted_rows}")
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
    if edge_mode == "undirected":
        verify_bidirectional_edges_plain(cursor)
    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("TRIANGULAR GRID DATASET GENERATOR (OPTIMAL MOTIF FEATURES)")
    print("=" * 70)
    print("Supports 2 TRIANGULAR_GRID schema modes:")
    print("  1. with-features    - distance_to_boundary, num_3cycles, num_hexagons,")
    print("                        edge_direction, edge_hexagons, edge_triangle_count")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    edge_modes = prompt_edge_modes(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    graphs = build_triangular_grid_graphs()

    debug_edges = args.debug_edges or args.debug_all_edges
    debug_graph_limit = None if args.debug_all_edges else args.debug_graph_limit
    debug_edge_limit = None if args.debug_all_edges else args.debug_edge_limit

    created_dbs = []
    for edge_mode in edge_modes:
        announce_edge_mode(edge_mode)
        db_name = build_db_name(args.db_name, args.feature_mode, edge_mode)

        source_edge_stats = analyze_source_edge_direction(graphs)
        print_source_edge_direction_analysis("TRIANGULAR_GRID", source_edge_stats, edge_mode)

        if args.feature_mode == "with-features":
            create_triangular_grid_database_with_features(
                db_name,
                graphs,
                edge_mode,
                debug_edges,
                debug_graph_limit,
                debug_edge_limit,
            )
        else:
            create_database_no_features(
                db_name,
                graphs,
                edge_mode,
                debug_edges,
                debug_graph_limit,
                debug_edge_limit,
            )
        created_dbs.append((db_name, edge_mode))

    print("\n" + "=" * 70)
    print("ALL DATABASES CREATED SUCCESSFULLY!")
    print("=" * 70)
    for index, (db_name, edge_mode) in enumerate(created_dbs, start=1):
        if args.feature_mode == "with-features":
            print(f"  {index}. {db_name} ({edge_mode}) (3 node + 3 edge features) [TRIANGULAR_GRID]")
            print("     node: distance_to_boundary, num_3cycles, num_hexagons")
            print("     edge: edge_direction, edge_hexagons, edge_triangle_count")
        else:
            print(f"  {index}. {db_name} ({edge_mode}) (structure only, no features) [TRIANGULAR_GRID]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
