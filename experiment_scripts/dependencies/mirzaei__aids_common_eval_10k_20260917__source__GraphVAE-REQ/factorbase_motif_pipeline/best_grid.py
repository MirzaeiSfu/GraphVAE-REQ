#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid graphs to MySQL database converter for FactorBase, using the
evidence-grounded "optimal" motif-aware feature schema.

This schema was derived by inspecting (a) the FactorBase GES/BDeu learning
mechanism (ca.sfu.cs.factorbase.jbn.BayesNet_Learning_main.java: GesCT with
samplePrior=10.0, structurePrior=1.0 -- high cardinality dilutes the prior
across more cells, and deterministic degree-relabelings waste learned edges
on zero-information "motifs") and (b) the actual learned Bayesian network and
empirical category counts for the existing `to_db_grid.py` schema
(`grid_undir_feat_snap_*` / `grid_undir_feat_snap_*_BN`).

Findings that drove this schema:
- `struct_type` (Corner/Edge/Interior) is purely a relabeling of node degree.
  It is already implicitly captured by the adjacency reconstruction loss and
  the degree-histogram kernel statistic in main.py / GlobalProperties.py, so
  it is DROPPED here -- keeping it only spends FactorBase's schema-restricted
  search budget on a feature with near-zero marginal information.
- `distance_to_boundary` is NOT a function of local degree (interior nodes of
  different depths all have degree 4), so it is the one node feature that
  injects genuinely new, non-redundant structural signal. KEPT unchanged.
- `edge_axis` and `edge_square_count` (from new_grid.py) are low-cardinality
  (r=2 each) and tie directly to a real, already-tracked graph statistic
  (4-cycle / "square" count in graph_statistics.py and eval/stats.py's
  motif_to_indices['4cycle']). KEPT unchanged.
- New addition: `edge_boundary_band` (min boundary-depth of the edge's two
  endpoints, r=5). This lets FactorBase relate edge orientation/square-count
  to *where in the grid* the edge sits, without raising joint cardinality
  much (r=5, same scale already used for the node feature).

The DB creation logic mirrors `to_db_grid.py` / `new_grid.py`:
- nodes.distance_to_boundary: 1-based 5-bucket distance-to-boundary category
- edges.edge_axis: horizontal vs vertical grid edge
- edges.edge_square_count: number of unit squares containing the edge
- edges.edge_boundary_band: min boundary-depth bucket of the two endpoints
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import networkx as nx
from pymysql import connect

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DB_NAME = "grid_optimal"
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

BOUNDARY_DEPTH_LABELS = {
    1: "Boundary",
    2: "Near-Boundary",
    3: "Near-Center",
    4: "Center",
    5: "Deep-Center",
}

EDGE_AXIS_LABELS = {
    1: "Horizontal",
    2: "Vertical",
}

EDGE_SQUARE_COUNT_LABELS = {
    1: "One-Square",
    2: "Two-Squares",
}

EDGE_BOUNDARY_BAND_LABELS = BOUNDARY_DEPTH_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated GRID graphs into MySQL with the optimal motif feature schema."
    )
    parser.add_argument(
        "--db-name",
        help="MySQL database base name (edge-mode suffix, e.g. '_dir'/'_undir', is appended automatically)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the GRID schema with or without features",
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


def build_grid_graphs():
    print("\n" + "=" * 70)
    print("GENERATING SQUARE GRID GRAPHS")
    print("=" * 70)

    graphs = []
    for width in range(10, 20):
        for height in range(10, 20):
            graphs.append(nx.grid_2d_graph(width, height))
            if len(graphs) % 20 == 0:
                print(f"  Generated {len(graphs)}/100 grid graphs...")
    print(f"Created {len(graphs)} square grid graphs")
    return graphs


def get_grid_dimensions(graph):
    rows = [node[0] for node in graph.nodes()]
    cols = [node[1] for node in graph.nodes()]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def compute_boundary_depth(node, width: int, height: int) -> int:
    row, col = node
    distance = min(row, width - 1 - row, col, height - 1 - col)
    if distance == 0:
        return 1
    if distance == 1:
        return 2
    if distance <= 3:
        return 3
    if distance <= 5:
        return 4
    return 5


def compute_edge_axis(source_node, target_node) -> int:
    source_row, source_col = source_node
    target_row, target_col = target_node
    if source_row == target_row and source_col != target_col:
        return 1
    if source_col == target_col and source_row != target_row:
        return 2
    raise ValueError(f"Unexpected non-grid edge: {source_node} -> {target_node}")


def compute_edge_square_count(source_node, target_node, width: int, height: int) -> int:
    source_row, source_col = source_node
    target_row, target_col = target_node
    if source_row == target_row:
        return int(source_row > 0) + int(source_row < width - 1)
    if source_col == target_col:
        return int(source_col > 0) + int(source_col < height - 1)
    raise ValueError(f"Unexpected non-grid edge: {source_node} -> {target_node}")


def compute_edge_boundary_band(
    source_node, target_node, width: int, height: int
) -> int:
    """
    Min boundary-depth bucket of the edge's two endpoints. Lets FactorBase
    relate edge_axis/edge_square_count to *where* in the grid the edge sits,
    without introducing a new, independently-redundant degree-derived feature.
    """
    source_depth = compute_boundary_depth(source_node, width, height)
    target_depth = compute_boundary_depth(target_node, width, height)
    return min(source_depth, target_depth)


def add_feature_edge_rows(
    edge_rows,
    seen_edges,
    source_node_id,
    target_node_id,
    edge_axis,
    edge_square_count,
    edge_boundary_band,
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
            (source_id, target_id, edge_axis, edge_square_count, edge_boundary_band)
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


def expected_edge_rows_for_mode(source_edge_count, edge_mode):
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


def print_graph_edge_check(graph_id, source_edge_count, inserted_edge_count, edge_mode):
    expected_rows = expected_edge_rows_for_mode(source_edge_count, edge_mode)
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
              AND e2.edge_axis = e1.edge_axis
              AND e2.edge_square_count = e1.edge_square_count
              AND e2.edge_boundary_band = e1.edge_boundary_band
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
            f"  {value:2d} ({name:14s}): {count:8,} "
            f"({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {total:,} rows (100.00%)")


def create_grid_database_with_features(
    db_name,
    graphs,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (GRID WITH OPTIMAL MOTIF FEATURES)")
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
            INDEX idx_distance_to_boundary (distance_to_boundary)
        )
        """
    )
    print("\nNODES table created")
    print("  - distance_to_boundary: INT (1=Boundary, 2=Near-Boundary, 3=Near-Center, 4=Center, 5=Deep-Center)")
    print("    (struct_type intentionally OMITTED -- proven to be a pure degree relabeling;")
    print("     see grid_undir_feat_snap_* BN: struct_type's learned parents add no info")
    print("     beyond distance_to_boundary and the adjacency-implied degree histogram.)")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            edge_axis INT NOT NULL,
            edge_square_count INT NOT NULL,
            edge_boundary_band INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
            INDEX idx_edge_axis (edge_axis),
            INDEX idx_edge_square_count (edge_square_count),
            INDEX idx_edge_boundary_band (edge_boundary_band)
        )
        """
    )
    print("\nEDGES table created")
    print("  - edge_axis: INT (1=Horizontal, 2=Vertical)")
    print("  - edge_square_count: INT (1=Boundary edge in one square, 2=Interior edge in two squares)")
    print("  - edge_boundary_band: INT (min boundary_depth bucket of the two endpoints)")
    print(f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}")

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    boundary_counts = defaultdict(int)
    edge_axis_counts = defaultdict(int)
    edge_square_counts = defaultdict(int)
    edge_boundary_band_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 20 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        width, height = get_grid_dimensions(graph)
        local_to_global = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id
            boundary_depth = compute_boundary_depth(node, width, height)
            boundary_counts[boundary_depth] += 1
            node_rows.append((global_id, boundary_depth))
            global_node_id += 1

        cursor.executemany(
            "INSERT INTO nodes (node_id, distance_to_boundary) VALUES (%s, %s)",
            node_rows,
        )

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_axis = compute_edge_axis(source_node, target_node)
            edge_square_count = compute_edge_square_count(source_node, target_node, width, height)
            edge_boundary_band = compute_edge_boundary_band(source_node, target_node, width, height)
            before = len(edge_rows)
            inserted_rows = add_feature_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_axis,
                edge_square_count,
                edge_boundary_band,
                edge_mode,
            )
            edge_axis_counts[edge_axis] += inserted_rows
            edge_square_counts[edge_square_count] += inserted_rows
            edge_boundary_band_counts[edge_boundary_band] += inserted_rows
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
                source_node_id, target_node_id, edge_axis, edge_square_count, edge_boundary_band
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

    print("\nDATASET SUMMARY: Square Grid (Optimal Motif Features)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_with_features(cursor)

    print_counter("NODE FEATURE: DISTANCE_TO_BOUNDARY", boundary_counts, node_count, BOUNDARY_DEPTH_LABELS)
    print_counter("EDGE FEATURE: EDGE_AXIS", edge_axis_counts, edge_count, EDGE_AXIS_LABELS)
    print_counter("EDGE FEATURE: EDGE_SQUARE_COUNT", edge_square_counts, edge_count, EDGE_SQUARE_COUNT_LABELS)
    print_counter("EDGE FEATURE: EDGE_BOUNDARY_BAND", edge_boundary_band_counts, edge_count, EDGE_BOUNDARY_BAND_LABELS)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | distance_to_boundary")
    print("  " + "-" * 44)
    for row in cursor.fetchall():
        print(f"  {row[0]:7d} | {row[1]:2d} ({BOUNDARY_DEPTH_LABELS.get(row[1], str(row[1]))})")

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_axis | edge_square_count | edge_boundary_band")
    print("  " + "-" * 80)
    for row in cursor.fetchall():
        print(
            f"  {row[0]:6d} | {row[1]:6d} | "
            f"{row[2]:2d} ({EDGE_AXIS_LABELS.get(row[2], str(row[2])):10s}) | "
            f"{row[3]:2d} ({EDGE_SQUARE_COUNT_LABELS.get(row[3], str(row[3])):11s}) | "
            f"{row[4]:2d} ({EDGE_BOUNDARY_BAND_LABELS.get(row[4], str(row[4]))})"
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
    print(f"CREATING DATABASE: {db_name} (GRID STRUCTURE ONLY)")
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
    print("GRID DATASET GENERATOR (OPTIMAL MOTIF FEATURES)")
    print("=" * 70)
    print("Supports 2 GRID schema modes:")
    print("  1. with-features    - distance_to_boundary, edge_axis, edge_square_count, edge_boundary_band")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    edge_modes = prompt_edge_modes(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    graphs = build_grid_graphs()

    debug_edges = args.debug_edges or args.debug_all_edges
    debug_graph_limit = None if args.debug_all_edges else args.debug_graph_limit
    debug_edge_limit = None if args.debug_all_edges else args.debug_edge_limit

    created_dbs = []
    for edge_mode in edge_modes:
        announce_edge_mode(edge_mode)
        db_name = build_db_name(args.db_name, args.feature_mode, edge_mode)

        source_edge_stats = analyze_source_edge_direction(graphs)
        print_source_edge_direction_analysis("GRID", source_edge_stats, edge_mode)

        if args.feature_mode == "with-features":
            create_grid_database_with_features(
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
            print(f"  {index}. {db_name} ({edge_mode}) (1 node + 3 edge features) [GRID]")
            print("     node: distance_to_boundary")
            print("     edge: edge_axis, edge_square_count, edge_boundary_band")
        else:
            print(f"  {index}. {db_name} ({edge_mode}) (structure only, no features) [GRID]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
