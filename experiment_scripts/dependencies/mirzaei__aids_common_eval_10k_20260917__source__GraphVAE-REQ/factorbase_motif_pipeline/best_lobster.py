#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lobster graphs to MySQL database converter for FactorBase, using the
evidence-grounded "optimal" motif-aware feature schema.

This schema was derived by inspecting (a) the FactorBase GES/BDeu learning
mechanism (ca.sfu.cs.factorbase.jbn.BayesNet_Learning_main.java: GesCT with
samplePrior=10.0, structurePrior=1.0 -- high cardinality dilutes the prior
across more cells, hurting evidence strength per category) and (b) the
actual learned Bayesian network and empirical category counts for the
existing `to_db_lobster.py` schema (`lobster_undir_feat_snap_*` / `..._BN`)
and the `new_lob.py` schema (`lobster_best_features` design intent).

Findings that drove this schema:
- `node_degree` (Leaf/Branch/Hub/SuperHub, r=4) was confirmed via the live
  learned BN to carry REAL, non-deterministic information (it has learned
  edges to `subtree_size` and `edge_type` that are not simple 1:1 relabelings,
  unlike grid/triangular_grid's degree-derived features). KEPT at r=4 --
  new_lob.py's `degree_bucket` (r=6, exact degree up to 5 then 6+) thins the
  same information across more cells for no new signal, so that change is
  NOT adopted here.
- `distance_to_spine` (On-Spine/Near-Spine/Mid-Spine/Far-Spine, r=4) has a
  DEAD category at this dataset's actual generation scale: querying
  `lobster_undir_feat_snap_85093d.nodes` shows distance_to_spine values
  {1, 2, 3} occur (counts 1034 / 1670 / 2658) but value 4 (Far-Spine,
  distance>3) NEVER occurs -- p1=p2=0.7 in `nx.random_lobster` structurally
  caps how far any node can be from the spine. Keeping a permanently-empty
  4th category wastes BDeu prior mass on a cell that will never be observed.
  This schema instead merges that empty tail and reuses the freed category
  slot for new_lob.py's genuinely useful idea: distinguishing the two spine
  ENDPOINTS (structurally special -- they are degree-1 within the spine
  itself) from internal spine nodes, which the old schema could not express
  at all. Result: `spine_role` (r=4): Spine-Endpoint / Spine-Internal /
  Near-Spine(1 hop) / Off-Spine(2+ hops) -- no dead categories, more
  structural distinctions, same cardinality as before.
- `subtree_size` and `eccentricity` (r=4 each) were checked empirically and
  are NOT degenerate (subtree_size: 1275/2939/1008/140; eccentricity:
  644/3332/1249/137) but each has one persistently thin tail bucket
  (3.6% and 1.7% respectively). Merged down to r=3 each to keep every
  category well-populated under the samplePrior=10 prior.
- `edge_type` (Spine-Edge/Branch-Edge/Leaf-Edge, r=3) was well-distributed
  empirically (1868/3340/5316) and has real learned BN edges to
  `distance_to_spine`/`node_degree`. KEPT unchanged.
- `depth_pair` (new_lob.py's idea: pairwise capped spine-distance relation
  between an edge's two endpoints, r=6) is a genuinely good RELATIONAL
  feature -- it is naturally a function of two nodes, not one, which is
  exactly the kind of pattern FactorBase's relational learning is built to
  exploit. KEPT.
- `endpoint_degree_pair` (new_lob.py's sorted degree-bucket pair, r=21 from
  C(6,2)+6) is DROPPED: it is the highest-cardinality feature in either
  schema, on the SMALLEST of the three datasets (lobster has ~5x fewer
  edges than grid/triangular_grid: ~7,900 vs ~39,000+), it is still purely
  degree-derived (no information beyond node_degree x node_degree), and the
  combinatorial blow-up makes most of its 21 cells too sparse for the
  samplePrior=10 BDeu score to support confident structure. Replaced with a
  much cheaper, lower-cardinality `terminal_edge` (r=2: does this edge touch
  at least one Leaf node) -- this directly captures where the tree
  terminates, which is exactly the structural pattern that matters for
  "lobster-ness" (leaf placement), at minimal parameter cost.

The DB creation logic mirrors `to_db_lobster.py` / `new_lob.py`:
- nodes.node_degree: 1-based 4-bucket degree category (unchanged from to_db_lobster.py)
- nodes.spine_role: spine endpoint / spine internal / near-spine / off-spine
- nodes.subtree_size: 1-based 3-bucket branch-component-size category
- nodes.eccentricity: 1-based 3-bucket eccentricity category
- edges.edge_type: spine-edge / branch-edge / leaf-edge (unchanged from to_db_lobster.py)
- edges.depth_pair: pairwise capped spine-distance relation (from new_lob.py)
- edges.terminal_edge: does this edge touch a Leaf (degree-1) node
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


DEFAULT_DB_NAME = "lobster_optimal"
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

LOBSTER_P1 = 0.7
LOBSTER_P2 = 0.7
LOBSTER_MIN_NODES = 10
LOBSTER_MAX_NODES = 100
LOBSTER_MEAN_NODES = 80
LOBSTER_NUM_GRAPHS = 100
LOBSTER_RANDOM_SEED = 1234

NODE_DEGREE_LABELS = {
    1: "Leaf",
    2: "Branch",
    3: "Hub",
    4: "SuperHub",
}

SPINE_ROLE_LABELS = {
    1: "Spine-Endpoint",
    2: "Spine-Internal",
    3: "Near-Spine",
    4: "Off-Spine",
}

SUBTREE_SIZE_LABELS = {
    1: "1-5",
    2: "6-20",
    3: "21+",
}

ECCENTRICITY_LABELS = {
    1: "1-5",
    2: "6-10",
    3: "11+",
}

EDGE_TYPE_LABELS = {
    1: "Spine-Edge",
    2: "Branch-Edge",
    3: "Leaf-Edge",
}

DEPTH_PAIR_TO_ID = {
    (0, 0): 1,
    (0, 1): 2,
    (1, 2): 3,
    (1, 1): 4,
    (0, 2): 5,
    (2, 2): 6,
}

DEPTH_PAIR_LABELS = {
    1: "Spine-Spine",
    2: "Spine-Branch",
    3: "Branch-Leaf",
    4: "Branch-Branch",
    5: "Spine-Leaf",
    6: "Leaf-Leaf",
}

TERMINAL_EDGE_LABELS = {
    1: "Non-Terminal",
    2: "Terminal (touches a Leaf)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load generated LOBSTER graphs into MySQL with the optimal motif feature schema."
    )
    parser.add_argument(
        "--db-name",
        help="MySQL database base name (edge-mode suffix, e.g. '_dir'/'_undir', is appended automatically)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_FEATURE_MODE,
        help="Choose whether to create the LOBSTER schema with or without features",
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


def build_lobster_graphs():
    print("\n" + "=" * 70)
    print("GENERATING LOBSTER GRAPHS")
    print("=" * 70)
    print("Config:")
    print(f"  p1 = {LOBSTER_P1}, p2 = {LOBSTER_P2}")
    print(f"  mean_node = {LOBSTER_MEAN_NODES}")
    print(f"  min_node = {LOBSTER_MIN_NODES}, max_node = {LOBSTER_MAX_NODES}")
    print(f"  num_graphs = {LOBSTER_NUM_GRAPHS}")

    graphs = []
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


def farthest_node(graph, start_node):
    distances = nx.single_source_shortest_path_length(graph, start_node)
    return max(distances, key=distances.get)


def find_spine_path(graph):
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_nodes() <= 2:
        return list(graph.nodes())
    endpoint_a = farthest_node(graph, next(iter(graph.nodes())))
    endpoint_b = farthest_node(graph, endpoint_a)
    return nx.shortest_path(graph, endpoint_a, endpoint_b)


def compute_distance_to_spine(graph, spine_path):
    if not spine_path:
        return {node: 2 for node in graph.nodes()}
    return nx.multi_source_dijkstra_path_length(graph, spine_path)


def compute_node_degree(graph, node) -> int:
    degree = graph.degree(node)
    if degree == 1:
        return 1
    if degree in (2, 3):
        return 2
    if degree in (4, 5):
        return 3
    return 4


def compute_spine_role(node, spine_path, spine_nodes, distance_to_spine) -> int:
    """
    1 = Spine-Endpoint, 2 = Spine-Internal, 3 = Near-Spine (1 hop),
    4 = Off-Spine (2+ hops). Replaces the old On/Near/Mid/Far-Spine scheme,
    whose 4th bucket (Far-Spine, distance>3) is empirically empty for this
    dataset's p1=p2=0.7 generation -- that dead category slot is reused here
    to distinguish the two structurally-special spine endpoints from
    internal spine nodes instead.
    """
    if node in spine_nodes:
        if not spine_path or node in (spine_path[0], spine_path[-1]):
            return 1
        return 2
    distance = distance_to_spine.get(node, 2)
    if distance == 1:
        return 3
    return 4


def _bucket_subtree_size(value: int) -> int:
    if value <= 5:
        return 1
    if value <= 20:
        return 2
    return 3


def compute_branch_component_sizes(graph, spine_path):
    branch_graph = graph.copy()
    spine_edges = list(zip(spine_path, spine_path[1:]))
    branch_graph.remove_edges_from(spine_edges)

    component_sizes: dict[int, int] = {}
    for component in nx.connected_components(branch_graph):
        component_bucket = _bucket_subtree_size(len(component))
        for node in component:
            component_sizes[node] = component_bucket
    return component_sizes


def compute_eccentricity(graph, node) -> int:
    distances = nx.single_source_shortest_path_length(graph, node)
    raw_value = max(distances.values()) if distances else 0
    if raw_value <= 5:
        return 1
    if raw_value <= 10:
        return 2
    return 3


def compute_edge_type(source_node, target_node, spine_nodes) -> int:
    source_on_spine = source_node in spine_nodes
    target_on_spine = target_node in spine_nodes
    if source_on_spine and target_on_spine:
        return 1
    if source_on_spine or target_on_spine:
        return 2
    return 3


def compute_depth_pair(source_node, target_node, distance_to_spine) -> int:
    source_depth = min(int(distance_to_spine.get(source_node, 2)), 2)
    target_depth = min(int(distance_to_spine.get(target_node, 2)), 2)
    return DEPTH_PAIR_TO_ID[tuple(sorted((source_depth, target_depth)))]


def compute_terminal_edge(graph, source_node, target_node) -> int:
    source_is_leaf = graph.degree(source_node) == 1
    target_is_leaf = graph.degree(target_node) == 1
    return 2 if (source_is_leaf or target_is_leaf) else 1


def add_feature_edge_rows(
    edge_rows,
    seen_edges,
    source_node_id,
    target_node_id,
    edge_type,
    depth_pair,
    terminal_edge,
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
        edge_rows.append((source_id, target_id, edge_type, depth_pair, terminal_edge))
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
    print("EXPECTED LOBSTER DATABASE COUNTS")
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
              AND e2.edge_type = e1.edge_type
              AND e2.depth_pair = e1.depth_pair
              AND e2.terminal_edge = e1.terminal_edge
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
            f"  {value:2d} ({name:26s}): {count:8,} "
            f"({pct:6.2f}%) [cumulative: {cumulative:6.2f}%]"
        )
    print(f"  TOTAL: {total:,} rows (100.00%)")


def create_lobster_database_with_features(
    db_name,
    graphs,
    edge_mode,
    debug_edges=False,
    debug_graph_limit=2,
    debug_edge_limit=20,
):
    print("\n" + "=" * 70)
    print(f"CREATING DATABASE: {db_name} (LOBSTER WITH OPTIMAL MOTIF FEATURES)")
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
            spine_role INT NOT NULL,
            subtree_size INT NOT NULL,
            eccentricity INT NOT NULL,
            INDEX idx_degree (node_degree),
            INDEX idx_spine_role (spine_role),
            INDEX idx_subtree (subtree_size),
            INDEX idx_eccentricity (eccentricity)
        )
        """
    )
    print("\nNODES table created")
    print("  - node_degree: INT (1=Leaf, 2=Branch, 3=Hub, 4=SuperHub)")
    print("  - spine_role: INT (1=Spine-Endpoint, 2=Spine-Internal, 3=Near-Spine, 4=Off-Spine)")
    print("    (replaces On/Near/Mid/Far-Spine -- the old Far-Spine bucket was empirically")
    print("     empty at this dataset's scale; the freed category now distinguishes the")
    print("     two structurally-special spine endpoints instead)")
    print("  - subtree_size: INT (1=1-5, 2=6-20, 3=21+) -- merged thin 41+ tail into 21+")
    print("  - eccentricity: INT (1=1-5, 2=6-10, 3=11+) -- merged thin 16+ tail into 11+")

    cursor.execute(
        """
        CREATE TABLE edges (
            source_node_id INT NOT NULL,
            target_node_id INT NOT NULL,
            edge_type INT NOT NULL,
            depth_pair INT NOT NULL,
            terminal_edge INT NOT NULL,
            PRIMARY KEY (source_node_id, target_node_id),
            FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
            FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
            INDEX idx_edge_type (edge_type),
            INDEX idx_depth_pair (depth_pair),
            INDEX idx_terminal_edge (terminal_edge)
        )
        """
    )
    print("\nEDGES table created")
    print("  - edge_type: INT (1=Spine-Edge, 2=Branch-Edge, 3=Leaf-Edge)")
    print("  - depth_pair: INT (1=Spine-Spine, 2=Spine-Branch, 3=Branch-Leaf, ...)")
    print("  - terminal_edge: INT (1=Non-Terminal, 2=Terminal, touches a Leaf)")
    print("    (replaces endpoint_degree_pair, r=21 in new_lob.py -- too sparse for")
    print("     this dataset's ~7,900 edges under BDeu samplePrior=10; terminal_edge")
    print("     captures the structurally relevant leaf-placement pattern at r=2)")
    print(f"  - edge mode: {EDGE_MODE_LABELS[edge_mode]}")

    print("\n" + "=" * 70)
    print("POPULATING DATABASE")
    print("=" * 70)

    global_node_id = 0
    node_degree_counts = defaultdict(int)
    spine_role_counts = defaultdict(int)
    subtree_size_counts = defaultdict(int)
    eccentricity_counts = defaultdict(int)
    edge_type_counts = defaultdict(int)
    depth_pair_counts = defaultdict(int)
    terminal_edge_counts = defaultdict(int)

    for graph_id, graph in enumerate(graphs):
        if graph_id % 10 == 0:
            progress = graph_id / len(graphs) * 100
            print(f"Progress: {graph_id}/{len(graphs)} graphs ({progress:.1f}%)")

        spine_path = find_spine_path(graph)
        spine_nodes = set(spine_path)
        distance_to_spine = compute_distance_to_spine(graph, spine_path)
        subtree_sizes = compute_branch_component_sizes(graph, spine_path)

        local_to_global = {}
        node_rows = []
        for node in sorted(graph.nodes()):
            global_id = global_node_id
            local_to_global[node] = global_id
            node_degree = compute_node_degree(graph, node)
            spine_role = compute_spine_role(node, spine_path, spine_nodes, distance_to_spine)
            subtree_size = subtree_sizes.get(node, 1)
            eccentricity = compute_eccentricity(graph, node)
            node_degree_counts[node_degree] += 1
            spine_role_counts[spine_role] += 1
            subtree_size_counts[subtree_size] += 1
            eccentricity_counts[eccentricity] += 1
            node_rows.append((global_id, node_degree, spine_role, subtree_size, eccentricity))
            global_node_id += 1

        cursor.executemany(
            """
            INSERT INTO nodes (node_id, node_degree, spine_role, subtree_size, eccentricity)
            VALUES (%s, %s, %s, %s, %s)
            """,
            node_rows,
        )

        edge_rows = []
        seen_edges = set()
        debug_this_graph = should_debug_graph(debug_edges, graph_id, debug_graph_limit)
        for edge_index, (source_node, target_node) in enumerate(graph.edges()):
            source_node_id = local_to_global[source_node]
            target_node_id = local_to_global[target_node]
            edge_type = compute_edge_type(source_node, target_node, spine_nodes)
            depth_pair = compute_depth_pair(source_node, target_node, distance_to_spine)
            terminal_edge = compute_terminal_edge(graph, source_node, target_node)
            before = len(edge_rows)
            inserted_rows = add_feature_edge_rows(
                edge_rows,
                seen_edges,
                source_node_id,
                target_node_id,
                edge_type,
                depth_pair,
                terminal_edge,
                edge_mode,
            )
            edge_type_counts[edge_type] += inserted_rows
            depth_pair_counts[depth_pair] += inserted_rows
            terminal_edge_counts[terminal_edge] += inserted_rows
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
            INSERT INTO edges (source_node_id, target_node_id, edge_type, depth_pair, terminal_edge)
            VALUES (%s, %s, %s, %s, %s)
            """,
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

    print("\nDATASET SUMMARY: Lobster Tree (Optimal Motif Features)")
    print("  " + "=" * 70)
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Total nodes: {node_count:,}")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Average nodes per graph: {node_count / len(graphs):.2f}")
    print(f"  Average edges per graph: {edge_count / len(graphs):.2f}")
    if edge_mode == "undirected":
        verify_bidirectional_edges_with_features(cursor)

    print_counter("NODE FEATURE: NODE_DEGREE", node_degree_counts, node_count, NODE_DEGREE_LABELS)
    print_counter("NODE FEATURE: SPINE_ROLE", spine_role_counts, node_count, SPINE_ROLE_LABELS)
    print_counter("NODE FEATURE: SUBTREE_SIZE", subtree_size_counts, node_count, SUBTREE_SIZE_LABELS)
    print_counter("NODE FEATURE: ECCENTRICITY", eccentricity_counts, node_count, ECCENTRICITY_LABELS)
    print_counter("EDGE FEATURE: EDGE_TYPE", edge_type_counts, edge_count, EDGE_TYPE_LABELS)
    print_counter("EDGE FEATURE: DEPTH_PAIR", depth_pair_counts, edge_count, DEPTH_PAIR_LABELS)
    print_counter("EDGE FEATURE: TERMINAL_EDGE", terminal_edge_counts, edge_count, TERMINAL_EDGE_LABELS)

    print("\nSAMPLE NODES (First 10):")
    cursor.execute("SELECT * FROM nodes LIMIT 10")
    print("\n  node_id | node_degree | spine_role | subtree_size | eccentricity")
    print("  " + "-" * 76)
    for row in cursor.fetchall():
        print(
            f"  {row[0]:7d} | {row[1]:2d} ({NODE_DEGREE_LABELS.get(row[1], str(row[1])):8s}) | "
            f"{row[2]:2d} ({SPINE_ROLE_LABELS.get(row[2], str(row[2])):14s}) | "
            f"{row[3]:2d} ({SUBTREE_SIZE_LABELS.get(row[3], str(row[3])):4s}) | "
            f"{row[4]:2d} ({ECCENTRICITY_LABELS.get(row[4], str(row[4]))})"
        )

    print("\nSAMPLE EDGES (First 10):")
    cursor.execute("SELECT * FROM edges LIMIT 10")
    print("\n  source | target | edge_type | depth_pair | terminal_edge")
    print("  " + "-" * 78)
    for row in cursor.fetchall():
        print(
            f"  {row[0]:6d} | {row[1]:6d} | "
            f"{row[2]:2d} ({EDGE_TYPE_LABELS.get(row[2], str(row[2])):11s}) | "
            f"{row[3]:2d} ({DEPTH_PAIR_LABELS.get(row[3], str(row[3])):13s}) | "
            f"{row[4]:2d} ({TERMINAL_EDGE_LABELS.get(row[4], str(row[4]))})"
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
    print(f"CREATING DATABASE: {db_name} (LOBSTER STRUCTURE ONLY)")
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
    if edge_mode == "undirected":
        verify_bidirectional_edges_plain(cursor)
    cursor.close()
    connection.close()
    print(f"\nDATABASE '{db_name}' COMPLETE!\n")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("LOBSTER DATASET GENERATOR (OPTIMAL MOTIF FEATURES)")
    print("=" * 70)
    print("Supports 2 LOBSTER schema modes:")
    print("  1. with-features    - node_degree, spine_role, subtree_size, eccentricity,")
    print("                        edge_type, depth_pair, terminal_edge")
    print("  2. without-features - structure only")
    print("=" * 70 + "\n")

    edge_modes = prompt_edge_modes(args)
    print(f"Selected feature mode: {args.feature_mode}\n")

    graphs = build_lobster_graphs()

    debug_edges = args.debug_edges or args.debug_all_edges
    debug_graph_limit = None if args.debug_all_edges else args.debug_graph_limit
    debug_edge_limit = None if args.debug_all_edges else args.debug_edge_limit

    created_dbs = []
    for edge_mode in edge_modes:
        announce_edge_mode(edge_mode)
        db_name = build_db_name(args.db_name, args.feature_mode, edge_mode)

        source_edge_stats = analyze_source_edge_direction(graphs)
        print_source_edge_direction_analysis("LOBSTER", source_edge_stats, edge_mode)

        if args.feature_mode == "with-features":
            create_lobster_database_with_features(
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
            print(f"  {index}. {db_name} ({edge_mode}) (4 node + 3 edge features) [LOBSTER]")
            print("     node: node_degree, spine_role, subtree_size, eccentricity")
            print("     edge: edge_type, depth_pair, terminal_edge")
        else:
            print(f"  {index}. {db_name} ({edge_mode}) (structure only, no features) [LOBSTER]")
    print("\nREADY FOR MOTIF FINDING ALGORITHMS!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
