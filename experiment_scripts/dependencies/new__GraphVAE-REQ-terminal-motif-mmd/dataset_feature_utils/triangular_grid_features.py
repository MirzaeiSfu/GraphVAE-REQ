from __future__ import annotations

from collections import Counter
from math import sqrt


STRUCT_TYPE_TO_ID = {
    "Vertex": 1,
    "Boundary": 2,
    "Edge-Corner": 3,
    "Edge-Transition": 4,
    "Interior": 5,
}

DISTANCE_TO_BOUNDARY_TO_ID = {
    "Boundary": 1,
    "Near-Boundary": 2,
    "Near-Center": 3,
    "Center": 4,
    "Deep-Center": 5,
}

EDGE_ORBIT_TO_ID = {
    "Boundary": 1,
    "Interior": 2,
}

NUM_6CYCLES_TO_ID = {
    0: 1,
    1: 2,
}

STRUCT_TYPE_LABELS = {
    value: key for key, value in STRUCT_TYPE_TO_ID.items()
}

DISTANCE_TO_BOUNDARY_LABELS = {
    value: key for key, value in DISTANCE_TO_BOUNDARY_TO_ID.items()
}

EDGE_ORBIT_LABELS = {
    value: key for key, value in EDGE_ORBIT_TO_ID.items()
}

NUM_6CYCLES_LABELS = {
    1: "No hexagon",
    2: "Has hexagon",
}


def get_lattice_bounds(graph):
    rows = [node[0] for node in graph.nodes()]
    cols = [node[1] for node in graph.nodes()]
    return min(rows), max(rows), min(cols), max(cols)


def compute_struct_type(graph, node):
    degree = graph.degree(node)
    if degree == 2:
        return STRUCT_TYPE_TO_ID["Vertex"]
    if degree == 3:
        return STRUCT_TYPE_TO_ID["Boundary"]
    if degree == 4:
        return STRUCT_TYPE_TO_ID["Edge-Corner"]
    if degree == 5:
        return STRUCT_TYPE_TO_ID["Edge-Transition"]
    return STRUCT_TYPE_TO_ID["Interior"]


def compute_distance_to_boundary(node, bounds):
    row, col = node
    min_row, max_row, min_col, max_col = bounds

    distance = min(
        row - min_row,
        max_row - row,
        col - min_col,
        max_col - col,
    )

    if distance == 0:
        return DISTANCE_TO_BOUNDARY_TO_ID["Boundary"]
    if distance == 1:
        return DISTANCE_TO_BOUNDARY_TO_ID["Near-Boundary"]
    if distance <= 3:
        return DISTANCE_TO_BOUNDARY_TO_ID["Near-Center"]
    if distance <= 5:
        return DISTANCE_TO_BOUNDARY_TO_ID["Center"]
    return DISTANCE_TO_BOUNDARY_TO_ID["Deep-Center"]


def _count_raw_num_3cycles(graph, node):
    neighbors = list(graph.neighbors(node))
    triangle_count = 0
    for index, left in enumerate(neighbors):
        for right in neighbors[index + 1:]:
            if graph.has_edge(left, right):
                triangle_count += 1
    return triangle_count


def compute_num_3cycles(graph, node):
    # Store as a 1-based categorical value so the local loader and FactorBase
    # schema use the same encoding convention.
    return _count_raw_num_3cycles(graph, node) + 1


def decode_num_3cycles(value):
    return int(value) - 1


def compute_num_6cycles(graph, node):
    raw_value = 1 if graph.degree(node) >= 4 else 0
    return NUM_6CYCLES_TO_ID[raw_value]


def decode_num_6cycles(value):
    return int(value) - 1


def compute_edge_orbit(source_node, target_node, bounds):
    min_row, max_row, min_col, max_col = bounds
    touches_boundary = (
        source_node[0] in (min_row, max_row)
        or source_node[1] in (min_col, max_col)
        or target_node[0] in (min_row, max_row)
        or target_node[1] in (min_col, max_col)
    )
    if touches_boundary:
        return EDGE_ORBIT_TO_ID["Boundary"]
    return EDGE_ORBIT_TO_ID["Interior"]


# --- "optimal" feature schema (see factorbase_motif_pipeline/best_triangular_grid.py) ---
#
# compute_struct_type() and compute_num_6cycles() above are kept only so
# existing to_db_triangular_grid.py/new_tri.py callers don't break. Verified
# (via direct SQL on a learned FactorBase BN) that struct_type/num_3cycles/
# num_6cycles in that old schema are a 100% deterministic relabeling of node
# degree -- struct_type is intentionally NOT reused below, and
# compute_num_6cycles() is replaced by a REAL induced-hexagon count
# (compute_induced_hexagon_participation()), not a degree>=4 proxy.

EDGE_DIRECTION_TO_ID = {
    "Horizontal": 1,
    "Positive-60": 2,
    "Negative-60": 3,
}

EDGE_DIRECTION_LABELS = {
    value: key for key, value in EDGE_DIRECTION_TO_ID.items()
}


def get_node_position(graph, node):
    position = graph.nodes[node].get("pos")
    if position is not None:
        return position
    col, row = node
    return 0.5 * (row % 2) + col, (sqrt(3) / 2) * row


def _canonical_cycle(cycle):
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
                    cycle = _canonical_cycle(path)
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
    """
    Real induced-6-cycle (hexagon) participation count, per node and per
    edge, for a whole graph. Call ONCE per graph (it's O(graph)), then index
    the returned Counters per node/edge -- do not call per node.

    Returns (node_counts, edge_counts), both raw (0-based) counts; callers
    add +1 to get the 1-based categorical convention used elsewhere.
    """
    node_counts = Counter({node: 0 for node in graph.nodes()})
    edge_counts = Counter({tuple(sorted(edge)): 0 for edge in graph.edges()})
    for cycle in induced_cycles_len_k(graph, 6):
        for node in cycle:
            node_counts[node] += 1
        for index, source_node in enumerate(cycle):
            target_node = cycle[(index + 1) % len(cycle)]
            edge_counts[tuple(sorted((source_node, target_node)))] += 1
    return node_counts, edge_counts


def compute_edge_direction(graph, source_node, target_node):
    """
    Triangular-lattice edge orientation category.

    Returns:
        1 = Horizontal
        2 = Positive-60
        3 = Negative-60
    """
    source_x, source_y = get_node_position(graph, source_node)
    target_x, target_y = get_node_position(graph, target_node)
    delta_x = target_x - source_x
    delta_y = target_y - source_y

    # Fold opposite directions together since edges are stored in both
    # source/target orders but the feature must remain symmetric.
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


def compute_edge_triangle_count(graph, source_node, target_node):
    """Number of triangles (0/1/2) the edge participates in."""
    return len(set(graph.neighbors(source_node)).intersection(graph.neighbors(target_node)))
