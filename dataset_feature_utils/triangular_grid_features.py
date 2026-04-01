from __future__ import annotations


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
