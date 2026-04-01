from __future__ import annotations


STRUCT_TYPE_TO_ID = {
    "Corner": 1,
    "Edge": 2,
    "Interior": 3,
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

STRUCT_TYPE_LABELS = {
    value: key for key, value in STRUCT_TYPE_TO_ID.items()
}

DISTANCE_TO_BOUNDARY_LABELS = {
    value: key for key, value in DISTANCE_TO_BOUNDARY_TO_ID.items()
}

EDGE_ORBIT_LABELS = {
    value: key for key, value in EDGE_ORBIT_TO_ID.items()
}


def get_grid_dimensions(graph):
    """Extract width/height from a NetworkX grid_2d_graph."""
    nodes = list(graph.nodes())
    rows = [node[0] for node in nodes]
    cols = [node[1] for node in nodes]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    width = max_row - min_row + 1
    height = max_col - min_col + 1
    return width, height


def compute_struct_type(graph, node):
    """
    Structural node category.

    Returns:
        1 = Corner
        2 = Edge
        3 = Interior
    """
    degree = graph.degree(node)
    if degree == 2:
        return STRUCT_TYPE_TO_ID["Corner"]
    if degree == 3:
        return STRUCT_TYPE_TO_ID["Edge"]
    return STRUCT_TYPE_TO_ID["Interior"]


def compute_distance_to_boundary(node, grid_size):
    """
    Distance-to-boundary category.

    Returns:
        1 = Boundary
        2 = Near-Boundary
        3 = Near-Center
        4 = Center
        5 = Deep-Center
    """
    row, col = node
    dist_to_top = row
    dist_to_bottom = grid_size - 1 - row
    dist_to_left = col
    dist_to_right = grid_size - 1 - col
    distance = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)

    if distance == 0:
        return DISTANCE_TO_BOUNDARY_TO_ID["Boundary"]
    if distance == 1:
        return DISTANCE_TO_BOUNDARY_TO_ID["Near-Boundary"]
    if distance <= 3:
        return DISTANCE_TO_BOUNDARY_TO_ID["Near-Center"]
    if distance <= 5:
        return DISTANCE_TO_BOUNDARY_TO_ID["Center"]
    return DISTANCE_TO_BOUNDARY_TO_ID["Deep-Center"]


def compute_edge_orbit(node_u, node_v, grid_size):
    """
    Edge orbit category.

    Returns:
        1 = Boundary
        2 = Interior
    """
    row_u, col_u = node_u
    row_v, col_v = node_v
    touches_boundary = (
        row_u in [0, grid_size - 1]
        or col_u in [0, grid_size - 1]
        or row_v in [0, grid_size - 1]
        or col_v in [0, grid_size - 1]
    )
    if touches_boundary:
        return EDGE_ORBIT_TO_ID["Boundary"]
    return EDGE_ORBIT_TO_ID["Interior"]
