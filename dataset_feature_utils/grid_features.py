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


# --- "optimal" feature schema (see factorbase_motif_pipeline/best_grid.py) ---
#
# compute_distance_to_boundary() above takes a single grid_size = max(width,
# height) used for BOTH axes -- a real bug for non-square grids (the shorter
# axis's true boundary distance gets systematically overestimated). The
# functions below take width/height separately and are the ones that should
# be used going forward; compute_distance_to_boundary()/compute_struct_type()
# are kept only so existing to_db_grid.py/new_grid.py callers don't break.

EDGE_AXIS_TO_ID = {
    "Horizontal": 1,
    "Vertical": 2,
}

EDGE_SQUARE_COUNT_LABELS = {
    1: "One-Square",
    2: "Two-Squares",
}

EDGE_AXIS_LABELS = {
    value: key for key, value in EDGE_AXIS_TO_ID.items()
}


def compute_boundary_depth(node, width, height):
    """
    Distance-to-boundary category, computed correctly per-axis (unlike
    compute_distance_to_boundary(), which uses a single grid_size for both
    row and column distance and over-estimates boundary distance on the
    shorter axis of a non-square grid).

    Returns:
        1 = Boundary
        2 = Near-Boundary
        3 = Near-Center
        4 = Center
        5 = Deep-Center
    """
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


def compute_edge_axis(source_node, target_node):
    """
    Edge orientation category.

    Returns:
        1 = Horizontal
        2 = Vertical
    """
    source_row, source_col = source_node
    target_row, target_col = target_node
    if source_row == target_row and source_col != target_col:
        return EDGE_AXIS_TO_ID["Horizontal"]
    if source_col == target_col and source_row != target_row:
        return EDGE_AXIS_TO_ID["Vertical"]
    raise ValueError(f"Unexpected non-grid edge: {source_node} -> {target_node}")


def compute_edge_square_count(source_node, target_node, width, height):
    """
    Number of unit grid-squares (4-cycles) the edge is a side of.

    Returns:
        1 = boundary edge, touches one square
        2 = interior edge, touches two squares
    """
    source_row, source_col = source_node
    target_row, target_col = target_node
    if source_row == target_row:
        return int(source_row > 0) + int(source_row < width - 1)
    if source_col == target_col:
        return int(source_col > 0) + int(source_col < height - 1)
    raise ValueError(f"Unexpected non-grid edge: {source_node} -> {target_node}")


def compute_edge_boundary_band(source_node, target_node, width, height):
    """
    Min boundary-depth bucket of the edge's two endpoints. Relates
    edge_axis/edge_square_count to *where* in the grid the edge sits.
    """
    source_depth = compute_boundary_depth(source_node, width, height)
    target_depth = compute_boundary_depth(target_node, width, height)
    return min(source_depth, target_depth)
