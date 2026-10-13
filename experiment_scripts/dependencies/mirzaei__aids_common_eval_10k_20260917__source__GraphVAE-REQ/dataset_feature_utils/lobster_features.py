from __future__ import annotations

import networkx as nx


NODE_DEGREE_TO_ID = {
    "Leaf": 1,
    "Branch": 2,
    "Hub": 3,
    "SuperHub": 4,
}

DISTANCE_TO_SPINE_TO_ID = {
    "On-Spine": 1,
    "Near-Spine": 2,
    "Mid-Spine": 3,
    "Far-Spine": 4,
}

SUBTREE_SIZE_BUCKET_TO_ID = {
    "1-5": 1,
    "6-20": 2,
    "21-40": 3,
    "41+": 4,
}

ECCENTRICITY_BUCKET_TO_ID = {
    "1-5": 1,
    "6-10": 2,
    "11-15": 3,
    "16+": 4,
}

EDGE_TYPE_TO_ID = {
    "Spine-Edge": 1,
    "Branch-Edge": 2,
    "Leaf-Edge": 3,
}

NODE_DEGREE_LABELS = {value: key for key, value in NODE_DEGREE_TO_ID.items()}
DISTANCE_TO_SPINE_LABELS = {
    value: key for key, value in DISTANCE_TO_SPINE_TO_ID.items()
}
SUBTREE_SIZE_BUCKET_LABELS = {
    value: key for key, value in SUBTREE_SIZE_BUCKET_TO_ID.items()
}
ECCENTRICITY_BUCKET_LABELS = {
    value: key for key, value in ECCENTRICITY_BUCKET_TO_ID.items()
}
EDGE_TYPE_LABELS = {value: key for key, value in EDGE_TYPE_TO_ID.items()}


def farthest_node(graph: nx.Graph, start_node: int) -> int:
    distances = nx.single_source_shortest_path_length(graph, start_node)
    return max(distances, key=distances.get)


def find_spine_path(graph: nx.Graph) -> list[int]:
    """
    Approximate the lobster spine with the tree diameter path.
    """
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_nodes() <= 2:
        return list(graph.nodes())

    start_node = next(iter(graph.nodes()))
    endpoint_a = farthest_node(graph, start_node)
    endpoint_b = farthest_node(graph, endpoint_a)
    return nx.shortest_path(graph, endpoint_a, endpoint_b)


def compute_node_degree(graph: nx.Graph, node: int) -> int:
    degree = graph.degree(node)
    if degree == 1:
        return NODE_DEGREE_TO_ID["Leaf"]
    if degree in (2, 3):
        return NODE_DEGREE_TO_ID["Branch"]
    if degree in (4, 5):
        return NODE_DEGREE_TO_ID["Hub"]
    return NODE_DEGREE_TO_ID["SuperHub"]


def compute_distance_to_spine_labels(
    graph: nx.Graph,
    spine_path: list[int],
) -> dict[int, int]:
    if not spine_path:
        return {
            node: DISTANCE_TO_SPINE_TO_ID["Far-Spine"]
            for node in graph.nodes()
        }

    distances = nx.multi_source_dijkstra_path_length(graph, spine_path)
    labels: dict[int, int] = {}
    for node, distance in distances.items():
        if distance == 0:
            labels[node] = DISTANCE_TO_SPINE_TO_ID["On-Spine"]
        elif distance == 1:
            labels[node] = DISTANCE_TO_SPINE_TO_ID["Near-Spine"]
        elif distance <= 3:
            labels[node] = DISTANCE_TO_SPINE_TO_ID["Mid-Spine"]
        else:
            labels[node] = DISTANCE_TO_SPINE_TO_ID["Far-Spine"]
    return labels


def _bucket_subtree_size(value: int) -> int:
    if value <= 5:
        return SUBTREE_SIZE_BUCKET_TO_ID["1-5"]
    if value <= 20:
        return SUBTREE_SIZE_BUCKET_TO_ID["6-20"]
    if value <= 40:
        return SUBTREE_SIZE_BUCKET_TO_ID["21-40"]
    return SUBTREE_SIZE_BUCKET_TO_ID["41+"]


def compute_branch_component_sizes(
    graph: nx.Graph,
    spine_path: list[int],
) -> dict[int, int]:
    """
    Use the size of the branch component attached to each spine segment as a
    stable tree-structure feature, then bucket it into a 1-based category.
    """
    branch_graph = graph.copy()
    spine_edges = list(zip(spine_path, spine_path[1:]))
    branch_graph.remove_edges_from(spine_edges)

    component_sizes: dict[int, int] = {}
    for component in nx.connected_components(branch_graph):
        component_bucket = _bucket_subtree_size(len(component))
        for node in component:
            component_sizes[node] = component_bucket
    return component_sizes


def compute_eccentricity(graph: nx.Graph, node: int) -> int:
    distances = nx.single_source_shortest_path_length(graph, node)
    raw_value = max(distances.values()) if distances else 0
    if raw_value <= 5:
        return ECCENTRICITY_BUCKET_TO_ID["1-5"]
    if raw_value <= 10:
        return ECCENTRICITY_BUCKET_TO_ID["6-10"]
    if raw_value <= 15:
        return ECCENTRICITY_BUCKET_TO_ID["11-15"]
    return ECCENTRICITY_BUCKET_TO_ID["16+"]


def compute_edge_type(
    source_node: int,
    target_node: int,
    spine_nodes: set[int],
) -> int:
    source_on_spine = source_node in spine_nodes
    target_on_spine = target_node in spine_nodes

    if source_on_spine and target_on_spine:
        return EDGE_TYPE_TO_ID["Spine-Edge"]
    if source_on_spine or target_on_spine:
        return EDGE_TYPE_TO_ID["Branch-Edge"]
    return EDGE_TYPE_TO_ID["Leaf-Edge"]


# --- "optimal" feature schema (see factorbase_motif_pipeline/best_lobster.py) ---
#
# compute_distance_to_spine_labels()/compute_branch_component_sizes()/
# compute_eccentricity() above are kept only so existing to_db_lobster.py
# callers don't break. The functions below fix two issues found by
# inspecting the actual learned FactorBase BN and empirical category counts:
# distance_to_spine's "Far-Spine" bucket is empirically EMPTY at this
# dataset's p1=p2=0.7 generation scale (dead category, wastes BDeu prior
# mass), and subtree_size/eccentricity each had one persistently thin tail
# bucket. compute_spine_role() reuses the freed category to add a
# distinction the old scheme couldn't express: the two spine endpoints are
# structurally special (degree-1 within the spine itself).

SPINE_ROLE_TO_ID = {
    "Spine-Endpoint": 1,
    "Spine-Internal": 2,
    "Near-Spine": 3,
    "Off-Spine": 4,
}

SPINE_ROLE_LABELS = {
    value: key for key, value in SPINE_ROLE_TO_ID.items()
}

SUBTREE_SIZE_BUCKET_V2_TO_ID = {
    "1-5": 1,
    "6-20": 2,
    "21+": 3,
}

SUBTREE_SIZE_BUCKET_V2_LABELS = {
    value: key for key, value in SUBTREE_SIZE_BUCKET_V2_TO_ID.items()
}

ECCENTRICITY_BUCKET_V2_TO_ID = {
    "1-5": 1,
    "6-10": 2,
    "11+": 3,
}

ECCENTRICITY_BUCKET_V2_LABELS = {
    value: key for key, value in ECCENTRICITY_BUCKET_V2_TO_ID.items()
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


def compute_distance_to_spine_raw(
    graph: nx.Graph,
    spine_path: list[int],
) -> dict[int, int]:
    """
    Un-bucketed hop distance from each node to the nearest spine node.
    Needed (rather than compute_distance_to_spine_labels()'s already-bucketed
    1-4 labels) by compute_spine_role()/compute_depth_pair(), which do their
    own capping/bucketing of the raw distance.
    """
    if not spine_path:
        return {node: 2 for node in graph.nodes()}
    return nx.multi_source_dijkstra_path_length(graph, spine_path)


def compute_spine_role(
    node: int,
    spine_path: list[int],
    spine_nodes: set[int],
    distance_to_spine: dict[int, int],
) -> int:
    """
    1 = Spine-Endpoint, 2 = Spine-Internal, 3 = Near-Spine (1 hop),
    4 = Off-Spine (2+ hops).
    """
    if node in spine_nodes:
        if not spine_path or node in (spine_path[0], spine_path[-1]):
            return SPINE_ROLE_TO_ID["Spine-Endpoint"]
        return SPINE_ROLE_TO_ID["Spine-Internal"]
    distance = distance_to_spine.get(node, 2)
    if distance == 1:
        return SPINE_ROLE_TO_ID["Near-Spine"]
    return SPINE_ROLE_TO_ID["Off-Spine"]


def _bucket_subtree_size_v2(value: int) -> int:
    if value <= 5:
        return SUBTREE_SIZE_BUCKET_V2_TO_ID["1-5"]
    if value <= 20:
        return SUBTREE_SIZE_BUCKET_V2_TO_ID["6-20"]
    return SUBTREE_SIZE_BUCKET_V2_TO_ID["21+"]


def compute_branch_component_sizes_v2(
    graph: nx.Graph,
    spine_path: list[int],
) -> dict[int, int]:
    """Same idea as compute_branch_component_sizes(), merged to 3 buckets."""
    branch_graph = graph.copy()
    spine_edges = list(zip(spine_path, spine_path[1:]))
    branch_graph.remove_edges_from(spine_edges)

    component_sizes: dict[int, int] = {}
    for component in nx.connected_components(branch_graph):
        component_bucket = _bucket_subtree_size_v2(len(component))
        for node in component:
            component_sizes[node] = component_bucket
    return component_sizes


def compute_eccentricity_v2(graph: nx.Graph, node: int) -> int:
    """Same idea as compute_eccentricity(), merged to 3 buckets."""
    distances = nx.single_source_shortest_path_length(graph, node)
    raw_value = max(distances.values()) if distances else 0
    if raw_value <= 5:
        return ECCENTRICITY_BUCKET_V2_TO_ID["1-5"]
    if raw_value <= 10:
        return ECCENTRICITY_BUCKET_V2_TO_ID["6-10"]
    return ECCENTRICITY_BUCKET_V2_TO_ID["11+"]


def compute_depth_pair(
    source_node: int,
    target_node: int,
    distance_to_spine: dict[int, int],
) -> int:
    """Pairwise capped spine-distance relation between an edge's endpoints."""
    source_depth = min(int(distance_to_spine.get(source_node, 2)), 2)
    target_depth = min(int(distance_to_spine.get(target_node, 2)), 2)
    return DEPTH_PAIR_TO_ID[tuple(sorted((source_depth, target_depth)))]


def compute_terminal_edge(graph: nx.Graph, source_node: int, target_node: int) -> int:
    """1 = Non-Terminal, 2 = Terminal (touches a Leaf/degree-1 node)."""
    source_is_leaf = graph.degree(source_node) == 1
    target_is_leaf = graph.degree(target_node) == 1
    return 2 if (source_is_leaf or target_is_leaf) else 1
