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
