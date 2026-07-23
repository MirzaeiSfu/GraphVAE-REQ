import numpy as np

from eval.attributed_gin import (
    AttributedGraph,
    categorical_groups,
    feature_view,
    graph_from_dense_attributes,
    grouped_argmax_onehot,
    validate_collection_dimensions,
)


def test_categorical_groups_and_argmax_are_per_original_feature():
    feature_info = {
        0: {"feature_name": "kind", "value": 0},
        1: {"feature_name": "kind", "value": 1},
        2: {"feature_name": "state", "value": 0},
        3: {"feature_name": "state", "value": 1},
        4: {"feature_name": "state", "value": 2},
    }
    groups = categorical_groups(feature_info, 5)
    assert groups == ((0, 1), (2, 3, 4))

    logits = np.array(
        [
            [0.2, 0.8, 10.0, 1.0, 0.0],
            [3.0, 2.0, -1.0, 5.0, 4.0],
        ],
        dtype=np.float32,
    )
    decoded = grouped_argmax_onehot(logits, groups)
    np.testing.assert_array_equal(
        decoded,
        np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_dense_decoder_outputs_stay_aligned_after_largest_component():
    adjacency = np.zeros((5, 5), dtype=np.float32)
    for source, target in ((1, 3), (3, 4), (4, 1), (0, 2)):
        adjacency[source, target] = 0.9

    node_logits = np.array(
        [
            [9, 0, 9, 0],
            [0, 9, 8, 1],
            [9, 0, 0, 9],
            [8, 1, 0, 7],
            [0, 8, 6, 0],
        ],
        dtype=np.float32,
    )
    edge_logits = np.zeros((3, 5, 5), dtype=np.float32)
    # Averaging both orientations must still choose channel 2 for source edge 1--3.
    edge_logits[:, 1, 3] = [1, 0, 8]
    edge_logits[:, 3, 1] = [1, 0, 2]
    edge_logits[:, 3, 4] = [0, 7, 0]
    edge_logits[:, 4, 3] = [0, 3, 0]
    edge_logits[:, 4, 1] = [5, 0, 0]
    edge_logits[:, 1, 4] = [1, 0, 0]

    graph = graph_from_dense_attributes(
        adjacency,
        node_logits,
        edge_logits,
        node_feature_info={
            0: {"feature_name": "a", "value": 0},
            1: {"feature_name": "a", "value": 1},
            2: {"feature_name": "b", "value": 0},
            3: {"feature_name": "b", "value": 1},
        },
        edge_feature_info={
            0: {"feature_name": "relation", "value": 0},
            1: {"feature_name": "relation", "value": 1},
            2: {"feature_name": "relation", "value": 2},
        },
        values_are_logits=True,
    )

    assert graph is not None
    np.testing.assert_array_equal(graph.source_node_ids, [1, 3, 4])
    np.testing.assert_array_equal(graph.edges, [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_array_equal(
        graph.node_attributes,
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
    )
    np.testing.assert_array_equal(
        graph.edge_attributes,
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
    )


def test_reference_directional_onehot_is_hardened_after_symmetrizing():
    adjacency = np.array([[0, 1], [1, 0]], dtype=np.float32)
    node_attributes = np.array([[1, 0], [0, 1]], dtype=np.float32)
    edge_attributes = np.zeros((2, 2, 2), dtype=np.float32)
    edge_attributes[1, 0, 1] = 1.0

    graph = graph_from_dense_attributes(
        adjacency,
        node_attributes,
        edge_attributes,
        node_feature_info={
            0: {"feature_name": "node_kind", "value": 0},
            1: {"feature_name": "node_kind", "value": 1},
        },
        edge_feature_info={
            0: {"feature_name": "edge_kind", "value": 0},
            1: {"feature_name": "edge_kind", "value": 1},
        },
        values_are_logits=False,
    )

    assert graph is not None
    np.testing.assert_array_equal(graph.edge_attributes, [[0, 1]])


def test_feature_ablations_keep_dimensions_and_do_not_create_degree_features():
    graph = AttributedGraph(
        edges=np.array([[0, 1]], dtype=np.int64),
        node_attributes=np.array([[0.2, 1.5], [3.0, -2.0]], dtype=np.float32),
        edge_attributes=np.array([[0.25, 2.75, -1.0]], dtype=np.float32),
        source_node_ids=np.array([4, 7], dtype=np.int64),
    )

    topology_nodes, topology_edges = feature_view(graph, "topology_control")
    np.testing.assert_array_equal(topology_nodes, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(topology_edges, [[0, 0, 0]])

    edge_nodes, edge_edges = feature_view(graph, "decoded_edge")
    np.testing.assert_array_equal(edge_nodes, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(edge_edges, graph.edge_attributes)

    both_nodes, both_edges = feature_view(graph, "decoded_node_edge")
    np.testing.assert_array_equal(both_nodes, graph.node_attributes)
    np.testing.assert_array_equal(both_edges, graph.edge_attributes)


def test_collection_validation_accepts_real_valued_edge_attributes():
    graph = AttributedGraph(
        edges=np.array([[0, 1]], dtype=np.int64),
        node_attributes=np.array([[0.1], [0.2]], dtype=np.float32),
        edge_attributes=np.array([[0.125, -3.5]], dtype=np.float32),
        source_node_ids=np.array([0, 1], dtype=np.int64),
    )
    assert validate_collection_dimensions([graph], [graph]) == (1, 2)
