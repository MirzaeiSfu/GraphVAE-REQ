"""Tests for the strict PyG collection boundary."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from ggm_eval.contract import (
    apply_feature_mode,
    normalize_pyg_graph,
    prepare_collection,
    validate_collection,
    validate_pyg_graph,
)


def make_path(
    node_count=3,
    *,
    node_dim=2,
    edge_dim=1,
    offset=0.0,
):
    """Construct a contract-compliant bidirectional path."""

    forward = torch.arange(node_count - 1, dtype=torch.int64)
    reverse = forward + 1
    edge_index = torch.stack(
        (
            torch.cat((forward, reverse)),
            torch.cat((reverse, forward)),
        )
    )
    values = torch.arange(node_count, dtype=torch.float32)[:, None] + offset
    x = torch.cat([values + index for index in range(node_dim)], dim=1)
    edge_attr = None
    if edge_dim:
        rows = torch.arange(node_count - 1, dtype=torch.float32)[:, None]
        rows = torch.cat([rows + index for index in range(edge_dim)], dim=1)
        edge_attr = torch.cat((rows, rows), dim=0)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=node_count,
    )


def test_valid_graph_and_collection_summary():
    graph = make_path(node_count=4, node_dim=3, edge_dim=2)

    assert validate_pyg_graph(graph) == (3, 2)
    summary = validate_collection([graph, graph.clone()])

    assert summary.graph_count == 2
    assert summary.total_nodes == 8
    assert summary.directed_edge_count == 12
    assert summary.node_feature_dim == 3
    assert summary.edge_feature_dim == 2


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (
            Data(
                x=torch.ones((2, 1)),
                edge_index=torch.tensor([[0], [1]], dtype=torch.int64),
                num_nodes=2,
            ),
            "missing reverse edge",
        ),
        (
            Data(
                x=torch.ones((2, 1)),
                edge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64),
                num_nodes=2,
            ),
            "self-loop",
        ),
        (
            Data(
                x=torch.ones((2, 1)),
                edge_index=torch.tensor(
                    [[0, 0, 1], [1, 1, 0]], dtype=torch.int64
                ),
                num_nodes=2,
            ),
            "duplicate directed edge",
        ),
        (
            Data(
                x=torch.ones((2, 1)),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
                edge_attr=torch.tensor([[0.0], [1.0]]),
                num_nodes=2,
            ),
            "conflicting edge_attr",
        ),
    ],
)
def test_contract_rejects_ambiguous_undirected_edges(graph, message):
    with pytest.raises(ValueError, match=message):
        validate_pyg_graph(graph)


def test_contract_rejects_batch_and_integer_features():
    graph = make_path()
    with pytest.raises(TypeError, match="PyG Batch"):
        validate_pyg_graph(Batch.from_data_list([graph, graph]))

    graph.x = graph.x.to(torch.int64)
    with pytest.raises(TypeError, match="floating point"):
        validate_pyg_graph(graph)


def test_normalization_keeps_deterministic_largest_component_and_alignment():
    graph = Data(
        x=torch.tensor([[10.0], [11.0], [12.0], [20.0], [21.0], [99.0]]),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 3, 4],
                [1, 0, 2, 1, 4, 3],
            ],
            dtype=torch.int64,
        ),
        edge_attr=torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]]),
        num_nodes=6,
    )

    normalized = normalize_pyg_graph(graph)

    torch.testing.assert_close(
        normalized.source_node_ids, torch.tensor([0, 1, 2])
    )
    torch.testing.assert_close(
        normalized.x, torch.tensor([[10.0], [11.0], [12.0]])
    )
    torch.testing.assert_close(
        normalized.edge_attr, torch.tensor([[1.0], [1.0], [2.0], [2.0]])
    )
    assert normalized.num_nodes == 3

    normalized_again = normalize_pyg_graph(normalized)
    torch.testing.assert_close(
        normalized_again.source_node_ids, torch.tensor([0, 1, 2])
    )


def test_normalization_preserves_producer_node_ids():
    graph = make_path()
    graph.source_node_ids = torch.tensor([10, 12, 15], dtype=torch.int64)

    normalized = normalize_pyg_graph(graph)

    torch.testing.assert_close(
        normalized.source_node_ids, torch.tensor([10, 12, 15])
    )


def test_feature_modes_are_explicit_and_dimensionally_stable():
    graph = make_path(node_dim=3, edge_dim=2)

    topology = apply_feature_mode(graph, "topology_control")
    assert topology.x.shape == (3, 1)
    assert topology.edge_attr is None

    node = apply_feature_mode(graph, "decoded_node")
    assert node.x.shape == (3, 3)
    assert node.edge_attr is None

    edge = apply_feature_mode(graph, "decoded_edge")
    assert edge.x.shape == (3, 1)
    assert edge.edge_attr.shape == (4, 2)

    both = apply_feature_mode(graph, "decoded_node_edge")
    assert both.x.shape == (3, 3)
    assert both.edge_attr.shape == (4, 2)


def test_collection_rejects_mixed_feature_dimensions():
    with pytest.raises(ValueError, match="inconsistent feature dimensions"):
        validate_collection(
            [
                make_path(node_dim=1, edge_dim=1),
                make_path(node_dim=2, edge_dim=1),
            ]
        )


def test_prepare_collection_applies_minimum_size():
    with pytest.raises(ValueError, match="at least 2 graphs"):
        prepare_collection(
            [make_path()],
            mode="decoded_node_edge",
            minimum_graphs=2,
        )
