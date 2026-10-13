"""Tests for direct, feature-preserving DGL/PyG conversion."""

import pytest
import torch

dgl = pytest.importorskip("dgl")

from ggm_eval.adapters import dgl_to_pyg, pyg_to_dgl
from ggm_eval.contract import validate_pyg_graph
from test_contract import make_path


def test_pyg_to_dgl_preserves_directed_edge_order_and_features():
    source = make_path(node_count=4, node_dim=2, edge_dim=2)

    converted = pyg_to_dgl(source)
    dgl_sources, dgl_targets = converted.edges(order="eid")

    torch.testing.assert_close(dgl_sources, source.edge_index[0])
    torch.testing.assert_close(dgl_targets, source.edge_index[1])
    torch.testing.assert_close(converted.ndata["attr"], source.x)
    torch.testing.assert_close(converted.edata["attr"], source.edge_attr)


def test_dgl_import_removes_loops_and_adds_reverse_direction():
    graph = dgl.graph(
        (
            torch.tensor([0, 1, 1], dtype=torch.int64),
            torch.tensor([1, 2, 1], dtype=torch.int64),
        ),
        num_nodes=3,
    )
    graph.ndata["attr"] = torch.tensor([[1.0], [2.0], [3.0]])
    graph.edata["attr"] = torch.tensor([[0.25], [0.75], [99.0]])

    converted = dgl_to_pyg(graph)

    assert validate_pyg_graph(converted) == (1, 1)
    torch.testing.assert_close(
        converted.edge_index,
        torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        converted.edge_attr,
        torch.tensor([[0.25], [0.75], [0.25], [0.75]]),
    )


def test_dgl_import_rejects_conflicting_reverse_attributes():
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata["attr"] = torch.ones((2, 1))
    graph.edata["attr"] = torch.tensor([[0.0], [1.0]])

    with pytest.raises(ValueError, match="conflicting attributes"):
        dgl_to_pyg(graph)
