"""Tests for backend-free attributed-array to PyG conversion."""

import numpy as np
import torch

from ggm_eval.adapters import attributed_arrays_to_pyg
from ggm_eval.contract import validate_pyg_graph


def test_attributed_arrays_add_reverse_edges_and_preserve_alignment():
    graph = attributed_arrays_to_pyg(
        edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        node_attributes=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        ),
        edge_attributes=np.asarray([[0.25], [0.75]], dtype=np.float32),
        source_node_ids=np.asarray([3, 7, 9], dtype=np.int64),
    )

    assert validate_pyg_graph(graph) == (2, 1)
    torch.testing.assert_close(
        graph.edge_index,
        torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        graph.edge_attr,
        torch.tensor([[0.25], [0.75], [0.25], [0.75]]),
    )
    torch.testing.assert_close(
        graph.source_node_ids,
        torch.tensor([3, 7, 9], dtype=torch.int64),
    )


def test_zero_width_edge_attributes_become_absent():
    graph = attributed_arrays_to_pyg(
        edges=np.asarray([[0, 1]], dtype=np.int64),
        node_attributes=np.ones((2, 1), dtype=np.float32),
        edge_attributes=np.empty((1, 0), dtype=np.float32),
    )

    assert graph.edge_attr is None
    assert validate_pyg_graph(graph) == (1, 0)
