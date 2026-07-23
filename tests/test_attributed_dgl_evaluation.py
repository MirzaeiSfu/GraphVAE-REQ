import dgl
import numpy as np
import pytest
import torch

from eval.attributed_gin import (
    attributed_graph_from_dgl,
    evaluate_dgl_feature_modes,
)
from scripts.evaluate_attributed_dgl_graphs import load_dgl_graphs
from scripts.evaluate_attributed_graph_realism_checkpoints import (
    GENERATED_DGL_FILENAME,
    REFERENCE_DGL_FILENAME,
    save_dgl_graph_collections,
)


def make_bidirectional_path(node_count: int, feature_offset: float = 0.0):
    forward_source = torch.arange(node_count - 1, dtype=torch.int64)
    forward_target = forward_source + 1
    source = torch.cat((forward_source, forward_target))
    target = torch.cat((forward_target, forward_source))
    graph = dgl.graph((source, target), num_nodes=node_count)
    node_index = torch.arange(node_count, dtype=torch.float32)
    graph.ndata["attr"] = torch.stack(
        (
            node_index / node_count + feature_offset,
            1.0 - node_index / node_count,
        ),
        dim=1,
    )
    edge_values = torch.arange(node_count - 1, dtype=torch.float32)[:, None]
    graph.edata["attr"] = torch.cat((edge_values, edge_values), dim=0)
    return graph


def test_dgl_normalization_keeps_features_aligned_and_uses_largest_component():
    source = torch.tensor([0, 1, 1, 2, 2, 3, 4], dtype=torch.int64)
    target = torch.tensor([1, 0, 2, 1, 2, 4, 3], dtype=torch.int64)
    graph = dgl.graph((source, target), num_nodes=6)
    graph.ndata["attr"] = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    )
    graph.edata["attr"] = torch.tensor(
        [[0.25], [0.25], [0.75], [0.75], [99.0], [1.5], [1.5]]
    )

    normalized = attributed_graph_from_dgl(graph)

    np.testing.assert_array_equal(normalized.source_node_ids, [0, 1, 2])
    np.testing.assert_array_equal(normalized.edges, [[0, 1], [1, 2]])
    np.testing.assert_allclose(normalized.node_attributes, [[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(normalized.edge_attributes, [[0.25], [0.75]])


def test_dgl_contract_rejects_integer_features_and_conflicting_reverse_edges():
    integer_features = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    integer_features.ndata["attr"] = torch.tensor([[0], [1]])
    with pytest.raises(TypeError, match="one-hot float"):
        attributed_graph_from_dgl(integer_features)

    conflicting = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    conflicting.ndata["attr"] = torch.tensor([[1.0], [2.0]])
    conflicting.edata["attr"] = torch.tensor([[0.0], [1.0]])
    with pytest.raises(ValueError, match="Conflicting"):
        attributed_graph_from_dgl(conflicting)


def test_dgl_api_explicitly_rejects_pyg_objects():
    fake_pyg_type = type("Data", (), {})
    fake_pyg_type.__module__ = "torch_geometric.data.data"

    with pytest.raises(TypeError, match="PyTorch Geometric inputs are not accepted"):
        evaluate_dgl_feature_modes(
            [fake_pyg_type()],
            [make_bidirectional_path(3) for _ in range(3)],
            repeats=1,
            nearest_k=1,
        )
    with pytest.raises(TypeError, match="received a PyTorch Geometric object"):
        evaluate_dgl_feature_modes(
            fake_pyg_type(),
            [make_bidirectional_path(3) for _ in range(3)],
            repeats=1,
            nearest_k=1,
        )


def test_dgl_api_runs_feature_aware_random_gin():
    reference = [
        make_bidirectional_path(node_count, feature_offset=index * 0.05)
        for index, node_count in enumerate((3, 4, 5))
    ]
    generated = [
        make_bidirectional_path(node_count, feature_offset=index * 0.05)
        for index, node_count in enumerate((3, 4, 5))
    ]

    result = evaluate_dgl_feature_modes(
        generated,
        reference,
        modes=("topology_control", "decoded_node_edge"),
        repeats=1,
        seed=7,
        nearest_k=1,
        device="cpu",
    )

    assert result["feature_dimensions"] == {"node": 2, "edge": 1}
    assert result["nearest_k"] == 1
    assert result["input_contract"]["graph_type"] == "DGLGraph"
    assert result["input_contract"]["pyg_inputs_accepted"] is False
    assert set(result["modes"]) == {"topology_control", "decoded_node_edge"}


def test_checkpoint_dgl_exports_are_accepted_by_file_evaluator(tmp_path):
    generated = [make_bidirectional_path(size) for size in (3, 4, 5)]
    reference = [
        make_bidirectional_path(size, feature_offset=0.1)
        for size in (3, 4, 5)
    ]

    paths = save_dgl_graph_collections(tmp_path, generated, reference)

    assert paths == {
        "generated": str((tmp_path / GENERATED_DGL_FILENAME).resolve()),
        "reference": str((tmp_path / REFERENCE_DGL_FILENAME).resolve()),
    }
    loaded_generated = load_dgl_graphs(tmp_path / GENERATED_DGL_FILENAME)
    loaded_reference = load_dgl_graphs(tmp_path / REFERENCE_DGL_FILENAME)
    assert len(loaded_generated) == len(generated)
    assert len(loaded_reference) == len(reference)
    torch.testing.assert_close(
        loaded_generated[1].ndata["attr"],
        generated[1].ndata["attr"],
    )
    torch.testing.assert_close(
        loaded_reference[2].edata["attr"],
        reference[2].edata["attr"],
    )
