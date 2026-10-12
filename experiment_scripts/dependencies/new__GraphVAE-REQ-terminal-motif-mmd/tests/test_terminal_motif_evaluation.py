from types import SimpleNamespace

import numpy as np
import torch

from terminal_motif_evaluation import (
    binarize_terminal_matrices,
    build_chain_metadata,
    terminal_adjacencies_to_graphs,
)


def _counter():
    return SimpleNamespace(
        rules=[
            ["edges(nodes0,nodes1)"],
            [
                "node_feature(nodes0)",
                "edges(nodes0,nodes1)",
                "node_feature(nodes1)",
            ],
            ["node_feature(nodes1)"],
        ],
        states=[[2], [1, 2, 1], [0]],
        values=[[["T"]], [["A"]], [["B"]]],
        rule_sources=["factorbase"] * 3,
        relation_keys=["edges"],
    )


def test_binarizes_before_union_and_excludes_raw_relation():
    mask = torch.zeros(3, 3, 3, dtype=torch.bool)
    mask[0, :, :] = True
    mask[1, :, :] = True
    mask[2, :, :1] = True

    metadata, selected = build_chain_metadata(
        _counter(), mask, matrix_size=3
    )
    assert selected == [1]
    assert metadata[0]["exclusion_reason"] == (
        "one_atom_raw_relation_is_not_a_motif_chain"
    )
    assert metadata[2]["exclusion_reason"] == "terminal_output_is_not_N_by_N"

    matrices = torch.zeros(1, 3, 3, 3)
    matrices[0, 1, 0, 1] = 0.50
    matrices[0, 1, 1, 2] = 0.49
    matrices[0, 1, 2, 2] = 0.99
    binary, union = binarize_terminal_matrices(
        matrices, mask, selected, threshold=0.5
    )

    assert binary.dtype == np.uint8
    assert binary.shape == (1, 1, 3, 3)
    assert binary[0, 0, 0, 1] == 1
    assert binary[0, 0, 1, 2] == 0
    # Per-chain output preserves the thresholded diagonal for auditing.
    assert binary[0, 0, 2, 2] == 1
    # The union graph has self loops removed before structural evaluation.
    assert union[0, 2, 2] == 0


def test_graph_cleanup_keeps_largest_component():
    adjacency = np.zeros((6, 6), dtype=np.uint8)
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1
    adjacency[3, 4] = adjacency[4, 3] = 1
    graphs, stats = terminal_adjacencies_to_graphs([adjacency])
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 3
    assert stats["empty_graph_count"] == 0
