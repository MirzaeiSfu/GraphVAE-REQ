import networkx as nx
import numpy as np

import mmd_rnn
from per_chain_terminal_motif_evaluation import (
    gaussian_tv_mmd,
    prepare_generated_adjacency,
)


def test_vectorized_mmd_matches_repository_implementation():
    reference = [
        np.asarray([1, 2, 1]),
        np.asarray([2, 1]),
        np.asarray([1, 3, 2, 1]),
    ]
    generated = [np.asarray([1, 1]), np.asarray([2, 2, 1])]
    expected = mmd_rnn.compute_mmd(
        reference,
        generated,
        kernel=mmd_rnn.gaussian_tv,
        sigma=0.1,
    )
    actual, _ = gaussian_tv_mmd(
        reference, generated, sigma=0.1, is_hist=True
    )
    assert np.isclose(actual, expected)


def test_chain_cleanup_binarizes_and_keeps_largest_component():
    adjacency = np.asarray(
        [
            [1.0, 0.8, 0.0, 0.0, 0.0],
            [0.8, 1.0, 0.7, 0.0, 0.0],
            [0.0, 0.7, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.0, 0.9, 1.0],
        ]
    )
    prepared = prepare_generated_adjacency(adjacency)
    assert prepared is not None
    assert prepared.graph.number_of_nodes() == 3
    assert prepared.graph.number_of_edges() == 2
    assert list(nx.selfloop_edges(prepared.graph)) == []
