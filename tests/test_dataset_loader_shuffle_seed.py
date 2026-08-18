import random

import numpy as np

from data import data_split_three_way, list_graph_loader


def _dense_graphs(graphs):
    return [adjacency.toarray() for adjacency in graphs]


def test_explicit_loader_seed_decouples_dataset_order_from_model_seed():
    random.seed(0)
    first = list_graph_loader(
        "small_lobster",
        return_labels=True,
        shuffle_seed=17,
    )[0]
    random.seed(999)
    second = list_graph_loader(
        "small_lobster",
        return_labels=True,
        shuffle_seed=17,
    )[0]

    for first_graph, second_graph in zip(
        _dense_graphs(first),
        _dense_graphs(second),
    ):
        np.testing.assert_array_equal(first_graph, second_graph)

    first_split = data_split_three_way(first, train_fraction=0.7, seed=123)
    second_split = data_split_three_way(second, train_fraction=0.7, seed=123)
    for first_partition, second_partition in zip(
        first_split[:3],
        second_split[:3],
    ):
        for first_graph, second_graph in zip(
            _dense_graphs(first_partition),
            _dense_graphs(second_partition),
        ):
            np.testing.assert_array_equal(first_graph, second_graph)
