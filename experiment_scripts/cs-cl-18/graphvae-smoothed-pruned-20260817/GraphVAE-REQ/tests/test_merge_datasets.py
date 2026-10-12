from types import SimpleNamespace

from data import merge_datasets


def _dataset(name, max_num_nodes):
    return SimpleNamespace(
        processed_adjs=[f"{name}-adj"],
        processed_Xs=[f"{name}-x"],
        processed_node_onehot=[f"{name}-node"],
        processed_edge_onehot=[f"{name}-edge"],
        max_num_nodes=max_num_nodes,
    )


def test_merge_datasets_accepts_train_validation_and_test_partitions():
    merged = merge_datasets(
        _dataset("train", 4),
        _dataset("validation", 6),
        _dataset("test", 5),
    )

    assert merged["processed_adjs"] == [
        "train-adj",
        "validation-adj",
        "test-adj",
    ]
    assert merged["max_num_nodes"] == 6
