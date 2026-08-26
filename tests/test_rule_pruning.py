from types import SimpleNamespace

import pytest
import torch

from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_store import RuleBasedMotifStore


VALUE_ROWS = [
    [1.0, 1, 1, 1, 1.0],
    [0.5, 1, 1, 1, 1.0],
]


def make_store(tmp_path, source):
    args = SimpleNamespace(
        device="cpu",
        motif_prune_max_values_per_rule=1,
    )
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = args
    store._initialize_structures()
    store.database_name = "single_atom"
    store.pickle_path = tmp_path / "single_atom.pkl"
    store.syntactic_literal_rule_mode = "both"
    store.use_syntactic_literal_rules = True
    store.entity_feature_columns = {"nodes": ["feature"]}
    store._add_processed_rule(
        ["feature(nodes0)"],
        VALUE_ROWS,
        relation_names=(),
        rule_source=source,
    )
    return store


@pytest.mark.parametrize("source", ["factorbase", "synthetic_literal"])
def test_single_atom_value_rows_are_never_pruned(tmp_path, source):
    store = make_store(tmp_path, source)

    assert store.values_full == [VALUE_ROWS]
    assert store.values_pruned == [VALUE_ROWS]
    assert store.values == [VALUE_ROWS]


def test_runtime_values_restore_single_atom_rows_from_an_old_pruned_cache(tmp_path):
    store = make_store(tmp_path, "factorbase")
    store.values_pruned = [[]]
    store._save_to_pickle()

    args = SimpleNamespace(
        device="cpu",
        motif_cache_dir=tmp_path,
        rule_prune=True,
        use_syntactic_literal_rules=True,
        syntactic_literal_rule_mode="both",
    )
    counter = RelationalMotifCounter("single_atom", args)

    assert counter.values == [VALUE_ROWS]


def test_two_hop_internal_feature_is_attached_once_to_outgoing_relation(tmp_path):
    args = SimpleNamespace(
        device="cpu",
        motif_prune_max_values_per_rule=None,
    )
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = args
    store._initialize_structures()
    store.database_name = "two_hop"
    store.pickle_path = tmp_path / "two_hop.pkl"
    store.syntactic_literal_rule_mode = "both"
    store.use_syntactic_literal_rules = True
    store.entity_feature_columns = {"nodes": ["node_feature"]}
    store.relations = {"edges": None}

    rule = [
        "node_feature(nodes0)",
        "edges(nodes0,nodes1)",
        "edges(nodes1,nodes2)",
        "node_feature(nodes1)",
        "node_feature(nodes2)",
    ]
    columns = [
        "MULT",
        *rule,
        "ParentSum",
        "local_mult",
        "CP",
        "likelihood",
        "prior",
    ]
    row = [1, 1, "T", "T", 1, 1, 1, 1, 1.0, 0.0, 1.0]
    store._add_processed_rule(
        rule,
        [row],
        relation_names=("edges",),
        value_columns=columns,
        smoothed_value_rows=[row],
        smoothed_value_columns=columns,
    )

    # node0 masks A01; node1 and node2 mask A12. In particular, node1
    # appears once rather than masking both A01 and A12.
    assert store.base_indices[0] == [0, 2]
    assert store.mask_indices[0] == [[0, 1], [1, 3], [1, 4]]
    assert store.masks[0][0] == [["edges", "nodes0", "nodes1"]]
    assert store.masks[0][3] == [["edges", "nodes1", "nodes2"]]
    assert store.masks[0][4] == [["edges", "nodes1", "nodes2"]]

    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    for name in (
        "rules",
        "values",
        "multiples",
        "functors",
        "variables",
        "states",
        "masks",
        "base_indices",
        "mask_indices",
        "sort_indices",
        "stack_indices",
        "entity_feature_columns",
        "relation_feature_columns",
    ):
        setattr(counter, name, getattr(store, name))
    counter.device = "cpu"

    adjacency = torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]
    )
    feature = torch.tensor([[[0.8], [0.6], [0.5]]])
    counts = counter._iteration_total_counts_batched(
        feat_b=feature,
        feat_onehot_b=feature,
        edge_b=None,
        adj_b={"edges": adjacency},
        feature_onehot_mapping={0: {1: 0}},
        B=1,
        N_max=3,
    )
    feature_vector = feature[0, :, 0]
    first_occurrence = feature_vector[:, None] * adjacency[0]
    second_occurrence = (
        feature_vector[:, None] * adjacency[0] * feature_vector[None, :]
    )
    expected = (first_occurrence @ second_occurrence).sum()
    torch.testing.assert_close(counts[0, 0], expected)
