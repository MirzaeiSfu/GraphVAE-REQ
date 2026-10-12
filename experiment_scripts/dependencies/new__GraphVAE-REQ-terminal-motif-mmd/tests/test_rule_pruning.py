from types import SimpleNamespace

import pytest

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


def test_global_pruned_value_cap_keeps_highest_scores_across_rules():
    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    counter.args = SimpleNamespace(
        rule_prune=True,
        motif_prune_max_total_values=2,
    )
    counter.multiples = [True, True]
    counter.values = [
        [
            [10.0, 0.9, 0.0, 0.1],
            [3.0, 0.8, 0.0, 0.2],
        ],
        [
            [8.0, 0.85, 0.0, 0.1],
            [1.0, 0.6, 0.0, 0.4],
        ],
    ]

    expected = [counter.values[0][0], counter.values[1][0]]
    counter._apply_global_pruned_value_cap()

    assert counter.values == [[expected[0]], [expected[1]]]
