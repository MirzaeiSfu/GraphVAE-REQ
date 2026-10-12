from types import SimpleNamespace

import pytest

from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_store import (
    RuleBasedMotifStore,
    get_motif_pickle_path,
)


VALUE_ROWS = [
    [1.0, 1, 1, 1, 1.0],
    [0.5, 1, 1, 1, 1.0],
]

SMOOTHED_COLUMNS = (
    "MULT",
    "child(nodes0)",
    "parent(nodes0)",
    "ParentSum",
    "local_mult",
    "CP",
    "likelihood",
    "prior",
)


def make_store(tmp_path, source):
    args = SimpleNamespace(
        device="cpu",
        motif_cache_dir=tmp_path,
        motif_prune_max_values_per_rule=1,
    )
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = args
    store._initialize_structures()
    store.database_name = "single_atom"
    store.pickle_path = get_motif_pickle_path("single_atom", args)
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


def test_smoothed_pruning_scores_rows_absent_from_normal_cp(tmp_path):
    args = SimpleNamespace(
        device="cpu",
        motif_cache_dir=tmp_path,
        factorbase_cp_table="CP_smoothed",
        motif_prune_max_values_per_rule=None,
    )
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = args
    store._initialize_structures()
    store.attributes = {}

    # These are four CP_smoothed rows with no accompanying normal-CP rows.
    # The first row deliberately stores CP=0.0; MULT / ParentSum is 0.5 and
    # must be used instead so the rare T child assignment can pass pruning.
    smoothed_rows = [
        (1, "T", "p0", 2, None, 0.0, None, None),
        (1, "F", "p0", 2, None, 0.5, None, None),
        (1, "T", "p1", 998, None, 0.001002, None, None),
        (997, "F", "p1", 998, None, 0.998998, None, None),
    ]
    store._add_processed_rule(
        ["child(nodes0)", "parent(nodes0)"],
        smoothed_rows,
        relation_names=(),
        rule_source="factorbase",
        value_columns=SMOOTHED_COLUMNS,
    )

    assert store.values_full == [smoothed_rows]
    assert store.values_pruned == [[smoothed_rows[0]]]


def test_smoothed_pruning_requires_count_columns(tmp_path):
    args = SimpleNamespace(
        device="cpu",
        motif_cache_dir=tmp_path,
        factorbase_cp_table="CP_smoothed",
        motif_prune_max_values_per_rule=None,
    )
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = args
    store._initialize_structures()
    store.attributes = {}

    with pytest.raises(RuntimeError, match="ParentSum"):
        store._add_processed_rule(
            ["child(nodes0)", "parent(nodes0)"],
            [(1, "T", "p0")],
            relation_names=(),
            rule_source="factorbase",
            value_columns=("MULT", "child(nodes0)", "parent(nodes0)"),
        )
