import pickle
from types import SimpleNamespace

import torch

from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_store import RuleBasedMotifStore


RULE = ["child(nodes0)", "parent(nodes1)"]
SMOOTHED_COLUMNS = [
    "MULT",
    *RULE,
    "ParentSum",
    "local_mult",
    "CP",
    "likelihood",
    "prior",
]
CP_COLUMNS = list(SMOOTHED_COLUMNS)


def test_smoothed_metadata_groups_by_parents_and_normalizes_over_children():
    smoothed_rows = [
        [1, "A", "X", None, None, None, None, None],
        [1, "B", "X", None, None, None, None, None],
        [1, "A", "Y", None, None, None, None, None],
        [1, "B", "Y", None, None, None, None, None],
    ]
    cp_rows = [
        [3, "A", "X", 4, 3, 0.75, 0.0, 0.6],
        [1, "B", "X", 4, 1, 0.25, 0.0, 0.4],
    ]

    rows = RelationalMotifCounter.derive_smoothed_rule_rows(
        rule=RULE,
        smoothed_rows=smoothed_rows,
        smoothed_columns=SMOOTHED_COLUMNS,
        cp_rows=cp_rows,
        cp_columns=CP_COLUMNS,
        local_mults=torch.tensor([3.0, 1.0, 0.0, 0.0]),
    )

    parent_sum_idx = SMOOTHED_COLUMNS.index("ParentSum")
    local_idx = SMOOTHED_COLUMNS.index("local_mult")
    cp_idx = SMOOTHED_COLUMNS.index("CP")
    prior_idx = SMOOTHED_COLUMNS.index("prior")
    assert [row[parent_sum_idx] for row in rows] == [4.0, 4.0, 0.0, 0.0]
    assert [row[local_idx] for row in rows] == [3.0, 1.0, 0.0, 0.0]
    assert [row[cp_idx] for row in rows] == [0.75, 0.25, 0.0, 0.0]
    assert [row[prior_idx] for row in rows] == [0.6, 0.4, 0.6, 0.4]


def test_data_driven_smoothed_pruning_counts_before_selecting_rows():
    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    counter.args = SimpleNamespace(
        rule_prune=True,
        motif_prune_max_values_per_rule=None,
    )
    counter.data_driven_smoothed_pruning_pending = True
    counter.rules = [RULE]
    counter.rule_sources = ["factorbase"]
    counter.values = [[
        [1, "A", "X", None, None, None, None, None],
        [1, "B", "X", None, None, None, None, None],
    ]]
    counter.value_columns = [SMOOTHED_COLUMNS]
    counter.cp_reference_values = [[
        [3, "A", "X", 4, 3, 0.75, 0.0, 0.6],
        [1, "B", "X", 4, 1, 0.25, 0.0, 0.4],
    ]]
    counter.cp_reference_columns = [CP_COLUMNS]
    counter.count_batch = lambda *args, **kwargs: torch.tensor([[3.0, 1.0]])
    counter._build_motif_group_masks = lambda: None

    summary = counter.prepare_data_driven_smoothed_pruning(
        preprocessor=object(),
        batch_size=8,
    )

    assert summary == {"full_combinations": 2, "pruned_combinations": 1}
    assert len(counter.values[0]) == 1
    assert counter.values[0][0][1:3] == ["A", "X"]
    assert counter.values[0][0][SMOOTHED_COLUMNS.index("local_mult")] == 3.0
    assert counter.values[0][0][SMOOTHED_COLUMNS.index("CP")] == 0.75
    assert counter.values[0][0][SMOOTHED_COLUMNS.index("prior")] == 0.6


def test_new_cache_payload_contains_both_cp_sources(tmp_path):
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.args = SimpleNamespace(device="cpu")
    store._initialize_structures()
    store.database_name = "dual_source"
    store.pickle_path = tmp_path / "dual_source.pkl"
    store.syntactic_literal_rule_mode = "both"
    store.use_syntactic_literal_rules = True
    store.entity_feature_columns = {"nodes": ["feature"]}
    store._add_processed_rule(
        ["feature(nodes0)"],
        [["A", 2, 2, 0.0, 1.0]],
        relation_names=(),
        value_columns=["feature(nodes0)", "MULT", "local_mult", "likelihood", "prior"],
        smoothed_value_rows=[["A", 3, 2, 0.0, 1.0]],
        smoothed_value_columns=["feature(nodes0)", "MULT", "local_mult", "likelihood", "prior"],
    )
    store._save_to_pickle()

    with store.pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["cache_schema_version"] == 3
    assert payload["values_full"][0][0][1] == 2
    assert payload["values_smoothed_full"][0][0][1] == 3
