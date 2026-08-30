from types import SimpleNamespace

from motif_counting.motif_selection_manifest import (
    apply_motif_selection_manifest,
    build_motif_selection_manifest,
)


class Counter:
    def __init__(self):
        self.rules = [["child", "parent"], ["literal"]]
        self.rule_sources = ["factorbase", "syntactic_literal"]
        self.value_columns = [
            ["child", "parent", "local_mult", "CP", "prior"],
            ["literal"],
        ]
        self.values = [
            [["T", "T", 8, 0.8, 0.5], ["F", "T", 2, 0.2, 0.5]],
            [["T"]],
        ]
        self.args = SimpleNamespace()
        self.data_driven_smoothed_pruning_pending = True
        self.mask_rebuilds = 0

    def _build_motif_group_masks(self):
        self.mask_rebuilds += 1


def test_manifest_restores_exact_post_pruning_rows_without_recounting():
    training_counter = Counter()
    manifest = build_motif_selection_manifest(
        training_counter,
        {0: [0], 1: [0]},
        database_name="demo",
        motif_cp_table_source="cp_smoothed",
        rule_prune=True,
        full_combinations=4,
        pruned_combinations=3,
        active_groups=[{"name": "non_literal"}],
    )

    evaluation_counter = Counter()
    evaluation_counter.values[0][0][2] = None
    selection = apply_motif_selection_manifest(
        evaluation_counter,
        manifest,
        database_name="demo",
        motif_cp_table_source="cp_smoothed",
        rule_prune=True,
    )

    assert selection == {0: [0], 1: [0]}
    assert evaluation_counter.values == [
        [["T", "T", 8, 0.8, 0.5]],
        [["T"]],
    ]
    assert evaluation_counter.data_driven_smoothed_pruning_pending is False
    assert evaluation_counter.mask_rebuilds == 1
