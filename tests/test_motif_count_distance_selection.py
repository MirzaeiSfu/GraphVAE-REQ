from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_motif_count_distance import (
    count_exact_graph_records,
    make_counter_args,
    motif_entry_metadata,
    metric_summary,
    prepare_training_motif_selection,
)


class SelectionCounter:
    def __init__(self):
        self.args = SimpleNamespace(
            rule_prune=True,
            motif_prune_max_total_values=None,
        )
        self.values = [["all-a", "all-b", "all-c"], ["all-d"]]
        self.rules = [["rule-a"], ["rule-b"]]
        self.rule_sources = ["factorbase", "syntactic_literal"]
        self.prepare_calls = []
        self.requires_data_driven_smoothed_pruning = True

    def prepare_data_driven_smoothed_pruning(self, preprocessor, batch_size):
        self.prepare_calls.append((preprocessor, batch_size))
        self.values = [["kept-a", "kept-c"], ["kept-d"]]
        self.requires_data_driven_smoothed_pruning = False
        return {"full_combinations": 4, "pruned_combinations": 3}

    def get_syntactic_literal_motif_mask(self):
        return torch.tensor([False, False, True])

    def get_unit_relation_motif_mask(self):
        return torch.zeros(3, dtype=torch.bool)

    def select_rule_values_from_motif_mask(self, motif_mask):
        assert motif_mask.tolist() == [True, True, False]
        return {0: [0, 1]}


def motif_config(**overrides):
    values = {
        "model": "GraphVAE",
        "motif_loss": True,
        "motif_output_mode": "total_count",
        "motif_loss_mode": "calibrated_gaussian",
        "motif_batch_size": 17,
        "alpha_motif_loss": 0.1,
        "alpha_syntactic_literal_motif_loss": 0.0,
        "alpha_unit_relation_edge_count_loss": 0.0,
    }
    values.update(overrides)
    return values


def test_training_selection_prunes_before_dropping_zero_weight_groups():
    counter = SelectionCounter()
    preprocessor = object()

    selection, summary = prepare_training_motif_selection(
        counter=counter,
        flat_config=motif_config(),
        pruning_preprocessor=preprocessor,
    )

    assert counter.prepare_calls == [(preprocessor, 17)]
    assert selection == {0: [0, 1]}
    assert summary == {
        "full_combinations": 4,
        "pruned_combinations": 3,
        "active_combinations": 2,
        "active_groups": [
            {
                "name": "non_literal",
                "motif_count": 2,
                "output_mode": "total_count",
                "loss_mode": "calibrated_gaussian",
                "weight": 0.1,
                "edge_count_weight": 0.0,
            }
        ],
        "motif_batch_size": 17,
    }


def test_motif_false_requires_an_explicit_motif_true_selection_config():
    counter = SelectionCounter()

    with pytest.raises(ValueError, match="motif_loss=false"):
        prepare_training_motif_selection(
            counter=counter,
            flat_config=motif_config(motif_loss=False),
            pruning_preprocessor=object(),
        )


class CountingCounter:
    relation_keys = ["edge"]
    values = [["a", "b"], ["c"]]
    rules = [["rule-a"], ["rule-b"]]
    rule_sources = ["factorbase", "factorbase"]

    def __init__(self):
        self.selections = []

    def count_batch(self, wrapper, batch_size, selected_rules_values=None):
        self.selections.append(selected_rules_values)
        width = sum(len(indices) for indices in selected_rules_values.values())
        return torch.arange(
            wrapper.num_graphs * width,
            dtype=torch.float32,
        ).reshape(wrapper.num_graphs, width)


def graph_record(node_count):
    return {
        "features": torch.zeros(node_count, 1),
        "feat_onehot": torch.zeros(node_count, 1),
        "adj": {"edge": torch.zeros(node_count, node_count)},
        "edge": None,
    }


def test_exact_graph_counting_and_metadata_use_only_selected_training_entries():
    counter = CountingCounter()
    selection = {0: [1], 1: [0]}

    counts = count_exact_graph_records(
        counter=counter,
        records=[graph_record(2), graph_record(2)],
        feature_onehot_mapping={},
        batch_size=8,
        device=torch.device("cpu"),
        selected_rules_values=selection,
    )
    metadata = motif_entry_metadata(counter, selection)

    assert counts.shape == (2, 2)
    assert counter.selections == [selection]
    assert [(entry["rule_index"], entry["value_index"]) for entry in metadata] == [
        (0, 1),
        (1, 0),
    ]


def test_counter_argument_defaults_match_main_training_defaults():
    args = make_counter_args(
        flat_config={},
        motif_cache_dir=Path("motif-cache"),
        device=torch.device("cpu"),
    )

    assert args.rule_prune is False
    assert args.motif_cp_table_source == "cp"
    assert args.use_syntactic_literal_rules is True
    assert args.syntactic_literal_rule_mode == "both"


def test_counter_arguments_reject_a_cap_not_supported_by_training():
    with pytest.raises(ValueError, match="not a training option supported"):
        make_counter_args(
            flat_config={"motif_prune_max_total_values": 12},
            motif_cache_dir=Path("motif-cache"),
            device=torch.device("cpu"),
        )


def test_robust_distance_reduces_single_high_count_outlier_domination():
    observed = torch.tensor([[1000.0, 2.0], [1000.0, 2.0]])
    generated = torch.tensor([[2000.0, 3.0], [2000.0, 3.0]])

    summary = metric_summary(observed, generated)

    assert summary["mean_vector_count_distance"] > 700.0
    assert summary["robust_count_distance"] < 1.0
    assert summary["top1_raw_squared_error_share"] > 0.999


def test_wasserstein_detects_distribution_change_hidden_by_equal_means():
    observed = torch.tensor([[0.0], [10.0]])
    generated = torch.tensor([[5.0], [5.0]])

    summary = metric_summary(observed, generated)

    assert summary["mean_vector_count_distance"] == 0.0
    assert summary["robust_count_distance"] > 0.0
