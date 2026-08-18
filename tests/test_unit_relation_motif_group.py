from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from motif_counting.motif_counter import RelationalMotifCounter


def make_counter(*, protect=True):
    counter = RelationalMotifCounter.__new__(RelationalMotifCounter)
    counter.args = SimpleNamespace(
        rule_prune=True,
        protect_unit_relation_motifs_from_pruning=protect,
    )
    counter.relations = {"edges": object()}
    counter.rules = [
        ["edges(nodes0,nodes1)"],
        ["node_degree(nodes1)"],
        ["edge_type(nodes0,nodes1)", "edges(nodes0,nodes1)"],
    ]
    counter.multiples = [0, 0, 0]
    return counter


def test_pruning_preserves_single_atom_rules_and_protects_unit_relations():
    counter = make_counter(protect=True)
    data = {
        "values_full": [
            [["T"], ["F"]],
            [["1"], ["2"]],
            [["type_a", "T"], ["type_b", "T"]],
        ],
        "values_pruned": [
            [["F"]],
            [],
            [["type_a", "T"]],
        ],
    }

    selected = counter._select_motif_values(data, Path("cache.pkl"))

    assert selected[0] == [["F"], ["T"]]
    assert selected[1] == [["1"], ["2"]]
    assert selected[2] == [["type_a", "T"]]
    assert selected is not data["values_full"]
    assert selected[0] is not data["values_full"][0]


def test_unit_relation_mask_tracks_restored_value_rows():
    counter = make_counter(protect=True)
    counter.values = [
        [["F"], ["T"]],
        [],
        [["type_a", "T"]],
    ]
    counter.use_syntactic_literal_rules = False

    counter._build_motif_group_masks()

    assert counter.unit_relation_rule_indices == [0]
    assert counter.get_unit_relation_motif_mask().tolist() == [False, True, False]
    assert counter.num_unit_relation_motifs == 1


def test_pruning_without_relation_protection_still_preserves_single_atom_rules():
    counter = make_counter(protect=False)
    data = {
        "values_full": [[["T"]], [["1"]], []],
        "values_pruned": [[], [], []],
    }

    selected = counter._select_motif_values(data, Path("cache.pkl"))

    assert selected == [[["T"]], [["1"]], []]


def test_flat_motif_mask_maps_back_to_ordered_rule_values():
    counter = make_counter(protect=True)
    counter.values = [
        [["F"], ["T"]],
        [["1"], ["2"]],
        [["type_a", "T"], ["type_b", "T"]],
    ]

    selection = counter.select_rule_values_from_motif_mask(
        torch.tensor([False, True, True, False, False, True])
    )

    assert selection == {0: [1], 1: [0], 2: [1]}
    with pytest.raises(ValueError, match="flattened rule/value"):
        counter.select_rule_values_from_motif_mask(torch.tensor([True]))
