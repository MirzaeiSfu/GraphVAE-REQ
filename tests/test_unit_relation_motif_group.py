from pathlib import Path
from types import SimpleNamespace

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


def test_protected_pruning_restores_only_bare_binary_relations():
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
    assert selected[1] == []
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


def test_pruning_without_protection_keeps_the_cached_selection():
    counter = make_counter(protect=False)
    data = {
        "values_full": [[["T"]], [["1"]], []],
        "values_pruned": [[], [], []],
    }

    selected = counter._select_motif_values(data, Path("cache.pkl"))

    assert selected == [[], [], []]
