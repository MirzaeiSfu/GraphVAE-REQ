import math
import types

from motif_counting.motif_counter import RelationalMotifCounter


def _counter(mode):
    counter = object.__new__(RelationalMotifCounter)
    counter.args = types.SimpleNamespace(
        rule_prune=True,
        rule_prune_score=mode,
        motif_prune_score_threshold=0.0,
        motif_prune_max_values_per_rule=None,
        protect_unit_relation_motifs_from_pruning=False,
    )
    counter.rules = [("a",), ("b", "a")]
    counter.rule_sources = ["factorbase", "factorbase"]
    counter.multiples = [0, 1]
    counter.value_columns = [
        ["a", "local_mult", "CP", "likelihood", "prior"],
        ["b", "a", "ParentSum", "local_mult", "CP", "likelihood", "prior"],
    ]
    return counter


def test_scores():
    assert _counter("original")._rule_prune_score(3.0, 0.5, 0.25) == 2 * 3 * math.log(2) - math.log(3)
    assert _counter("no_local_mult")._rule_prune_score(3.0, 0.5, 0.25) == 2 * math.log(2)


def test_no_local_mult_keeps_rare_rows_with_cp_above_prior():
    # local_mult=1, CP=0.3 > prior=0.29: original score 2*ln(0.3/0.29) - 0 > 0, kept by both;
    # local_mult=4, CP=0.3, prior=0.29: original 8*ln(0.3/0.29) - ln 4 < 0 (dropped), new > 0 (kept);
    # CP < prior: dropped by both.
    rows = [
        ("b1", "a1", 10, 1, 0.3, 0.0, 0.29),
        ("b1", "a2", 10, 4, 0.3, 0.0, 0.29),
        ("b2", "a1", 10, 5, 0.2, 0.0, 0.4),
    ]
    data = {
        "values_full": [[("a1", 1, 1.0, 0.0, 1.0)], rows],
        "values_pruned": [[("a1", 1, 1.0, 0.0, 1.0)], [rows[0]]],  # what the cache stores (original score)
    }
    original = _counter("original")._select_motif_values(data, "cache.pkl")
    new = _counter("no_local_mult")._select_motif_values(data, "cache.pkl")
    assert original[1] == [rows[0]]
    assert new[1] == [rows[0], rows[1]]
    assert original[0] == new[0] == data["values_full"][0]  # single-atom rules keep every row
