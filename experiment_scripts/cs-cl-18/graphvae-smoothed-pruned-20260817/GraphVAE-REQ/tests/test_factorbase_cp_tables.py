from types import SimpleNamespace

import pytest
import torch

from motif_counting.factorbase_tables import (
    factorbase_cp_table_name,
    normalize_factorbase_cp_table,
)
from motif_counting.motif_counter import get_motif_pickle_path as counter_pickle_path
from motif_counting.motif_store import RuleBasedMotifStore
from motif_counting.motif_store import get_motif_pickle_path as store_pickle_path
from motif_counting.sanity_check_compare import (
    _build_local_count_maps,
    _normalize_observed_count,
    _smoothed_mult_to_observed_count,
)


class RecordingCursor:
    def __init__(self):
        self.queries = []
        self._rows = []
        self.description = None

    def execute(self, query, params=None):
        self.queries.append((query, params))
        if query.startswith("SELECT DISTINCT child"):
            self._rows = [("edge(nodes0,nodes1)",)]
        elif query.startswith("SELECT parent"):
            self._rows = [("node_feature(nodes0)",)]
        elif query.startswith("SELECT *"):
            self.description = tuple(
                (name, None, None, None, None, None, None)
                for name in (
                    "MULT",
                    "edge(nodes0,nodes1)",
                    "node_feature(nodes0)",
                    "ParentSum",
                    "local_mult",
                    "CP",
                    "likelihood",
                    "prior",
                )
            )
            self._rows = [(1, "T", 1, 1, None, 1.0, None, None)]

    def fetchall(self):
        return self._rows


@pytest.mark.parametrize(
    ("source", "expected_table"),
    [
        ("CP", "edge(nodes0,nodes1)_CP"),
        ("CP_smoothed", "edge(nodes0,nodes1)_CP_smoothed"),
    ],
)
def test_rule_store_reads_selected_cp_tables(source, expected_table):
    store = RuleBasedMotifStore.__new__(RuleBasedMotifStore)
    store.relations = {}
    store.factorbase_cp_table = source
    captured = {}
    store._add_processed_rule = (
        lambda rule, value, relation_names, rule_source, value_columns=None: captured.update(
            rule=rule,
            value=value,
            relation_names=relation_names,
            rule_source=rule_source,
            value_columns=value_columns,
        )
    )
    store._ensure_entity_unary_literal_rules = lambda relation_names: None
    store._ensure_relation_literal_rules = lambda relation_names: None
    store._adjust_matrices = lambda: None
    cursor = RecordingCursor()

    store._process_rules({"cursor": cursor}, {"cursor": object()})

    assert (f"SELECT * FROM `{expected_table}`", None) in cursor.queries
    assert captured["rule_source"] == "factorbase"
    assert captured["value_columns"][0] == "MULT"
    assert not any(
        query == "SELECT * FROM `edge(nodes0,nodes1)_CP`"
        for query, _ in cursor.queries
        if source == "CP_smoothed"
    )


@pytest.mark.parametrize("source", ["CP", "CP_smoothed"])
def test_cp_sources_have_distinct_cache_filenames(tmp_path, source):
    args = SimpleNamespace(
        motif_cache_dir=tmp_path,
        factorbase_cp_table=source,
    )
    expected = tmp_path / f"example_{source}.pkl"

    assert store_pickle_path("example", args) == expected
    assert counter_pickle_path("example", args) == expected


@pytest.mark.parametrize("source", ["CP", "CP_smoothed"])
def test_sanity_check_uses_selected_cp_table_names(source):
    motif_counter = SimpleNamespace(
        factorbase_cp_table=source,
        rules=[["edge(nodes0,nodes1)"]],
        multiples=[1],
        values=[[(0.25, "T")]],
    )

    local_maps, metadata = _build_local_count_maps(
        aggregated_counts=torch.tensor([3.0]),
        motif_counter=motif_counter,
    )

    table_name = factorbase_cp_table_name("edge(nodes0,nodes1)", source)
    assert local_maps == {table_name: {("T",): 3.0}}
    assert table_name in metadata


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [("CP", "CP"), ("_CP", "CP"), ("CP_smoothed", "CP_smoothed")],
)
def test_cp_table_source_normalization(configured, normalized):
    assert normalize_factorbase_cp_table(configured) == normalized


def test_factorbase_observed_count_normalization():
    assert _normalize_observed_count(None) == 0.0
    assert _normalize_observed_count(7) == 7.0
    assert _smoothed_mult_to_observed_count(1) == 0.0
    assert _smoothed_mult_to_observed_count(8) == 7.0
