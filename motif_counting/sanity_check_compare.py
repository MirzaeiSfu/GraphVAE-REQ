# Notes about use_cache and sanity-check comparison:
#
# No, not exactly.
#
# The comparison function in this file will still run even if use_cache = True,
# as long as:
# - sanity mode is enabled
# - main.py reaches the motif-counting block
# - and the needed pickle/DB data exists
#
# What changes with use_cache is not whether the comparison runs, but what data
# it compares.
#
# With use_cache = True:
# - main.py may load cached processed graphs from cache_datasets/<dataset>.pkl
# - then sanity check counts those cached graphs
# - then this comparator still compares those counts against the live
#   FactorBase database
#
# So yes, the real comparison still happens.
# But the local side may be stale if the cache was built before the latest
# data.py changes.
#
# That is the important risk:
# - use_cache = True can make you compare old local preprocessing
# - against current database values
#
# Recommendation:
# - for trustworthy sanity checking after changing data.py, use use_cache = False
# - or delete the relevant cache file first, then cache can be turned back on later
#
# Examples:
# - cache_datasets/PROTEINS.pkl
# - cache_datasets/QM9.pkl
#
# Also remember there is a second cache-like artifact:
# - cache_motifs/<database_name>.pkl
#
# That file stores motif/rule metadata read from FactorBase. If the database is
# rebuilt or changed, that pickle can also become stale.
#
# Practical summary:
# - the comparison function still runs with cache on
# - but for a fresh and meaningful sanity check, use_cache should usually be
#   False after loader or database changes
#
from __future__ import annotations

from decimal import Decimal
from math import isclose
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from pymysql import connect

from factorbase_motif_pipeline.factorbase_utils import (
    parse_mysql_address,
    quote_mysql_identifier,
    read_config_values,
)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def _row_has_na(row: Sequence[Any]) -> bool:
    return any(_normalize_scalar(value) == "N/A" for value in row)


def _assignment_key(row: Sequence[Any], start_idx: int, rule_len: int) -> Tuple[Any, ...]:
    return tuple(_normalize_scalar(row[start_idx + offset]) for offset in range(rule_len))


def _load_mysql_connection_settings(config_path: Path) -> Dict[str, Any]:
    config_values = read_config_values(config_path)
    required_keys = ("dbaddress", "dbusername", "dbpassword")
    missing = [key for key in required_keys if key not in config_values]
    if missing:
        raise KeyError(
            f"Missing required config keys in {config_path}: {', '.join(missing)}"
        )

    host, port = parse_mysql_address(config_values["dbaddress"])
    return {
        "host": host,
        "port": port,
        "user": config_values["dbusername"],
        "password": config_values["dbpassword"],
    }


def _build_local_count_maps(
    aggregated_counts: torch.Tensor,
    motif_counter,
) -> Tuple[Dict[str, Dict[Tuple[Any, ...], float]], Dict[str, Dict[str, Any]]]:
    counts_list = aggregated_counts.detach().cpu().tolist()
    local_maps: Dict[str, Dict[Tuple[Any, ...], float]] = {}
    rule_metadata: Dict[str, Dict[str, Any]] = {}

    count_idx = 0
    for rule_idx, rule in enumerate(motif_counter.rules):
        cp_table_name = f"{rule[0]}_CP"
        start_idx = motif_counter.multiples[rule_idx]
        syntactic_literal_rule_mask = getattr(
            motif_counter, "syntactic_literal_rule_mask", []
        )
        is_syntactic_literal_rule = (
            rule_idx < len(syntactic_literal_rule_mask)
            and bool(syntactic_literal_rule_mask[rule_idx])
        )
        table_map: Dict[Tuple[Any, ...], float] = {}

        for table_row in motif_counter.values[rule_idx]:
            key = _assignment_key(table_row, start_idx, len(rule))
            table_map[key] = float(counts_list[count_idx])
            count_idx += 1

        local_maps[cp_table_name] = table_map
        rule_metadata[cp_table_name] = {
            "rule": tuple(rule),
            "start_idx": start_idx,
            "is_syntactic_literal_rule": is_syntactic_literal_rule,
        }

    return local_maps, rule_metadata


def _table_exists(cursor, database_name: str, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (database_name, table_name),
    )
    return int(cursor.fetchone()[0]) > 0


def _fetch_database_count_maps(
    database_name: str,
    rule_metadata: Dict[str, Dict[str, Any]],
    config_path: Path,
) -> Tuple[
    Dict[str, Dict[Tuple[Any, ...], float]],
    Dict[str, str],
    Dict[str, str],
]:
    connection_settings = _load_mysql_connection_settings(config_path)
    db_bn = f"{database_name}_BN"

    connection = connect(db=db_bn, **connection_settings)
    try:
        with connection.cursor() as cursor:
            db_maps: Dict[str, Dict[Tuple[Any, ...], float]] = {}
            graph_only_tables: Dict[str, str] = {}
            unavailable_tables: Dict[str, str] = {}

            for cp_table_name, meta in rule_metadata.items():
                rule_len = len(meta["rule"])
                start_idx = int(meta["start_idx"])
                is_syntactic_literal_rule = bool(
                    meta.get("is_syntactic_literal_rule", False)
                )

                if not _table_exists(cursor, db_bn, cp_table_name):
                    message = (
                        f"{cp_table_name}: no FactorBase CP table found in {db_bn}"
                    )
                    if is_syntactic_literal_rule:
                        graph_only_tables[cp_table_name] = (
                            f"{message}; using graph-computed count only"
                        )
                    else:
                        unavailable_tables[cp_table_name] = message
                    continue

                cursor.execute(
                    f"SHOW COLUMNS FROM {quote_mysql_identifier(cp_table_name)}"
                )
                column_names = [row[0] for row in cursor.fetchall()]
                if "local_mult" not in column_names:
                    unavailable_tables[cp_table_name] = (
                        f"'local_mult' column not found in {db_bn}.{cp_table_name}"
                    )
                    continue
                local_mult_idx = column_names.index("local_mult")

                cursor.execute(
                    f"SELECT * FROM {quote_mysql_identifier(cp_table_name)}"
                )
                rows = cursor.fetchall()

                table_map: Dict[Tuple[Any, ...], float] = {}
                for row in rows:
                    if _row_has_na(row):
                        continue
                    key = _assignment_key(row, start_idx, rule_len)
                    table_map[key] = float(_normalize_scalar(row[local_mult_idx]))

                db_maps[cp_table_name] = table_map

            return db_maps, graph_only_tables, unavailable_tables
    finally:
        connection.close()


def _format_assignment(rule: Sequence[str], key: Sequence[Any]) -> str:
    return ", ".join(f"{functor}={value}" for functor, value in zip(rule, key))


def compare_aggregated_counts_to_factorbase_detailed(
    aggregated_counts: torch.Tensor,
    motif_counter,
    database_name: str,
    config_path: str | Path = Path("factorbase_motif_pipeline/config.tmp"),
    atol: float = 1e-4,
    include_graph_only: bool = False,
) -> Tuple[bool, List[str]] | Tuple[bool, List[str], List[str]]:
    """
    Compare sanity-check motif counts against live FactorBase local_mult values.

    The comparison is performed only for the rule/value rows currently active in
    motif_counter.values. That means it respects rule pruning if --rule_prune is
    enabled. Syntactic literal rules injected by --syntactic_literal_rule_mode
    may not have FactorBase CP tables; those rules are kept as graph-only checks
    instead of being treated as DB mismatches.

    Returns
    -------
    (matches, mismatches)
        matches    : True if every local count matches the DB local_mult value
                     for DB-backed rules
        mismatches : human-readable mismatch descriptions for DB-backed rules

    When include_graph_only=True, returns (matches, mismatches, graph_only).
    """
    config_path = Path(config_path)
    local_maps, rule_metadata = _build_local_count_maps(aggregated_counts, motif_counter)
    db_maps, graph_only_tables, unavailable_tables = _fetch_database_count_maps(
        database_name, rule_metadata, config_path
    )

    mismatches: List[str] = list(unavailable_tables.values())
    for cp_table_name, meta in rule_metadata.items():
        if cp_table_name in graph_only_tables or cp_table_name in unavailable_tables:
            continue

        rule = meta["rule"]
        local_table = local_maps.get(cp_table_name, {})
        db_table = db_maps.get(cp_table_name, {})

        for key, local_count in local_table.items():
            if key not in db_table:
                mismatches.append(
                    f"{cp_table_name}: missing DB row for {_format_assignment(rule, key)}"
                )
                continue

            db_count = db_table[key]
            if not isclose(local_count, db_count, rel_tol=0.0, abs_tol=atol):
                mismatches.append(
                    f"{cp_table_name}: {_format_assignment(rule, key)} "
                    f"local={local_count:.4f} db={db_count:.4f}"
                )

    matches = len(mismatches) == 0
    if include_graph_only:
        return matches, mismatches, list(graph_only_tables.values())
    return matches, mismatches


def compare_aggregated_counts_to_factorbase(
    aggregated_counts: torch.Tensor,
    motif_counter,
    database_name: str,
    config_path: str | Path = Path("factorbase_motif_pipeline/config.tmp"),
    atol: float = 1e-4,
) -> bool:
    """
    Return True when sanity-check counts match FactorBase local_mult values.
    """
    matches, _ = compare_aggregated_counts_to_factorbase_detailed(
        aggregated_counts=aggregated_counts,
        motif_counter=motif_counter,
        database_name=database_name,
        config_path=config_path,
        atol=atol,
    )
    return matches
