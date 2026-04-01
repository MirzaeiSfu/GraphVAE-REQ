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
        table_map: Dict[Tuple[Any, ...], float] = {}

        for table_row in motif_counter.values[rule_idx]:
            key = _assignment_key(table_row, start_idx, len(rule))
            table_map[key] = float(counts_list[count_idx])
            count_idx += 1

        local_maps[cp_table_name] = table_map
        rule_metadata[cp_table_name] = {
            "rule": tuple(rule),
            "start_idx": start_idx,
        }

    return local_maps, rule_metadata


def _fetch_database_count_maps(
    database_name: str,
    rule_metadata: Dict[str, Dict[str, Any]],
    config_path: Path,
) -> Dict[str, Dict[Tuple[Any, ...], float]]:
    connection_settings = _load_mysql_connection_settings(config_path)
    db_bn = f"{database_name}_BN"

    connection = connect(db=db_bn, **connection_settings)
    try:
        with connection.cursor() as cursor:
            db_maps: Dict[str, Dict[Tuple[Any, ...], float]] = {}

            for cp_table_name, meta in rule_metadata.items():
                rule_len = len(meta["rule"])
                start_idx = int(meta["start_idx"])

                cursor.execute(
                    f"SHOW COLUMNS FROM {quote_mysql_identifier(cp_table_name)}"
                )
                column_names = [row[0] for row in cursor.fetchall()]
                if "local_mult" not in column_names:
                    raise KeyError(
                        f"'local_mult' column not found in {db_bn}.{cp_table_name}"
                    )
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

            return db_maps
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
) -> Tuple[bool, List[str]]:
    """
    Compare sanity-check motif counts against live FactorBase local_mult values.

    The comparison is performed only for the rule/value rows currently active in
    motif_counter.values. That means it respects rule pruning if --rule_prune is
    enabled.

    Returns
    -------
    (matches, mismatches)
        matches    : True if every local count matches the DB local_mult value
        mismatches : human-readable mismatch descriptions
    """
    config_path = Path(config_path)
    local_maps, rule_metadata = _build_local_count_maps(aggregated_counts, motif_counter)
    db_maps = _fetch_database_count_maps(database_name, rule_metadata, config_path)

    mismatches: List[str] = []
    for cp_table_name, meta in rule_metadata.items():
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

    return len(mismatches) == 0, mismatches


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
