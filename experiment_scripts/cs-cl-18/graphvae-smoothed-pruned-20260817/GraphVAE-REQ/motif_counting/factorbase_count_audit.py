"""Audit observed counts shared by FactorBase CP and CP_smoothed tables."""

from __future__ import annotations

from math import isclose
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pymysql import connect
from pymysql.cursors import SSCursor

from factorbase_motif_pipeline.factorbase_utils import quote_mysql_identifier
from motif_counting.sanity_check_compare import (
    _load_mysql_connection_settings,
    _normalize_observed_count,
    _normalize_scalar,
)


FACTORBASE_COUNT_COLUMNS = {
    "cp",
    "likelihood",
    "local_mult",
    "mult",
    "parentsum",
    "prior",
}


def _assignment_columns(column_names: Sequence[str]) -> List[str]:
    return [
        column_name
        for column_name in column_names
        if column_name.lower() not in FACTORBASE_COUNT_COLUMNS
    ]


def _assignment_key(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(_normalize_scalar(value) for value in values)


def compare_cp_count_rows(
    cp_rows: Iterable[Sequence[Any]],
    smoothed_rows: Iterable[Sequence[Any]],
    *,
    atol: float = 1e-9,
    max_mismatches: int = 20,
) -> Dict[str, Any]:
    """Compare rows shaped as ``(*assignment_values, local_mult)``."""
    cp_counts: Dict[Tuple[Any, ...], float] = {}
    duplicate_cp_assignments = 0
    cp_row_count = 0
    for row in cp_rows:
        cp_row_count += 1
        key = _assignment_key(row[:-1])
        if key in cp_counts:
            duplicate_cp_assignments += 1
        cp_counts[key] = _normalize_observed_count(row[-1])

    mismatches: List[str] = []
    smoothed_row_count = 0
    added_zero_rows = 0
    duplicate_smoothed_assignments = 0
    seen_smoothed = set()
    value_mismatch_count = 0

    for row in smoothed_rows:
        smoothed_row_count += 1
        key = _assignment_key(row[:-1])
        if key in seen_smoothed:
            duplicate_smoothed_assignments += 1
        seen_smoothed.add(key)

        assignment_was_observed = key in cp_counts
        expected_count = cp_counts.pop(key, 0.0)
        actual_count = _normalize_observed_count(row[-1])
        if not assignment_was_observed and actual_count == 0.0:
            added_zero_rows += 1
        if not isclose(actual_count, expected_count, rel_tol=0.0, abs_tol=atol):
            value_mismatch_count += 1
            if len(mismatches) < max_mismatches:
                mismatches.append(
                    f"assignment={key!r} CP={expected_count} "
                    f"CP_smoothed={actual_count}"
                )

    missing_smoothed_assignments = len(cp_counts)
    if cp_counts and len(mismatches) < max_mismatches:
        for key, expected_count in cp_counts.items():
            mismatches.append(
                f"assignment={key!r} CP={expected_count} missing from CP_smoothed"
            )
            if len(mismatches) >= max_mismatches:
                break

    mismatch_count = (
        value_mismatch_count
        + missing_smoothed_assignments
        + duplicate_cp_assignments
        + duplicate_smoothed_assignments
    )
    return {
        "matches": mismatch_count == 0 and missing_smoothed_assignments == 0,
        "cp_rows": cp_row_count,
        "smoothed_rows": smoothed_row_count,
        "added_zero_rows": added_zero_rows,
        "missing_smoothed_assignments": missing_smoothed_assignments,
        "duplicate_cp_assignments": duplicate_cp_assignments,
        "duplicate_smoothed_assignments": duplicate_smoothed_assignments,
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
    }


def _select_count_rows_sql(
    database_name: str,
    table_name: str,
    assignment_columns: Sequence[str],
    count_expression: str,
) -> str:
    assignment_sql = ", ".join(
        quote_mysql_identifier(column_name) for column_name in assignment_columns
    )
    selected_sql = (
        f"{assignment_sql}, {count_expression} AS `observed_count`"
        if assignment_sql
        else f"{count_expression} AS `observed_count`"
    )
    return (
        f"SELECT {selected_sql} FROM {quote_mysql_identifier(database_name)}."
        f"{quote_mysql_identifier(table_name)}"
    )


def _table_columns(connection, database_name: str, table_name: str) -> List[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"SHOW COLUMNS FROM {quote_mysql_identifier(database_name)}."
            f"{quote_mysql_identifier(table_name)}"
        )
        return [row[0] for row in cursor.fetchall()]


def _database_names(connection) -> List[str]:
    with connection.cursor() as cursor:
        cursor.execute("SHOW DATABASES")
        return sorted(row[0][:-3] for row in cursor.fetchall() if row[0].endswith("_BN"))


def _bn_tables(connection, database_name: str) -> List[str]:
    bn_database_name = f"{database_name}_BN"
    with connection.cursor() as cursor:
        cursor.execute(f"SHOW TABLES FROM {quote_mysql_identifier(bn_database_name)}")
        return sorted(row[0] for row in cursor.fetchall())


def audit_factorbase_database_counts(
    connection,
    database_name: str,
    *,
    atol: float = 1e-9,
    max_mismatches: int = 20,
) -> Dict[str, Any]:
    """Compare every paired CP/CP_smoothed table in one FactorBase database."""
    bn_database_name = f"{database_name}_BN"
    tables = _bn_tables(connection, database_name)
    cp_tables = {table for table in tables if table.endswith("_CP")}
    smoothed_tables = {
        table[:-len("_smoothed")]
        for table in tables
        if table.endswith("_CP_smoothed")
    }
    missing_smoothed_tables = sorted(cp_tables - smoothed_tables)
    missing_cp_tables = sorted(smoothed_tables - cp_tables)

    result: Dict[str, Any] = {
        "database": database_name,
        "status": "SKIP" if not cp_tables and not smoothed_tables else "PASS",
        "tables_checked": 0,
        "cp_rows": 0,
        "smoothed_rows": 0,
        "added_zero_rows": 0,
        "missing_smoothed_tables": missing_smoothed_tables,
        "missing_cp_tables": missing_cp_tables,
        "mismatch_count": len(missing_smoothed_tables) + len(missing_cp_tables),
        "mismatches": [],
    }
    if missing_smoothed_tables or missing_cp_tables:
        result["status"] = "FAIL"

    for cp_table_name in sorted(cp_tables & smoothed_tables):
        smoothed_table_name = f"{cp_table_name}_smoothed"
        cp_columns = _table_columns(connection, bn_database_name, cp_table_name)
        smoothed_columns = _table_columns(
            connection,
            bn_database_name,
            smoothed_table_name,
        )
        cp_assignment_columns = _assignment_columns(cp_columns)
        smoothed_assignment_columns = _assignment_columns(smoothed_columns)
        if cp_assignment_columns != smoothed_assignment_columns:
            result["status"] = "FAIL"
            result["mismatch_count"] += 1
            result["mismatches"].append(
                f"{cp_table_name}: assignment columns differ: "
                f"CP={cp_assignment_columns!r}, "
                f"CP_smoothed={smoothed_assignment_columns!r}"
            )
            continue
        if "local_mult" not in cp_columns or "MULT" not in smoothed_columns:
            result["status"] = "FAIL"
            result["mismatch_count"] += 1
            result["mismatches"].append(
                f"{cp_table_name}: CP.local_mult or CP_smoothed.MULT is missing"
            )
            continue

        cp_sql = _select_count_rows_sql(
            bn_database_name,
            cp_table_name,
            cp_assignment_columns,
            quote_mysql_identifier("local_mult"),
        )
        with connection.cursor() as cursor:
            cursor.execute(cp_sql)
            cp_rows = cursor.fetchall()

        smoothed_sql = _select_count_rows_sql(
            bn_database_name,
            smoothed_table_name,
            smoothed_assignment_columns,
            f"({quote_mysql_identifier('MULT')} - 1)",
        )
        with connection.cursor(SSCursor) as cursor:
            cursor.execute(smoothed_sql)
            table_result = compare_cp_count_rows(
                cp_rows,
                cursor,
                atol=atol,
                max_mismatches=max_mismatches,
            )

        result["tables_checked"] += 1
        result["cp_rows"] += table_result["cp_rows"]
        result["smoothed_rows"] += table_result["smoothed_rows"]
        result["added_zero_rows"] += table_result["added_zero_rows"]
        result["mismatch_count"] += table_result["mismatch_count"]
        if not table_result["matches"]:
            result["status"] = "FAIL"
            remaining_slots = max_mismatches - len(result["mismatches"])
            result["mismatches"].extend(
                f"{cp_table_name}: {message}"
                for message in table_result["mismatches"][:remaining_slots]
            )

    return result


def audit_all_factorbase_database_counts(
    config_path: str | Path = Path("factorbase_motif_pipeline/config.tmp"),
    database_names: Optional[Sequence[str]] = None,
    *,
    atol: float = 1e-9,
    max_mismatches: int = 20,
) -> List[Dict[str, Any]]:
    """Audit all available BN databases, or only the requested base names."""
    connection_settings = _load_mysql_connection_settings(Path(config_path))
    connection = connect(**connection_settings)
    try:
        selected_databases = (
            sorted({name[:-3] if name.endswith("_BN") else name for name in database_names})
            if database_names
            else _database_names(connection)
        )
        results = []
        for database_name in selected_databases:
            try:
                results.append(
                    audit_factorbase_database_counts(
                        connection,
                        database_name,
                        atol=atol,
                        max_mismatches=max_mismatches,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "database": database_name,
                        "status": "ERROR",
                        "tables_checked": 0,
                        "cp_rows": 0,
                        "smoothed_rows": 0,
                        "added_zero_rows": 0,
                        "missing_smoothed_tables": [],
                        "missing_cp_tables": [],
                        "mismatch_count": 0,
                        "mismatches": [str(exc)],
                    }
                )
        return results
    finally:
        connection.close()
