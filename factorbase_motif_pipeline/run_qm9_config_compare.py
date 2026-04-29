#!/usr/bin/env python3
"""
Create QM9 once, clone the prepared database for each config template, and run
the standard FactorBase pipeline on each clone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from factorbase_utils import (
    parse_mysql_address,
    print_section,
    quote_mysql_identifier,
    read_config_values,
    sanitize_path_component,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "run_factorbase_pipeline.py"
DROP_DATABASES_SCRIPT = SCRIPT_DIR / "drop_factorbase_databases.sh"
DEFAULT_CONFIGS = (
    Path("config.tmp"),
    Path("config1.tmp"),
    Path("config2.tmp"),
)
DEFAULT_PREFIX = "qm9_cfg_compare"
DEFAULT_EDGE_MODE = "directed"
DEFAULT_JAR = "snapshot"
REQUIRED_BASE_TABLES = ("nodes", "edges")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare QM9 once, clone the base database per config, and run "
            "FactorBase once for each config."
        )
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=(
            "Prefix for generated database names. The script uses "
            "<prefix>_base, <prefix>_config, <prefix>_config1, and <prefix>_config2."
        ),
    )
    parser.add_argument(
        "--base-db",
        help="Prepared QM9 database name. Defaults to <prefix>_base.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        type=Path,
        help=(
            "Config template to test. Repeat this flag for multiple configs. "
            "Defaults to config.tmp, config1.tmp, and config2.tmp."
        ),
    )
    parser.add_argument(
        "--edge-mode",
        choices=("directed", "undirected"),
        default=DEFAULT_EDGE_MODE,
        help="Edge storage mode used only when building the base QM9 database.",
    )
    parser.add_argument(
        "--jar",
        choices=("snapshot", "patched"),
        default=DEFAULT_JAR,
        help="FactorBase jar choice. 'snapshot' is the standard unpatched jar.",
    )
    parser.add_argument(
        "--drop-base-at-end",
        action="store_true",
        help="Drop the prepared base QM9 database after all comparison runs finish.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR / path


def require_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def resolve_config_paths(configs: list[Path] | None) -> list[Path]:
    selected = list(configs) if configs else list(DEFAULT_CONFIGS)
    resolved = [resolve_input_path(path) for path in selected]
    for config_path in resolved:
        require_path_exists(config_path, "config template")
    return resolved


def build_target_db_name(prefix: str, config_path: Path) -> str:
    return f"{prefix}_{sanitize_path_component(config_path.stem)}"


def build_run_log_path(db_name: str) -> Path:
    return SCRIPT_DIR / "log" / f"{sanitize_path_component(db_name)}_run.log"


def run_command(command: list[str], step_name: str) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def drop_factorbase_databases(db_name: str) -> None:
    run_command(
        [str(DROP_DATABASES_SCRIPT), "--yes", db_name],
        step_name=f"Drop databases for {db_name}",
    )


def prepare_base_qm9_database(
    base_db_name: str,
    edge_mode: str,
    config_path: Path,
) -> None:
    print_section("PREPARING BASE QM9 DATABASE")
    drop_factorbase_databases(base_db_name)
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "QM9",
        base_db_name,
        f"--{edge_mode}",
        "--prepare-only",
        "--config-template",
        str(config_path),
    ]
    run_command(
        command,
        step_name="Prepare base QM9 database",
    )


def load_mysql_connection_settings(config_path: Path) -> dict[str, str | int]:
    config_values = read_config_values(config_path)
    required_keys = ("dbaddress", "dbusername", "dbpassword")
    missing_keys = [key for key in required_keys if key not in config_values]
    if missing_keys:
        raise RuntimeError(
            f"Config file is missing required MySQL settings {missing_keys}: {config_path}"
        )

    host, port = parse_mysql_address(config_values["dbaddress"])
    if host == "localhost":
        host = "127.0.0.1"

    return {
        "host": host,
        "port": port,
        "user": config_values["dbusername"],
        "password": config_values["dbpassword"],
    }


def clone_database(connection_settings: dict[str, str | int], source_db: str, target_db: str) -> None:
    from pymysql import connect

    source_db_id = quote_mysql_identifier(source_db)
    target_db_id = quote_mysql_identifier(target_db)

    connection = connect(
        host=str(connection_settings["host"]),
        port=int(connection_settings["port"]),
        user=str(connection_settings["user"]),
        password=str(connection_settings["password"]),
        autocommit=False,
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME "
                "FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                (source_db,),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError(f"Source database '{source_db}' does not exist.")

            charset_name, collation_name = row
            cursor.execute(f"DROP DATABASE IF EXISTS {target_db_id}")
            cursor.execute(
                f"CREATE DATABASE {target_db_id} "
                f"CHARACTER SET {charset_name} COLLATE {collation_name}"
            )
            cursor.execute("SET FOREIGN_KEY_CHECKS=0")

            try:
                cursor.execute(f"USE {target_db_id}")
                for table_name in REQUIRED_BASE_TABLES:
                    table_id = quote_mysql_identifier(table_name)
                    cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.TABLES "
                        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                        (source_db, table_name),
                    )
                    if cursor.fetchone()[0] != 1:
                        raise RuntimeError(
                            f"Source database '{source_db}' is missing required table '{table_name}'."
                        )

                    cursor.execute(f"SHOW CREATE TABLE {source_db_id}.{table_id}")
                    create_table_row = cursor.fetchone()
                    create_table_sql = create_table_row[1]
                    cursor.execute(create_table_sql)
                    cursor.execute(
                        f"INSERT INTO {target_db_id}.{table_id} "
                        f"SELECT * FROM {source_db_id}.{table_id}"
                    )
            finally:
                cursor.execute("SET FOREIGN_KEY_CHECKS=1")

        connection.commit()
    finally:
        connection.close()


def run_factorbase_for_config(db_name: str, config_path: Path, jar_choice: str) -> bool:
    print_section(f"RUNNING {db_name}")
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "QM9",
        db_name,
        "--use-existing-db",
        "--jar",
        jar_choice,
        "--config-template",
        str(config_path),
    ]

    try:
        run_command(command, step_name=f"Run FactorBase for {db_name}")
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = parse_args()
    config_paths = resolve_config_paths(args.configs)
    base_db_name = args.base_db or f"{sanitize_path_component(args.prefix)}_base"
    connection_settings = load_mysql_connection_settings(config_paths[0])

    target_specs: list[tuple[Path, str]] = []
    seen_db_names = {base_db_name}
    for config_path in config_paths:
        target_db_name = build_target_db_name(args.prefix, config_path)
        if target_db_name in seen_db_names:
            raise ValueError(
                f"Config-derived database name collision for '{config_path.name}': {target_db_name}"
            )
        seen_db_names.add(target_db_name)
        target_specs.append((config_path, target_db_name))

    prepare_base_qm9_database(
        base_db_name,
        args.edge_mode,
        config_paths[0],
    )

    results: list[tuple[Path, str, bool]] = []
    for config_path, target_db_name in target_specs:
        print_section(f"CLONING {base_db_name} -> {target_db_name}")
        drop_factorbase_databases(target_db_name)
        clone_database(connection_settings, base_db_name, target_db_name)
        success = run_factorbase_for_config(target_db_name, config_path, args.jar)
        results.append((config_path, target_db_name, success))

    if args.drop_base_at_end:
        print_section("DROPPING BASE QM9 DATABASE")
        drop_factorbase_databases(base_db_name)

    print_section("SUMMARY")
    for config_path, target_db_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(
            f"{status:7s} | config={config_path.name:10s} | db={target_db_name} | "
            f"log={build_run_log_path(target_db_name)}"
        )

    if not args.drop_base_at_end:
        print(f"Base database kept: {base_db_name}")

    return 0 if all(success for _, _, success in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())




  
#result summary printed by the script on the qm9 CFG_COMPARE_CONFIG configs with the default settings:
#DEFAULT_EDGE_MODE = "directed" DEFAULT_JAR = "snapshot"
    # SUCCESS | config=config.tmp | db=qm9_cfg_compare_config | log=/local-scratch/localhome/mirzaei/GraphVAE-REQ/factorbase_motif_pipeline/log/qm9_cfg_compare_config_run.log
    # FAILED  | config=config1.tmp | db=qm9_cfg_compare_config1 | log=/local-scratch/localhome/mirzaei/GraphVAE-REQ/factorbase_motif_pipeline/log/qm9_cfg_compare_config1_run.log
    # SUCCESS | config=config2.tmp | db=qm9_cfg_compare_config2 | log=/local-scratch/localhome/mirzaei/GraphVAE-REQ/factorbase_motif_pipeline/log/qm9_cfg_compare_config2_run.log
    # Base database kept: qm9_cfg_compare_base
