#!/usr/bin/env python3
"""
Prepare a dataset database, write a database-specific config file, and launch FactorBase.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pymysql import connect

from factorbase_utils import (
    parse_mysql_address,
    print_section,
    quote_mysql_identifier,
    read_config_values,
    resolve_edge_mode,
    sanitize_path_component,
    update_config_dbname,
)


SCRIPT_DIR = Path(__file__).resolve().parent

# Default values you can edit in one place.
DEFAULT_DATASET = "GRID" # "PROTEINS" "QM9"
DEFAULT_DB_NAME = "grid_experiment1" #"proteins_experiment" "qm9_experiment"
DEFAULT_CONFIG_TEMPLATE = SCRIPT_DIR / "config.tmp"
DEFAULT_JAR = None        # "snapshot" or "patched"
DEFAULT_EDGE_MODE = None  # "directed", "undirected", or None to prompt
DEFAULT_GRID_FEATURE_MODE = "with-features"

DATASET_SCRIPTS = {
    "PROTEINS": SCRIPT_DIR / "PROTEINS_db.py",
    "QM9": SCRIPT_DIR / "QM9_db.py",
    "GRID": SCRIPT_DIR / "GRID_db.py",
}
JAR_FILES = {
    "snapshot": "factorbase-1.0-SNAPSHOT.jar",
    "patched": "factorbase-1.0-patched.jar",
}
EXPECTED_DATASET_TABLES = ("nodes", "edges")
REQUIRED_CONFIG_KEYS = ("dbaddress", "dbname", "dbusername", "dbpassword")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dataset import script, create a database-specific config file, "
            "and launch the selected FactorBase JAR."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Dataset name, for example PROTEINS or QM9 (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "db_name",
        nargs="?",
        default=DEFAULT_DB_NAME,
        help=f"MySQL database name to create and place into the generated config file (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--config-template",
        type=Path,
        default=DEFAULT_CONFIG_TEMPLATE,
        help="Template config file used to generate the database-specific config file",
    )
    parser.add_argument(
        "--jar",
        choices=sorted(JAR_FILES),
        default=DEFAULT_JAR,
        help="Jar to launch after preparation; if omitted, the script prompts",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the database and run directory without launching Java",
    )
    parser.add_argument(
        "--grid-feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_GRID_FEATURE_MODE,
        help="For GRID dataset only, choose whether to create a schema with or without features",
    )

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store both directions for each edge during dataset import",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store one canonical edge per undirected pair during dataset import",
    )
    args = parser.parse_args()

    if not args.directed and not args.undirected:
        if DEFAULT_EDGE_MODE == "directed":
            args.directed = True
        elif DEFAULT_EDGE_MODE == "undirected":
            args.undirected = True

    return args


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().upper()
    if normalized not in DATASET_SCRIPTS:
        supported = ", ".join(sorted(DATASET_SCRIPTS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return normalized


def build_generated_config_path(db_name: str) -> Path:
    config_name = f"{sanitize_path_component(db_name)}_config.cfg"
    return SCRIPT_DIR / config_name


def require_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def load_template_config(template_path: Path) -> str:
    require_path_exists(template_path, "config template")
    return template_path.read_text(encoding="utf-8")


def build_import_command(dataset_name: str, db_name: str, directed: bool) -> list[str]:
    command = [
        sys.executable,
        str(DATASET_SCRIPTS[dataset_name]),
        "--db-name",
        db_name,
        "--directed" if directed else "--undirected",
    ]
    return command


def append_dataset_specific_args(
    command: list[str],
    dataset_name: str,
    grid_feature_mode: str,
) -> list[str]:
    if dataset_name == "GRID":
        command.extend(["--feature-mode", grid_feature_mode])
    return command


def write_generated_config(config_path: Path, config_text: str, db_name: str) -> None:
    rendered_config = update_config_dbname(config_text, db_name)
    config_path.write_text(rendered_config, encoding="utf-8")


def normalize_mysql_host(host: str) -> str:
    return "127.0.0.1" if host == "localhost" else host


def load_and_validate_config_values(config_path: Path, expected_db_name: str) -> dict[str, str]:
    require_path_exists(config_path, "generated FactorBase config")
    config_values = read_config_values(config_path)
    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config_values]
    if missing_keys:
        raise RuntimeError(
            f"Generated config is missing required keys {missing_keys}: {config_path}"
        )

    actual_db_name = config_values["dbname"]
    if actual_db_name != expected_db_name:
        raise RuntimeError(
            f"Generated config points to '{actual_db_name}', expected '{expected_db_name}': "
            f"{config_path}"
        )

    return config_values


def verify_dataset_database(config_path: Path, expected_db_name: str) -> None:
    config_values = load_and_validate_config_values(config_path, expected_db_name)
    host, port = parse_mysql_address(config_values["dbaddress"])
    host = normalize_mysql_host(host)

    try:
        connection = connect(
            host=host,
            port=port,
            user=config_values["dbusername"],
            password=config_values["dbpassword"],
        )
    except Exception as exc:  # pragma: no cover - depends on local MySQL access
        raise RuntimeError(
            "Dataset import finished, but the pipeline could not connect to MySQL "
            f"to verify database '{expected_db_name}' at {host}:{port}."
        ) from exc

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s",
                (expected_db_name,),
            )
            database_exists = cursor.fetchone()[0] == 1
            if not database_exists:
                raise RuntimeError(
                    f"Dataset import completed, but MySQL database '{expected_db_name}' was not created."
                )

            for table_name in EXPECTED_DATASET_TABLES:
                table_identifier = quote_mysql_identifier(table_name)
                db_identifier = quote_mysql_identifier(expected_db_name)
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                    (expected_db_name, table_name),
                )
                table_exists = cursor.fetchone()[0] == 1
                if not table_exists:
                    raise RuntimeError(
                        f"Dataset database '{expected_db_name}' is missing required table '{table_name}'."
                    )

                cursor.execute(f"SELECT COUNT(*) FROM {db_identifier}.{table_identifier}")
                row_count = cursor.fetchone()[0]
                if row_count <= 0:
                    raise RuntimeError(
                        f"Dataset database '{expected_db_name}' has an empty required table '{table_name}'."
                    )
    finally:
        connection.close()


def run_subprocess_step(
    step_name: str,
    command: list[str],
    cwd: Path,
    required_markers: tuple[str, ...] = (),
) -> None:
    marker_seen = {marker: False for marker in required_markers}

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise RuntimeError(
            f"{step_name} could not be started: {command}"
        ) from exc

    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
            for marker in marker_seen:
                if not marker_seen[marker] and marker in line:
                    marker_seen[marker] = True
    finally:
        process.stdout.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{step_name} failed with exit code {return_code}: {command}")

    missing_markers = [marker for marker, seen in marker_seen.items() if not seen]
    if missing_markers:
        raise RuntimeError(
            f"{step_name} finished, but did not emit expected success markers: {missing_markers}"
        )


def choose_jar(jar_choice: str | None) -> str:
    if jar_choice is not None:
        return JAR_FILES[jar_choice]

    while True:
        choice = input(
            "Which FactorBase jar should be run?\n"
            "  1 - factorbase-1.0-SNAPSHOT.jar\n"
            "  2 - factorbase-1.0-patched.jar\n"
            "Choice: "
        ).strip()
        if choice == "1":
            return JAR_FILES["snapshot"]
        if choice == "2":
            return JAR_FILES["patched"]
        print("Please enter 1 or 2.")


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name != "GRID" and args.grid_feature_mode != DEFAULT_GRID_FEATURE_MODE:
        raise ValueError("--grid-feature-mode can only be used with the GRID dataset.")

    directed = resolve_edge_mode(args.directed, args.undirected)
    dataset_script_path = DATASET_SCRIPTS[dataset_name]
    require_path_exists(dataset_script_path, f"{dataset_name} dataset import script")
    template_text = load_template_config(args.config_template)
    config_path = build_generated_config_path(args.db_name)

    print_section("RUNNING DATASET IMPORT")
    import_command = build_import_command(dataset_name, args.db_name, directed)
    import_command = append_dataset_specific_args(
        import_command,
        dataset_name,
        args.grid_feature_mode,
    )
    run_subprocess_step("Dataset import", import_command, SCRIPT_DIR)

    print_section("WRITING FACTORBASE CONFIG")
    write_generated_config(config_path, template_text, args.db_name)
    load_and_validate_config_values(config_path, args.db_name)
    verify_dataset_database(config_path, args.db_name)
    print(f"Config file written: {config_path}")

    if args.prepare_only:
        print("\nPreparation complete. Skipping FactorBase launch because --prepare-only was used.")
        return

    jar_filename = choose_jar(args.jar)
    jar_path = SCRIPT_DIR / jar_filename
    require_path_exists(jar_path, "FactorBase JAR")

    print_section("LAUNCHING FACTORBASE")
    print(f"Running {jar_filename} with config {config_path.name}")
    run_subprocess_step(
        "FactorBase launch",
        ["java", f"-Dconfig={config_path.name}", "-jar", jar_filename],
        SCRIPT_DIR,
        required_markers=(f"Input Database: {args.db_name}", "Program Done!"),
    )


if __name__ == "__main__":
    main()
