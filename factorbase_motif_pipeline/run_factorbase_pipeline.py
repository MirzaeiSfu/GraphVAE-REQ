#!/usr/bin/env python3
"""
Prepare or reuse a dataset database, write a database-specific config file, and launch
FactorBase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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
RUNS_DIR = SCRIPT_DIR / "runs"

# Default values you can edit in one place.
DEFAULT_DATASET = "LOBSTER" #"TRIANGULAR_GRID"#"LOBSTER" #""GRID"" # "PROTEINS" "QM9"
DEFAULT_CONFIG_TEMPLATE = SCRIPT_DIR / "config.tmp"
DEFAULT_JAR = "snapshot"        # "snapshot" or "patched"
DEFAULT_EDGE_MODE = "directed"  # "directed", "undirected", or None to prompt
DEFAULT_SYNTHETIC_EDGE_MODE = "undirected"
DEFAULT_GRID_FEATURE_MODE = "with-features"
DEFAULT_LOBSTER_FEATURE_MODE = "with-features"
DEFAULT_TRIANGULAR_GRID_FEATURE_MODE = "with-features"
AUTO_DB_NAME_HASH_LENGTH = 6

DATASET_SCRIPTS = {
    "PROTEINS": SCRIPT_DIR / "PROTEINS_db.py",
    "QM9": SCRIPT_DIR / "QM9_db.py",
    "GRID": SCRIPT_DIR / "GRID_db.py",
    "LOBSTER": SCRIPT_DIR / "LOBSTER_db.py",
    "TRIANGULAR_GRID": SCRIPT_DIR / "TRIANGULAR_GRID_db.py",
}
SYNTHETIC_DATASETS = {"GRID", "LOBSTER", "TRIANGULAR_GRID"}

# Edge-mode semantics:
# QM9/PROTEINS --directed:
#   source gives A->B and B->A
#   DB stores both
# Synthetic --directed:
#   source gives A-B once
#   DB stores one row
SYNTHETIC_DIRECTED_MISMATCH_ERROR = (
    "Synthetic datasets cannot be imported with --directed: using --directed, "
    "DB will store only one row per undirected edge, while main.py uses symmetric "
    "adjacency. That would be a mismatch. Use --undirected."
)

JAR_FILES = {
    "snapshot": "factorbase-1.0-SNAPSHOT.jar",
    "patched": "factorbase-1.0-patched.jar",
}
EXPECTED_DATASET_TABLES = ("nodes", "edges")
REQUIRED_CONFIG_KEYS = ("dbaddress", "dbname", "dbusername", "dbpassword")
RUN_METADATA_TABLE = "run_metadata"


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
        default=None,
        help=(
            "Optional MySQL database name. If omitted, a reproducible name is "
            "computed from dataset, edge mode, feature mode, config template, and JAR."
        ),
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
        help=f"Jar to launch after preparation (default: {DEFAULT_JAR})",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the database and run directory without launching Java",
    )
    parser.add_argument(
        "--use-existing-db",
        action="store_true",
        help="Skip dataset import and reuse an existing database with populated nodes/edges tables",
    )
    parser.add_argument(
        "--grid-feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_GRID_FEATURE_MODE,
        help="For GRID dataset only, choose whether to create a schema with or without features",
    )
    parser.add_argument(
        "--lobster-feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_LOBSTER_FEATURE_MODE,
        help="For LOBSTER dataset only, choose whether to create a schema with or without features",
    )
    parser.add_argument(
        "--triangular-grid-feature-mode",
        choices=("with-features", "without-features"),
        default=DEFAULT_TRIANGULAR_GRID_FEATURE_MODE,
        help="For TRIANGULAR_GRID dataset only, choose whether to create a schema with or without features",
    )
    parser.add_argument(
        "--debug-db-edges",
        action="store_true",
        help="For GRID/LOBSTER/TRIANGULAR_GRID only: print source-to-database edge mappings for a limited set of graphs",
    )
    parser.add_argument(
        "--debug-all-db-edges",
        action="store_true",
        help="For GRID/LOBSTER/TRIANGULAR_GRID only: print every source-to-database edge mapping",
    )
    parser.add_argument(
        "--debug-db-graph-limit",
        type=int,
        default=2,
        help="When --debug-db-edges is set, print edge mappings for this many synthetic graphs",
    )
    parser.add_argument(
        "--debug-db-edge-limit",
        type=int,
        default=20,
        help="When --debug-db-edges is set, print this many source edges per synthetic graph",
    )

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store exactly the source graph edge rows",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store both directions for each source edge pair",
    )
    args = parser.parse_args()

    return args


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().upper()
    if normalized not in DATASET_SCRIPTS:
        supported = ", ".join(sorted(DATASET_SCRIPTS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return normalized


def build_run_dir(db_name: str) -> Path:
    return RUNS_DIR / sanitize_path_component(db_name)


def build_run_config_path(run_dir: Path) -> Path:
    return run_dir / "factorbase_config.cfg"


def ensure_generated_output_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def require_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def load_template_config(template_path: Path) -> str:
    require_path_exists(template_path, "config template")
    return template_path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git_command(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=SCRIPT_DIR.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_git_metadata() -> dict[str, str | bool | None]:
    status = run_git_command(["status", "--short"])
    return {
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status or "",
    }


def canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_manifest_hash(manifest: dict) -> str:
    material = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_import_command(
    dataset_name: str,
    db_name: str,
    edge_mode: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(DATASET_SCRIPTS[dataset_name]),
        "--db-name",
        db_name,
    ]
    if edge_mode is not None:
        command.append(f"--{edge_mode}")
    return command


def append_dataset_specific_args(
    command: list[str],
    dataset_name: str,
    grid_feature_mode: str,
    lobster_feature_mode: str,
    triangular_grid_feature_mode: str,
) -> list[str]:
    if dataset_name == "GRID":
        command.extend(["--feature-mode", grid_feature_mode])
    elif dataset_name == "LOBSTER":
        command.extend(["--feature-mode", lobster_feature_mode])
    elif dataset_name == "TRIANGULAR_GRID":
        command.extend(["--feature-mode", triangular_grid_feature_mode])
    return command


def append_synthetic_debug_args(command: list[str], dataset_name: str, args: argparse.Namespace) -> list[str]:
    if dataset_name not in SYNTHETIC_DATASETS:
        return command

    if args.debug_db_edges:
        command.append("--debug-edges")
    if args.debug_all_db_edges:
        command.append("--debug-all-edges")
    if args.debug_db_graph_limit is not None:
        command.extend(["--debug-graph-limit", str(args.debug_db_graph_limit)])
    if args.debug_db_edge_limit is not None:
        command.extend(["--debug-edge-limit", str(args.debug_db_edge_limit)])
    return command


def write_generated_config(config_path: Path, config_text: str, db_name: str) -> None:
    rendered_config = update_config_dbname(config_text, db_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered_config, encoding="utf-8")


def write_command_file(
    run_dir: Path,
    wrapper_command: list[str],
    import_command: list[str] | None,
    factorbase_command: list[str] | None,
) -> Path:
    command_path = run_dir / "command.txt"
    lines = [
        "# Wrapper command",
        shlex.join(wrapper_command),
        "",
    ]
    if import_command is not None:
        lines.extend(["# Dataset import command", shlex.join(import_command), ""])
    if factorbase_command is not None:
        lines.extend(["# FactorBase command", shlex.join(factorbase_command), ""])
    command_path.write_text("\n".join(lines), encoding="utf-8")
    return command_path


def extract_source_edge_analysis_from_log(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}

    wanted_prefixes = (
        "Dataset:",
        "Graphs analyzed:",
        "Graphs with edges:",
        "Source edge rows:",
        "Unique undirected edge pairs:",
        "Rows missing reverse edge:",
        "Typed rows missing reverse edge with same bond_type:",
    )
    key_map = {
        "Dataset": "dataset",
        "Graphs analyzed": "graphs_analyzed",
        "Graphs with edges": "graphs_with_edges",
        "Source edge rows": "source_edge_rows",
        "Unique undirected edge pairs": "source_undirected_pairs",
        "Rows missing reverse edge": "source_missing_reverse_rows",
        "Typed rows missing reverse edge with same bond_type": "source_missing_typed_reverse_rows",
    }

    stats = {}
    in_section = False
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "SOURCE EDGE DIRECTION ANALYSIS":
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("=") and stats:
            break
        if not line.startswith(wanted_prefixes) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key_map.get(key)
        if normalized_key:
            stats[normalized_key] = value.strip()
    return stats


SOURCE_EDGE_ANALYSIS_DESCRIPTIONS = {
    "source_edge_rows": (
        "Number of edge rows exposed by the dataset loader before the importer "
        "adds or removes anything."
    ),
    "source_undirected_pairs": (
        "Number of unique unordered source edge pairs after treating A->B and "
        "B->A as the same relation."
    ),
    "source_missing_reverse_rows": (
        "Number of source rows A->B that do not also have B->A in the source."
    ),
    "effective_db_edge_relation": (
        "What the DB edge table represents after applying the selected edge mode."
    ),
}


def parse_manifest_count(value: str | None) -> int | None:
    if value is None:
        return None
    normalized_value = value.replace(",", "").strip()
    if not normalized_value:
        return None
    try:
        return int(normalized_value)
    except ValueError:
        return None


def describe_effective_db_edge_relation(
    edge_mode_label: str,
    source_edge_analysis: dict[str, str],
) -> str:
    if edge_mode_label == "reused existing database":
        return "existing database reused; importer did not create or transform edge rows"

    source_edge_rows = parse_manifest_count(source_edge_analysis.get("source_edge_rows"))
    missing_reverse_rows = parse_manifest_count(
        source_edge_analysis.get("source_missing_reverse_rows")
    )

    if edge_mode_label == "undirected":
        return "bidirectional; importer writes A->B and B->A for each source edge pair"

    if edge_mode_label == "directed":
        if source_edge_rows == 0:
            return "source rows preserved; no source edge rows were observed"
        if missing_reverse_rows == 0:
            return (
                "bidirectional source rows preserved; directed mode has no edge-row "
                "effect compared with undirected"
            )
        if missing_reverse_rows is not None:
            return "source rows preserved; DB may contain one-way edge relations"
        return "source rows preserved; source bidirectionality was not available"

    return f"{edge_mode_label}; effective DB edge relation was not classified"


def append_log_message(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message)
        if not message.endswith("\n"):
            log_file.write("\n")


def initialize_run_log(
    log_path: Path,
    dataset_name: str,
    db_name: str,
    edge_mode_label: str,
    prepare_only: bool,
    use_existing_db: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    append_log_message(log_path, "=" * 60)
    append_log_message(log_path, "FACTORBASE PIPELINE RUN")
    append_log_message(log_path, "=" * 60)
    append_log_message(log_path, f"Dataset: {dataset_name}")
    append_log_message(log_path, f"Database: {db_name}")
    append_log_message(log_path, f"Edge mode: {edge_mode_label}")
    append_log_message(log_path, f"Use existing database: {use_existing_db}")
    append_log_message(log_path, f"Prepare only: {prepare_only}")
    append_log_message(log_path, "")


def choose_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def archive_factorbase_jar_log_to_run_dir(jar_filename: str, run_dir: Path) -> Path | None:
    source_log_path = SCRIPT_DIR / f"{Path(jar_filename).stem}.log"
    if not source_log_path.exists():
        return None

    destination_log_path = choose_available_path(run_dir / source_log_path.name)
    destination_log_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_log_path), str(destination_log_path))
    return destination_log_path


def archive_factorbase_output_dir_to_run_dir(db_name: str, run_dir: Path) -> Path | None:
    output_dir_candidates = [
        SCRIPT_DIR / db_name,
        SCRIPT_DIR / sanitize_path_component(db_name),
    ]
    for source_output_dir in output_dir_candidates:
        if not source_output_dir.exists() or not source_output_dir.is_dir():
            continue
        if source_output_dir.resolve() == run_dir.resolve():
            continue

        destination_output_dir = choose_available_path(run_dir / "factorbase_output")
        destination_output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_output_dir), str(destination_output_dir))
        return destination_output_dir

    return None


def archive_factorbase_loose_outputs_to_run_dir(db_name: str, run_dir: Path) -> list[Path]:
    output_candidates = [
        SCRIPT_DIR / f"Bif_{db_name}.xml",
        SCRIPT_DIR / f"Bif_{sanitize_path_component(db_name)}.xml",
        SCRIPT_DIR / "dag_.txt",
    ]
    archived_paths = []
    for source_output_path in output_candidates:
        if not source_output_path.exists() or not source_output_path.is_file():
            continue

        destination_path = choose_available_path(run_dir / source_output_path.name)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_output_path), str(destination_path))
        archived_paths.append(destination_path)

    return archived_paths


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
    from pymysql import connect

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


def write_manifest_to_database(
    config_values: dict[str, str],
    db_name: str,
    manifest: dict,
) -> None:
    from pymysql import connect

    host, port = parse_mysql_address(config_values["dbaddress"])
    host = normalize_mysql_host(host)
    connection = connect(
        host=host,
        port=port,
        user=config_values["dbusername"],
        password=config_values["dbpassword"],
        database=db_name,
    )
    metadata_table = quote_mysql_identifier(RUN_METADATA_TABLE)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {metadata_table} (
                    meta_key VARCHAR(128) PRIMARY KEY,
                    meta_value MEDIUMTEXT NOT NULL
                )
                """
            )
            rows = []
            for key, value in sorted(manifest.items()):
                if isinstance(value, (dict, list)):
                    value_text = json.dumps(value, sort_keys=True)
                elif value is None:
                    value_text = "null"
                else:
                    value_text = str(value)
                rows.append((key, value_text))
            rows.append(("manifest_json", json.dumps(manifest, sort_keys=True)))
            cursor.executemany(
                f"""
                REPLACE INTO {metadata_table} (meta_key, meta_value)
                VALUES (%s, %s)
                """,
                rows,
            )
        connection.commit()
    finally:
        connection.close()


def drop_manifest_metadata_table_from_database(
    config_values: dict[str, str],
    db_name: str,
) -> None:
    from pymysql import connect

    host, port = parse_mysql_address(config_values["dbaddress"])
    host = normalize_mysql_host(host)
    connection = connect(
        host=host,
        port=port,
        user=config_values["dbusername"],
        password=config_values["dbpassword"],
        database=db_name,
    )
    metadata_table = quote_mysql_identifier(RUN_METADATA_TABLE)

    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {metadata_table}")
        connection.commit()
    finally:
        connection.close()


def run_subprocess_step(
    step_name: str,
    command: list[str],
    cwd: Path,
    required_markers: tuple[str, ...] = (),
    log_path: Path | None = None,
) -> None:
    marker_seen = {marker: False for marker in required_markers}
    log_file = None

    if log_path is not None:
        append_log_message(log_path, "=" * 60)
        append_log_message(log_path, step_name.upper())
        append_log_message(log_path, "=" * 60)
        append_log_message(log_path, f"CWD: {cwd}")
        append_log_message(log_path, f"COMMAND: {shlex.join(command)}")
        append_log_message(log_path, "")

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
    if log_path is not None:
        log_file = log_path.open("a", encoding="utf-8")
    try:
        for line in process.stdout:
            print(line, end="")
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
            for marker in marker_seen:
                if not marker_seen[marker] and marker in line:
                    marker_seen[marker] = True
    finally:
        process.stdout.close()
        if log_file is not None:
            log_file.close()

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


def dataset_feature_mode(dataset_name: str, args: argparse.Namespace) -> str | None:
    if dataset_name == "GRID":
        return args.grid_feature_mode
    if dataset_name == "LOBSTER":
        return args.lobster_feature_mode
    if dataset_name == "TRIANGULAR_GRID":
        return args.triangular_grid_feature_mode
    return None


def feature_mode_alias(feature_mode: str | None) -> str:
    if feature_mode == "without-features":
        return "nofeat"
    return "feat"


def edge_mode_alias(edge_mode_label: str) -> str:
    if edge_mode_label == "directed":
        return "dir"
    if edge_mode_label == "undirected":
        return "undir"
    if edge_mode_label == "reused existing database":
        return "reuse"
    return sanitize_path_component(edge_mode_label.lower())


def jar_choice_alias(jar_choice: str | None) -> str:
    if jar_choice == "snapshot":
        return "snap"
    if jar_choice == "patched":
        return "patch"
    return "jar"


def resolve_pipeline_edge_mode(
    dataset_name: str,
    args: argparse.Namespace,
) -> tuple[str | None, str]:
    directed_flag = args.directed
    undirected_flag = args.undirected
    if dataset_name in SYNTHETIC_DATASETS and not directed_flag and not undirected_flag:
        if DEFAULT_SYNTHETIC_EDGE_MODE == "undirected":
            undirected_flag = True
        else:
            raise ValueError(SYNTHETIC_DIRECTED_MISMATCH_ERROR)
    elif not directed_flag and not undirected_flag:
        if DEFAULT_EDGE_MODE == "directed":
            directed_flag = True
        elif DEFAULT_EDGE_MODE == "undirected":
            undirected_flag = True

    edge_mode = resolve_edge_mode(directed_flag, undirected_flag)
    if args.use_existing_db:
        return None, edge_mode
    return edge_mode, edge_mode


def build_auto_db_name_material(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    edge_mode_label: str,
) -> dict:
    jar_filename = JAR_FILES.get(args.jar) if args.jar is not None else None
    jar_path = SCRIPT_DIR / jar_filename if jar_filename is not None else None
    return {
        "dataset": dataset_name,
        "edge_mode": edge_mode_label,
        "feature_mode": dataset_feature_mode(dataset_name, args),
        "config_template_sha256": sha256_file(args.config_template),
        "jar_choice": args.jar,
        "jar_filename": jar_filename,
        "jar_sha256": sha256_file(jar_path) if jar_path is not None else None,
        "synthetic_dataset": dataset_name in SYNTHETIC_DATASETS,
    }


def build_auto_db_name(
    *,
    dataset_name: str,
    edge_mode_label: str,
    feature_mode: str | None,
    jar_choice: str | None,
    material: dict,
) -> str:
    digest = hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()
    name_parts = [
        sanitize_path_component(dataset_name.lower()),
        edge_mode_alias(edge_mode_label),
        feature_mode_alias(feature_mode),
        jar_choice_alias(jar_choice),
        digest[:AUTO_DB_NAME_HASH_LENGTH],
    ]
    return sanitize_path_component("_".join(name_parts))


def resolve_db_name(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    edge_mode_label: str,
) -> tuple[str, str, dict]:
    material = build_auto_db_name_material(
        args=args,
        dataset_name=dataset_name,
        edge_mode_label=edge_mode_label,
    )
    if args.db_name:
        return args.db_name, "provided", material

    return (
        build_auto_db_name(
            dataset_name=dataset_name,
            edge_mode_label=edge_mode_label,
            feature_mode=dataset_feature_mode(dataset_name, args),
            jar_choice=args.jar,
            material=material,
        ),
        "auto",
        material,
    )


def build_rule_manifest(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    db_name: str,
    db_name_source: str,
    auto_db_name_material: dict,
    edge_mode_label: str,
    run_dir: Path,
    run_log_path: Path,
    config_template_path: Path,
    run_config_path: Path,
    import_command: list[str] | None,
    factorbase_command: list[str] | None,
    command_file_path: Path,
    jar_filename: str | None,
    factorbase_status: str,
    jar_log_archive_path: Path | None = None,
    factorbase_output_dir_archive_path: Path | None = None,
    factorbase_loose_output_archive_paths: list[Path] | None = None,
) -> dict:
    jar_path = SCRIPT_DIR / jar_filename if jar_filename is not None else None
    source_edge_analysis = extract_source_edge_analysis_from_log(run_log_path)
    manifest = {
        "manifest_schema_version": "rule-learning-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_name": db_name,
        "db_name_source": db_name_source,
        "auto_db_name_material": auto_db_name_material,
        "dataset": dataset_name,
        "edge_mode": edge_mode_label,
        "effective_db_edge_relation": describe_effective_db_edge_relation(
            edge_mode_label,
            source_edge_analysis,
        ),
        "feature_mode": dataset_feature_mode(dataset_name, args),
        "use_existing_db": bool(args.use_existing_db),
        "prepare_only": bool(args.prepare_only),
        "debug_db_edges": bool(args.debug_db_edges),
        "debug_all_db_edges": bool(args.debug_all_db_edges),
        "debug_db_graph_limit": args.debug_db_graph_limit,
        "debug_db_edge_limit": args.debug_db_edge_limit,
        "synthetic_dataset": dataset_name in SYNTHETIC_DATASETS,
        "source_edge_analysis": source_edge_analysis,
        "source_edge_analysis_descriptions": SOURCE_EDGE_ANALYSIS_DESCRIPTIONS,
        "config_template_path": str(config_template_path),
        "config_template_sha256": sha256_file(config_template_path),
        "run_config_path": str(run_config_path),
        "run_config_sha256": sha256_file(run_config_path),
        "jar_choice": args.jar,
        "jar_filename": jar_filename,
        "jar_sha256": sha256_file(jar_path) if jar_path is not None else None,
        "factorbase_status": factorbase_status,
        "factorbase_jar_log_archive_path": (
            str(jar_log_archive_path) if jar_log_archive_path is not None else None
        ),
        "factorbase_output_dir_archive_path": (
            str(factorbase_output_dir_archive_path)
            if factorbase_output_dir_archive_path is not None
            else None
        ),
        "factorbase_loose_output_archive_paths": [
            str(path) for path in factorbase_loose_output_archive_paths or []
        ],
        "run_dir": str(run_dir),
        "run_log_path": str(run_log_path),
        "command_file_path": str(command_file_path),
        "wrapper_command": shlex.join([sys.executable, *sys.argv]),
        "import_command": shlex.join(import_command) if import_command is not None else None,
        "factorbase_command": (
            shlex.join(factorbase_command) if factorbase_command is not None else None
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": get_git_metadata(),
    }
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    manifest["manifest_hash_short"] = manifest["manifest_hash"][:8]
    return manifest


def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name != "GRID" and args.grid_feature_mode != DEFAULT_GRID_FEATURE_MODE:
        raise ValueError("--grid-feature-mode can only be used with the GRID dataset.")
    if dataset_name != "LOBSTER" and args.lobster_feature_mode != DEFAULT_LOBSTER_FEATURE_MODE:
        raise ValueError("--lobster-feature-mode can only be used with the LOBSTER dataset.")
    if (
        dataset_name != "TRIANGULAR_GRID"
        and args.triangular_grid_feature_mode != DEFAULT_TRIANGULAR_GRID_FEATURE_MODE
    ):
        raise ValueError(
            "--triangular-grid-feature-mode can only be used with the TRIANGULAR_GRID dataset."
        )
    if dataset_name in SYNTHETIC_DATASETS and args.directed:
        raise ValueError(SYNTHETIC_DIRECTED_MISMATCH_ERROR)

    ensure_generated_output_dirs()
    template_text = load_template_config(args.config_template)
    edge_mode, edge_mode_label = resolve_pipeline_edge_mode(dataset_name, args)
    db_name, db_name_source, auto_db_name_material = resolve_db_name(
        args=args,
        dataset_name=dataset_name,
        edge_mode_label=edge_mode_label,
    )
    args.db_name = db_name
    run_dir = build_run_dir(db_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = build_run_config_path(run_dir)
    run_log_path = run_dir / "run.log"
    import_command = None
    factorbase_command = None
    jar_filename = JAR_FILES.get(args.jar) if args.jar is not None else None
    command_file_path = run_dir / "command.txt"

    print_section("RESOLVED RUN IDENTITY")
    print(f"Dataset: {dataset_name}")
    print(f"Database: {db_name} ({db_name_source})")
    print(f"Run directory: {run_dir}")

    initialize_run_log(
        run_log_path,
        dataset_name,
        db_name,
        edge_mode_label,
        args.prepare_only,
        args.use_existing_db,
    )
    append_log_message(run_log_path, f"Database name source: {db_name_source}")
    append_log_message(run_log_path, f"Run directory: {run_dir}")
    append_log_message(run_log_path, "")

    if args.use_existing_db:
        print_section("REUSING EXISTING DATASET DATABASE")
        message = (
            f"Skipping dataset import and reusing existing database '{db_name}'."
        )
        print(message)
        append_log_message(run_log_path, message)
        append_log_message(run_log_path, "")
    else:
        dataset_script_path = DATASET_SCRIPTS[dataset_name]
        require_path_exists(dataset_script_path, f"{dataset_name} dataset import script")
        print_section("RUNNING DATASET IMPORT")
        import_command = build_import_command(
            dataset_name,
            db_name,
            edge_mode,
        )
        import_command = append_dataset_specific_args(
            import_command,
            dataset_name,
            args.grid_feature_mode,
            args.lobster_feature_mode,
            args.triangular_grid_feature_mode,
        )
        import_command = append_synthetic_debug_args(import_command, dataset_name, args)
        run_subprocess_step("Dataset import", import_command, SCRIPT_DIR, log_path=run_log_path)

    print_section("WRITING FACTORBASE CONFIG")
    write_generated_config(run_config_path, template_text, db_name)
    config_values = load_and_validate_config_values(run_config_path, db_name)
    verify_dataset_database(run_config_path, db_name)
    print(f"Run config written: {run_config_path}")
    append_log_message(run_log_path, f"Run config written: {run_config_path}")
    append_log_message(run_log_path, "")

    if args.prepare_only:
        command_file_path = write_command_file(
            run_dir,
            [sys.executable, *sys.argv],
            import_command,
            factorbase_command,
        )
        manifest = build_rule_manifest(
            args=args,
            dataset_name=dataset_name,
            db_name=db_name,
            db_name_source=db_name_source,
            auto_db_name_material=auto_db_name_material,
            edge_mode_label=edge_mode_label,
            run_dir=run_dir,
            run_log_path=run_log_path,
            config_template_path=args.config_template,
            run_config_path=run_config_path,
            import_command=import_command,
            factorbase_command=factorbase_command,
            command_file_path=command_file_path,
            jar_filename=jar_filename,
            factorbase_status="prepare_only",
        )
        manifest_path = run_dir / "rule_manifest.json"
        write_json(manifest_path, manifest)
        append_log_message(run_log_path, f"Rule manifest written: {manifest_path}")
        append_log_message(
            run_log_path,
            (
                f"SQL metadata table not written during --prepare-only because "
                f"FactorBase scans source tables during setup."
            ),
        )
        print("\nPreparation complete. Skipping FactorBase launch because --prepare-only was used.")
        append_log_message(
            run_log_path,
            "Preparation complete. Skipping FactorBase launch because --prepare-only was used.",
        )
        return

    jar_filename = choose_jar(args.jar)
    jar_path = SCRIPT_DIR / jar_filename
    require_path_exists(jar_path, "FactorBase JAR")
    factorbase_command = ["java", f"-Dconfig={run_config_path}", "-jar", jar_filename]
    command_file_path = write_command_file(
        run_dir,
        [sys.executable, *sys.argv],
        import_command,
        factorbase_command,
    )
    manifest_path = run_dir / "rule_manifest.json"
    manifest = build_rule_manifest(
        args=args,
        dataset_name=dataset_name,
        db_name=db_name,
        db_name_source=db_name_source,
        auto_db_name_material=auto_db_name_material,
        edge_mode_label=edge_mode_label,
        run_dir=run_dir,
        run_log_path=run_log_path,
        config_template_path=args.config_template,
        run_config_path=run_config_path,
        import_command=import_command,
        factorbase_command=factorbase_command,
        command_file_path=command_file_path,
        jar_filename=jar_filename,
        factorbase_status="pending",
    )
    write_json(manifest_path, manifest)
    drop_manifest_metadata_table_from_database(config_values, db_name)
    append_log_message(
        run_log_path,
        f"SQL metadata table cleared before FactorBase launch: {RUN_METADATA_TABLE}",
    )

    print_section("LAUNCHING FACTORBASE")
    print(f"Running {jar_filename} with config {run_config_path}")
    run_subprocess_step(
        "FactorBase launch",
        factorbase_command,
        SCRIPT_DIR,
        required_markers=(f"Input Database: {db_name}", "Program Done!"),
        log_path=run_log_path,
    )
    jar_log_archive_path = archive_factorbase_jar_log_to_run_dir(jar_filename, run_dir)
    if jar_log_archive_path is not None:
        append_log_message(run_log_path, f"FactorBase JAR log moved to: {jar_log_archive_path}")
    factorbase_output_dir_archive_path = archive_factorbase_output_dir_to_run_dir(db_name, run_dir)
    if factorbase_output_dir_archive_path is not None:
        append_log_message(
            run_log_path,
            f"FactorBase output directory moved to: {factorbase_output_dir_archive_path}",
        )
    factorbase_loose_output_archive_paths = archive_factorbase_loose_outputs_to_run_dir(
        db_name,
        run_dir,
    )
    for archive_path in factorbase_loose_output_archive_paths:
        append_log_message(run_log_path, f"FactorBase output file moved to: {archive_path}")
    manifest = build_rule_manifest(
        args=args,
        dataset_name=dataset_name,
        db_name=db_name,
        db_name_source=db_name_source,
        auto_db_name_material=auto_db_name_material,
        edge_mode_label=edge_mode_label,
        run_dir=run_dir,
        run_log_path=run_log_path,
        config_template_path=args.config_template,
        run_config_path=run_config_path,
        import_command=import_command,
        factorbase_command=factorbase_command,
        command_file_path=command_file_path,
        jar_filename=jar_filename,
        factorbase_status="completed",
        jar_log_archive_path=jar_log_archive_path,
        factorbase_output_dir_archive_path=factorbase_output_dir_archive_path,
        factorbase_loose_output_archive_paths=factorbase_loose_output_archive_paths,
    )
    write_json(manifest_path, manifest)
    write_manifest_to_database(config_values, db_name, manifest)
    append_log_message(run_log_path, f"Rule manifest written: {manifest_path}")
    append_log_message(run_log_path, f"SQL metadata table updated: {RUN_METADATA_TABLE}")


if __name__ == "__main__":
    main()
