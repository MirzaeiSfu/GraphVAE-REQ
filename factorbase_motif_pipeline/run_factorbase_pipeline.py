#!/usr/bin/env python3
"""
Prepare a dataset database, write a database-specific config file, and launch FactorBase.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from factorbase_utils import (
    print_section,
    resolve_edge_mode,
    sanitize_path_component,
    update_config_dbname,
)


SCRIPT_DIR = Path(__file__).resolve().parent

# Default values you can edit in one place.
DEFAULT_DATASET = "QM9" # "PROTEINS" "QM9"
DEFAULT_DB_NAME = "qm9_experiment1" #"proteins_experiment" "qm9_experiment"
DEFAULT_CONFIG_TEMPLATE = SCRIPT_DIR / "config.tmp"
DEFAULT_JAR = None        # "snapshot" or "patched"
DEFAULT_EDGE_MODE = None  # "directed", "undirected", or None to prompt

DATASET_SCRIPTS = {
    "PROTEINS": SCRIPT_DIR / "PROTEINS_db.py",
    "QM9": SCRIPT_DIR / "QM9_db.py",
}
JAR_FILES = {
    "snapshot": "factorbase-1.0-SNAPSHOT.jar",
    "patched": "factorbase-1.0-patched.jar",
}


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


def load_template_config(template_path: Path) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found: {template_path}")

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


def write_generated_config(config_path: Path, config_text: str, db_name: str) -> None:
    rendered_config = update_config_dbname(config_text, db_name)
    config_path.write_text(rendered_config, encoding="utf-8")


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
    directed = resolve_edge_mode(args.directed, args.undirected)
    template_text = load_template_config(args.config_template)
    config_path = build_generated_config_path(args.db_name)

    print_section("RUNNING DATASET IMPORT")
    import_command = build_import_command(dataset_name, args.db_name, directed)
    subprocess.run(import_command, cwd=SCRIPT_DIR, check=True)

    print_section("WRITING FACTORBASE CONFIG")
    write_generated_config(config_path, template_text, args.db_name)
    print(f"Config file written: {config_path}")

    if args.prepare_only:
        print("\nPreparation complete. Skipping FactorBase launch because --prepare-only was used.")
        return

    jar_filename = choose_jar(args.jar)
    jar_path = SCRIPT_DIR / jar_filename
    if not jar_path.exists():
        raise FileNotFoundError(f"Required JAR not found: {jar_path}")

    print_section("LAUNCHING FACTORBASE")
    print(f"Running {jar_filename} with config {config_path.name}")
    subprocess.run(
        ["java", f"-Dconfig={config_path.name}", "-jar", jar_filename],
        cwd=SCRIPT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
