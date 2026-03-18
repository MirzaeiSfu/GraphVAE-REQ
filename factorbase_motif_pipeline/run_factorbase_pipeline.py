#!/usr/bin/env python3
"""
Prepare a dataset database and a FactorBase run directory, then launch FactorBase.
"""

from __future__ import annotations

import argparse
import shutil
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
DEFAULT_RUN_DIR = None
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
            "Run a dataset import script, create a database-specific FactorBase run folder, "
            "write config.cfg from config.tmp, copy both JARs, and launch the selected JAR."
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
        help=f"MySQL database name to create and place into config.cfg (default: {DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--config-template",
        type=Path,
        default=DEFAULT_CONFIG_TEMPLATE,
        help="Template config file used to generate config.cfg",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Optional explicit run directory path",
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


def build_run_directory(dataset_name: str, db_name: str, explicit_run_dir: Path | None) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir.resolve()

    folder_name = f"factorbase_{sanitize_path_component(db_name)}_run"
    return SCRIPT_DIR / folder_name


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


def prepare_run_directory(run_dir: Path, config_text: str, db_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    rendered_config = update_config_dbname(config_text, db_name)
    (run_dir / "config.cfg").write_text(rendered_config, encoding="utf-8")

    for jar_filename in JAR_FILES.values():
        jar_source = SCRIPT_DIR / jar_filename
        if not jar_source.exists():
            raise FileNotFoundError(f"Required JAR not found: {jar_source}")
        shutil.copy2(jar_source, run_dir / jar_filename)


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
    run_dir = build_run_directory(dataset_name, args.db_name, args.run_dir)

    print_section("RUNNING DATASET IMPORT")
    import_command = build_import_command(dataset_name, args.db_name, directed)
    subprocess.run(import_command, cwd=SCRIPT_DIR, check=True)

    print_section("PREPARING FACTORBASE RUN DIRECTORY")
    prepare_run_directory(run_dir, template_text, args.db_name)
    print(f"Run directory ready: {run_dir}")
    print(f"Config file written: {run_dir / 'config.cfg'}")

    if args.prepare_only:
        print("\nPreparation complete. Skipping FactorBase launch because --prepare-only was used.")
        return

    jar_filename = choose_jar(args.jar)
    print_section("LAUNCHING FACTORBASE")
    print(f"Running {jar_filename} from {run_dir}")
    subprocess.run(["java", "-jar", jar_filename], cwd=run_dir, check=True)


if __name__ == "__main__":
    main()
