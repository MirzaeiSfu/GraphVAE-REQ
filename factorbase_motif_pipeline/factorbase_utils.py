#!/usr/bin/env python3
"""
Shared helpers for the FactorBase motif pipeline scripts.
"""

from __future__ import annotations

import re
from pathlib import Path


def print_section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def prompt_non_empty(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Please enter a non-empty value.")


def resolve_edge_mode(directed_flag: bool, undirected_flag: bool) -> bool:
    if directed_flag:
        print("Selected: DIRECTED\n")
        return True

    if undirected_flag:
        print("Selected: UNDIRECTED\n")
        return False

    while True:
        choice = input(
            "Edge storage mode?\n"
            "  1 - DIRECTED (A->B and B->A)\n"
            "  2 - UNDIRECTED (only one stored edge per pair)\n"
            "Choice: "
        ).strip()
        if choice == "1":
            print("Selected: DIRECTED\n")
            return True
        if choice == "2":
            print("Selected: UNDIRECTED\n")
            return False
        print("Please enter 1 or 2.")


def quote_mysql_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def read_config_values(config_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    with config_path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or "=" not in raw_line:
                continue

            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()

    return values


def parse_mysql_address(dbaddress: str) -> tuple[str, int]:
    normalized = dbaddress
    for prefix in ("mysql://", "mariadb://"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    normalized = normalized.split("/", 1)[0]

    if ":" in normalized:
        host, port_text = normalized.rsplit(":", 1)
        return host, int(port_text)

    return normalized, 3306


def update_config_dbname(template_text: str, db_name: str) -> str:
    lines = template_text.splitlines()
    updated_lines = []
    replaced = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key == "dbname":
            prefix = line[: line.index("=") + 1]
            updated_lines.append(f"{prefix} {db_name}")
            replaced = True
        else:
            updated_lines.append(line)

    if not replaced:
        updated_lines.append(f"dbname = {db_name}")

    return "\n".join(updated_lines) + "\n"


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "run"
