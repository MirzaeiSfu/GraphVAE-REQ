#!/usr/bin/env python3
"""Check observed-count consistency across all FactorBase CP table pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motif_counting.factorbase_count_audit import (  # noqa: E402
    audit_all_factorbase_database_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("factorbase_motif_pipeline/config.tmp"),
        help="FactorBase config containing the MySQL connection settings.",
    )
    parser.add_argument(
        "--database",
        action="append",
        dest="databases",
        help="Base database name to check; repeat to select multiple. Default: all.",
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--max-mismatches", type=int, default=20)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = audit_all_factorbase_database_counts(
        config_path=args.config,
        database_names=args.databases,
        atol=args.atol,
        max_mismatches=max(0, args.max_mismatches),
    )

    for result in results:
        print(
            f"{result['status']:5} {result['database']}: "
            f"tables={result['tables_checked']} "
            f"CP_rows={result['cp_rows']} "
            f"CP_smoothed_rows={result['smoothed_rows']} "
            f"added_zero_rows={result['added_zero_rows']} "
            f"mismatches={result['mismatch_count']}"
        )
        for mismatch in result["mismatches"]:
            print(f"      {mismatch}")

    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("PASS", "FAIL", "ERROR", "SKIP")
    }
    summary["databases"] = len(results)
    summary["tables_checked"] = sum(result["tables_checked"] for result in results)
    summary["cp_rows"] = sum(result["cp_rows"] for result in results)
    summary["smoothed_rows"] = sum(result["smoothed_rows"] for result in results)
    summary["added_zero_rows"] = sum(
        result["added_zero_rows"] for result in results
    )
    print(f"SUMMARY {json.dumps(summary, sort_keys=True)}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )

    return 1 if summary["FAIL"] or summary["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
