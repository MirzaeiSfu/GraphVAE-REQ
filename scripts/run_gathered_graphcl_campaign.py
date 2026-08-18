#!/usr/bin/env python3
"""Evaluate one shard of a prepared gathered-checkpoint GraphCL campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_EVALUATION_SRC = REPO_ROOT / "graph_evaluation" / "src"
if str(GRAPH_EVALUATION_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_EVALUATION_SRC))

from ggm_eval.runner import evaluate_contrastive_checkpoints  # noqa: E402
from ggm_eval.trained import evaluate_with_trained_gnns  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "/local-scratch2/new/gather/pretrained_graphcl_evaluation"
)
DEFAULT_UPSTREAM = Path(
    "/local-scratch2/mirzaei/Abdolreza/upstreams/"
    "Self-Supervised-Models-for-GGM-Evaluation"
)
DEFAULT_PYTHON = Path(
    "/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python"
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_completed_evaluation(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("checkpoint_count", -1)) == 3
        and len(payload.get("per_checkpoint", ())) == 3
        and bool(payload.get("summary"))
    )


def write_status_csv(path: Path, rows: list[dict]):
    fieldnames = (
        "dataset",
        "setting",
        "generator_seed",
        "status",
        "error",
        "evaluation_json",
        "device",
        "shard_index",
        "shard_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--upstream-repo", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count).")

    output_root = args.output_root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output_root / "campaign_manifest.csv"
    )
    upstream = args.upstream_repo.expanduser().resolve()
    python = args.python.expanduser().resolve()
    all_rows = read_rows(manifest)
    selected = [
        row
        for index, row in enumerate(all_rows)
        if index % args.shard_count == args.shard_index
    ]
    custom_enzymes_checkpoints = [
        (
            output_root
            / "evaluators"
            / "ENZYMES_exact_schema"
            / f"seed_{seed}"
            / "checkpoint.pt"
        ).resolve()
        for seed in (0, 1, 2)
    ]
    statuses = []
    for row in selected:
        evaluation_dir = Path(row["evaluation_dir"]).resolve()
        evaluation_json = evaluation_dir / "evaluation.json"
        status = {
            "dataset": row["dataset"],
            "setting": row["setting"],
            "generator_seed": row["generator_seed"],
            "status": "",
            "error": "",
            "evaluation_json": str(evaluation_json),
            "device": args.device,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        }
        if row["status"] != "prepared":
            status["status"] = "not_prepared"
            status["error"] = row.get("error", "")
            statuses.append(status)
            continue
        if not args.force and valid_completed_evaluation(evaluation_json):
            status["status"] = "already_complete"
            statuses.append(status)
            print(
                f"[already_complete] {row['dataset']}/{row['setting']}/"
                f"seed_{row['generator_seed']}"
            )
            continue
        try:
            if row["dataset"] == "ENZYMES":
                missing = [
                    str(path)
                    for path in custom_enzymes_checkpoints
                    if not path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        "Missing exact-schema ENZYMES GraphCL checkpoints: "
                        + ", ".join(missing)
                    )
                evaluate_contrastive_checkpoints(
                    generated=row["generated"],
                    reference=row["reference"],
                    checkpoints=custom_enzymes_checkpoints,
                    upstream_repository=upstream,
                    output_dir=evaluation_dir,
                    python_executable=python,
                    device=args.device,
                    nearest_k=args.nearest_k,
                    max_graphs=args.max_graphs,
                )
            else:
                evaluate_with_trained_gnns(
                    dataset=row["dataset"],
                    generated=row["generated"],
                    reference=row["reference"],
                    output_dir=evaluation_dir,
                    upstream_repository=upstream,
                    seeds=(0, 1, 2),
                    python_executable=python,
                    device=args.device,
                    nearest_k=args.nearest_k,
                    max_graphs=args.max_graphs,
                )
            if not valid_completed_evaluation(evaluation_json):
                raise RuntimeError(
                    f"Evaluation did not produce a complete result: {evaluation_json}"
                )
            failure_path = evaluation_dir / "failure.json"
            if failure_path.exists():
                failure_path.unlink()
            status["status"] = "complete"
            print(
                f"[complete] {row['dataset']}/{row['setting']}/"
                f"seed_{row['generator_seed']}"
            )
        except Exception as exc:
            status["status"] = "failed"
            status["error"] = f"{type(exc).__name__}: {exc}"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            (evaluation_dir / "failure.json").write_text(
                json.dumps(
                    {
                        **status,
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[failed] {row['dataset']}/{row['setting']}/"
                f"seed_{row['generator_seed']}: {status['error']}",
                file=sys.stderr,
            )
        statuses.append(status)
        write_status_csv(
            output_root / f"evaluation_status_shard_{args.shard_index}.csv",
            statuses,
        )

    status_path = (
        output_root / f"evaluation_status_shard_{args.shard_index}.csv"
    )
    write_status_csv(status_path, statuses)
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "device": args.device,
        "selected_count": len(selected),
        "complete_count": sum(
            row["status"] in {"complete", "already_complete"}
            for row in statuses
        ),
        "failed_count": sum(row["status"] == "failed" for row in statuses),
        "status_csv": str(status_path.resolve()),
    }
    (
        output_root / f"evaluation_summary_shard_{args.shard_index}.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_count"]:
        raise SystemExit(f"{summary['failed_count']} evaluation(s) failed.")


if __name__ == "__main__":
    main()
