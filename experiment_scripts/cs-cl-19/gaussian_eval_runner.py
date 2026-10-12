#!/usr/bin/env python3
"""Re-run motif-count artifacts with the Gaussian-aligned v7 evaluator."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-config-template")
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_text in args.sources:
        source = Path(source_text)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        artifacts = payload["artifacts"]
        seed = int(payload["seed"])
        if source.name == "count_distance_v5.json":
            if "motif_false_3seed" in source.parts:
                setting_name = "false"
            elif "total_count" in source.parts:
                setting_name = "total"
            elif "full_matrix" in source.parts:
                setting_name = "full"
            else:
                raise ValueError(f"Cannot infer setting from {source}")
            output = output_dir / f"{setting_name}_seed{seed}.json"
        else:
            output = output_dir / source.name
        if output.is_file():
            try:
                with output.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if (
                    existing.get("schema_version")
                    == "graphvae-motif-count-distance-v7"
                    and "gaussian_aligned_score" in existing.get("soft", {})
                ):
                    print(f"[runner] complete, skipping {output}", flush=True)
                    continue
            except (OSError, ValueError):
                pass

        selection_config = artifacts.get("motif_selection_config")
        if not selection_config and args.selection_config_template:
            selection_config = args.selection_config_template.format(seed=seed)
        if not selection_config:
            selection_config = artifacts["config"]

        command = [
            args.python,
            args.evaluator,
            "--config",
            artifacts["config"],
            "--motif-selection-config",
            selection_config,
            "--checkpoint",
            artifacts["checkpoint"],
            "--dataset-cache",
            artifacts["dataset_cache"],
            "--motif-cache-dir",
            str(Path(artifacts["motif_cache"]).parent),
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--generation-batch-size",
            "16",
            "--count-batch-size",
            "256",
        ]
        if payload.get("dataset"):
            command.extend(["--dataset-label", str(payload["dataset"])])
        if payload.get("setting"):
            command.extend(["--setting", str(payload["setting"])])
        print(f"[runner] evaluating {source} -> {output}", flush=True)
        subprocess.run(command, cwd=args.repo, check=True)


if __name__ == "__main__":
    main()
