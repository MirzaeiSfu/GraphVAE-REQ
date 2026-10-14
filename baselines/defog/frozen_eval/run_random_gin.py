#!/usr/bin/env python3
"""Run the pinned third-party Random-GIN on one verified DeFoG seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GRAPH_EVAL_SRC = ROOT / "graph_evaluation" / "src"
if str(GRAPH_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPH_EVAL_SRC))

from ggm_eval.runner import evaluate_legacy_random_gin  # noqa: E402
from ggm_eval.reporting import write_json  # noqa: E402

from verify_campaign import (  # noqa: E402
    generated_path,
    load_yaml,
    verify_evaluator_files,
    verify_generated_artifact,
    verify_reference_artifact,
    artifact_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    manifest = load_yaml(manifest_path)
    verify_evaluator_files(manifest)
    dataset = args.dataset.upper()
    seed = int(args.training_seed)
    if seed not in [int(value) for value in manifest["protocol"]["training_seeds"]]:
        raise ValueError(f"Training seed {seed} is outside the frozen campaign")
    spec = manifest["datasets"][dataset]
    verification_spec = {
        **spec,
        "defog_commit": manifest["repositories"]["defog_benchmark_commit"],
    }
    reference = artifact_path(artifact_root, dataset, "reference")
    generated = generated_path(artifact_root, dataset, seed)
    reference_record = verify_reference_artifact(
        reference, dataset, "reference", spec
    )
    generated_record = verify_generated_artifact(
        generated,
        dataset=dataset,
        training_seed=seed,
        spec=verification_spec,
        protocol=manifest["protocol"],
    )
    if generated_record["summary"]["graph_count"] != reference_record["summary"]["graph_count"]:
        raise ValueError("Generated and reference counts must be exactly equal")

    modes = (
        ["decoded_node", "topology_control"]
        if spec["feature_mode"] == "decoded_node"
        else ["topology_control"]
    )
    output_dir = (
        args.output_root.expanduser().resolve()
        / dataset.lower()
        / f"seed_{seed}"
    )
    result = evaluate_legacy_random_gin(
        generated=generated,
        reference=reference,
        legacy_repository=ROOT,
        output_dir=output_dir,
        python_executable=args.python,
        modes=modes,
        repeats=10,
        evaluator_seed=0,
        nearest_k=5,
        max_graphs=0,
        device=args.device,
        trusted_input=False,
    )
    result["campaign"] = {
        "schema_version": manifest["schema_version"],
        "dataset": dataset,
        "training_seed": seed,
        "generation_seed": manifest["protocol"]["generation_seed"],
        "evaluator_seeds": manifest["evaluator"]["evaluator_seeds"],
        "aggregation_stage": "within_training_seed",
        "generated_provenance": generated_record["metadata"],
    }
    write_json(output_dir / "evaluation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
