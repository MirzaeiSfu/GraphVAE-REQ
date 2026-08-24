#!/usr/bin/env python3
"""Audit and summarize the frozen six-checkpoint LOBSTER held-out comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if __package__:
    from .graphvae_attr_bo_distributed import (
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        sha256_file,
    )
    from .tune_graphvae_attribute_weights import parse_attr_f1pr_file
else:
    sys.path.insert(0, str(SCRIPT_DIR))
    from graphvae_attr_bo_distributed import (  # noqa: E402
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        sha256_file,
    )
    from tune_graphvae_attribute_weights import parse_attr_f1pr_file  # noqa: E402

EXPECTED_OBJECTIVE = "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
EXPECTED_PLAN_SHA256 = "831b1c35169cb48bb8c1b06d03b800bf0ed07119b21ebfd7f07689d1d272d881"
EXPECTED_FROZEN_SHA256 = "cd03b7413f158505b1230c65c16a7638d4f03fb36a2be3845532e7ee2a3c59fb"
EXPECTED_CONFIRMATION_REPORT_SHA256 = "557f1d7e5ae596cb243c427682a18558a6da69b4686aed4586639e507100c9a0"
EXPECTED_SNAPSHOT_SHA256 = "7a8354cbc8e101197b9aa681fb056941f665e70ba2318ef51fc783a14b46a35f"
EXPECTED_TRIALS_CSV_SHA256 = "d74ce60bcd6b637bd72a9f5df5399419f9aafecda1aa0322663ef4347715aaab"


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DistributedContractError(f"Expected a JSON object: {path}")
    return value


def build_heldout_report(study_dir: Path, plan_path: Path) -> dict[str, Any]:
    study = study_dir.expanduser().resolve()
    plan_file = plan_path.expanduser().resolve()
    plan = _read_json(plan_file)
    definition = _read_json(study / "study_definition.json")
    frozen = _read_json(study / "FROZEN.json")
    confirmation = _read_json(study / "confirmation_report.json")
    contract_hash = canonical_contract_hash(definition)
    source_hashes = {
        "FROZEN.json": sha256_file(study / "FROZEN.json"),
        "confirmation_report.json": sha256_file(study / "confirmation_report.json"),
        "study_snapshot.sqlite3": sha256_file(study / "study_snapshot.sqlite3"),
        "trials.csv": sha256_file(study / "trials.csv"),
    }
    if (
        sha256_file(plan_file) != EXPECTED_PLAN_SHA256
        or source_hashes["FROZEN.json"] != EXPECTED_FROZEN_SHA256
        or source_hashes["confirmation_report.json"]
        != EXPECTED_CONFIRMATION_REPORT_SHA256
        or source_hashes["study_snapshot.sqlite3"] != EXPECTED_SNAPSHOT_SHA256
        or source_hashes["trials.csv"] != EXPECTED_TRIALS_CSV_SHA256
        or plan.get("study_contract_sha256") != contract_hash
        or frozen.get("study_contract_sha256") != contract_hash
        or frozen.get("lifecycle") != "FROZEN"
        or confirmation.get("study_contract_sha256") != contract_hash
        or confirmation.get("conclusion") != "no_improvement"
        or confirmation.get("no_reranking") is not True
        or confirmation.get("held_out_access") is not False
        or confirmation.get("test_access") is not False
        or confirmation.get("selection_split") != "validation"
        or confirmation.get("objective_json_path") != EXPECTED_OBJECTIVE
    ):
        raise DistributedContractError("Frozen validation inputs changed before held-out audit.")
    if (
        plan.get("schema_version")
        != "graphvae-attr-f1pr-confirmation-heldout-plan-v1"
        or plan.get("frozen_confirmation_conclusion") != "no_improvement"
        or plan.get("no_reranking") is not True
        or plan.get("allow_training") is not False
        or plan.get("allow_new_trials") is not False
        or plan.get("split") != "test"
        or plan.get("primary_mode") != "decoded_node_edge"
        or plan.get("objective_json_path") != EXPECTED_OBJECTIVE
        or plan.get("expected_test_graphs") != 20
        or plan.get("max_graphs") != 0
        or plan.get("max_parallel") != 3
    ):
        raise DistributedContractError("Held-out audit plan is not exact.")

    entries = plan.get("trials")
    if not isinstance(entries, list) or [row.get("trial_number") for row in entries] != list(range(6)):
        raise DistributedContractError("Held-out audit requires exactly trials 0 through 5.")
    heldout_root = study / "heldout_confirmation"
    actual_dirs = sorted(path.name for path in heldout_root.iterdir() if path.is_dir())
    expected_dirs = [f"trial_{number:05d}" for number in range(6)]
    if actual_dirs != expected_dirs or any((heldout_root / name / "FAILED").exists() for name in expected_dirs):
        raise DistributedContractError("Held-out result directory identities are not exact.")

    results = []
    by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        number = int(entry["trial_number"])
        result_dir = heldout_root / f"trial_{number:05d}"
        result = _read_json(result_dir / "heldout_result.json")
        evaluator_path = result_dir / "attributed_random_gin.json"
        metrics = parse_attr_f1pr_file(
            evaluator_path,
            expected_split="test",
            expected_graph_count=int(plan["expected_test_graphs"]),
            expected_node_feature_dimension=int(plan["node_feature_dimension"]),
            expected_edge_feature_dimension=int(plan["edge_feature_dimension"]),
            expected_cache_sha256=plan["cache_sha256"],
            expected_split_fingerprint=plan["test_split_fingerprint"],
            expected_node_schema_fingerprint=plan["node_schema_fingerprint"],
            expected_edge_schema_fingerprint=plan["edge_schema_fingerprint"],
            expected_generation_seed=int(plan["generation_seed"]),
            expected_evaluator_seed=int(plan["evaluator_seed"]),
            expected_repeats=int(plan["evaluator_repeats"]),
        )
        expected = {
            "schema_version": "graphvae-attr-f1pr-confirmation-heldout-result-v1",
            "study_name": plan["study_name"],
            "study_contract_sha256": contract_hash,
            "trial_number": number,
            "candidate": entry["candidate"],
            "training_seed": entry["training_seed"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "split": "test",
            "primary_mode": "decoded_node_edge",
            "objective_json_path": EXPECTED_OBJECTIVE,
            "accepted_test_graphs": 20,
            "generation_seed": plan["generation_seed"],
            "evaluator_seed": plan["evaluator_seed"],
            "evaluator_repeats": plan["evaluator_repeats"],
            "hostname": entry["hostname"],
            "physical_gpu": entry["physical_gpu"],
            "logical_device": "cuda:0",
            "no_reranking": True,
            "selection_changed": False,
            "new_trial_created": False,
            "training_ran": False,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise DistributedContractError(f"Held-out result {number} differs from its plan.")
        if (
            result.get("test_attr_f1pr") != metrics.f1_pr
            or result.get("test_precision") != metrics.precision
            or result.get("test_recall") != metrics.recall
            or result.get("evaluator_output_sha256") != sha256_file(evaluator_path)
        ):
            raise DistributedContractError(f"Held-out result {number} metrics or hash differ.")
        identity = (str(entry["candidate"]), int(entry["training_seed"]))
        if identity in by_identity:
            raise DistributedContractError("Duplicate held-out candidate/seed identity.")
        by_identity[identity] = result
        results.append(dict(result))

    pairs = []
    selected_values = []
    uniform_values = []
    differences = []
    for seed in (0, 1, 2):
        try:
            selected = by_identity[("selected", seed)]
            uniform = by_identity[("uniform", seed)]
        except KeyError as exc:
            raise DistributedContractError("Missing held-out candidate/seed identity.") from exc
        selected_value = float(selected["test_attr_f1pr"])
        uniform_value = float(uniform["test_attr_f1pr"])
        difference = selected_value - uniform_value
        selected_values.append(selected_value)
        uniform_values.append(uniform_value)
        differences.append(difference)
        pairs.append(
            {
                "training_seed": seed,
                "selected_trial_number": int(selected["trial_number"]),
                "uniform_trial_number": int(uniform["trial_number"]),
                "selected_test_attr_f1pr": selected_value,
                "uniform_test_attr_f1pr": uniform_value,
                "paired_difference": difference,
            }
        )

    return {
        "schema_version": "graphvae-attr-f1pr-confirmation-heldout-report-v1",
        "study_name": plan["study_name"],
        "study_contract_sha256": contract_hash,
        "heldout_plan_sha256": sha256_file(plan_file),
        "source_artifact_sha256": source_hashes,
        "validation_conclusion": "no_improvement",
        "validation_conclusion_changed": False,
        "heldout_role": "secondary_descriptive_evidence_only",
        "no_reranking": True,
        "training_ran": False,
        "new_trial_created": False,
        "postgresql_access": False,
        "split": "test",
        "primary_mode": "decoded_node_edge",
        "objective_json_path": EXPECTED_OBJECTIVE,
        "pairs": pairs,
        "candidate_summary": {
            "selected_mean_test_attr_f1pr": statistics.mean(selected_values),
            "uniform_mean_test_attr_f1pr": statistics.mean(uniform_values),
            "mean_paired_difference": statistics.mean(differences),
            "positive_paired_differences": sum(value > 0 for value in differences),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_heldout_report(args.study_dir, args.plan)
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
