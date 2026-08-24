#!/usr/bin/env python3
"""Apply the frozen paired LOBSTER Attr-F1PR confirmation policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .graphvae_attr_bo_distributed import (
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        sha256_file,
    )
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR))
    from graphvae_attr_bo_distributed import (  # noqa: E402
        DistributedContractError,
        atomic_write_json,
        canonical_contract_hash,
        sha256_file,
    )

T_CRITICAL_95_DF2 = 4.302652729696142
EXPECTED_OBJECTIVE = "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
EXPECTED_SUPERIORITY_RULE = (
    "all three paired differences are positive and the 95% paired t interval "
    "lower bound is greater than zero"
)
EXPECTED_DIRECTIONAL_RULE = (
    "mean paired difference is positive but the superiority rule is not met"
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DistributedContractError(f"Expected a JSON object: {path}")
    return value


def _weights(row: Mapping[str, str]) -> dict[str, float]:
    return {
        "alpha_node_feat": float(row["alpha_node_feat"]),
        "alpha_edge_feat": float(row["alpha_edge_feat"]),
    }


def build_confirmation_report(study_dir: Path, policy_path: Path) -> dict[str, Any]:
    root = study_dir.expanduser().resolve()
    policy_file = policy_path.expanduser().resolve()
    definition = _read_json(root / "study_definition.json")
    frozen = _read_json(root / "FROZEN.json")
    policy = _read_json(policy_file)
    contract_hash = canonical_contract_hash(definition)

    if (
        frozen.get("lifecycle") != "FROZEN"
        or frozen.get("study_name") != definition.get("study_name")
        or frozen.get("study_contract_sha256") != contract_hash
    ):
        raise DistributedContractError("Confirmation analysis requires a matching frozen study.")
    if (
        definition.get("reserved_trials") != 6
        or definition.get("objective", {}).get("json_path") != EXPECTED_OBJECTIVE
        or definition.get("objective", {}).get("split") != "validation"
        or definition.get("objective", {}).get("test_access") is not False
        or definition.get("evaluator", {}).get("test_access") is not False
        or definition.get("training", {}).get("epoch_number")
        != policy.get("training_epoch_number")
    ):
        raise DistributedContractError("Frozen study differs from the confirmation policy.")
    if (
        policy.get("schema_version") != "graphvae-attr-f1pr-confirmation-analysis-v1"
        or policy.get("objective_json_path") != EXPECTED_OBJECTIVE
        or policy.get("selection_split") != "validation"
        or policy.get("test_access_during_confirmation") is not False
        or policy.get("held_out_access_before_freeze") is not False
        or policy.get("no_reranking") is not True
        or policy.get("paired_difference")
        != "selected_validation_attr_f1pr - uniform_validation_attr_f1pr"
        or policy.get("uncertainty")
        != "two-sided 95% paired t interval with 2 degrees of freedom"
        or policy.get("superiority_rule") != EXPECTED_SUPERIORITY_RULE
        or policy.get("directional_rule") != EXPECTED_DIRECTIONAL_RULE
    ):
        raise DistributedContractError("Confirmation analysis policy is not exact.")

    seeds = [int(seed) for seed in policy.get("paired_training_seeds", [])]
    if seeds != [0, 1, 2]:
        raise DistributedContractError("Confirmation seeds must be exactly [0, 1, 2].")
    selected_weights = dict(policy.get("selected_weights") or {})
    uniform_weights = dict(policy.get("uniform_weights") or {})
    expected_plan = []
    for seed in seeds:
        expected_plan.extend(
            (
                {"parameters": selected_weights, "training_seed": seed},
                {"parameters": uniform_weights, "training_seed": seed},
            )
        )
    actual_plan = definition.get("reservation_plan")
    if not isinstance(actual_plan, list) or len(actual_plan) != 6:
        raise DistributedContractError("Confirmation reservation plan must contain six rows.")
    for budget_index, (actual, expected) in enumerate(zip(actual_plan, expected_plan)):
        if (
            actual.get("budget_index") != budget_index
            or actual.get("parameters") != expected["parameters"]
            or actual.get("training_seed") != expected["training_seed"]
        ):
            raise DistributedContractError("Confirmation reservation plan identity mismatch.")

    with (root / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise DistributedContractError("Confirmation trials.csv must contain six trials.")

    by_identity: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        if (
            row.get("state") != "COMPLETE"
            or row.get("reserved") != "True"
            or row.get("unreserved_guard") != "False"
        ):
            raise DistributedContractError("Every confirmation row must be reserved COMPLETE.")
        value = float(row["validation_attr_f1pr"])
        if not math.isfinite(value):
            raise DistributedContractError("Confirmation objectives must be finite.")
        weights = _weights(row)
        if weights == selected_weights:
            candidate = "selected"
        elif weights == uniform_weights:
            candidate = "uniform"
        else:
            raise DistributedContractError("Confirmation row has an uncontracted weight map.")
        identity = (candidate, int(row["training_seed"]))
        if identity in by_identity:
            raise DistributedContractError("Duplicate candidate/seed confirmation identity.")
        by_identity[identity] = row

    pairs = []
    selected_values = []
    uniform_values = []
    differences = []
    for seed in seeds:
        try:
            selected = by_identity[("selected", seed)]
            uniform = by_identity[("uniform", seed)]
        except KeyError as exc:
            raise DistributedContractError("Missing candidate/seed confirmation identity.") from exc
        selected_value = float(selected["validation_attr_f1pr"])
        uniform_value = float(uniform["validation_attr_f1pr"])
        difference = selected_value - uniform_value
        selected_values.append(selected_value)
        uniform_values.append(uniform_value)
        differences.append(difference)
        pairs.append(
            {
                "training_seed": seed,
                "selected_trial_number": int(selected["trial_number"]),
                "uniform_trial_number": int(uniform["trial_number"]),
                "selected_validation_attr_f1pr": selected_value,
                "uniform_validation_attr_f1pr": uniform_value,
                "paired_difference": difference,
            }
        )

    mean_difference = statistics.mean(differences)
    sample_sd = statistics.stdev(differences)
    standard_error = sample_sd / math.sqrt(len(differences))
    half_width = T_CRITICAL_95_DF2 * standard_error
    ci_lower = mean_difference - half_width
    ci_upper = mean_difference + half_width
    all_positive = all(value > 0.0 for value in differences)
    if all_positive and ci_lower > 0.0:
        conclusion = "superiority_supported"
    elif mean_difference > 0.0:
        conclusion = "directional_improvement_only"
    else:
        conclusion = "no_improvement"

    return {
        "schema_version": "graphvae-attr-f1pr-confirmation-report-v1",
        "study_name": definition["study_name"],
        "study_contract_sha256": contract_hash,
        "lifecycle": "FROZEN",
        "policy_path": policy_file.name,
        "policy_sha256": sha256_file(policy_file),
        "selection_source_study": policy["selection_source_study"],
        "selection_source_contract_sha256": policy[
            "selection_source_contract_sha256"
        ],
        "objective_json_path": EXPECTED_OBJECTIVE,
        "selection_split": "validation",
        "test_access": False,
        "training_epoch_number": int(policy["training_epoch_number"]),
        "selected_weights": selected_weights,
        "uniform_weights": uniform_weights,
        "pairs": pairs,
        "candidate_summary": {
            "selected_mean_validation_attr_f1pr": statistics.mean(selected_values),
            "uniform_mean_validation_attr_f1pr": statistics.mean(uniform_values),
        },
        "paired_summary": {
            "count": len(differences),
            "mean_difference": mean_difference,
            "sample_standard_deviation": sample_sd,
            "standard_error": standard_error,
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "confidence_interval_95": [ci_lower, ci_upper],
            "all_differences_positive": all_positive,
        },
        "conclusion": conclusion,
        "superiority_rule_met": conclusion == "superiority_supported",
        "no_reranking": True,
        "held_out_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_confirmation_report(args.study_dir, args.policy)
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
