from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_lobster_attr_f1pr_confirmation import build_confirmation_report
from scripts.graphvae_attr_bo_distributed import (
    DistributedContractError,
    canonical_contract_hash,
)


SELECTED = {
    "alpha_node_feat": 5.229045672015893,
    "alpha_edge_feat": 0.05386414830134693,
}
UNIFORM = {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}


def _write_fixture(tmp_path: Path, differences=(0.03, 0.04, 0.05)):
    root = tmp_path / "study"
    root.mkdir()
    plan = []
    rows = []
    for seed, difference in enumerate(differences):
        for weights, value in ((SELECTED, 0.7 + difference), (UNIFORM, 0.7)):
            budget_index = len(plan)
            plan.append(
                {
                    "budget_index": budget_index,
                    "parameters": weights,
                    "training_seed": seed,
                }
            )
            rows.append(
                {
                    "trial_number": budget_index,
                    "budget_index": budget_index,
                    "reserved": "True",
                    "unreserved_guard": "False",
                    "state": "COMPLETE",
                    "validation_attr_f1pr": value,
                    "alpha_node_feat": weights["alpha_node_feat"],
                    "alpha_edge_feat": weights["alpha_edge_feat"],
                    "training_seed": seed,
                }
            )
    definition = {
        "study_name": "confirmation",
        "reserved_trials": 6,
        "objective": {
            "json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
            "split": "validation",
            "test_access": False,
        },
        "evaluator": {"test_access": False},
        "training": {"epoch_number": 10000},
        "reservation_plan": plan,
    }
    contract_hash = canonical_contract_hash(definition)
    (root / "study_definition.json").write_text(json.dumps(definition))
    (root / "FROZEN.json").write_text(
        json.dumps(
            {
                "study_name": "confirmation",
                "study_contract_sha256": contract_hash,
                "lifecycle": "FROZEN",
            }
        )
    )
    with (root / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    policy = {
        "schema_version": "graphvae-attr-f1pr-confirmation-analysis-v1",
        "selection_source_study": "search",
        "selection_source_contract_sha256": "search-contract",
        "selection_split": "validation",
        "objective_json_path": "evaluation.modes.decoded_node_edge.summary.f1_pr.mean",
        "test_access_during_confirmation": False,
        "training_epoch_number": 10000,
        "paired_training_seeds": [0, 1, 2],
        "selected_weights": SELECTED,
        "uniform_weights": UNIFORM,
        "paired_difference": (
            "selected_validation_attr_f1pr - uniform_validation_attr_f1pr"
        ),
        "primary_summary": "mean paired difference",
        "uncertainty": "two-sided 95% paired t interval with 2 degrees of freedom",
        "superiority_rule": (
            "all three paired differences are positive and the 95% paired t interval "
            "lower bound is greater than zero"
        ),
        "directional_rule": (
            "mean paired difference is positive but the superiority rule is not met"
        ),
        "no_reranking": True,
        "held_out_access_before_freeze": False,
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return root, policy_path


def test_confirmation_report_supports_superiority(tmp_path):
    root, policy = _write_fixture(tmp_path)
    report = build_confirmation_report(root, policy)
    assert report["conclusion"] == "superiority_supported"
    assert report["superiority_rule_met"] is True
    assert report["paired_summary"]["confidence_interval_95"][0] > 0
    assert report["test_access"] is False


def test_confirmation_report_distinguishes_no_improvement(tmp_path):
    root, policy = _write_fixture(tmp_path, differences=(0.02, -0.01, -0.03))
    report = build_confirmation_report(root, policy)
    assert report["conclusion"] == "no_improvement"
    assert report["superiority_rule_met"] is False
    assert report["paired_summary"]["all_differences_positive"] is False


def test_confirmation_report_rejects_uncontracted_row(tmp_path):
    root, policy = _write_fixture(tmp_path)
    rows = list(csv.DictReader((root / "trials.csv").open()))
    rows[0]["alpha_node_feat"] = "9.0"
    with (root / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DistributedContractError, match="uncontracted weight"):
        build_confirmation_report(root, policy)
