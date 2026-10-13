from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import evaluate_lobster_attr_f1pr_confirmation_heldout as heldout
from scripts.graphvae_attr_bo_distributed import (
    DistributedContractError,
    canonical_contract_hash,
    sha256_file,
)


def _write_fixture(tmp_path: Path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    definition = {
        "study_name": "confirmation",
        "reserved_trials": 6,
        "objective": {
            "json_path": heldout.EXPECTED_OBJECTIVE,
            "split": "validation",
            "test_access": False,
        },
        "evaluator": {"test_access": False},
    }
    contract = canonical_contract_hash(definition)
    (study / "study_definition.json").write_text(json.dumps(definition))
    (study / "FROZEN.json").write_text(
        json.dumps(
            {
                "lifecycle": "FROZEN",
                "study_contract_sha256": contract,
            }
        )
    )
    report = {
        "study_contract_sha256": contract,
        "conclusion": "no_improvement",
        "no_reranking": True,
        "held_out_access": False,
        "test_access": False,
        "objective_json_path": heldout.EXPECTED_OBJECTIVE,
        "selection_split": "validation",
    }
    report_path = study / "confirmation_report.json"
    report_path.write_text(json.dumps(report))
    trials = []
    for trial_number in range(6):
        trials.append(
            {
                "trial_number": trial_number,
                "candidate": "selected" if trial_number % 2 == 0 else "uniform",
                "training_seed": trial_number // 2,
                "hostname": "node",
                "physical_gpu": 0,
                "checkpoint_sha256": f"checkpoint-{trial_number}",
                "resolved_config_sha256": f"config-{trial_number}",
            }
        )
    plan = {
        "schema_version": "graphvae-attr-f1pr-confirmation-heldout-plan-v1",
        "study_name": "confirmation",
        "study_contract_sha256": contract,
        "frozen_confirmation_report_sha256": sha256_file(report_path),
        "frozen_confirmation_conclusion": "no_improvement",
        "no_reranking": True,
        "allow_training": False,
        "allow_new_trials": False,
        "split": "test",
        "primary_mode": "decoded_node_edge",
        "objective_json_path": heldout.EXPECTED_OBJECTIVE,
        "max_graphs": 0,
        "expected_test_graphs": 20,
        "generation_seed": 123,
        "evaluator_seed": 0,
        "evaluator_repeats": 10,
        "max_parallel": 3,
        "trials": trials,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(heldout, "EXPECTED_PLAN_SHA256", sha256_file(plan_path))

    trial_dir = study / "trials" / "trial_00000"
    trial_dir.mkdir(parents=True)
    (trial_dir / "trial_result.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "trial_number": 0,
                "study_contract_sha256": contract,
                "training_seed": 0,
                "hostname": "node",
                "physical_gpu": 0,
                "sampled_weights": heldout.EXPECTED_SELECTED,
                "checkpoint_sha256": "checkpoint-0",
                "resolved_config_sha256": "config-0",
                "objective_json_path": heldout.EXPECTED_OBJECTIVE,
            }
        )
    )
    return study, plan_path


def test_heldout_inputs_accept_exact_frozen_identity(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    loaded_plan, entry, trial = heldout.validate_heldout_inputs(
        study, plan, 0, hostname="node", physical_gpu=0
    )
    assert loaded_plan["no_reranking"] is True
    assert entry["candidate"] == "selected"
    assert trial["training_seed"] == 0


def test_heldout_inputs_reject_plan_mutation(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    payload = json.loads(plan.read_text())
    payload["evaluator_repeats"] = 9
    plan.write_text(json.dumps(payload))
    with pytest.raises(DistributedContractError, match="Held-out plan differs"):
        heldout.validate_heldout_inputs(
            study, plan, 0, hostname="node", physical_gpu=0
        )


def test_heldout_inputs_reject_wrong_launch_slot(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    with pytest.raises(DistributedContractError, match="launch identity"):
        heldout.validate_heldout_inputs(
            study, plan, 0, hostname="other-node", physical_gpu=0
        )
