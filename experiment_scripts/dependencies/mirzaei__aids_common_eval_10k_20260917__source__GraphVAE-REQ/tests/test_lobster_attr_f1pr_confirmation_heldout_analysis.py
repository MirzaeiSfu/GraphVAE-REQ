from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import analyze_lobster_attr_f1pr_confirmation_heldout as analysis
from scripts.graphvae_attr_bo_distributed import (
    DistributedContractError,
    canonical_contract_hash,
    sha256_file,
)
from scripts.tune_graphvae_attribute_weights import (
    _mock_evaluator_payload,
    parse_attr_f1pr_file,
)


def _write_fixture(tmp_path: Path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    definition = {
        "study_name": "confirmation",
        "reserved_trials": 6,
        "objective": {
            "json_path": analysis.EXPECTED_OBJECTIVE,
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
    (study / "confirmation_report.json").write_text(
        json.dumps(
            {
                "study_contract_sha256": contract,
                "conclusion": "no_improvement",
                "no_reranking": True,
                "held_out_access": False,
                "test_access": False,
                "selection_split": "validation",
                "objective_json_path": analysis.EXPECTED_OBJECTIVE,
            }
        )
    )
    (study / "study_snapshot.sqlite3").write_bytes(b"snapshot")
    (study / "trials.csv").write_text("trials")

    integrity = {
        "cache_sha256": "cache",
        "split_fingerprint": "test-split",
        "node_schema_fingerprint": "node-schema",
        "edge_schema_fingerprint": "edge-schema",
    }
    entries = []
    heldout_root = study / "heldout_confirmation"
    for number in range(6):
        candidate = "selected" if number % 2 == 0 else "uniform"
        entry = {
            "trial_number": number,
            "candidate": candidate,
            "training_seed": number // 2,
            "hostname": "node",
            "physical_gpu": number % 2,
            "checkpoint_sha256": f"checkpoint-{number}",
        }
        entries.append(entry)
        result_dir = heldout_root / f"trial_{number:05d}"
        result_dir.mkdir(parents=True)
        weights = (
            {"alpha_node_feat": 2.0, "alpha_edge_feat": 0.5}
            if candidate == "selected"
            else {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0}
        )
        payload = _mock_evaluator_payload(
            weights,
            split="test",
            graph_count=20,
            generation_seed=123,
            evaluator_seed=0,
            evaluator_repeats=10,
            integrity=integrity,
            node_feature_dimension=14,
            edge_feature_dimension=11,
        )
        evaluator_path = result_dir / "attributed_random_gin.json"
        evaluator_path.write_text(json.dumps(payload))
        metrics = parse_attr_f1pr_file(evaluator_path, expected_split="test")
        (result_dir / "heldout_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "graphvae-attr-f1pr-confirmation-heldout-result-v1",
                    "study_name": "confirmation",
                    "study_contract_sha256": contract,
                    "trial_number": number,
                    "candidate": candidate,
                    "training_seed": number // 2,
                    "checkpoint_sha256": f"checkpoint-{number}",
                    "split": "test",
                    "primary_mode": "decoded_node_edge",
                    "objective_json_path": analysis.EXPECTED_OBJECTIVE,
                    "test_attr_f1pr": metrics.f1_pr,
                    "test_precision": metrics.precision,
                    "test_recall": metrics.recall,
                    "accepted_test_graphs": 20,
                    "generation_seed": 123,
                    "evaluator_seed": 0,
                    "evaluator_repeats": 10,
                    "hostname": "node",
                    "physical_gpu": number % 2,
                    "logical_device": "cuda:0",
                    "evaluator_output_sha256": sha256_file(evaluator_path),
                    "no_reranking": True,
                    "selection_changed": False,
                    "new_trial_created": False,
                    "training_ran": False,
                }
            )
        )
    plan = {
        "schema_version": "graphvae-attr-f1pr-confirmation-heldout-plan-v1",
        "study_name": "confirmation",
        "study_contract_sha256": contract,
        "frozen_confirmation_conclusion": "no_improvement",
        "no_reranking": True,
        "allow_training": False,
        "allow_new_trials": False,
        "split": "test",
        "primary_mode": "decoded_node_edge",
        "objective_json_path": analysis.EXPECTED_OBJECTIVE,
        "expected_test_graphs": 20,
        "max_graphs": 0,
        "max_parallel": 3,
        "node_feature_dimension": 14,
        "edge_feature_dimension": 11,
        "cache_sha256": "cache",
        "test_split_fingerprint": "test-split",
        "node_schema_fingerprint": "node-schema",
        "edge_schema_fingerprint": "edge-schema",
        "generation_seed": 123,
        "evaluator_seed": 0,
        "evaluator_repeats": 10,
        "trials": entries,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(analysis, "EXPECTED_PLAN_SHA256", sha256_file(plan_path))
    monkeypatch.setattr(analysis, "EXPECTED_FROZEN_SHA256", sha256_file(study / "FROZEN.json"))
    monkeypatch.setattr(
        analysis,
        "EXPECTED_CONFIRMATION_REPORT_SHA256",
        sha256_file(study / "confirmation_report.json"),
    )
    monkeypatch.setattr(
        analysis, "EXPECTED_SNAPSHOT_SHA256", sha256_file(study / "study_snapshot.sqlite3")
    )
    monkeypatch.setattr(
        analysis, "EXPECTED_TRIALS_CSV_SHA256", sha256_file(study / "trials.csv")
    )
    return study, plan_path


def test_heldout_report_is_secondary_and_keeps_validation_conclusion(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    report = analysis.build_heldout_report(study, plan)
    assert report["validation_conclusion"] == "no_improvement"
    assert report["validation_conclusion_changed"] is False
    assert report["heldout_role"] == "secondary_descriptive_evidence_only"
    assert report["new_trial_created"] is False
    assert len(report["pairs"]) == 3


def test_heldout_report_rejects_selection_change(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    result_path = study / "heldout_confirmation/trial_00000/heldout_result.json"
    result = json.loads(result_path.read_text())
    result["selection_changed"] = True
    result_path.write_text(json.dumps(result))
    with pytest.raises(DistributedContractError, match="differs from its plan"):
        analysis.build_heldout_report(study, plan)


def test_heldout_report_rejects_changed_validation_conclusion(tmp_path, monkeypatch):
    study, plan = _write_fixture(tmp_path, monkeypatch)
    report_path = study / "confirmation_report.json"
    report = json.loads(report_path.read_text())
    report["conclusion"] = "superiority_supported"
    report_path.write_text(json.dumps(report))
    with pytest.raises(DistributedContractError, match="validation inputs changed"):
        analysis.build_heldout_report(study, plan)
