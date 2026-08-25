import argparse
import json
import sys

import optuna
import pytest

from scripts.graphvae_attr_bo_distributed import (
    BUDGET_INDEX_ATTR,
    RESERVED_ATTR,
    TRIAL_CONTRACT_ATTR,
    audit_trial_result,
    build_study_definition,
    canonical_contract_hash,
)
from scripts.tune_graphvae_attribute_weights import (
    SearchRanges,
    TrialExecutionError,
    _mock_graphcl_payload,
    execute_grouped_graphcl_trial,
    parse_graphcl_f1pr_payload,
)


ENCODERS = [
    {"seed": seed, "sha256": str(seed) * 64}
    for seed in (101, 202, 303, 404, 505)
]


def graphcl_contract():
    return {
        "schema_version": "lobster-graphcl-f1pr-distributed-backend-v1",
        "backend": "graphcl_f1pr",
        "objective_json_path": "summary.f1_pr.mean",
        "compatibility_objective_json_path": (
            "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        ),
        "encoder_bundle_sha256": "bundle",
        "encoder_bundle_manifest_sha256": "manifest",
        "encoder_checkpoints": ENCODERS,
        "graphcl_runtime_sha256": "runtime",
        "upstream_revision": "upstream",
        "validation_collection_sha256": "validation",
        "validation_reference_file_sha256": "reference-file",
        "validation_split_fingerprint": "split",
        "paths": {
            "reference": "reference.pt",
            "encoder_bundle_manifest": "bundle.json",
            "campaign_root": "campaign",
            "dependency_root": "dependencies",
            "upstream_repo": "upstream",
        },
        "training_seeds": [0, 1],
        "checkpoint_count": 5,
        "nearest_k": 5,
        "test_access": False,
    }


def definition():
    base = {"experiment": {"epoch_number": 1}, "loss": {}}
    result = build_study_definition(
        study_name="graphcl-unit",
        base_config=base,
        base_config_sha256="base",
        ranges={
            "alpha_node_feat": {"low": 0.25, "high": 4.0, "log": True},
            "alpha_edge_feat": {"low": 0.25, "high": 4.0, "log": True},
            "alpha_motif_loss": None,
        },
        reserved_trials=1,
        seeds={
            "study_seed": 7,
            "split_seed": 123,
            "generation_seed": 123,
            "evaluator_seed": 0,
            "training_seeds": [0, 1],
        },
        evaluator={
            "backend": "graphcl_f1pr",
            "backend_contract": graphcl_contract(),
            "repeat_count": 5,
            "max_graphs": 10,
        },
        training={"epoch_number": 1, "mock": True},
        source={"tree_sha256": "source"},
        environment={"sha256": "environment"},
        dataset_cache={"sha256": "cache", "split_fingerprint": "synthetic-cache-split"},
        feature_schemas={"node_sha256": "node", "edge_sha256": "edge"},
        hardware_policy={},
        heartbeat_interval=60,
        grace_period=600,
        max_parallel=1,
        study_uuid="00000000-0000-0000-0000-000000000009",
    )
    return base, result


class FakeTrial:
    number = 0

    def __init__(self, contract_hash):
        self.params = {}
        self.user_attrs = {
            RESERVED_ATTR: True,
            BUDGET_INDEX_ATTR: 0,
            TRIAL_CONTRACT_ATTR: contract_hash,
            "budget_index": 0,
        }
        self.value = None
        self.state = optuna.trial.TrialState.RUNNING

    def suggest_float(self, name, low, high, *, log):
        if name not in self.params:
            self.params[name] = 1.0
        return self.params[name]

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def execution_args(contract_hash, *, fail_seed=()):
    return argparse.Namespace(
        distributed=True,
        evaluator_backend="graphcl_f1pr",
        study_contract_sha256=contract_hash,
        budget_index=0,
        worker_id="worker",
        worker_run_id="run",
        hostname="host",
        physical_gpu=None,
        gpu_model=None,
        gpu_vram_bytes=None,
        dispatch_sequence=0,
        sampler_constant_liar=True,
        sampler_seed=7,
        tpe_startup_trials=6,
        optuna_version="4.2.1",
        db_driver_version="2.9.10",
        training_seed=0,
        training_seeds=(0, 1),
        generation_seed=123,
        evaluator_seed=0,
        evaluator_repeats=5,
        max_graphs=10,
        generation_batch_size=4,
        nearest_k=5,
        adjacency_threshold=0.5,
        device="cpu",
        python_bin=sys.executable,
        training_timeout=5,
        evaluation_timeout=5,
        process_termination_grace=1,
        mock=True,
        mock_fail_trial=[],
        mock_fail_training_seed=list(fail_seed),
        expected_validation_graph_count=10,
        expected_node_feature_dimension=14,
        expected_edge_feature_dimension=11,
        integrity={
            "cache_sha256": "cache",
            "split_fingerprint": "synthetic-cache-split",
            "node_schema_fingerprint": "node",
            "edge_schema_fingerprint": "edge",
            "source_tree_sha256": "source",
            "environment_sha256": "environment",
        },
        graphcl_bundle_sha256="bundle",
        graphcl_runtime_sha256="runtime",
        graphcl_encoder_checkpoints=ENCODERS,
        graphcl_validation_collection_sha256="validation",
        graphcl_validation_split_fingerprint="split",
    )


def test_graphcl_parser_rejects_tampered_compatibility_objective():
    payload = _mock_graphcl_payload(
        {"alpha_node_feat": 1.0, "alpha_edge_feat": 1.0},
        generation_seed=123,
        bundle_sha256="bundle",
        runtime_sha256="runtime",
        encoder_checkpoints=ENCODERS,
        cache_sha256="cache",
        split_fingerprint="split",
        validation_collection_sha256="validation",
    )
    metrics = parse_graphcl_f1pr_payload(
        payload,
        expected_generation_seed=123,
        expected_bundle_sha256="bundle",
        expected_runtime_sha256="runtime",
        expected_encoder_checkpoints=ENCODERS,
        expected_cache_sha256="cache",
        expected_split_fingerprint="split",
        expected_validation_collection_sha256="validation",
    )
    assert metrics.graph_count == 10
    payload["evaluation"]["modes"]["decoded_node_edge"]["summary"]["f1_pr"][
        "mean"
    ] = 0.1
    with pytest.raises(TrialExecutionError, match="objective views"):
        parse_graphcl_f1pr_payload(
            payload,
            expected_generation_seed=123,
            expected_bundle_sha256="bundle",
            expected_runtime_sha256="runtime",
            expected_encoder_checkpoints=ENCODERS,
            expected_cache_sha256="cache",
            expected_split_fingerprint="split",
            expected_validation_collection_sha256="validation",
        )


def test_grouped_graphcl_trial_requires_both_seeds_and_audits(tmp_path):
    base, study_definition = definition()
    contract_hash = canonical_contract_hash(study_definition)
    trial = FakeTrial(contract_hash)
    value = execute_grouped_graphcl_trial(
        trial,
        args=execution_args(contract_hash),
        base_config=base,
        ranges=SearchRanges((0.25, 4.0), (0.25, 4.0)),
        output_dir=tmp_path,
        split_seed=123,
    )
    trial.value = value
    trial.state = optuna.trial.TrialState.COMPLETE
    audited = audit_trial_result(
        trial, study_root=tmp_path, definition=study_definition
    )
    assert audited["status"] == "COMPLETE"
    assert [item["training_seed"] for item in audited["replicates"]] == [0, 1]
    assert value == pytest.approx(sum(trial.user_attrs["replicate_values"]) / 2)


def test_second_seed_failure_consumes_group_without_partial_score(tmp_path):
    base, study_definition = definition()
    contract_hash = canonical_contract_hash(study_definition)
    trial = FakeTrial(contract_hash)
    with pytest.raises(TrialExecutionError, match="Mock failure"):
        execute_grouped_graphcl_trial(
            trial,
            args=execution_args(contract_hash, fail_seed=(1,)),
            base_config=base,
            ranges=SearchRanges((0.25, 4.0), (0.25, 4.0)),
            output_dir=tmp_path,
            split_seed=123,
        )
    trial.state = optuna.trial.TrialState.FAIL
    audited = audit_trial_result(
        trial, study_root=tmp_path, definition=study_definition
    )
    assert audited["status"] == "FAIL"
    assert [item["status"] for item in audited["replicates"]] == ["COMPLETE", "FAIL"]
    assert audited["validation_attr_f1pr"] is None
    assert "replicate_values" not in trial.user_attrs
