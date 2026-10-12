import json
from pathlib import Path

import pytest

from graph_evaluation.src.ggm_eval.io import load_pyg_collection_with_metadata
from scripts import export_aids_graphcl_f1pr_splits as exporter
from scripts import analyze_aids_evaluator_bakeoff as analyzer


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_evaluator_bakeoff_split_qualification.json"
)
CONTRACT = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_evaluator_bakeoff_contract.json"
)
COMPLETION = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_evaluator_bakeoff_completion.json"
)


def test_frozen_aids_export_is_exact_deterministic_and_test_free(tmp_path):
    cache_path = REPO_ROOT / exporter.CACHE_RELATIVE_PATH
    first = exporter.export_frozen_splits(cache_path, tmp_path / "first")
    second = exporter.export_frozen_splits(cache_path, tmp_path / "second")

    assert first["test_access"] is False
    assert first["exported_splits"] == ["train", "validation"]
    assert first["split_overlap_count"] == 0
    assert first["cache"] == {
        "byte_length": exporter.CACHE_BYTE_LENGTH,
        "mode": "0444",
        "sha256": exporter.CACHE_SHA256,
    }
    for split, count in (("train", 1294), ("validation", 184)):
        first_artifact = first["artifacts"][split]
        second_artifact = second["artifacts"][split]
        assert first_artifact["collection_sha256"] == second_artifact[
            "collection_sha256"
        ]
        assert first_artifact["split_fingerprint"] == second_artifact[
            "split_fingerprint"
        ]
        graphs, metadata = load_pyg_collection_with_metadata(first_artifact["path"])
        assert len(graphs) == count
        assert metadata["split"] == split
        assert metadata["test_access"] is False
        assert metadata["feature_schema"] == exporter.FEATURE_SCHEMA
        assert graphs[0].x.shape[1] == 56
        assert graphs[0].edge_attr.shape[1] == 3
    assert not list((tmp_path / "first").glob("*test*"))


def test_aids_test_export_guard_precedes_cache_access(tmp_path):
    with pytest.raises(
        exporter.AidsGraphCLExportError,
        match="Held-out/test export is forbidden",
    ):
        exporter.export_frozen_splits(
            tmp_path / "missing.pkl", tmp_path / "output", include_test=True
        )


def test_aids_cache_hash_guard_precedes_pickle_loading(tmp_path, monkeypatch):
    relative = Path("cache_datasets/frozen.pkl")
    cache_path = tmp_path / relative
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"not a trusted pickle")
    cache_path.chmod(0o444)
    monkeypatch.setattr(exporter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "CACHE_RELATIVE_PATH", relative)
    monkeypatch.setattr(exporter, "CACHE_BYTE_LENGTH", cache_path.stat().st_size)
    monkeypatch.setattr(exporter, "CACHE_SHA256", "0" * 64)

    with pytest.raises(exporter.AidsGraphCLExportError, match="SHA-256"):
        exporter.export_frozen_splits(cache_path, tmp_path / "output")


def test_aids_split_contract_rejects_overlap_and_test_name():
    with pytest.raises(exporter.AidsGraphCLExportError, match="overlap"):
        exporter.assert_disjoint_splits(["same"], ["same"])
    with pytest.raises(exporter.AidsGraphCLExportError, match="Only training"):
        exporter._split_values({}, "test")


def test_aids_split_qualification_records_exact_test_free_inputs():
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert qualification["contract"] == {
        "dataset": "AIDS",
        "feature_schema": exporter.FEATURE_SCHEMA,
        "node_feature_dimension": 56,
        "edge_feature_dimension": 3,
        "exported_splits": ["train", "validation"],
        "test_access": False,
        "split_overlap_count": 0,
    }
    assert qualification["artifacts"]["train"]["collection_sha256"] == (
        "3d979eb9fec89967832df54547fff6a9dfa2d6f9da8a5d357200dc5963f41650"
    )
    assert qualification["artifacts"]["validation"]["collection_sha256"] == (
        "c0eaa2e545525e4a7a321eaa1937f6c68db76738906e2aa7f514b46dcd907e74"
    )
    assert qualification["verification"]["generated_collections_committed"] is False


def test_aids_evaluator_bakeoff_contract_is_matched_bounded_and_test_free():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["scientific_contract"]["split"] == "validation"
    assert contract["scientific_contract"]["validation_graphs"] == 184
    assert contract["scientific_contract"]["test_access"] is False
    assert contract["scientific_contract"]["held_out_access"] is False
    assert contract["source"]["new_graphvae_training"] is False
    assert contract["source"]["new_bo_trials"] is False
    assert contract["source"]["minimum_generation_implementation_commit"] == (
        "b80aeacc633a1ed17b0cb37ed43dc60661283cf6"
    )
    assert contract["sampling"]["training_seeds"] == [0, 1, 2]
    assert contract["sampling"]["generation_seeds"] == [123, 124, 125]
    assert contract["sampling"]["generated_collection_count"] == 18
    assert len(contract["sampling"]["random_gin"]["fixed_evaluator_seeds"]) == 10
    assert contract["sampling"]["random_gin"]["fresh_ensemble_per_job"] is False
    assert len(
        contract["sampling"]["graphcl"]["fixed_train_only_encoder_seeds"]
    ) == 10
    assert contract["sampling"]["graphcl"]["retrain_per_job"] is False
    assert contract["matched_collection_rule"][
        "same_generated_collection_for_both_evaluators"
    ] is True
    assert contract["decision_rule"]["no_weight_improvement_claim_from_bakeoff"] is True
    assert contract["decision_rule"]["graphcl_selected_only_if_all_conditions_hold"][
        0
    ] == "both evaluators replay exactly on a frozen job"

    checkpoint_rows = [
        checkpoint
        for candidate in contract["candidates"].values()
        for checkpoint in candidate["checkpoints"]
    ]
    assert len(checkpoint_rows) == 6
    assert {row["host"] for row in checkpoint_rows} == {"cs-cl-13", "cs-cl-17"}
    assert len({row["sha256"] for row in checkpoint_rows}) == 6
    assert all(row["relative_path"].endswith("/model_249_6") for row in checkpoint_rows)


def test_aids_evaluator_bakeoff_completion_obeys_predeclared_decision():
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    assert completion["sampling"] == {
        "candidate_checkpoint_count": 6,
        "training_seeds": [0, 1, 2],
        "generation_seeds": [123, 124, 125],
        "matched_jobs": 18,
        "random_gin_evaluators_per_job": 10,
        "graphcl_encoders_per_job": 10,
        "graphcl_checkpoint_evaluations": 180,
    }
    assert completion["test_access"] is False
    assert completion["held_out_access"] is False
    assert completion["random_gin"]["exact_replay"] is True
    assert completion["graphcl"]["exact_replay"] is True
    conditions = completion["decision_conditions"]
    assert conditions["graphcl_paired_dispersion_at_least_20_percent_lower"] is True
    assert conditions["graphcl_mean_generation_range_no_greater"] is False
    assert conditions["graphcl_sign_stability_no_worse"] is False
    assert completion["decision"]["selected_primary_evaluator"] == "random_gin"
    assert completion["decision"]["weight_improvement_claim"] is False
    assert completion["integrity"]["reference_collection_unique_digest_count"] == 1
    assert completion["integrity"]["generated_collection_unique_digest_count"] == 18
    assert completion["integrity"]["credential_marker_files"] == 0
    assert completion["integrity"]["test_or_held_out_true_files"] == 0


def _summary_jobs(random_gin: bool):
    jobs = {}
    evaluator_offsets = [index * 0.01 for index in range(10)]
    for candidate in ("selected", "uniform"):
        for training_seed in (0, 1, 2):
            for generation_seed in (123, 124, 125):
                generation_offset = (generation_seed - 123) * (
                    0.03 if random_gin else 0.01
                )
                values = [0.4 + generation_offset] * 10
                if candidate == "selected":
                    values = [
                        value + (offset if random_gin else 0.04)
                        for value, offset in zip(values, evaluator_offsets)
                    ]
                jobs[(candidate, training_seed, generation_seed)] = {
                    "method": {"values": values, "mean": sum(values) / len(values)}
                }
    return jobs


def test_bakeoff_summary_rewards_lower_paired_and_generation_dispersion():
    random_summary = analyzer._method_summary(_summary_jobs(True), "method")
    graphcl_summary = analyzer._method_summary(_summary_jobs(False), "method")

    assert graphcl_summary["mean_paired_difference_population_sd"] < (
        0.8 * random_summary["mean_paired_difference_population_sd"]
    )
    assert graphcl_summary["mean_generation_seed_range"] < random_summary[
        "mean_generation_seed_range"
    ]
    assert graphcl_summary["sign_stability_count_of_9"] == 9


def test_bakeoff_manifest_rejects_test_access(tmp_path):
    manifest = {
        "collection_sha256": "a" * 64,
        "summary": {
            "graph_count": 184,
            "node_feature_dim": 56,
            "edge_feature_dim": 3,
        },
        "metadata": {
            "dataset": "AIDS",
            "feature_mode": "decoded_node_edge",
            "feature_schema": analyzer.FEATURE_SCHEMA,
            "split": "validation",
            "test_access": True,
            "generation_seed": 123,
            "source_cache_sha256": analyzer.CACHE_SHA256,
            "split_fingerprint": analyzer.VALIDATION_SPLIT_FINGERPRINT,
            "checkpoint_sha256": "b" * 64,
            "collection_role": "generated",
        },
    }
    path = tmp_path / "generated.pt.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(analyzer.DistributedContractError, match="test_access"):
        analyzer._manifest(
            path, role="generated", generation_seed=123, checkpoint_sha="b" * 64
        )
