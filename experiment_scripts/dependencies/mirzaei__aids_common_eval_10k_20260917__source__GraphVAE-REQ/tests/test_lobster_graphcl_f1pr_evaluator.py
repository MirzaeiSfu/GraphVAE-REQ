import copy
import hashlib
import json

import pytest

from scripts import evaluate_lobster_graphcl_f1pr_checkpoint as evaluator


def _payload():
    metrics = [
        {"f1_pr": 0.40, "precision": 0.50, "recall": 0.35, "fid": 2.0},
        {"f1_pr": 0.45, "precision": 0.55, "recall": 0.40, "fid": 1.9},
        {"f1_pr": 0.50, "precision": 0.60, "recall": 0.45, "fid": 1.8},
        {"f1_pr": 0.55, "precision": 0.65, "recall": 0.50, "fid": 1.7},
        {"f1_pr": 0.60, "precision": 0.70, "recall": 0.55, "fid": 1.6},
    ]
    summary = {
        name: evaluator.summarize_values([entry[name] for entry in metrics])
        for name in metrics[0]
    }
    return {
        "schema_version": "lobster-graphcl-f1pr-evaluation-v1",
        "engine": "contrastive-pyg-upstream",
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "checkpoint_count": 5,
        "split": "validation",
        "test_access": False,
        "skip_final_evaluation": True,
        "generation_seed": 12345,
        "nearest_k": 5,
        "objective_json_path": "summary.f1_pr.mean",
        "encoder_bundle_sha256": "b" * 64,
        "graphcl_runtime_sha256": "r" * 64,
        "graph_counts": {
            "generated_accepted": 10,
            "reference_accepted": 10,
            "validation_cache_count": 10,
            "generation_attempts": 12,
        },
        "feature_dimensions": {"node": 14, "edge": 11},
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "frozen LOBSTER validation node and edge one-hot attributes",
            "same_latent_decoding": True,
            "hand_made_topology_features": False,
        },
        "summary": summary,
        "per_checkpoint": [
            {
                "seed": seed,
                "checkpoint_sha256": str(seed) * 16,
                "metrics": metric,
            }
            for seed, metric in zip(evaluator.EXPECTED_SEEDS, metrics)
        ],
        "evaluation": {
            "modes": {"decoded_node_edge": {"summary": copy.deepcopy(summary)}}
        },
        "integrity": {
            "cache_sha256": evaluator.CACHE_SHA256,
            "validation_split_fingerprint": evaluator.VALIDATION_SPLIT_FINGERPRINT,
            "validation_collection_sha256": evaluator.VALIDATION_COLLECTION_SHA256,
            "upstream_revision": evaluator.EXPECTED_UPSTREAM_REVISION,
        },
    }


def _parse(payload):
    return evaluator.parse_graphcl_f1pr_payload(
        payload,
        expected_bundle_sha256="b" * 64,
        expected_runtime_sha256="r" * 64,
        expected_generation_seed=12345,
    )


def test_graphcl_payload_accepts_exact_validation_ensemble():
    assert _parse(_payload()) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("encoder",), "gin-random", "encoder"),
        (("feature_mode",), "topology_control", "feature_mode"),
        (("checkpoint_count",), 4, "checkpoint_count"),
        (("split",), "test", "split"),
        (("test_access",), True, "test_access"),
        (("skip_final_evaluation",), False, "skip_final_evaluation"),
        (("graph_counts", "generated_accepted"), 9, "graph counts"),
        (("feature_dimensions", "edge"), 0, "feature dimensions"),
        (
            ("integrity", "upstream_revision"),
            "changed",
            "upstream_revision",
        ),
        (
            ("integrity", "validation_collection_sha256"),
            "tampered",
            "validation_collection_sha256",
        ),
    ],
)
def test_graphcl_payload_rejects_contract_substitutions(path, value, message):
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(evaluator.DistributedContractError, match=message):
        _parse(payload)


def test_graphcl_payload_rejects_missing_decoder_and_repeated_checkpoint():
    missing_decoder = _payload()
    missing_decoder["feature_source"]["generated"] = "GraphVAE node_feature_decoder"
    with pytest.raises(evaluator.DistributedContractError, match="feature provenance"):
        _parse(missing_decoder)

    repeated = _payload()
    repeated["per_checkpoint"][1]["checkpoint_sha256"] = repeated[
        "per_checkpoint"
    ][0]["checkpoint_sha256"]
    with pytest.raises(evaluator.DistributedContractError, match="repeats"):
        _parse(repeated)


def test_graphcl_payload_rejects_nonfinite_and_inconsistent_summary():
    nonfinite = _payload()
    nonfinite["per_checkpoint"][2]["metrics"]["f1_pr"] = float("nan")
    with pytest.raises(evaluator.DistributedContractError, match="nonfinite"):
        _parse(nonfinite)

    inconsistent = _payload()
    inconsistent["summary"]["f1_pr"]["mean"] += 0.01
    with pytest.raises(evaluator.DistributedContractError, match="differs"):
        _parse(inconsistent)


def test_graphcl_payload_preserves_upstream_f1pr_epsilon_ceiling():
    payload = _payload()
    for entry in payload["per_checkpoint"]:
        entry["metrics"]["f1_pr"] = 1.00001
    payload["summary"]["f1_pr"] = evaluator.summarize_values([1.00001] * 5)
    payload["evaluation"]["modes"]["decoded_node_edge"]["summary"][
        "f1_pr"
    ] = copy.deepcopy(payload["summary"]["f1_pr"])
    assert _parse(payload) == pytest.approx(1.00001)


def test_encoder_bundle_rejects_unsafe_or_tampered_checkpoint(tmp_path, monkeypatch):
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    encoders = []
    for seed in evaluator.EXPECTED_SEEDS:
        checkpoint = campaign_root / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint.chmod(0o444)
        encoders.append(
            {
                "seed": seed,
                "checkpoint": {
                    "path": checkpoint.relative_to(campaign_root).as_posix(),
                    "byte_length": checkpoint.stat().st_size,
                    "mode": "0444",
                    "sha256": evaluator.sha256_file(checkpoint),
                },
            }
        )
    runtime = {"sha256": "r" * 64}
    manifest = {
        "schema_version": "lobster-graphcl-f1pr-encoder-bundle-v1",
        "encoder": "graphcl",
        "feature_mode": "decoded_node_edge",
        "feature_schema": evaluator.EXPECTED_FEATURE_SCHEMA,
        "training_split": "train",
        "test_access": False,
        "seeds": list(evaluator.EXPECTED_SEEDS),
        "checkpoint_count": 5,
        "upstream": {
            "revision": evaluator.EXPECTED_UPSTREAM_REVISION,
            "worktree_dirty": False,
        },
        "runtime": runtime,
        "encoders": encoders,
    }
    manifest["bundle_sha256"] = hashlib.sha256(
        evaluator.canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_path = campaign_root / "bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)
    manifest_file_sha = evaluator.sha256_file(manifest_path)
    monkeypatch.setattr(
        evaluator, "graphcl_runtime_fingerprint", lambda _root: runtime
    )

    _, checkpoints = evaluator.load_encoder_bundle(
        manifest_path,
        campaign_root=campaign_root,
        expected_manifest_sha256=manifest_file_sha,
        dependency_root=tmp_path,
        expected_runtime_sha256="r" * 64,
    )
    assert len(checkpoints) == 5

    checkpoint = checkpoints[0]
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"tampered")
    checkpoint.chmod(0o444)
    with pytest.raises(evaluator.DistributedContractError, match="checkpoint differs"):
        evaluator.load_encoder_bundle(
            manifest_path,
            campaign_root=campaign_root,
            expected_manifest_sha256=manifest_file_sha,
            dependency_root=tmp_path,
            expected_runtime_sha256="r" * 64,
        )


def test_validation_reference_is_exact_real_collection():
    path = (
        evaluator.REPO_ROOT
        / "runs/graphcl_f1pr_lobster_20260825/inputs/real_validation_graphs.pt"
    )
    graphs, metadata = evaluator.validate_validation_reference(path)
    assert len(graphs) == 10
    assert metadata["split"] == "validation"
    assert metadata["test_access"] is False
