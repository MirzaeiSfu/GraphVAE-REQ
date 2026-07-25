"""Tests for provenance validation and multi-checkpoint reporting."""

from pathlib import Path

import pytest

from ggm_eval.reporting import aggregate_checkpoint_results
from ggm_eval.upstreams import (
    CONTRASTIVE_REQUIRED_FILES,
    validate_contrastive_upstream,
)


def test_checkpoint_metrics_are_aggregated_across_independent_runs():
    common = {
        "engine": "contrastive-pyg-upstream",
        "encoder": "graphcl",
        "feature_mode": "decoded_node",
        "model": {"hidden_dim": 16},
        "training": {"trained": True, "epochs": 100},
        "training_metadata": {
            "dataset": "PROTEINS",
            "feature_schema": "proteins-v1",
        },
        "schema_identity": {
            "dataset": "PROTEINS",
            "feature_schema": "proteins-v1",
        },
        "upstream_revision": "revision",
        "generated_sha256": "generated",
        "reference_sha256": "reference",
    }
    result = aggregate_checkpoint_results(
        [
            {
                **common,
                "checkpoint_seed": 0,
                "metrics": {"fid": 1.0, "mmd_linear": 0.2},
            },
            {
                **common,
                "checkpoint_seed": 1,
                "metrics": {"fid": 3.0, "mmd_linear": 0.4},
            },
        ]
    )

    assert result["checkpoint_count"] == 2
    assert result["summary"]["fid"] == {
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert result["summary"]["mmd_linear"]["mean"] == pytest.approx(0.3)


def test_checkpoint_aggregation_rejects_mixed_encoders():
    common = {
        "engine": "contrastive-pyg-upstream",
        "feature_mode": "decoded_node",
        "model": {"hidden_dim": 16},
        "training": {"trained": True, "epochs": 100},
        "training_metadata": {
            "dataset": "PROTEINS",
            "feature_schema": "proteins-v1",
        },
        "schema_identity": {
            "dataset": "PROTEINS",
            "feature_schema": "proteins-v1",
        },
        "upstream_revision": "revision",
        "generated_sha256": "generated",
        "reference_sha256": "reference",
        "metrics": {"fid": 1.0},
    }
    with pytest.raises(ValueError, match="not comparable"):
        aggregate_checkpoint_results(
            [
                {**common, "encoder": "graphcl", "checkpoint_seed": 0},
                {**common, "encoder": "infograph", "checkpoint_seed": 1},
            ]
        )


def test_upstream_checkout_requires_expected_files(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        validate_contrastive_upstream(tmp_path, allow_unpinned=True)


def test_unpinned_checkout_requires_explicit_opt_in(tmp_path):
    for relative in CONTRASTIVE_REQUIRED_FILES:
        path = Path(tmp_path, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test fixture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not the clean pinned source"):
        validate_contrastive_upstream(tmp_path)

    provenance = validate_contrastive_upstream(
        tmp_path,
        allow_unpinned=True,
    )
    assert provenance["revision"] is None
    assert provenance["revision_matches"] is False
    assert provenance["worktree_dirty"] is None
