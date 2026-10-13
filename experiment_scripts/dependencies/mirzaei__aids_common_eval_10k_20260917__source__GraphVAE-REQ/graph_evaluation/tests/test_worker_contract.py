"""Tests for metadata checks performed at the isolated worker boundary."""

import numpy as np
import pytest

from ggm_eval.worker import (
    _install_numpy_legacy_aliases,
    _validate_schema_identity,
)


def test_schema_identity_accepts_matching_declared_metadata():
    identity = _validate_schema_identity(
        training={"dataset": "PROTEINS", "feature_schema": "proteins-v1"},
        generated={"dataset": "PROTEINS", "feature_schema": "proteins-v1"},
        reference={"dataset": "PROTEINS", "feature_schema": "proteins-v1"},
    )

    assert identity == {
        "dataset": "PROTEINS",
        "feature_schema": "proteins-v1",
    }


def test_schema_identity_rejects_mismatch_or_missing_declaration():
    with pytest.raises(ValueError, match="feature_schema"):
        _validate_schema_identity(
            generated={"feature_schema": "proteins-v1"},
            reference={"feature_schema": "proteins-v2"},
        )

    with pytest.raises(ValueError, match="dataset"):
        _validate_schema_identity(
            generated={"dataset": "PROTEINS"},
            reference={},
        )


def test_upstream_shims_restore_numpy_aliases_needed_by_pygcl():
    aliases = ("bool", "float", "int", "object", "str")
    removed = {
        alias: np.__dict__.pop(alias)
        for alias in aliases
        if alias in np.__dict__
    }
    try:
        _install_numpy_legacy_aliases()
        assert all(alias in np.__dict__ for alias in aliases)
    finally:
        for alias in aliases:
            np.__dict__.pop(alias, None)
        np.__dict__.update(removed)
