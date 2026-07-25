"""Tests for the bundled GraphCL-GIN registry and convenience wrapper."""

import pytest

import ggm_eval.trained as trained
from ggm_eval import (
    available_trained_datasets,
    evaluate_with_trained_gnns,
    resolve_trained_checkpoints,
)
from ggm_eval.datasets import normalize_dataset_name


EXPECTED_DATASETS = {
    "AIDS",
    "ENZYMES",
    "MUTAG",
    "PROTEINS",
    "PTC",
    "QM9",
    "ogbg-molbbbp",
}


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("protein", "PROTEINS"),
        ("PROTEINS", "PROTEINS"),
        ("enzymez", "ENZYMES"),
        ("ogbg", "ogbg-molbbbp"),
        ("ogbg_molbbbp", "ogbg-molbbbp"),
    ],
)
def test_dataset_aliases_have_stable_canonical_names(alias, canonical):
    assert normalize_dataset_name(alias) == canonical


def test_every_dataset_has_three_integrity_checked_checkpoints():
    assert set(available_trained_datasets()) == EXPECTED_DATASETS

    resolved = []
    for dataset in available_trained_datasets():
        checkpoints = resolve_trained_checkpoints(dataset)
        assert len(checkpoints) == 3
        assert all(path.is_file() for path in checkpoints)
        resolved.extend(checkpoints)

    assert len(resolved) == 21
    assert len(set(resolved)) == 21


def test_checkpoint_seed_selection_preserves_requested_order():
    checkpoints = resolve_trained_checkpoints("protein", seeds=[2, 0])

    assert [path.parent.name for path in checkpoints] == ["seed_2", "seed_0"]


def test_checkpoint_seed_selection_rejects_missing_or_repeated_seeds():
    with pytest.raises(ValueError, match="unique"):
        resolve_trained_checkpoints("MUTAG", seeds=[0, 0])
    with pytest.raises(ValueError, match="Available seeds"):
        resolve_trained_checkpoints("MUTAG", seeds=[9])


def test_upstream_resolution_prefers_explicit_then_environment(
    tmp_path, monkeypatch
):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    monkeypatch.setenv(trained.CONTRASTIVE_REPOSITORY_ENV, str(configured))

    assert trained.resolve_contrastive_upstream(explicit) == explicit.resolve()
    assert trained.resolve_contrastive_upstream() == configured.resolve()


def test_wrapper_delegates_to_existing_runner(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"checkpoint_count": len(kwargs["checkpoints"])}

    monkeypatch.setattr(
        trained, "evaluate_contrastive_checkpoints", fake_evaluate
    )
    upstream = tmp_path / "upstream"
    result = evaluate_with_trained_gnns(
        dataset="protein",
        generated="generated.pt",
        reference="reference.pt",
        output_dir="report",
        upstream_repository=upstream,
        seeds=[1],
        device="cpu",
    )

    assert result == {"checkpoint_count": 1}
    assert captured["upstream_repository"] == upstream.resolve()
    assert captured["checkpoints"][0].parent.name == "seed_1"
    assert captured["device"] == "cpu"
    assert captured["generated"] == "generated.pt"
    assert captured["reference"] == "reference.pt"
    assert captured["output_dir"] == "report"
