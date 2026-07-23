import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from scripts import select_lobster_checkpoints_per_run as posthoc


def valid_posthoc_options(**overrides):
    values = {
        "third_party_repeats": 10,
        "third_party_max_graphs": 1000,
        "model_filename": posthoc.DEFAULT_MODEL_FILENAME,
        "metadata_filename": posthoc.DEFAULT_METADATA_FILENAME,
        "generated_filename": posthoc.DEFAULT_GENERATED_FILENAME,
        "reference_filename": posthoc.DEFAULT_REFERENCE_FILENAME,
        "third_party_json_filename": posthoc.DEFAULT_THIRD_PARTY_JSON_FILENAME,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def candidate(run_dir: Path, checkpoint: str, score: float) -> dict:
    return {
        "run": run_dir.name,
        "artifact_dir": str(run_dir),
        "validation_graphs": str(run_dir / "validationGraphs_adj_.npy"),
        "checkpoint": checkpoint,
        "checkpoint_path": str(run_dir / checkpoint),
        "dense_threshold": 10.0,
        "reference_mean_edges": 5.0,
        "edge_mean_log_error": 0.0,
        "selection_score": score,
        "validation": {
            "score": {"mean": score, "std": 0.1, "median": score},
            "dense_rate": 0.0,
            "mean_raw_edges": {"mean": 5.0},
        },
    }


def test_select_best_per_run_does_not_collapse_to_one_global_winner(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    candidates = [
        candidate(run_a, "periodic_epoch_04000.pt", 2.0),
        candidate(run_a, "periodic_epoch_08000.pt", 1.0),
        candidate(run_b, "periodic_epoch_04000.pt", 0.5),
        candidate(run_b, "periodic_epoch_08000.pt", 0.8),
    ]

    winners = posthoc.select_best_per_run(candidates)

    assert len(winners) == 2
    assert {winner["checkpoint"] for winner in winners} == {
        "periodic_epoch_08000.pt",
        "periodic_epoch_04000.pt",
    }
    assert {winner["artifact_dir"] for winner in winners} == {
        str(run_a),
        str(run_b),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("third_party_repeats", 0, "repeats must be positive"),
        ("third_party_repeats", -1, "repeats must be positive"),
        ("third_party_max_graphs", 0, "max-graphs must be positive"),
        ("third_party_max_graphs", -1, "max-graphs must be positive"),
    ],
)
def test_posthoc_options_require_positive_random_gin_counts(field, value, message):
    with pytest.raises(ValueError, match=message):
        posthoc.validate_posthoc_options(valid_posthoc_options(**{field: value}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"generated_filename": posthoc.DEFAULT_REFERENCE_FILENAME},
        {"metadata_filename": posthoc.DEFAULT_MODEL_FILENAME},
        {"third_party_json_filename": posthoc.DEFAULT_REFERENCE_FILENAME},
        {"generated_filename": posthoc.VALIDATION_REFERENCE_FILENAME},
    ],
)
def test_posthoc_options_reject_artifact_filename_collisions(overrides):
    with pytest.raises(ValueError, match="collide"):
        posthoc.validate_posthoc_options(valid_posthoc_options(**overrides))


def test_posthoc_options_reject_validation_graphs_as_heldout_reference():
    with pytest.raises(ValueError, match="must not name the validation input"):
        posthoc.validate_posthoc_options(
            valid_posthoc_options(
                reference_filename=posthoc.VALIDATION_REFERENCE_FILENAME
            )
        )


@pytest.mark.parametrize(
    "unsafe_name",
    ["nested/output.npy", "../output.npy", "/tmp/output.npy"],
)
def test_posthoc_options_reject_nonlocal_artifact_paths(unsafe_name):
    with pytest.raises(ValueError, match="plain filename"):
        posthoc.validate_posthoc_options(
            valid_posthoc_options(generated_filename=unsafe_name)
        )


def test_run_freezes_selection_manifest_before_materializing_heldout(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_a"
    selected = candidate(run_dir, "periodic_epoch_08000.pt", 1.0)
    output_dir = tmp_path / "selection"
    observed = []

    monkeypatch.setattr(posthoc, "select_candidates", lambda args, device: [selected])

    def fake_materialize(winners, **kwargs):
        frozen_path = output_dir / "selection.json"
        assert frozen_path.is_file()
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen["selection_frozen_before_heldout_load"] is True
        assert frozen["heldout_materialized"] is False
        assert frozen["winners"][0]["checkpoint"] == "periodic_epoch_08000.pt"
        observed.append("heldout_phase")
        return [{"artifact_dir": str(run_dir), "run": run_dir.name}]

    monkeypatch.setattr(posthoc, "materialize_selected_runs", fake_materialize)
    args = SimpleNamespace(
        runs_root=[tmp_path],
        output_dir=output_dir,
        validation_rollouts=3,
        seed=123,
        generation_seed=456,
        latent_dim=1024,
        stability_weight=0.25,
        dense_penalty_weight=1.0,
        edge_mean_penalty_weight=0.25,
        device="cpu",
        skip_materialization=False,
        model_filename=posthoc.DEFAULT_MODEL_FILENAME,
        metadata_filename=posthoc.DEFAULT_METADATA_FILENAME,
        generated_filename=posthoc.DEFAULT_GENERATED_FILENAME,
        reference_filename=posthoc.DEFAULT_REFERENCE_FILENAME,
        run_third_party_eval=False,
        third_party_repeats=2,
        third_party_max_graphs=100,
        third_party_seed=0,
        third_party_device="cpu",
        third_party_json_filename=posthoc.DEFAULT_THIRD_PARTY_JSON_FILENAME,
        no_third_party_structural_features=False,
    )

    payload = posthoc.run(args)

    assert observed == ["heldout_phase"]
    assert payload["heldout_materialized"] is True
    assert (output_dir / "run_random_gin.sh").is_file()


def test_materialization_copies_only_selected_model_and_writes_random_gin_input(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    selected_path = run_dir / "periodic_epoch_08000.pt"
    selected_path.write_bytes(b"selected checkpoint")
    reference_path = run_dir / posthoc.DEFAULT_REFERENCE_FILENAME
    np.save(
        reference_path,
        np.array([np.eye(3), np.eye(4)], dtype=object),
        allow_pickle=True,
    )
    selected = candidate(run_dir, selected_path.name, 1.0)
    loaded = []

    def fake_load_decoder(path, device, latent_dim):
        loaded.append(Path(path))
        return object()

    def fake_generate(decoder, count, latent_dim, device, seed):
        assert count == 2
        assert seed == 999
        # Equal shapes catch accidental (N, n, n) object arrays, which the
        # Random-GIN loader cannot convert back to NetworkX graphs.
        graphs = [nx.path_graph(3), nx.path_graph(3)]
        return graphs, graphs

    monkeypatch.setattr(posthoc.legacy_selector, "load_decoder", fake_load_decoder)
    monkeypatch.setattr(posthoc.legacy_selector, "generate", fake_generate)

    materialized = posthoc.materialize_selected_runs(
        [selected],
        device=torch.device("cpu"),
        latent_dim=1024,
        generation_seed=999,
        model_filename=posthoc.DEFAULT_MODEL_FILENAME,
        metadata_filename=posthoc.DEFAULT_METADATA_FILENAME,
        generated_filename=posthoc.DEFAULT_GENERATED_FILENAME,
        reference_filename=posthoc.DEFAULT_REFERENCE_FILENAME,
    )

    assert loaded == [selected_path]
    model_copy = run_dir / posthoc.DEFAULT_MODEL_FILENAME
    assert model_copy.read_bytes() == b"selected checkpoint"
    generated = np.load(
        run_dir / posthoc.DEFAULT_GENERATED_FILENAME, allow_pickle=True
    )
    assert generated.shape == (2,)
    assert len(generated) == 2
    from scripts.evaluate_graph_realism_batch import item_to_graph

    assert item_to_graph(generated[0]).number_of_edges() == 2
    metadata = json.loads(
        (run_dir / posthoc.DEFAULT_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["selection_split"] == "validation"
    assert metadata["selection_frozen_before_heldout_load"] is True
    assert metadata["num_generated_graphs"] == 2
    assert materialized[0]["selected_checkpoint"] == str(selected_path)


def test_random_gin_command_uses_posthoc_graphs_and_heldout_references(tmp_path):
    run_dirs = [tmp_path / "run a", tmp_path / "run_b"]
    command = posthoc.build_random_gin_command(
        run_dirs,
        generated_filename=posthoc.DEFAULT_GENERATED_FILENAME,
        reference_filename=posthoc.DEFAULT_REFERENCE_FILENAME,
        json_filename=posthoc.DEFAULT_THIRD_PARTY_JSON_FILENAME,
        summary_csv=tmp_path / "summary.csv",
        repeats=10,
        max_graphs=1000,
        seed=0,
        device="cuda",
        structural_features=True,
        python_executable="python",
    )

    assert command.count("--run-dir") == 2
    assert posthoc.DEFAULT_GENERATED_FILENAME in command
    assert posthoc.DEFAULT_REFERENCE_FILENAME in command
    assert "Single_comp_generatedGraphs_adj_final_eval.npy" not in command
    assert "--no-structural-features" not in command
