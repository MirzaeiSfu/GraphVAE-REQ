import ast
import copy
import csv
from pathlib import Path

import pytest
import yaml

from scripts.tune_graphvae_attribute_weights import (
    PRIMARY_MODE,
    SearchRanges,
    TrialExecutionError,
    build_evaluator_command,
    completed_finite_trials,
    create_or_load_study,
    create_trial_directory,
    ensure_study_definition,
    flatten_config,
    inject_sampled_parameters,
    parse_attr_f1pr_payload,
    remaining_trial_count,
    resolve_trial_config,
    sample_search_space,
    validate_base_config,
    validate_feature_head_keys,
    write_study_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_main_training_plot_is_scoped_to_graph_artifact_directory():
    tree = ast.parse((REPO_ROOT / "main.py").read_text(encoding="utf-8"))
    plotter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "plotter"
        and node.func.attr == "Plotter"
    ]
    assert len(plotter_calls) == 1
    save_keyword = next(
        keyword
        for keyword in plotter_calls[0].keywords
        if keyword.arg == "save_to_filepath"
    )
    assert isinstance(save_keyword.value, ast.BinOp)
    assert isinstance(save_keyword.value.left, ast.Name)
    assert save_keyword.value.left.id == "graph_save_path"
    assert isinstance(save_keyword.value.right, ast.Constant)
    assert save_keyword.value.right.value == "kernelVGAE_Log"


def base_config():
    return {
        "data": {
            "dataset": "QM9",
            "split_mode": "paper_70_10_20",
            "split_seed": 123,
            "train_fraction": 0.7,
            "val_fraction": 0.1,
        },
        "experiment": {"epoch_number": 2, "task": "graphGeneration"},
        "motif": {"motif_loss": False},
        "loss": {
            "alpha_node_feat": 1.0,
            "alpha_edge_feat": 1.0,
            "alpha_motif_loss": 0.25,
            "alpha_adj_recon": 7.0,
        },
        "runtime": {
            "disable_dataset_cache": False,
            "ideal_Evalaution": False,
            "tiny_overfit": False,
            "sanity_check_only": False,
        },
    }


def attributed_payload(split="validation", primary_mode=PRIMARY_MODE):
    return {
        "split": split,
        "primary_mode": primary_mode,
        "graph_counts": {"accepted_per_collection": 12},
        "feature_source": {
            "generated": "GraphVAE node_feature_decoder and edge_feature_decoder",
            "reference": "cached dataset node and edge one-hot attributes",
            "hand_made_topology_features": False,
        },
        "evaluation": {
            "feature_dimensions": {"node": 5, "edge": 3},
            "modes": {
                PRIMARY_MODE: {
                    "summary": {
                        "f1_pr": {"mean": 0.75},
                        "precision": {"mean": 0.8},
                        "recall": {"mean": 0.7},
                    }
                }
            },
        },
    }


class RecordingTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, *, log):
        self.calls.append((name, low, high, log))
        return (low * high) ** 0.5


def test_search_space_sampling_and_parameter_injection():
    trial = RecordingTrial()
    ranges = SearchRanges((1e-3, 1e2), (1e-4, 1e1), None)
    sampled = sample_search_space(trial, ranges)

    assert trial.calls == [
        ("alpha_node_feat", 1e-3, 1e2, True),
        ("alpha_edge_feat", 1e-4, 1e1, True),
    ]
    original = base_config()
    original_copy = copy.deepcopy(original)
    resolved = inject_sampled_parameters(original, sampled)

    assert original == original_copy
    assert resolved["loss"]["alpha_node_feat"] == pytest.approx(sampled["alpha_node_feat"])
    assert resolved["loss"]["alpha_edge_feat"] == pytest.approx(sampled["alpha_edge_feat"])
    assert resolved["loss"]["alpha_motif_loss"] == 0.25
    assert resolved["loss"]["alpha_adj_recon"] == 7.0


def test_optional_motif_weight_is_sampled_only_when_requested():
    trial = RecordingTrial()
    sampled = sample_search_space(
        trial,
        SearchRanges((1e-3, 1e2), (1e-3, 1e2), (1e-2, 1e1)),
    )
    assert set(sampled) == {
        "alpha_node_feat",
        "alpha_edge_feat",
        "alpha_motif_loss",
    }
    assert trial.calls[-1] == ("alpha_motif_loss", 1e-2, 1e1, True)

    config = base_config()
    with pytest.raises(ValueError, match="motif_loss=true"):
        validate_base_config(config, tune_alpha_motif=True)
    config["motif"]["motif_loss"] = True
    validate_base_config(config, tune_alpha_motif=True)


def test_objective_parser_is_structural_and_deterministic():
    first = parse_attr_f1pr_payload(attributed_payload(), expected_split="validation")
    second = parse_attr_f1pr_payload(attributed_payload(), expected_split="validation")

    assert first == second
    assert first.f1_pr == 0.75
    assert first.precision == 0.8
    assert first.recall == 0.7
    assert first.graph_count == 12


def test_validation_split_isolation_and_training_only_config(tmp_path):
    config = base_config()
    validate_base_config(config, tune_alpha_motif=False)
    legacy = copy.deepcopy(config)
    legacy["data"]["split_mode"] = "legacy_80_20"
    with pytest.raises(ValueError, match="distinct"):
        validate_base_config(legacy, tune_alpha_motif=False)

    resolved = resolve_trial_config(
        config,
        {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0},
        trial_number=4,
        trial_directory=tmp_path / "trial_00004",
        training_seed=9,
        split_seed=321,
        device="cpu",
    )
    flat = flatten_config(resolved)
    assert flat["split_seed"] == 321
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["plot_testGraphs"] is False

    command = build_evaluator_command(
        python_bin="python",
        run_dir=tmp_path / "run",
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint",
        output_dir=tmp_path / "evaluation",
        split="validation",
        generation_seed=1,
        evaluator_seed=2,
        evaluator_repeats=3,
        max_graphs=0,
        generation_batch_size=4,
        nearest_k=2,
        adjacency_threshold=0.5,
        device="cpu",
    )
    assert command[command.index("--split") + 1] == "validation"
    assert command[command.index("--modes") + 1] == PRIMARY_MODE
    assert "test" not in command


def test_trial_directories_are_unique(tmp_path):
    first = create_trial_directory(tmp_path, 0)
    second = create_trial_directory(tmp_path, 1)
    assert first != second
    assert first.name == "trial_00000"
    assert second.name == "trial_00001"
    with pytest.raises(FileExistsError):
        create_trial_directory(tmp_path, 0)


def test_resume_rejects_changed_study_definition(tmp_path):
    path = tmp_path / "study_definition.json"
    definition = {"objective": "Attr-F1PR", "training_seed": 0}
    ensure_study_definition(path, definition, existing_trial_count=0)
    ensure_study_definition(path, definition, existing_trial_count=2)
    with pytest.raises(ValueError, match="training_seed"):
        ensure_study_definition(
            path,
            {"objective": "Attr-F1PR", "training_seed": 1},
            existing_trial_count=2,
        )


def test_persistent_study_resumption_does_not_repeat_finished_trials(tmp_path):
    pytest.importorskip("optuna")
    database = tmp_path / "study.sqlite3"
    study = create_or_load_study(
        database_path=database,
        study_name="resume-test",
        sampler_seed=17,
        startup_trials=1,
    )
    study.optimize(lambda trial: trial.suggest_float("x", 1e-3, 1e2, log=True), n_trials=1)
    assert remaining_trial_count(study, 2) == 1

    resumed = create_or_load_study(
        database_path=database,
        study_name="resume-test",
        sampler_seed=17,
        startup_trials=1,
    )
    assert [trial.number for trial in resumed.trials] == [0]
    resumed.optimize(
        lambda trial: trial.suggest_float("x", 1e-3, 1e2, log=True),
        n_trials=remaining_trial_count(resumed, 2),
    )
    assert [trial.number for trial in resumed.trials] == [0, 1]
    assert resumed.trials[1].params != resumed.trials[0].params
    assert remaining_trial_count(resumed, 2) == 0


def test_failed_trial_is_preserved_and_best_config_is_written(tmp_path):
    optuna = pytest.importorskip("optuna")
    database = tmp_path / "study.sqlite3"
    study = create_or_load_study(
        database_path=database,
        study_name="failure-test",
        sampler_seed=4,
        startup_trials=1,
    )
    config_path = tmp_path / "winning_config.yaml"
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.write_bytes(b"checkpoint")

    def objective(trial):
        node_weight = trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
        edge_weight = trial.suggest_float("alpha_edge_feat", 1e-3, 1e2, log=True)
        if trial.number == 0:
            trial.set_user_attr("failure_reason", "synthetic failure")
            raise RuntimeError("synthetic failure")
        config = base_config()
        config["loss"]["alpha_node_feat"] = node_weight
        config["loss"]["alpha_edge_feat"] = edge_weight
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        trial.set_user_attr("resolved_config", str(config_path))
        trial.set_user_attr("checkpoint", str(checkpoint_path))
        trial.set_user_attr("checkpoint_sha256", "dummy")
        trial.set_user_attr("validation_precision", 0.8)
        trial.set_user_attr("validation_recall", 0.7)
        trial.set_user_attr("accepted_validation_graphs", 10)
        return 0.75

    study.optimize(objective, n_trials=2, catch=(RuntimeError,))
    best = write_study_outputs(study, output_dir=tmp_path, database_path=database)

    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert len(completed_finite_trials(study)) == 1
    assert best.number == 1
    assert (tmp_path / "best_config.yaml").is_file()
    assert (tmp_path / "best_trial.json").is_file()
    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["state"] == "FAIL"
    assert rows[0]["failure_reason"] == "synthetic failure"
    assert rows[1]["state"] == "COMPLETE"
    written = yaml.safe_load((tmp_path / "best_config.yaml").read_text(encoding="utf-8"))
    assert written["loss"]["alpha_node_feat"] == pytest.approx(best.params["alpha_node_feat"])
    assert written["loss"]["alpha_edge_feat"] == pytest.approx(best.params["alpha_edge_feat"])


def test_topology_only_and_missing_feature_checkpoints_are_rejected():
    with pytest.raises(TrialExecutionError, match="node_feature_decoder"):
        validate_feature_head_keys(["decode.layers.0.weight"])
    with pytest.raises(TrialExecutionError, match="edge_feature_decoder"):
        validate_feature_head_keys(["node_feature_decoder.net.0.weight"])
    validate_feature_head_keys(
        [
            "node_feature_decoder.net.0.weight",
            "edge_feature_decoder.net.0.weight",
        ]
    )

    topology_payload = attributed_payload(primary_mode="topology_control")
    with pytest.raises(TrialExecutionError, match="decoded_node_edge"):
        parse_attr_f1pr_payload(topology_payload, expected_split="validation")
    missing_edges = attributed_payload()
    missing_edges["evaluation"]["feature_dimensions"]["edge"] = 0
    with pytest.raises(TrialExecutionError, match="positive matching"):
        parse_attr_f1pr_payload(missing_edges, expected_split="validation")


def test_evaluator_output_for_test_cannot_be_parsed_as_validation():
    with pytest.raises(TrialExecutionError, match="Expected 'validation'"):
        parse_attr_f1pr_payload(attributed_payload(split="test"), expected_split="validation")
