import json

import pytest

from scripts.evaluate_lobster_frozen_selections import (
    aggregate_conditions,
    load_frozen_winners,
    numeric_summary,
)


def test_numeric_summary_uses_sample_std_only_for_seed_level_values():
    population = numeric_summary([1.0, 2.0, 3.0])
    seed_level = numeric_summary([1.0, 2.0, 3.0], sample_std=True)

    assert population["std"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert seed_level["std"] == pytest.approx(1.0)


def test_load_frozen_winners_maps_remote_relative_run_to_local_collection(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = (
        runs_root
        / "seed_0"
        / "lobster_kiarash_parity_kia40_2000_legacy__worker_gpu0"
        / "seed_0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "periodic_epoch_12000.pt").write_bytes(b"checkpoint")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "winners": [
                    {
                        "run": str(run_dir.relative_to(runs_root)),
                        "artifact_dir": "/remote/path/that/does/not/exist",
                        "checkpoint": "periodic_epoch_12000.pt",
                        "selection_score": 0.25,
                        "validation": {"score": {"median": 0.2}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    winners = load_frozen_winners(
        [selection_path],
        [tmp_path / "empty", runs_root],
        expected_runs=1,
    )

    assert len(winners) == 1
    assert winners[0]["condition"] == (
        "lobster_kiarash_parity_kia40_2000_legacy"
    )
    assert winners[0]["seed"] == 0
    assert winners[0]["run_dir"] == str(run_dir.resolve())


def test_load_frozen_winners_uses_source_root_name_to_disambiguate(tmp_path):
    run = (
        "seed_1/"
        "lobster_kiarash_parity_kia40_2000_legacy__worker_gpu0/"
        "seed_1"
    )
    original_root = tmp_path / "lobster_kiarash_parity"
    fixed_root = tmp_path / "lobster_kiarash_parity_fixed_split"
    for runs_root in (original_root, fixed_root):
        run_dir = runs_root / run
        run_dir.mkdir(parents=True)
        (run_dir / "periodic_epoch_20000.pt").write_bytes(b"checkpoint")

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "winners": [
                    {
                        "run": run,
                        "artifact_dir": (
                            "/remote/repo/runs/20260724/"
                            f"{fixed_root.name}/{run}"
                        ),
                        "checkpoint": "periodic_epoch_20000.pt",
                        "selection_score": 0.25,
                        "validation": {"score": {"median": 0.2}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    winners = load_frozen_winners(
        [selection_path],
        [original_root, fixed_root],
        expected_runs=1,
    )

    assert winners[0]["run_dir"] == str((fixed_root / run).resolve())


def test_load_frozen_winners_accepts_custom_complete_condition_matrix(tmp_path):
    runs_root = tmp_path / "runs"
    winners = []
    condition_order = ("native_kernel", "motif_bundle")
    for condition in condition_order:
        for seed in range(3):
            run_dir = (
                runs_root
                / f"seed_{seed}"
                / f"{condition}__worker_gpu0"
                / f"seed_{seed}"
            )
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "periodic_epoch_20000.pt"
            checkpoint.write_bytes(b"checkpoint")
            winners.append(
                {
                    "run": str(run_dir.relative_to(runs_root)),
                    "artifact_dir": str(run_dir),
                    "checkpoint": checkpoint.name,
                    "selection_score": 0.25,
                    "validation": {"score": {"median": 0.2}},
                }
            )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"winners": winners}),
        encoding="utf-8",
    )

    loaded = load_frozen_winners(
        [selection_path],
        [runs_root],
        expected_runs=6,
        condition_order=condition_order,
    )

    assert [
        (winner["condition"], winner["seed"])
        for winner in loaded
    ] == [
        (condition, seed)
        for condition in condition_order
        for seed in range(3)
    ]


def test_aggregate_conditions_reports_seed_and_rollout_uncertainty():
    run_results = []
    for seed, degree_values in enumerate(([1.0, 3.0], [2.0, 4.0])):
        rollouts = [
            {
                "metrics": {
                    "degree": degree,
                    "clustering": degree + 1.0,
                    "orbit": degree + 2.0,
                    "spectral": degree + 3.0,
                    "diameter": degree + 4.0,
                }
            }
            for degree in degree_values
        ]
        run_results.append(
            {
                "condition": "example",
                "seed": seed,
                "rollouts": rollouts,
                "test_summary": {
                    "metrics": {
                        metric: {"mean": sum(row["metrics"][metric] for row in rollouts) / 2}
                        for metric in (
                            "degree",
                            "clustering",
                            "orbit",
                            "spectral",
                            "diameter",
                        )
                    },
                    "lcc_nodes": {"mean": 10.0 + seed},
                    "raw_nodes": {"mean": 12.0 + seed},
                    "lcc_edges": {"mean": 9.0 + seed},
                },
                "reference_nodes": {"mean": 20.0},
                "reference_edges": {"mean": 19.0},
            }
        )

    summary = aggregate_conditions(run_results)["example"]

    assert summary["metrics_across_all_rollouts"]["degree"]["mean"] == 2.5
    assert summary["metrics_across_seed_means"]["degree"]["mean"] == 2.5
    assert summary["metrics_across_seed_means"]["degree"]["std"] == pytest.approx(
        2 ** -0.5
    )
