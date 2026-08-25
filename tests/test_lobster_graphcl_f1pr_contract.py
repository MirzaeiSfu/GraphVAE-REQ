import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "bayesian_optimization"
QUALIFICATION = CONFIG_ROOT / "lobster_graphcl_f1pr_prerequisite_qualification.json"
REPO_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_REPO_PATHS.txt"
PYTHON_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_PYTHON_PATHS.txt"
SLOTS = REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_SLOTS.txt"
CREDENTIAL_PATHS = (
    REPO_ROOT / "CLUSTER_GRAPHVAE_GRAPHCL_F1PR_LOBSTER_CREDENTIAL_ENV_PATHS.txt"
)


def _data_rows(path):
    return [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_graphcl_f1pr_prerequisite_contract_is_test_free():
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["dataset_cache"] == {
        "relative_path": (
            "cache_datasets/LOBSTER_split-paper_70_10_20_train0p7_val0p1_"
            "test0p2_seed123_loaderseed-0_bfs-legacy_first_component_"
            "features-lobster-optimal_v2.pkl"
        ),
        "byte_length": 59295793,
        "sha256": (
            "928852f9402119e6d1f261ef364de5679d7f92f8c6408cf254e03d3dd27a8660"
        ),
        "mode": "0444",
        "split_counts": {"train": 70, "validation": 10, "held_out_test": 20},
        "node_feature_dimension": 14,
        "edge_feature_dimension": 11,
        "test_access": False,
    }
    assert qualification["contrastive_upstream"]["revision"] == (
        "fb6bc26237eb21d7617fd41b22b4bb26ab29bf95"
    )
    assert qualification["selected_host"]["host"] == "cs-cl-09"
    assert qualification["concurrency"]["candidate_max_parallel"] == 2
    assert qualification["concurrency"]["hardware_qualified_for_new_objective"] is False
    assert qualification["checkpoint_reuse"] == {
        "bundled_lobster_graphcl_checkpoint_exists": False,
        "new_training_only_encoder_bundle_required": True,
    }
    assert qualification["execution"] == {
        "graphcl_f1pr_study_created": False,
        "graphcl_encoder_training_started": False,
        "held_out_or_test_access": False,
    }


def test_graphcl_f1pr_cluster_mappings_are_dedicated_and_exact():
    assert _data_rows(REPO_PATHS) == [[
        "cs-cl-09",
        "/local-scratch2/graphvae-req-work/GraphVAE-REQ-lobster-graphcl-f1pr",
    ]]
    assert _data_rows(PYTHON_PATHS) == [[
        "cs-cl-09",
        "/localhome/mirzaei/miniconda3/envs/micro/bin/python",
    ]]
    assert _data_rows(SLOTS) == [
        ["cs-cl-09", "0", "cs-cl-09-lobster-graphcl-gpu0"],
        ["cs-cl-09", "1", "cs-cl-09-lobster-graphcl-gpu1"],
    ]
    assert _data_rows(CREDENTIAL_PATHS) == [[
        "cs-cl-09",
        "/localhome/mirzaei/.graphvae-bo-credentials/lobster-production/worker.env",
    ]]
    credential_path = Path(_data_rows(CREDENTIAL_PATHS)[0][1])
    repository_path = Path(_data_rows(REPO_PATHS)[0][1])
    try:
        credential_path.relative_to(repository_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Credential path must remain outside the repository root")
