import json
from pathlib import Path

from factorbase_motif_pipeline.tu_dataset_to_db import TU_DATASET_SPECS
from scripts.graphvae_attr_bo_distributed import parse_slots
from scripts.resample_grid_checkpoints import (
    build_dataset_cache_metadata,
    build_dataset_cache_name,
)
from scripts.run_distributed_graphvae_attr_bo import _load_mapping
from scripts.tune_graphvae_attribute_weights import (
    flatten_config,
    load_yaml_mapping,
    validate_base_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_graphvae_attr_f1pr_calibration.yaml"
)
PROVENANCE_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_dataset_provenance.json"
)
CACHE_MANIFEST_PATH = (
    REPO_ROOT
    / "configs"
    / "bayesian_optimization"
    / "aids_attr_f1pr_cache_manifest.json"
)
REPO_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_REPO_PATHS.txt"
PYTHON_PATHS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_PYTHON_PATHS.txt"
CREDENTIAL_PATHS = (
    REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_CREDENTIAL_ENV_PATHS.txt"
)
SLOTS = REPO_ROOT / "CLUSTER_GRAPHVAE_ATTR_BO_AIDS_SLOTS.txt"


def test_aids_calibration_config_is_validation_only_and_bounded():
    config = load_yaml_mapping(CONFIG_PATH)
    validate_base_config(config, tune_alpha_motif=False)
    flat = flatten_config(config)

    assert flat["dataset"] == "AIDS"
    assert flat["split_mode"] == "paper_70_10_20"
    assert flat["train_fraction"] == 0.7
    assert flat["val_fraction"] == 0.1
    assert flat["split_seed"] == 123
    assert flat["dataset_loader_seed"] == 0
    assert flat["bfs_strategy"] == "all_components"
    assert flat["tu_attribute_bins"] == 8
    assert flat["tu_max_nodes"] == 40
    assert flat["data_dir"] is None
    assert flat["expected_total_graphs"] == 1849
    assert flat["max_graphs"] == 0
    assert flat["alpha_node_feat"] == 1.0
    assert flat["alpha_edge_feat"] == 1.0
    assert flat["require_existing_dataset_cache"] is True
    assert flat["skip_final_evaluation"] is True
    assert flat["third_party_eval"] is False
    assert flat["ideal_Evalaution"] is False
    assert flat["tiny_overfit"] is False


def test_aids_cache_identity_is_exact():
    flat = flatten_config(load_yaml_mapping(CONFIG_PATH))
    metadata = build_dataset_cache_metadata(flat)

    assert metadata == {
        "cache_schema_version": "dataset-cache-v4",
        "dataset": "AIDS",
        "split_mode": "paper_70_10_20",
        "bfs_strategy": "all_components",
        "split_kind": "three_way",
        "train_fraction": 0.7,
        "val_fraction": 0.1,
        "test_fraction": 0.20000000000000004,
        "split_seed": 123,
        "dataset_loader_seed": 0,
        "feature_schema": "tu-quantile8-max40",
    }
    assert build_dataset_cache_name(metadata) == (
        "AIDS_split-paper_70_10_20_train0p7_val0p1_test0p2_seed123_"
        "loaderseed-0_bfs-all_components_features-tu-quantile8-max40.pkl"
    )


def test_aids_provenance_requires_real_edge_labels_and_exact_splits():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert TU_DATASET_SPECS["AIDS"].has_edge_labels is True
    assert provenance["source"]["dataset"] == "AIDS"
    assert provenance["expected_contract"]["source_graphs"] == 2000
    assert provenance["expected_contract"]["retained_graphs"] == 1849
    assert provenance["expected_contract"]["split_counts"] == {
        "train": 1294,
        "validation": 184,
        "test": 371,
    }
    assert provenance["expected_contract"]["edge_feature_fields"] == {
        "edge_label": [0, 1, 2]
    }
    assert len(provenance["archive"]["sha256"]) == 64
    assert set(provenance["files"]) == {
        "AIDS_A.txt",
        "AIDS_edge_labels.txt",
        "AIDS_graph_indicator.txt",
        "AIDS_graph_labels.txt",
        "AIDS_node_attributes.txt",
        "AIDS_node_labels.txt",
    }
    assert all(len(value) == 64 for value in provenance["files"].values())


def test_aids_cache_manifest_freezes_full_attributed_validation_contract():
    manifest = json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["relative_path"] == (
        "cache_datasets/AIDS_split-paper_70_10_20_train0p7_val0p1_"
        "test0p2_seed123_loaderseed-0_bfs-all_components_"
        "features-tu-quantile8-max40.pkl"
    )
    assert manifest["byte_length"] == 73822456
    assert manifest["sha256"] == (
        "6edcc3309fb1c3d366b0f87065aa1b2e2c7d23cbff92bc729053f44e874909bb"
    )
    assert {
        split: manifest["splits"][split]["graph_count"]
        for split in ("train", "validation", "test")
    } == {"train": 1294, "validation": 184, "test": 371}
    assert manifest["expected_validation_graphs"] == 184
    assert manifest["node_feature_dimension"] == 56
    assert manifest["edge_feature_dimension"] == 3
    assert manifest["node_schema_fingerprint"] == (
        "630400b38c74e0a51e505e57b7e41ee986deea1eb06e27099ea792cf6876c2c9"
    )
    assert manifest["edge_schema_fingerprint"] == (
        "1fee16532276d512d7143669f2d3c80a0140ee3c2ded035fa206c323320bf772"
    )
    assert len(manifest["splits"]["validation"]["graph_fingerprints"]) == 184


def test_aids_deployment_mappings_are_dedicated_and_homogeneous():
    repositories = _load_mapping(REPO_PATHS)
    pythons = _load_mapping(PYTHON_PATHS)
    credentials = _load_mapping(CREDENTIAL_PATHS)
    slots = parse_slots(SLOTS, known_hosts=sorted(repositories))

    assert set(repositories) == {"cs-cl-13", "cs-cl-17"}
    assert set(repositories) == set(pythons) == set(credentials)
    assert all(
        path.endswith("GraphVAE-REQ-aids-attr-bo")
        for path in repositories.values()
    )
    assert all("lobster" not in path.lower() for path in repositories.values())
    assert all(path.endswith("/envs/micro/bin/python") for path in pythons.values())
    assert all(
        path.endswith("/aids-production/worker.env")
        for path in credentials.values()
    )
    assert [(slot["host"], slot["physical_gpu"]) for slot in slots] == [
        ("cs-cl-13", 0),
        ("cs-cl-17", 0),
        ("cs-cl-17", 1),
    ]
    assert all("aids-prod" in slot["worker_id"] for slot in slots)
