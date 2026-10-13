#!/usr/bin/env python3
"""Generate the fixed configs for the 2026-07-23 GraphVAE-MM count runs."""

from copy import deepcopy
from pathlib import Path

import yaml


OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "experiments"
    / "count_s02_s09_20260723"
)

DATASETS = {
    "aids": {
        "dataset": "AIDS",
        "directed": False,
        "database_name": "aids_undir_feat",
        "train_batch_size": 200,
        "motif_batch_size": 50,
        "motif_cache_dir": "cache_motifs",
        "motif_prune_max_total_values": 100,
    },
    "enzymes": {
        "dataset": "ENZYMES",
        "directed": False,
        "database_name": "enzymes_top4_undir_feat",
        "train_batch_size": 200,
        "motif_batch_size": 50,
        "motif_cache_dir": "cache_motifs",
        "motif_prune_max_total_values": 100,
    },
    "mutag": {
        "dataset": "MUTAG",
        "directed": True,
        "database_name": "mutag_undir_feat",
        "train_batch_size": 131,
        "motif_batch_size": 131,
        "motif_cache_dir": "cache_motifs",
        "motif_prune_max_total_values": 11,
    },
    "ptc": {
        "dataset": "PTC",
        "directed": True,
        "database_name": "ptc_undir_feat",
        "train_batch_size": 240,
        "motif_batch_size": 240,
        "motif_cache_dir": "cache_motifs",
        "motif_prune_max_total_values": 43,
    },
    "proteins": {
        "dataset": "PROTEINS",
        "directed": True,
        "database_name": "proteins_dir_feat_snap_0094da",
        "train_batch_size": 200,
        "motif_batch_size": 50000,
        "motif_cache_dir": "cache_motifs",
        # Deliberately unpruned: all 13 original motif/value combinations.
        "motif_prune_max_total_values": None,
    },
    "ogbg_molbbbp": {
        "dataset": "ogbg-molbbbp",
        "directed": True,
        "database_name": "ogbg_molbbbp_feat_snap",
        "train_batch_size": 500,
        "motif_batch_size": 100000,
        "motif_cache_dir": "cache_motifs_ogb_cap8",
        "motif_prune_max_total_values": 99,
    },
}


def make_config(dataset_key, dataset_spec, setting):
    motif_enabled = setting == "09"
    # PROTEINS is the requested exception: keep every original cache row.
    rule_prune = motif_enabled and dataset_key != "proteins"

    return {
        "data": {
            "dataset": dataset_spec["dataset"],
            "use_feature": True,
            "bfsOrdering": True,
            "bfs_strategy": "legacy_first_component",
            "directed": dataset_spec["directed"],
            "database_name": dataset_spec["database_name"],
            "graph_type": "homogeneous",
            "graph_index_start": None,
            "graph_index_end": None,
            "data_dir": "data_raw",
            "tu_attribute_bins": 8,
            "tu_max_nodes": None,
            "split_mode": "paper_70_10_20",
            "split_seed": 123,
            "train_fraction": 0.7,
            "val_fraction": 0.1,
            "seed": 0,
            "deterministic": True,
            "deterministic_warn_only": False,
        },
        "model": {
            "model": "GraphVAE-MM",
            "encoder_type": "AvePool",
            "decoder": "FC",
            "graphEmDim": 1024,
            "beta": None,
        },
        "experiment": {
            "epoch_number": 20000,
            "Vis_step": 1000,
            "redraw": False,
            "lr": 0.0003,
            "train_batch_size": dataset_spec["train_batch_size"],
            "task": "graphGeneration",
        },
        "motif": {
            "motif_loss": motif_enabled,
            "motif_output_mode": "count",
            "use_syntactic_literal_rules": False,
            "syntactic_literal_rule_mode": "original",
            "motif_loss_mode": "calibrated_gaussian",
            "motif_temperature_start": 1.0,
            "motif_temperature_end": 1.0,
            "motif_temperature_anneal_start_frac": 1.0,
            "motif_temperature_guard_ratio": 0.0,
            "rule_prune": rule_prune,
            "motif_prune_max_values_per_rule": None,
            "motif_prune_max_total_values": (
                dataset_spec["motif_prune_max_total_values"]
                if rule_prune
                else None
            ),
            "motif_batch_size": dataset_spec["motif_batch_size"],
            "prepare_motif_cache_only": False,
        },
        "loss": {
            "alpha_kernel_cost": 1.0,
            "alpha_node_feat": 40.0,
            "alpha_edge_feat": 40.0,
            "alpha_motif_loss": 1.0 if motif_enabled else 0.0,
            "alpha_syntactic_literal_motif_loss": (
                1.0 if motif_enabled else 0.0
            ),
            "use_graphvae_mm_bce_kl_weights": False,
            "alpha_adj_recon": 0.0,
        },
        "runtime": {
            "run_label": (
                f"{dataset_key}-setting-{setting}-graphvae-mm-count"
                + ("-all13-original-combinations" if dataset_key == "proteins" and motif_enabled else "")
            ),
            "graph_save_path": None,
            "dataset_cache_dir": None,
            "disable_dataset_cache": False,
            "motif_cache_dir": dataset_spec["motif_cache_dir"],
            "PATH": "model",
            "UseGPU": True,
            "device": "cuda:0",
            "plot_testGraphs": True,
            "ideal_Evalaution": False,
            "keep_best_validation_mmd": True,
            "best_validation_mmd_metric": "normalized_table2_table3",
            "save_validation_checkpoints": False,
            "checkpoint_interval_epochs": 1000,
            "third_party_eval": True,
            "third_party_eval_repeats": 10,
            "third_party_eval_max_graphs": 1000,
            "third_party_eval_seed": 0,
            "third_party_eval_device": "same",
            "third_party_eval_structural_features": True,
            "third_party_eval_json_filename": "graph_realism_random_gin.json",
            "third_party_eval_strict": True,
            "tiny_overfit": False,
            "tiny_overfit_size": 32,
            "interactive": False,
            "sanity_check": False,
            "sanity_check_only": False,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key, dataset_spec in DATASETS.items():
        for setting in ("02", "09"):
            config = make_config(dataset_key, deepcopy(dataset_spec), setting)
            output_path = OUTPUT_DIR / f"{dataset_key}_setting_{setting}.yaml"
            output_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            print(output_path)


if __name__ == "__main__":
    main()
