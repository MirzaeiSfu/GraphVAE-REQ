#!/usr/bin/env python3
"""Materialize the effective YAMLs used by the 2026-08-31 motif=False runs."""

from pathlib import Path

import yaml


SOURCE = Path("/tmp/motif_false_src")
OUTPUT = Path("/local-scratch/localhome/mirzaei/MOTIF_FALSE_EFFECTIVE_YAMLS_20260903")
RUN_ROOT = "runs/kia_motif_false_2seed"

# These are the resolved coefficients selected by
# use_graphvae_mm_bce_kl_weights=true. They are recorded as comments because
# the training code derives them from the dataset rather than reading separate
# scalar YAML keys.
WEIGHTS = {
    "grid": (50, 2000),
    "lobster": (40, 2000),
    "triangular_grid": (50, 2000),
    "mutag": (4, 60),
    "proteins": (50, 2000),
}


def dump_config(dataset: str, seed: int) -> None:
    source = SOURCE / f"{dataset}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["data"]["seed"] = seed

    motif = config["motif"]
    motif["motif_loss"] = False
    motif["use_syntactic_literal_rules"] = False

    loss = config["loss"]
    loss["alpha_motif_loss"] = 0.0
    loss["alpha_syntactic_literal_motif_loss"] = 0.0
    loss["use_graphvae_mm_bce_kl_weights"] = True

    runtime = config["runtime"]
    runtime["run_label"] = f"kia-false-{dataset}-seed{seed}"
    runtime["graph_save_path"] = f"{RUN_ROOT}/{dataset}/seed_{seed}"
    runtime["dataset_cache_dir"] = f"{RUN_ROOT}/{dataset}/dataset_cache"
    runtime["third_party_eval_seed"] = seed
    runtime["device"] = "cuda:0"

    bce, kl = WEIGHTS[dataset]
    preamble = (
        "# Effective configuration for the completed motif=False run.\n"
        "# This resolves the command-line overrides used by run_kia_false_queue.sh.\n"
        f"# Resolved adjacency BCE weight: {bce}\n"
        f"# Resolved KL weight: {kl}\n"
        "# Motif loss weight: 0\n"
    )
    target = OUTPUT / dataset / f"seed_{seed}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        preamble + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


OUTPUT.mkdir(parents=True, exist_ok=True)
for dataset in WEIGHTS:
    for seed in (0, 1):
        dump_config(dataset, seed)

print(OUTPUT)
