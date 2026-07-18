#!/usr/bin/env python3
"""Freshly regenerate and evaluate the three archived PROTEINS best models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.chdir(ROOT)

# stat_rnn reads these settings at import time.
os.environ.setdefault("GRAPHVAE_GGM_GNN_EVAL_RUNS", "10")
os.environ.setdefault("GRAPHVAE_GGM_GNN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

from evaluate_graph_realism_batch import evaluate_graph_collections  # noqa: E402
from reproduce_table2_grid import locked_orca_tmp, to_graphs  # noqa: E402
from resample_grid_checkpoints import (  # noqa: E402
    build_model,
    generate_graphs,
    load_cached_dataset,
    load_config,
)
from stat_rnn import mmd_eval  # noqa: E402


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LABELS = {
    "degree": "degree",
    "clustering": "clustering",
    "orbit": "orbits",
    "spectral": "Spec",
    "diameter": "diameter",
    "mmd_rbf": "mmd_rbf",
    "mmd_rbf_std": "mmd_rbf_std",
    "precision": "precision",
    "precision_std": "precision_std",
    "recall": "recall",
    "recall_std": "recall_std",
    "f1_pr": "f1_pr",
    "f1_pr_std": "f1_pr_std",
}


def parse_metrics(result: str) -> dict[str, float | None]:
    parsed = {}
    for name, label in LABELS.items():
        match = re.search(rf"{re.escape(label)}\s*:\s*({FLOAT})", result)
        parsed[name] = float(match.group(1)) if match else None
    return parsed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_graphs(path: Path, graphs: list[nx.Graph]) -> None:
    matrices = np.array([nx.to_numpy_array(graph) for graph in graphs], dtype=object)
    np.save(path, matrices, allow_pickle=True)


def local_evaluate(generated: list[nx.Graph], reference: list[nx.Graph], seed: int) -> dict:
    seed_everything(seed)
    raw = mmd_eval([nx.Graph(graph) for graph in generated], [nx.Graph(graph) for graph in reference], diam=True)
    return {"metrics": parse_metrics(raw), "raw_result": raw}


def evaluate_seed(
    seed: int,
    archive_root: Path,
    cache_dir: Path,
    output_root: Path,
    device: torch.device,
    generation_seed: int,
    gin_repeats: int,
) -> dict:
    run_dir = archive_root / f"seed_{seed}"
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir / "run_config_used.yaml")
    config["dataset_cache_dir"] = str(cache_dir)
    cache = load_cached_dataset(config)
    validation_refs = to_graphs(cache["val_adj"], keep_largest_component=False)
    test_refs = to_graphs(cache["test_list_adj"], keep_largest_component=False)

    model = build_model(config, cache, device)
    checkpoint = run_dir / "best_validation_mmd_model"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    generated_by_split = {}
    references = {"validation": validation_refs, "test": test_refs}
    for split_index, (split, refs) in enumerate(references.items()):
        split_seed = generation_seed + split_index * 100_000
        seed_everything(split_seed)
        generated = generate_graphs(model, len(refs), device)["largest_component"]
        generated_by_split[split] = generated
        save_graphs(output_dir / f"generated_{split}.npy", generated)
        save_graphs(output_dir / f"reference_{split}.npy", refs)

    with locked_orca_tmp():
        validation_local = local_evaluate(
            generated_by_split["validation"], validation_refs, generation_seed + 200_000
        )
        test_local = local_evaluate(
            generated_by_split["test"], test_refs, generation_seed + 300_000
        )

    seed_everything(generation_seed + 400_000)
    third_party = evaluate_graph_collections(
        generated_graphs=generated_by_split["test"],
        reference_graphs=test_refs,
        repeats=gin_repeats,
        seed=generation_seed + 400_000,
        device=device,
        use_structural_features=True,
    )

    payload = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "generation_seed": generation_seed,
        "validation_graphs": len(generated_by_split["validation"]),
        "test_graphs": len(generated_by_split["test"]),
        "validation": validation_local,
        "test": test_local,
        "third_party": third_party,
    }
    (output_dir / "fresh_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def flatten(payload: dict) -> dict:
    val = payload["validation"]["metrics"]
    test = payload["test"]["metrics"]
    third = payload["third_party"]["metrics"]
    return {
        "seed": payload["seed"],
        "val_f1_pr": val["f1_pr"],
        "val_mmd_rbf": val["mmd_rbf"],
        "test_f1_pr": test["f1_pr"],
        "test_mmd_rbf": test["mmd_rbf"],
        "precision": test["precision"],
        "recall": test["recall"],
        "degree": test["degree"],
        "clustering": test["clustering"],
        "orbit": test["orbit"],
        "spectral": test["spectral"],
        "diameter": test["diameter"],
        "third_party_f1_pr": third["f1_pr"]["mean"],
        "third_party_mmd_rbf": third["mmd_rbf"]["mean"],
        "third_party_linear_mmd_trimmed": third["mmd_linear"]["trimmed_mean"],
    }


def write_summary(output_root: Path, payloads: list[dict]) -> None:
    rows = [flatten(payload) for payload in payloads]
    fields = list(rows[0])
    with (output_root / "fresh_requested_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Freshly recomputed PROTEINS metrics",
        "",
        "All values below were recomputed from newly generated graphs; no previous metric JSON was reused.",
        "",
        "| Seed | Val F1-PR ↑ | Val MMD RBF ↓ | Test F1-PR ↑ | Test MMD RBF ↓ | Precision ↑ | Recall ↑ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['val_f1_pr']:.6f} | {row['val_mmd_rbf']:.6f} | "
            f"{row['test_f1_pr']:.6f} | {row['test_mmd_rbf']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} |"
        )
    lines += [
        "",
        "| Seed | Degree ↓ | Clustering ↓ | Orbit ↓ | Spectral ↓ | Diameter ↓ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['degree']:.6f} | {row['clustering']:.6f} | "
            f"{row['orbit']:.6f} | {row['spectral']:.6f} | {row['diameter']:.6f} |"
        )
    lines += [
        "",
        "| Seed | Third-party F1-PR ↑ | Third-party MMD RBF ↓ | Third-party linear MMD trimmed ↓ |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['third_party_f1_pr']:.6f} | "
            f"{row['third_party_mmd_rbf']:.6f} | "
            f"{row['third_party_linear_mmd_trimmed']:.6f} |"
        )
    (output_root / "fresh_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-seed", type=int, default=20260716)
    parser.add_argument("--gin-repeats", type=int, default=10)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payloads = []
    for seed in range(3):
        print(f"[fresh] seed {seed}", flush=True)
        payloads.append(
            evaluate_seed(
                seed, args.archive_root, args.cache_dir, args.output_root,
                device, args.generation_seed, args.gin_repeats,
            )
        )
    write_summary(args.output_root, payloads)


if __name__ == "__main__":
    main()
